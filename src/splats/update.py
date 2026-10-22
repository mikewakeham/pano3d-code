from PIL import Image
from typing import Optional
from splats.lifting import convert_rgbd_equi_to_3dgs
import gsplat
import torch.nn as nn
import numpy as np
import torch


def align_depth_scale(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    mask: np.ndarray,
    percentile: float = 60.0,
) -> float:
    valid_source = np.where(mask, source_depth, np.nan)
    valid_target = np.where(mask, target_depth, np.nan)

    ratios = valid_target / valid_source
    q1, q2 = np.nanpercentile(ratios, [50 - (percentile / 2), 50 + (percentile / 2)])
    middle_mask = (ratios >= q1) & (ratios <= q2) & ~np.isnan(ratios)

    middle_source = valid_source[middle_mask]
    middle_target = valid_target[middle_mask]

    scale = np.nanmedian(middle_target / middle_source)

    print(f"Depth alignment (scale): percentile range [{q1:.6f}, {q2:.6f}], scale: {scale:.6f}")

    return scale


def merge_splats(
    existing_gsplat,
    new_splats: nn.ParameterDict,
) -> nn.ParameterDict:

    existing_means = existing_gsplat["means"].data
    existing_scales = existing_gsplat["scales"].data
    existing_quats = existing_gsplat["quats"].data
    existing_opacities = existing_gsplat["opacities"].data
    existing_sh0 = existing_gsplat["sh0"].data
    existing_shN = existing_gsplat["shN"].data
    
    new_means = new_splats["means"].data
    new_scales = new_splats["scales"].data
    new_quats = new_splats["quats"].data
    new_opacities = new_splats["opacities"].data
    new_sh0 = new_splats["sh0"].data
    new_shN = new_splats["shN"].data
    
    device = existing_means.device
    new_means = new_means.to(device)
    new_scales = new_scales.to(device)
    new_quats = new_quats.to(device)
    new_opacities = new_opacities.to(device)
    new_sh0 = new_sh0.to(device)
    new_shN = new_shN.to(device)
    
    if existing_means.numel() > 0 and new_means.numel() > 0:
        merged_means = torch.cat([existing_means, new_means], dim=0)
        merged_scales = torch.cat([existing_scales, new_scales], dim=0)
        merged_quats = torch.cat([existing_quats, new_quats], dim=0)
        merged_opacities = torch.cat([existing_opacities, new_opacities], dim=0)
        merged_sh0 = torch.cat([existing_sh0, new_sh0], dim=0)
        merged_shN = torch.cat([existing_shN, new_shN], dim=0)
    elif existing_means.numel() > 0:
        merged_means = existing_means
        merged_scales = existing_scales
        merged_quats = existing_quats
        merged_opacities = existing_opacities
        merged_sh0 = existing_sh0
        merged_shN = existing_shN
    elif new_means.numel() > 0:
        merged_means = new_means
        merged_scales = new_scales
        merged_quats = new_quats
        merged_opacities = new_opacities
        merged_sh0 = new_sh0
        merged_shN = new_shN
    else:
        device = existing_means.device
        merged_means = torch.empty((0, 3), device=device, dtype=torch.float32)
        merged_scales = torch.empty((0, 3), device=device, dtype=torch.float32)
        merged_quats = torch.empty((0, 4), device=device, dtype=torch.float32)
        merged_opacities = torch.empty((0,), device=device, dtype=torch.float32)
        merged_sh0 = torch.empty((0, 1, 3), device=device, dtype=torch.float32)
        merged_shN = torch.empty((0, 8, 3), device=device, dtype=torch.float32)
    
    merged_splats = nn.ParameterDict({
        "means": nn.Parameter(merged_means, requires_grad=True),
        "scales": nn.Parameter(merged_scales, requires_grad=True),
        "quats": nn.Parameter(merged_quats, requires_grad=True),
        "opacities": nn.Parameter(merged_opacities, requires_grad=True),
        "sh0": nn.Parameter(merged_sh0, requires_grad=True),
        "shN": nn.Parameter(merged_shN, requires_grad=True),
    })
    
    print(f"Merged splats: {len(existing_means)} existing + {len(new_means)} new = {len(merged_means)} total")
    
    return merged_splats


def update_3dgs_with_outpaint(
    splats,
    outpainted_image: Image.Image,
    depth: np.ndarray,
    rendered_mask: np.ndarray,
    interpolated_camera_poses: np.ndarray,
    frame_idx: int = 0,
    depth_valid_mask: Optional[np.ndarray] = None,
    rendered_depth: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    depth_align_percentile: float = 60.0,
    dis_threshold: float = 0.0,
    epsilon: float = 1e-3,
    scale_rate: float = 1.0,
    prune_floaters: bool = True,
    prune_enclosing: bool = True,
    save_new_splats_path: Optional[str] = None,
    return_new_splats: bool = False,
) -> nn.ParameterDict:
    import cv2
    from splats.pruning import (
        prune_reference_view_floaters,
        prune_splats_enclosing_cameras,
    )
    from geometry.cameras import c2w_to_w2c

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if prune_floaters:
        splats = prune_reference_view_floaters(
            splats=splats,
            w2c_pose=c2w_to_w2c(interpolated_camera_poses[frame_idx]),
            equi_height=1024,
            equi_width=2048,
            device=device,
            radius_percentile=99.0,
        )

    cam_c2w = interpolated_camera_poses[frame_idx]
    cam_c2w_torch = torch.from_numpy(cam_c2w).float().to(device)

    if isinstance(outpainted_image, Image.Image):
        image_np = np.array(outpainted_image)
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
    else:
        image_np = np.array(outpainted_image)

    H, W = image_np.shape[:2]

    if depth.shape != (H, W):
        print(f"depth map dimensions {depth.shape} does not match panoramic image dimensions {image_np.shape[:2]}")
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

    if depth_valid_mask is not None and depth_valid_mask.shape != (H, W):
        print(f"depth valid mask dimensions {depth_valid_mask.shape} does not match panoramic image dimensions {image_np.shape[:2]}")
        depth_valid_mask = cv2.resize(
            depth_valid_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
        ) > 0.5

    align_mask = rendered_mask & (rendered_depth > 0) & (depth > 0) if rendered_depth is not None else None
    if align_mask is not None and depth_valid_mask is not None:
        align_mask = align_mask & depth_valid_mask

    if align_mask is not None and align_mask.sum() > 100:
        scale = align_depth_scale(depth, rendered_depth, align_mask, percentile=depth_align_percentile)
        depth_scaled = depth * scale
    else:
        depth_scaled = depth.copy()

    if depth_valid_mask is not None:
        sky_far_plane = float(depth_scaled[depth_valid_mask].max())
        depth_scaled[~depth_valid_mask] = sky_far_plane
        print(f"Sky: {(~depth_valid_mask).sum()} px ({100.0 * (~depth_valid_mask).mean():.1f}%) set to far plane ({sky_far_plane:.2f})")

    new_content_mask = ~rendered_mask & (depth_scaled > 0.01)

    image_torch = torch.from_numpy(image_np).float().to(device)
    depth_torch = torch.from_numpy(depth_scaled).float().to(device)
    mask_torch  = torch.from_numpy(new_content_mask).bool().to(device)

    new_splats = convert_rgbd_equi_to_3dgs(
        rgb=image_torch,
        distance=depth_torch,
        mask=mask_torch,
        camera_pose_c2w=cam_c2w_torch,
        dis_threshold=dis_threshold,
        epsilon=epsilon,
        scale_rate=scale_rate,
        save_path=None,
    )

    if save_new_splats_path is not None:
        gsplat.export_splats(
            means=new_splats["means"],
            sh0=new_splats["sh0"],
            shN=new_splats["shN"],
            scales=new_splats["scales"],
            quats=new_splats["quats"],
            opacities=new_splats["opacities"],
            format="ply",
            save_to=str(save_new_splats_path),
        )
        print(f"Saved pano-only splats to {save_new_splats_path}")

    merged_splats = merge_splats(splats, new_splats)

    if prune_enclosing:
        merged_splats = prune_splats_enclosing_cameras(
            merged_splats,
            camera_positions=interpolated_camera_poses[:, :3, 3],
            device=device,
        )

    if return_new_splats:
        return merged_splats, new_splats

    return merged_splats
