from concurrent.futures import ThreadPoolExecutor
import cv2
from geometry.cameras import get_panorama_cameras_icosahedron
from geometry.cameras import interpolate_w2c_poses
import numpy as np
import os
from gsplat.rendering import rasterization
from geometry.projection import spherical_uv_to_directions
import torch
from tqdm import tqdm
import utils3d


def render_icosahedron(
    splats: dict,
    w2c_poses: np.ndarray,
    num_frames: int,
    face_resolution: int = 512,
    batch_size: int = 6,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"].to(device))
    opacities = torch.sigmoid(splats["opacities"].to(device))
    colors = torch.cat([splats["sh0"].to(device), splats["shN"].to(device)], dim=1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    extrinsics_ico, intrinsics_ico = get_panorama_cameras_icosahedron()
    num_views = len(extrinsics_ico)

    uv = utils3d.np.uv_map(face_resolution, face_resolution)
    rays = utils3d.np.unproject_cv(
        uv,
        np.ones((face_resolution, face_resolution), dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32),
        intrinsics=intrinsics_ico[0],
    )
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    z_to_ray_distance = torch.tensor(
        1.0 / np.clip(rays[..., 2], 1e-6, None),
        dtype=torch.float32,
        device=device,
    )
    
    w2c_interp = interpolate_w2c_poses(w2c_poses, num_frames)
    w2c_interp = torch.tensor(w2c_interp, dtype=torch.float32, device=device)

    opencv_to_moge_format = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ], dtype=np.float32)
    opencv_to_moge = torch.eye(4, dtype=torch.float32, device=device)
    opencv_to_moge[:3, :3] = torch.tensor(
        opencv_to_moge_format,
        dtype=torch.float32,
        device=device,
    )

    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = K[1, 1] = face_resolution / 2.0
    K[0, 2] = K[1, 2] = face_resolution / 2.0
    K = K.unsqueeze(0)

    all_frames = np.zeros((num_frames, num_views, face_resolution, face_resolution, 3), dtype=np.uint8)
    all_alphas = np.zeros((num_frames, num_views, face_resolution, face_resolution), dtype=np.float32)
    all_depths = np.zeros((num_frames, num_views, face_resolution, face_resolution), dtype=np.float32)
    
    pbar = tqdm(total=num_frames, desc="Rendering")
    
    for frame_idx in range(num_frames):
        cam_w2c = w2c_interp[frame_idx].unsqueeze(0)  
        
        for batch_start in range(0, num_views, batch_size):
            batch_end = min(batch_start + batch_size, num_views)
            batch_views = batch_end - batch_start
            
            view_extrinsics_batch = [extrinsics_ico[i] for i in range(batch_start, batch_end)]
            view_w2c_batch = torch.tensor(
                np.stack(view_extrinsics_batch), 
                dtype=torch.float32, 
                device=device
            )  
            
            relative_views = torch.bmm(
                view_w2c_batch,
                opencv_to_moge.expand(batch_views, -1, -1),
            )

            combined = torch.bmm(
                relative_views,
                cam_w2c.expand(batch_views, -1, -1),
            )
            
            K_batch = K.expand(batch_views, -1, -1)
            
            with torch.no_grad():
                rgbd, alpha, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=combined,
                    Ks=K_batch,
                    width=face_resolution,
                    height=face_resolution,
                    sh_degree=sh_degree,
                    packed=True,
                    render_mode="RGB+D",
                )

            imgs = (rgbd[..., :3].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            depths = (rgbd[..., 3] * z_to_ray_distance[None]).cpu().numpy()
            alphas = alpha[..., 0].cpu().numpy()
            
            all_frames[frame_idx, batch_start:batch_end] = imgs
            all_alphas[frame_idx, batch_start:batch_end] = alphas
            all_depths[frame_idx, batch_start:batch_end] = depths
        
        pbar.update(1)
    
    pbar.close()
    
    return all_frames, all_alphas, all_depths


def convert_icosahedron_frame_to_equirectangular(args):
    frame_views, view_mappings, eq_h, eq_w, blend_mode = args
    num_views, _, _, C = frame_views.shape

    if blend_mode == "weighted":
        output = np.zeros((eq_h, eq_w, C), dtype=np.float32)
        weight_sum = np.zeros((eq_h, eq_w, 1), dtype=np.float32)

        for view_idx in range(num_views):
            mapping = view_mappings[view_idx]
            sampled = cv2.remap(
                frame_views[view_idx],
                mapping["pixels"][..., 0],
                mapping["pixels"][..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            if sampled.ndim == 2:
                sampled = sampled[..., None]

            output += sampled * mapping["weight"]
            weight_sum += mapping["weight"]

        valid = weight_sum[..., 0] > 1e-6
        output[valid] /= weight_sum[valid]

    else:
        output = np.zeros((eq_h, eq_w, C), dtype=np.float32)
        best_weights = np.zeros((eq_h, eq_w), dtype=np.float32)
        best_view = np.zeros((eq_h, eq_w), dtype=np.int32)

        for view_idx, mapping in enumerate(view_mappings):
            weight = mapping["weight"][..., 0]
            mask = weight > best_weights
            best_weights[mask] = weight[mask]
            best_view[mask] = view_idx

        valid = best_weights > 0
        for view_idx in range(num_views):
            mask = valid & (best_view == view_idx)
            if not np.any(mask):
                continue

            mapping = view_mappings[view_idx]
            sampled = cv2.remap(
                frame_views[view_idx],
                mapping["pixels"][..., 0],
                mapping["pixels"][..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            if sampled.ndim == 2:
                sampled = sampled[..., None]

            output[mask] = sampled[mask]

    if C == 3:
        output = np.clip(output, 0, 255).astype(np.uint8)

    return output.squeeze()


def convert_icosahedron_video_to_equirectangular(
    ico_views: np.ndarray,
    equi_height: int = None,
    equi_width: int = None,
    blend_mode: str = "weighted",
    num_workers: int = 1,
):
    if ico_views.ndim != 5:
        raise ValueError(f"ico_views must be [N, num_views, H, W, C], got shape {ico_views.shape}")

    N, num_views, H, W, C = ico_views.shape
    eq_h = equi_height or H
    eq_w = equi_width or (H * 2)
    
    extrinsics_ico, intrinsics_ico = get_panorama_cameras_icosahedron()
    if len(extrinsics_ico) != num_views:
        raise ValueError(f"Expected {len(extrinsics_ico)} views but got {num_views}")
    
    uv_equi = utils3d.np.uv_map(eq_h, eq_w)  
    
    spherical_directions = spherical_uv_to_directions(uv_equi)  
    view_mappings = []
    for view_idx in range(num_views):
        projected_uv, projected_depth = utils3d.np.project_cv(
            spherical_directions,
            extrinsics=extrinsics_ico[view_idx],
            intrinsics=intrinsics_ico[view_idx]
        )
        
        eps = 1e-4
        projection_valid = (
            (projected_depth > 0) & 
            (projected_uv[..., 0] >= -eps) & (projected_uv[..., 0] <= 1 + eps) &
            (projected_uv[..., 1] >= -eps) & (projected_uv[..., 1] <= 1 + eps)
        )
        
        projected_pixels = utils3d.np.uv_to_pixel(
            np.clip(projected_uv, 0, 1), 
            (H, W)
        ).astype(np.float32)
        
        view_forward = extrinsics_ico[view_idx][2, :3]
        cos_angle = np.sum(spherical_directions * view_forward, axis=-1)
        cos_angle = np.clip(cos_angle, 0, 1)
        weight = (cos_angle ** 2) * projection_valid.astype(np.float32)
        
        view_mappings.append({
            'pixels': projected_pixels,
            'valid': projection_valid,
            'weight': weight[..., None]  
        })
    
    frame_args = [
        (ico_views[i], view_mappings, eq_h, eq_w, blend_mode)
        for i in range(N)
    ]

    if num_workers <= 1:
        equi_frames = [
            convert_icosahedron_frame_to_equirectangular(frame_args[i])
            for i in tqdm(range(N), desc="Converting to equirectangular", leave=False)
        ]
    else:
        w = min(num_workers, N, os.cpu_count() or 1)

        old_cv_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=w) as executor:
                equi_frames = list(tqdm(
                    executor.map(convert_icosahedron_frame_to_equirectangular, frame_args),
                    total=N,
                    desc="Converting to equirectangular",
                    leave=False,
                ))
        finally:
            cv2.setNumThreads(old_cv_threads)

    result = np.stack(equi_frames, axis=0)
    if C == 1 and result.ndim == 3:
        result = result[..., None]
    return result
