from geometry.cameras import CUBE_FACE_DIRS
from typing import Optional
from geometry.cameras import face_w2c_relative
from geometry.cameras import interpolate_w2c_poses
import numpy as np
import py360convert
from gsplat.rendering import rasterization
import torch
from tqdm import tqdm


def render_cubemap(
    splats: dict,
    w2c_poses: np.ndarray,
    num_frames: int,
    face_resolution: int = 512,
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

    w2c_interp = interpolate_w2c_poses(w2c_poses, num_frames)
    w2c_interp = torch.tensor(w2c_interp, dtype=torch.float32, device=device)

    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = K[1, 1] = face_resolution / 2.0
    K[0, 2] = K[1, 2] = face_resolution / 2.0
    K = K.unsqueeze(0)

    frames = []
    alpha_frames = []
    depth_frames = []
    for i in tqdm(range(num_frames)):
        cam_w2c = w2c_interp[i].unsqueeze(0)
        face_imgs = []
        face_alphas = []
        face_depths = []
        for face_name, fwd, up in CUBE_FACE_DIRS:
            face_w2c = face_w2c_relative(fwd, up)
            face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

            R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
            R_face = face_w2c[:, :3, :3]

            combined = cam_w2c.clone()
            combined[:, :3, :3] = torch.bmm(R_face, R_cam)
            combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                rgbd, alpha, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=combined,
                    Ks=K,
                    width=face_resolution,
                    height=face_resolution,
                    sh_degree=sh_degree,
                    packed=True,
                    render_mode="RGB+D",
                )

            img = (rgbd[0, ..., :3].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            depth = rgbd[0, ..., 3].cpu().numpy()
            a = alpha[0, ..., 0].cpu().numpy()

            face_imgs.append(img)
            face_alphas.append(a)
            face_depths.append(depth)

        frames.append(np.stack(face_imgs, axis=0))
        alpha_frames.append(np.stack(face_alphas, axis=0))
        depth_frames.append(np.stack(face_depths, axis=0))

    result = np.stack(frames, axis=0)  
    outputs = [result]
    outputs.append(np.stack(alpha_frames, axis=0))
    outputs.append(np.stack(depth_frames, axis=0))

    return tuple(outputs)


def cubemap_frames_to_equirectangular(
    cubemap_frames: np.ndarray,
    equi_height: int = None,
    equi_width: int = None,
):
    if cubemap_frames.ndim != 5 or cubemap_frames.shape[1] != 6:
        raise ValueError("cubemap_frames must be [N, 6, H, W, 3]")

    N, _, H, W, _ = cubemap_frames.shape
    eq_h = equi_height or H
    eq_w = equi_width or (H * 2)

    equi_frames = []
    for i in range(N):
        faces = [cubemap_frames[i, f] for f in range(6)]
        equi = py360convert.c2e(faces, eq_h, eq_w, cube_format="list")
        equi_frames.append(equi.astype(np.uint8))

    return np.stack(equi_frames, axis=0)


def cubemap_alpha_to_equirectangular_mask(
    cubemap_alpha: np.ndarray,
    equi_height: int = None,
    equi_width: int = None,
    alpha_threshold: Optional[float] = 0.5,
):
    if cubemap_alpha.ndim != 4 or cubemap_alpha.shape[1] != 6:
        raise ValueError("cubemap_alpha must be [N, 6, H, W]")

    N, _, H, W = cubemap_alpha.shape
    eq_h = equi_height or H
    eq_w = equi_width or (H * 2)

    results = []
    for i in range(N):
        faces = [cubemap_alpha[i, f] for f in range(6)]
        equi_alpha = py360convert.c2e(faces, eq_h, eq_w, cube_format="list")
        
        if alpha_threshold is not None:
            results.append(equi_alpha > alpha_threshold)
        else:
            results.append(equi_alpha)

    return np.stack(results, axis=0)


def cubemap_depth_to_equirectangular(
    cubemap_depth: np.ndarray,
    equi_height: int = None,
    equi_width: int = None,
):
    if cubemap_depth.ndim != 4 or cubemap_depth.shape[1] != 6:
        raise ValueError("cubemap_depth must be [N, 6, H, W]")

    N, _, H, W = cubemap_depth.shape
    eq_h = equi_height or H
    eq_w = equi_width or (H * 2)

    depth_frames = []
    for i in range(N):
        faces = [cubemap_depth[i, f] for f in range(6)]
        equi_depth = py360convert.c2e(faces, eq_h, eq_w, cube_format="list")
        depth_frames.append(equi_depth.astype(np.float32))

    return np.stack(depth_frames, axis=0)


def render_3dgs_alpha_with_radius_filter(
    splats: dict,
    w2c_pose: np.ndarray,
    equi_height: int,
    equi_width: int,
    face_resolution: int = 512,
    device: str = None,
    max_gaussian_radius_px: float = 64.0,
    radius_percentile: float = 99.0,
    scale_percentile: float = 99.5,
) -> np.ndarray:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"].to(device))
    opacities = torch.sigmoid(splats["opacities"].to(device))
    colors = torch.cat([splats["sh0"].to(device), splats["shN"].to(device)], dim=1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    cam_w2c = torch.tensor(w2c_pose, dtype=torch.float32, device=device).unsqueeze(0)

    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = K[1, 1] = face_resolution / 2.0
    K[0, 2] = K[1, 2] = face_resolution / 2.0
    K = K.unsqueeze(0)

    scale_metric = torch.amax(scales, dim=1).to(torch.float32)
    scale_cutoff = torch.quantile(
        scale_metric, torch.tensor(scale_percentile / 100.0, device=device)
    )

    face_alphas = []
    for face_name, fwd, up in CUBE_FACE_DIRS:
        face_w2c = face_w2c_relative(fwd, up)
        face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

        R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
        R_face = face_w2c[:, :3, :3]

        combined = cam_w2c.clone()
        combined[:, :3, :3] = torch.bmm(R_face, R_cam)
        combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            _, _, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=combined,
                Ks=K,
                width=face_resolution,
                height=face_resolution,
                sh_degree=sh_degree,
                packed=False,
            )

        radii = info["radii"]
        if radii.ndim == 3:
            if radii.shape[0] != 1:
                raise ValueError(f"Expected single-camera radii, got shape {tuple(radii.shape)}")
            radii = radii[0]
        if radii.ndim != 2 or radii.shape[-1] != 2:
            raise ValueError(f"Unexpected gsplat radii shape {tuple(radii.shape)}")

        visible = (radii > 0).all(dim=-1)
        footprint = radii.to(torch.float32).amax(dim=-1)

        if torch.any(visible):
            radius_cutoff = torch.quantile(
                footprint[visible], torch.tensor(radius_percentile / 100.0, device=device)
            )
            radius_cutoff = torch.minimum(
                radius_cutoff,
                torch.tensor(max_gaussian_radius_px, dtype=footprint.dtype, device=device),
            )
            keep = visible & (footprint <= radius_cutoff) & (scale_metric <= scale_cutoff)
        else:
            keep = torch.ones_like(scale_metric, dtype=torch.bool)

        if not torch.any(keep):
            keep = visible & (scale_metric <= scale_cutoff)
        if not torch.any(keep):
            keep = visible
        if not torch.any(keep):
            keep = torch.ones_like(scale_metric, dtype=torch.bool)

        with torch.no_grad():
            _, alpha, _ = rasterization(
                means=means[keep],
                quats=quats[keep],
                scales=scales[keep],
                opacities=opacities[keep],
                colors=colors[keep],
                viewmats=combined,
                Ks=K,
                width=face_resolution,
                height=face_resolution,
                sh_degree=sh_degree,
                packed=False,
            )

        a = alpha[0, ..., 0].cpu().numpy()
        face_alphas.append(a)

    cubemap_alpha = np.expand_dims(np.stack(face_alphas, axis=0), axis=0)
    equi_alpha = cubemap_alpha_to_equirectangular_mask(
        cubemap_alpha=cubemap_alpha,
        equi_height=equi_height,
        equi_width=equi_width,
        alpha_threshold=None,
    )
    return equi_alpha[0]
