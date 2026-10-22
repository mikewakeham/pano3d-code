from geometry.cameras import CUBE_FACE_DIRS
from typing import Optional
from rendering.cubemap import cubemap_alpha_to_equirectangular_mask
from geometry.cameras import face_w2c_relative
import torch.nn as nn
import numpy as np
from gsplat.rendering import rasterization
import torch


def prune_splats_enclosing_cameras(
    splats: dict,
    camera_positions: np.ndarray,
    scale_sigmas: float = 5.0,
    safety_margin: float = 0.0,
    device: Optional[str] = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    means = splats["means"].to(device)                    
    scales_act = torch.exp(splats["scales"].to(device))   
    max_scale = scales_act.amax(dim=-1)                   

    cam_pos = torch.tensor(camera_positions, dtype=torch.float32, device=device)
    if cam_pos.ndim == 1:
        cam_pos = cam_pos.unsqueeze(0)                    

    dists = torch.cdist(cam_pos, means)                   
    min_dist = dists.amin(dim=0)                          

    extent = scale_sigmas * max_scale + safety_margin     
    bad = min_dist < extent
    keep = ~bad

    n_total = means.shape[0]
    n_bad = int(bad.sum().item())
    print(f"prune_splats_enclosing_cameras: removed {n_bad}/{n_total} "
          f"({100.0 * n_bad / max(n_total, 1):.2f}%) "
          f"(scale_sigmas={scale_sigmas}, safety_margin={safety_margin})")

    if n_bad == 0:
        return splats

    pruned = nn.ParameterDict({
        k: nn.Parameter(v.detach()[keep].clone(), requires_grad=v.requires_grad)
        for k, v in splats.items()
    })
    return pruned


def prune_reference_view_floaters(
    splats: dict,
    w2c_pose: np.ndarray,
    equi_height: int,
    equi_width: int,
    face_resolution: int = 512,
    device: str = None,
    radius_percentile: float = 99.8,
    max_candidates: int = 128,
    support_alpha_threshold: float = 0.10,
    candidate_alpha_threshold: float = 0.08,
    outside_support_ratio_threshold: float = 0.35,
    solo_ratio_threshold: float = 0.50,
    overlap_eps: float = 0.05,
) -> dict:

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

    def _render_equi_alpha(keep_mask: torch.Tensor) -> np.ndarray:
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
                _, alpha, _ = rasterization(
                    means=means[keep_mask],
                    quats=quats[keep_mask],
                    scales=scales[keep_mask],
                    opacities=opacities[keep_mask],
                    colors=colors[keep_mask],
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

    full_keep = torch.ones(means.shape[0], dtype=torch.bool, device=device)
    full_alpha = _render_equi_alpha(full_keep)
    support_mask = full_alpha >= support_alpha_threshold

    max_footprint = torch.zeros(means.shape[0], dtype=torch.float32, device=device)

    for _, fwd, up in CUBE_FACE_DIRS:
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
            radii = radii[0]

        visible = (radii > 0).all(dim=-1)
        footprint = radii.to(torch.float32).amax(dim=-1)
        max_footprint = torch.maximum(
            max_footprint,
            torch.where(visible, footprint, torch.zeros_like(footprint)),
        )

    visible_any = max_footprint > 0
    if not torch.any(visible_any):
        return splats

    cutoff = torch.quantile(
        max_footprint[visible_any],
        torch.tensor(radius_percentile / 100.0, device=device),
    )

    candidate_idx = torch.nonzero(max_footprint >= cutoff, as_tuple=False).squeeze(1)
    if candidate_idx.numel() == 0:
        return splats

    if candidate_idx.numel() > max_candidates:
        candidate_scores = max_footprint[candidate_idx]
        topk = torch.topk(candidate_scores, k=max_candidates).indices
        candidate_idx = candidate_idx[topk]

    bad = torch.zeros(means.shape[0], dtype=torch.bool, device=device)

    for idx in candidate_idx.tolist():
        keep_mask = torch.zeros(means.shape[0], dtype=torch.bool, device=device)
        keep_mask[idx] = True

        cand_alpha = _render_equi_alpha(keep_mask)
        cand_mask = cand_alpha >= candidate_alpha_threshold
        cand_area = int(cand_mask.sum())
        if cand_area == 0:
            continue

        outside_support_ratio = float((cand_mask & (~support_mask)).sum()) / float(cand_area)
        solo_mask = cand_mask & ((full_alpha - cand_alpha) <= overlap_eps)
        solo_ratio = float(solo_mask.sum()) / float(cand_area)

        if outside_support_ratio >= outside_support_ratio_threshold and solo_ratio >= solo_ratio_threshold:
            bad[idx] = True

    keep = ~bad
    print(f"Reference-view floater pruning removed {(~keep).sum().item()} / {keep.numel()} gaussians")

    return {
        k: v[keep.to(v.device)] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == keep.numel() else v
        for k, v in splats.items()
    }
