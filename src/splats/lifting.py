from typing import Optional
from geometry.projection import equi_unit_rays
import gsplat
from splats.conversion import matrix_to_quaternion
import torch.nn as nn
import os.path as osp
from einops import rearrange
import torch


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def convert_rgbd_equi_to_3dgs(
    rgb: torch.Tensor,                       
    distance: torch.Tensor,                  
    rays: Optional[torch.Tensor] = None,     
    mask: Optional[torch.Tensor] = None,     
    camera_pose_c2w: Optional[torch.Tensor] = None,
    dis_threshold=0.,
    epsilon=1e-3,
    scale_rate=1.0,
    save_path: Optional[str] = None,
) -> nn.ParameterDict:
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rgb.shape[:2] == distance.shape[:2], "Input shapes must match"
    if rays is not None:
        assert rgb.shape[:2] == rays.shape[:2], "Input shapes must match"
        assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    if mask is not None:
        assert mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    H, W = rgb.shape[:2]
    device = rgb.device

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    if rays is None:
        rays = equi_unit_rays(rgb.shape[0], rgb.shape[1], device)
        rays[..., [0, 1]] *= -1
    
    valid_mask = distance > dis_threshold
    if mask is not None:
        valid_mask = valid_mask & mask.unsqueeze(0).unsqueeze(0)
    rays_flat = rays.view(-1, 3)
    rgbs_flat = rgb.view(-1, 3)
    distance_flat = distance.view(-1)
    valid_rays = rays_flat[valid_mask.view(-1)]
    valid_rgbs = rgbs_flat[valid_mask.view(-1)]
    valid_distance = distance_flat[valid_mask.view(-1)]
    centers = valid_rays * valid_distance[:, None]

    delta_phi = 2 * torch.pi / rgb.shape[1]
    delta_theta = torch.pi / rgb.shape[0]
    sigma_x = valid_distance * delta_phi * scale_rate
    sigma_y = valid_distance * delta_theta * scale_rate
    sigma_z = torch.ones_like(valid_distance) * epsilon * scale_rate

    S = torch.stack([sigma_x, sigma_y, sigma_z], dim=1)

    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    x_axis = torch.nn.functional.normalize(torch.cross(up, valid_rays, dim=-1), dim=1)
    fallback_up = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    degenerate_mask = torch.isnan(x_axis).any(dim=1)
    x_axis[degenerate_mask] = torch.nn.functional.normalize(torch.cross(fallback_up[degenerate_mask], valid_rays[degenerate_mask], dim=-1), dim=1)
    y_axis = torch.nn.functional.normalize(torch.cross(valid_rays, x_axis, dim=-1), dim=1)
    z_axis = valid_rays

    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  

    if camera_pose_c2w is not None:
        centers_h = torch.cat([centers, torch.ones((centers.shape[0], 1), device=centers.device)], dim=1)
        centers = (camera_pose_c2w @ centers_h.T).T[:, :3]
        R_c2w = camera_pose_c2w[:3, :3]
        R = torch.bmm(R_c2w.unsqueeze(0).expand(R.shape[0], -1, -1), R)

    sh_degree = 2
    num_sh_coeffs = (sh_degree + 1) ** 2  

    sh0 = rearrange(rgb_to_sh(valid_rgbs), 'n c -> n 1 c')  

    shN = torch.zeros((centers.shape[0], num_sh_coeffs - 1, 3), device=device)
    
    def inverse_sigmoid(x):
        return torch.log(x/(1-x))

    inverse_scaling_activation = torch.log    
    inverse_opacity_activation = inverse_sigmoid
    
    scales = inverse_scaling_activation(S)
    alphas = torch.ones((centers.shape[0],), device=device) * 0.99
    opacities = inverse_opacity_activation(alphas)
    
    quats = matrix_to_quaternion(R)

    splats = nn.ParameterDict({
        "means": nn.Parameter(centers, requires_grad=True),
        "sh0": nn.Parameter(sh0, requires_grad=True),
        "shN": nn.Parameter(shN, requires_grad=True),
        "scales": nn.Parameter(scales, requires_grad=True),
        "quats": nn.Parameter(quats, requires_grad=True),
        "opacities": nn.Parameter(opacities, requires_grad=True),
    })
    
    if save_path is not None:
        gsplat.export_splats(
            means=splats["means"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            format=osp.splitext(save_path)[-1].lower().lstrip('.'),
            save_to=save_path,
        )
    
    return splats
