from typing import Optional
from typing import Tuple
import cv2
import numpy as np
import torch


def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions


def render_equirectangular_torch(
    points: np.ndarray,
    colors: np.ndarray,
    H: int,
    W: int,
    camera_pose: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    colors = torch.as_tensor(colors, dtype=torch.float32, device=device)

    if camera_pose is not None:
        camera_pose = torch.as_tensor(camera_pose, dtype=torch.float32, device=device)
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=1)
        points_cam = (camera_pose @ points_h.T).T[:, :3]
    else:
        points_cam = points

    x, y, z = points_cam.T
    r = torch.norm(points_cam, dim=1)
    
    phi = torch.arcsin(y / r)
    theta = torch.atan2(x, z)
    
    u = torch.round((theta + torch.pi) / (2 * torch.pi) * (W - 1)).long()
    v = torch.round((phi + torch.pi/2) / torch.pi * (H - 1)).long()
    
    u = torch.clamp(u, 0, W - 1)
    v = torch.clamp(v, 0, H - 1)
    
    flat_indices = v * W + u
    
    sort_idx = torch.argsort(r, descending=True)
    flat_indices_sorted = flat_indices[sort_idx]
    colors_sorted = colors[sort_idx]
    depths_sorted = r[sort_idx]
    
    color_buffer = torch.zeros((H * W, 3), dtype=torch.float32, device=device)
    depth_buffer = torch.zeros(H * W, dtype=torch.float32, device=device)
    mask_buffer = torch.zeros(H * W, dtype=torch.bool, device=device)
    
    color_buffer[flat_indices_sorted] = colors_sorted
    depth_buffer[flat_indices_sorted] = depths_sorted
    mask_buffer[flat_indices_sorted] = True
    
    color_map = color_buffer.reshape(H, W, 3)
    depth_map = depth_buffer.reshape(H, W)
    mask = mask_buffer.reshape(H, W)
    
    color_map = color_map.cpu().numpy().astype(np.uint8)
    depth_map = depth_map.cpu().numpy()
    mask = mask.cpu().numpy()
    
    del points, colors, points_cam, x, y, z, r, phi, theta, u, v
    del flat_indices, sort_idx, flat_indices_sorted, colors_sorted, depths_sorted
    del color_buffer, depth_buffer, mask_buffer
    if camera_pose is not None:
        del camera_pose
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return color_map, depth_map, mask


def create_pointcloud(
    pts3d: torch.Tensor,
    pts3d_conf: torch.Tensor,
    input_images: torch.Tensor,
    num_views: Optional[int] = None,
    subsample_ratio: float = 1.0,
    confidence_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    
    pts3d = pts3d.cpu().numpy()
    pts3d_conf = pts3d_conf.cpu().numpy()
    images = input_images.permute(0, 2, 3, 1).cpu().numpy()
    
    S = pts3d.shape[0]
    if num_views is not None:
        step = max(S // num_views, 1)
        indices = np.arange(0, S, step)
    else:
        indices = np.arange(S)
    
    all_points = []
    all_colors = []
    
    for idx in indices:
        view_pts3d = pts3d[idx]  
        view_conf = pts3d_conf[idx]  
        view_image = images[min(idx, images.shape[0] - 1)]  
        
        if view_image.shape[:2] != view_pts3d.shape[:2]:
            H, W = view_pts3d.shape[:2]
            view_image = cv2.resize(
                view_image, (W, H), interpolation=cv2.INTER_LINEAR
            )
        
        valid_mask = view_conf > confidence_threshold
        
        if subsample_ratio < 1.0:
            H, W = valid_mask.shape
            step = int(1.0 / subsample_ratio) if subsample_ratio > 0 else 1
            grid_mask = np.zeros_like(valid_mask)
            grid_mask[::step, ::step] = valid_mask[::step, ::step]
            valid_mask = grid_mask
        
        points = view_pts3d[valid_mask]
        colors = view_image[valid_mask]
        
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        
        all_points.append(points)
        all_colors.append(colors)
    
    points = np.concatenate(all_points, axis=0)  
    colors = np.concatenate(all_colors, axis=0)  
    
    return points, colors


def equi_unit_rays(h: int, w: int, device):
    u = (torch.arange(w, device=device).float() + 0.5) / w
    v = (torch.arange(h, device=device).float() + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")   

    phi   = uu * 2 * torch.pi - torch.pi  
    theta = torch.pi / 2 - vv * torch.pi  

    x = - torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    return torch.stack((x, y, z), dim=-1)  
