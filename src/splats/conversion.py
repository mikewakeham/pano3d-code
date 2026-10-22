from typing import Any
from typing import Dict
import torch.nn as nn
import torch


def sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


def convert_hunyuanworld_to_gsplat(
    hunyuan_splats: Dict[str, Any],
) -> nn.ParameterDict:
    def extract_tensor(key, expected_shape_suffix):
        tensor = hunyuan_splats[key]
        if isinstance(tensor, list):
            tensor = tensor[0] if len(tensor) > 0 else torch.tensor([], dtype=torch.float32)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        while tensor.ndim > len(expected_shape_suffix):
            tensor = tensor[0]
        return tensor
    
    means = extract_tensor("means", (None, 3))  
    scales = extract_tensor("scales", (None, 3))  
    quats = extract_tensor("quats", (None, 4))  
    opacities = extract_tensor("opacities", (None,))  
    
    if opacities.ndim > 1:
        opacities = opacities.squeeze()
    if opacities.ndim == 0:
        opacities = opacities.unsqueeze(0)
    
    if "sh" in hunyuan_splats:
        sh = extract_tensor("sh", (None, None, 3))  
        
        if sh.ndim == 2:
            sh = sh.unsqueeze(1)  
        elif sh.ndim == 1:
            sh = sh.reshape(-1, 1, 3)
        
        if sh.shape[0] == 0:
            device = sh.device if sh.numel() > 0 else means.device
            sh0 = torch.zeros((0, 1, 3), device=device, dtype=sh.dtype)
            shN = torch.zeros((0, 8, 3), device=device, dtype=sh.dtype)
        elif sh.shape[1] == 1:
            sh0 = sh  
            shN = torch.zeros((sh0.shape[0], 8, 3), device=sh0.device, dtype=sh0.dtype)
        else:
            sh0 = sh[:, 0:1, :]  
            shN = sh[:, 1:, :]  
            if shN.shape[1] < 8:
                padding = torch.zeros((shN.shape[0], 8 - shN.shape[1], 3), 
                                     device=shN.device, dtype=shN.dtype)
                shN = torch.cat([shN, padding], dim=1)
            elif shN.shape[1] > 8:
                shN = shN[:, :8, :]
    elif "colors" in hunyuan_splats:
        colors = extract_tensor("colors", (None, 3))
        sh0 = colors.unsqueeze(1)  
        shN = torch.zeros((sh0.shape[0], 8, 3), device=sh0.device, dtype=sh0.dtype)
    else:
        raise ValueError("hunyuan_splats must have either 'sh' or 'colors'")
    
    def inverse_sigmoid(x):
        return torch.log(x / (1 - x + 1e-8))
    
    if opacities.numel() > 0:
        opacities = inverse_sigmoid(opacities.clamp(min=1e-8, max=1-1e-8))
    
    if scales.numel() > 0:
        scales = torch.log(scales.clamp(min=1e-8))
    
    return nn.ParameterDict({
        "means": nn.Parameter(means, requires_grad=True),
        "scales": nn.Parameter(scales, requires_grad=True),
        "quats": nn.Parameter(quats, requires_grad=True),
        "opacities": nn.Parameter(opacities, requires_grad=True),
        "sh0": nn.Parameter(sh0, requires_grad=True),
        "shN": nn.Parameter(shN, requires_grad=True),
    })
