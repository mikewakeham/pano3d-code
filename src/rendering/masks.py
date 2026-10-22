import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import Tuple

def compute_edge_mask(depth, abs_thresh=0.1, rel_thresh=0.1):
    assert depth.ndim == 2
    depth = depth.astype(np.float32, copy=False)

    valid = np.isfinite(depth) & (depth > 0)
    eps = 1e-6

    edge = np.zeros_like(valid, dtype=bool)

    d1 = depth[:, :-1]
    d2 = depth[:, 1:]
    v_pair = valid[:, :-1] & valid[:, 1:]

    diff = np.abs(d1 - d2)
    rel = diff / (np.minimum(d1, d2) + eps)
    edge_pair = v_pair & (diff > abs_thresh) & (rel > rel_thresh)

    edge[:, :-1] |= edge_pair
    edge[:, 1:] |= edge_pair

    d1 = depth[:-1, :]
    d2 = depth[1:, :]
    v_pair = valid[:-1, :] & valid[1:, :]

    diff = np.abs(d1 - d2)
    rel = diff / (np.minimum(d1, d2) + eps)
    edge_pair = v_pair & (diff > abs_thresh) & (rel > rel_thresh)

    edge[:-1, :] |= edge_pair
    edge[1:, :] |= edge_pair

    keep = valid & (~edge)
    return keep




def fill_holes_with_interpolation(
    combined_rgb: np.ndarray,
    rendered_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> np.ndarray:

    hole_mask = np.logical_and(refined_mask == 1, rendered_mask == 0)
    
    ys, xs = np.nonzero(rendered_mask == 1)
    if len(ys) == 0:
        return combined_rgb.copy()
    known_colors = combined_rgb[ys, xs]
    
    yy, xx = np.nonzero(hole_mask)
    coords_known = np.stack([ys, xs], axis=-1)
    coords_fill = np.stack([yy, xx], axis=-1)
    
    filled_rgb = combined_rgb.copy()
    for c in range(3):
        channel_values = known_colors[:, c]
        linear = griddata(coords_known, channel_values, coords_fill, method="linear")
        nearest = griddata(coords_known, channel_values, coords_fill, method="nearest")
        filled_values = np.where(np.isnan(linear), nearest, linear)
        filled_rgb[yy, xx, c] = filled_values
    
    nan_mask = np.isnan(filled_rgb)
    for c in range(3):
        filled_rgb[nan_mask[..., c], c] = combined_rgb[nan_mask[..., c], c]
    
    return filled_rgb


def harden_alpha_rgb(rgb, alpha, source_threshold=0.1, background=0):
    rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)

    keep = alpha >= source_threshold

    out = np.full_like(rgb_uint8, background)
    alpha_safe = np.maximum(alpha, 1e-6)

    out[keep] = (
        rgb_uint8[keep].astype(np.float32) / alpha_safe[keep, None]
    ).clip(0, 255).astype(np.uint8)

    return out


def harden_alpha_video(rgb_frames, alpha_frames, source_threshold=0.1, background=0):
    return np.stack([
        harden_alpha_rgb(rgb, alpha, source_threshold=source_threshold, background=background)
        for rgb, alpha in zip(rgb_frames, alpha_frames)
    ], axis=0)


def fill_holes_telea(rgb, alpha, source_threshold=0.1, radius=15):
    alpha = np.clip(alpha, 0, 1).astype(np.float32)

    keep = alpha >= source_threshold
    hole = ~keep

    hardened_rgb = harden_alpha_rgb(
        rgb,
        alpha,
        source_threshold=source_threshold,
        background=0,
    )

    mask = hole.astype(np.uint8) * 255

    pad = max(32, radius * 2)
    rgb_wrap = np.concatenate(
        [hardened_rgb[:, -pad:], hardened_rgb, hardened_rgb[:, :pad]],
        axis=1,
    )
    mask_wrap = np.concatenate(
        [mask[:, -pad:], mask, mask[:, :pad]],
        axis=1,
    )
    filled_wrap = cv2.inpaint(
        rgb_wrap,
        mask_wrap,
        inpaintRadius=radius,
        flags=cv2.INPAINT_TELEA,
    )
    inpainted = filled_wrap[:, pad:pad + hardened_rgb.shape[1]]

    result = inpainted.copy()
    result[keep] = hardened_rgb[keep]

    return result.astype(np.uint8)
    

def telea_inpaint_frame(rgb_alpha: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    rgb, alpha = rgb_alpha
    return fill_holes_telea(rgb=rgb, alpha=alpha)


def fill_video_sequence_telea(
    rgb_frames: np.ndarray,
    alpha_frames: np.ndarray,
    num_workers: int = 1,
    desc: str = "Filling frames",
) -> np.ndarray:
    n_frames = rgb_frames.shape[0]
    if n_frames == 0:
        return np.empty((0,) + rgb_frames.shape[1:], dtype=np.uint8)
    if alpha_frames.shape[0] != n_frames:
        raise ValueError(
            f"alpha_frames length {alpha_frames.shape[0]} != rgb_frames {n_frames}"
        )

    rgb_alpha_per_frame = [
        (rgb_frames[i], alpha_frames[i]) for i in range(n_frames)
    ]
    if num_workers <= 1:
        filled_list = [
            fill_holes_telea(rgb=rgb, alpha=alpha)
            for rgb, alpha in tqdm(rgb_alpha_per_frame, desc=desc)
        ]
    else:
        w = min(num_workers, n_frames, os.cpu_count() or 1)
        chunksize = max(1, n_frames // (w * 8))
        with ProcessPoolExecutor(max_workers=w) as ex:
            filled_list = list(
                tqdm(
                    ex.map(telea_inpaint_frame, rgb_alpha_per_frame, chunksize=chunksize),
                    total=n_frames,
                    desc=desc,
                )
            )
    out = np.stack(filled_list, axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)








def build_mask_from_alpha(
    alpha: np.ndarray,
    low_threshold: float = 0.55,
    high_threshold: float = 0.9,
    kernel_size: int = 3,
    min_region_size: int = 256,
    min_core_pixels: int = 64,
    min_core_ratio: float = 0.01,
    max_diffuse_area_ratio: float = 0.35,
) -> np.ndarray:

    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1)
    )

    candidate_mask = (alpha >= low_threshold).astype(np.uint8)
    core_mask = (alpha >= high_threshold).astype(np.uint8)

    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
    refined_mask = np.zeros_like(candidate_mask, dtype=np.uint8)

    image_area = alpha.shape[0] * alpha.shape[1]
    max_diffuse_area = max(int(image_area * max_diffuse_area_ratio), min_region_size)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_region_size:
            continue

        component = labels == lbl
        core_pixels = int(core_mask[component].sum())
        peak_alpha = float(alpha[component].max())

        if core_pixels == 0 or peak_alpha < high_threshold:
            continue

        core_ratio = core_pixels / float(area)
        if core_pixels < min_core_pixels and core_ratio < min_core_ratio:
            continue

        if area > max_diffuse_area and core_ratio < (min_core_ratio * 2.0):
            continue

        refined_mask[component] = 1

    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    return refined_mask.astype(bool)


def fill_holes_with_alpha_blending(
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_mask: np.ndarray,
    alpha_threshold: float = 0.99,
) -> np.ndarray:
    solid_mask = alpha > alpha_threshold
    
    interpolated_rgb = fill_holes_with_interpolation(
        combined_rgb=rgb,
        rendered_mask=solid_mask,
        refined_mask=target_mask,
    )
    
    filled_rgb = rgb.copy()
    alpha_3d = alpha[:, :, np.newaxis]
    
    filled_rgb[target_mask] = (
        rgb[target_mask] * alpha_3d[target_mask] + 
        interpolated_rgb[target_mask] * (1 - alpha_3d[target_mask])
    )
    
    return filled_rgb
