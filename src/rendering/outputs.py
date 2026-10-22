from moviepy.editor import ImageSequenceClip
import cv2
from rendering.masks import harden_alpha_video
import numpy as np


def save_equirectangular_outputs(
    output_dir,
    other_output_dir,
    prefix: str,
    rgb_frames: np.ndarray,
    alpha_frames: np.ndarray,
    depth_frames: np.ndarray,
    fps: int,
):
    name = f"{prefix}_" if prefix else ""

    alpha_harden_threshold = 0.1
    rgb_to_save = harden_alpha_video(
        rgb_frames,
        alpha_frames,
        source_threshold=alpha_harden_threshold,
    )

    render_path = output_dir / f"{name}equirectangular_render.mp4"
    ImageSequenceClip(list(rgb_to_save), fps=fps).write_videofile(
        str(render_path), codec="libx264", logger=None
    )
    print(f"Saved render video to {render_path}")

    binary_mask = alpha_frames >= alpha_harden_threshold
    mask_frames = (binary_mask.astype(np.uint8) * 255)[..., None].repeat(3, axis=-1)
    mask_path = other_output_dir / f"{name}equirectangular_mask.mp4"
    ImageSequenceClip(list(mask_frames), fps=fps).write_videofile(
        str(mask_path), codec="libx264", logger=None
    )
    print(f"Saved mask video to {mask_path}")

    alpha_vis = (np.clip(alpha_frames, 0, 1) * 255).astype(np.uint8)[..., None].repeat(3, axis=-1)
    alpha_path = other_output_dir / f"{name}equirectangular_alpha.mp4"
    ImageSequenceClip(list(alpha_vis), fps=fps).write_videofile(
        str(alpha_path), codec="libx264", logger=None
    )
    print(f"Saved alpha video to {alpha_path}")

    valid_depth = np.isfinite(depth_frames) & (depth_frames > 0) & binary_mask
    if np.any(valid_depth):
        lo = np.percentile(depth_frames[valid_depth], 2)
        hi = np.percentile(depth_frames[valid_depth], 98)
        depth_norm = np.clip((depth_frames - lo) / max(hi - lo, 1e-6), 0, 1)
        depth_gray = (depth_norm * 255).astype(np.uint8)
        depth_gray[~valid_depth] = 0
        depth_vis = np.stack([
            cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)[..., ::-1]
            for frame in depth_gray
        ], axis=0)
    else:
        depth_vis = np.zeros(rgb_frames.shape, dtype=np.uint8)

    depth_path = other_output_dir / f"{name}equirectangular_depth.mp4"
    ImageSequenceClip(list(depth_vis), fps=fps).write_videofile(
        str(depth_path), codec="libx264", logger=None
    )
    print(f"Saved depth video to {depth_path}")
