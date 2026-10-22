import json
import sys
import gc
import torch
import argparse
from pathlib import Path
from PIL import Image
import gsplat
import numpy as np
import cv2

_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from models.hunyuanworld_mirror import HunyuanWorldMirror
from models.dit360 import DiT360
from rendering.masks import fill_video_sequence_telea, compute_edge_mask
from splats.update import update_3dgs_with_outpaint
from splats.conversion import convert_hunyuanworld_to_gsplat
from rendering.panorama import render_panoramic_video
from geometry.cameras import get_interpolated_camera_poses, build_orientation_stabilized_w2c, c2w_to_w2c
from rendering.outputs import save_equirectangular_outputs
from geometry.projection import create_pointcloud, render_equirectangular_torch
from rendering.cubemap import render_3dgs_alpha_with_radius_filter
from rendering.masks import build_mask_from_alpha, fill_holes_with_alpha_blending
from typing import Dict, Tuple, Optional
from moviepy.editor import ImageSequenceClip


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all outputs")
    
    outpaint_group = parser.add_mutually_exclusive_group(required=True)
    outpaint_group.add_argument("--outpainted_image_path", type=str,
                               help="Path to pre-existing outpainted image (skips DiT360 generation)")
    outpaint_group.add_argument("--scene_prompt", type=str,
                               help="Text prompt for scene outpainting with DiT360")
    
    parser.add_argument("--input_fps", type=int, default=2,
                        help="FPS for input video processing")
    parser.add_argument("--render_fps", type=int, default=10,
                        help="FPS for final rendered output")
    parser.add_argument("--anchor_fps", type=int, default=None,
                        help="FPS of the trajectory used to create a precomputed outpaint")
    parser.add_argument("--frame_idx", type=int, default=-1,
                        help="Frame index for outpainting initialization. -1 = auto-select least covered.")
    
    parser.add_argument("--confidence_threshold", type=float, default=0.9,
                        help="Confidence threshold for prepare_dit360_inputs")
    parser.add_argument("--subsample_ratio", type=float, default=1.0,
                        help="Subsample ratio for prepare_dit360_inputs")
    parser.add_argument("--dit360_timestep", type=int, default=50,
                        help="Number of timesteps for DiT360 diffusion")
    parser.add_argument("--dit360_seed", type=int, default=29,
                        help="Random seed for DiT360")
    parser.add_argument("--dit360_tau", type=float, default=10.0,
                        help="Tau parameter for DiT360")
    parser.add_argument("--dit360_guidance_scale", type=float, default=2.5,
                        help="Guidance scale for DiT360")
    parser.add_argument("--dit360_source_key_scale", type=float, default=1.05,
                        help="Scale factor for source-region keys in the edited branch attention.",)
    parser.add_argument("--dit360_source_value_scale", type=float, default=1.0,
                        help="Scale factor for source-region values in the edited branch attention.",)
    
    parser.add_argument("--render_mode", type=str, default="icosahedron",
                        choices=["cubemap", "icosahedron"],
                        help="Rendering method: cubemap (6 faces) or icosahedron")
    parser.add_argument("--face_resolution", type=int, default=1024,
                        help="Resolution for cubemap faces")
    parser.add_argument("--equi_height", type=int, default=1024,
                        help="Height of equirectangular output")
    parser.add_argument("--equi_width", type=int, default=2048,
                        help="Width of equirectangular output")
    parser.add_argument("--blend_mode", type=str, default="weighted",
                        choices=["weighted", "nearest"],
                        help="Blending mode for icosahedron (ignored for cubemap)")
    parser.add_argument("--ico_convert_workers", type=int, default=12, 
                        help="Thread workers for icosahedron-to-equirectangular conversion")
    parser.add_argument("--depth_align_percentile", type=float, default=90.0,
                        help="Percentile for depth alignment")
    parser.add_argument("--save_splats", action="store_true",
                        help="Save updated 3D Gaussian Splats to .ply file")
    parser.add_argument("--fill_video_telea", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply telea inpainting to rendered video")
    parser.add_argument("--fill_workers", "--fill_telea_workers", dest="fill_workers", type=int, default=12,
                        help="Parallel processes for classical fill (1 = sequential)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    return parser.parse_args()


def find_least_covered_frame_idx(
    splats: Dict,
    camera_poses: torch.Tensor,
    interpolated_camera_poses: Optional[np.ndarray] = None,
    H: int = 512,
    W: int = 1024,
    face_resolution: int = 256,
    alpha_threshold: float = 0.5,
) -> int:
    sparse_poses = camera_poses.cpu().numpy()
    gsplat_splats = splats
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coverage_counts = []
    for cam_c2w in sparse_poses:
        gs_alpha = render_3dgs_alpha_with_radius_filter(
            splats=gsplat_splats,
            w2c_pose=c2w_to_w2c(cam_c2w),
            equi_height=H,
            equi_width=W,
            face_resolution=face_resolution,
            device=device,
            max_gaussian_radius_px=max(48.0, face_resolution * 0.12),
            radius_percentile=99.0,
            scale_percentile=99.5,
        )
        coverage_counts.append(int((gs_alpha > alpha_threshold).sum()))

    best_sparse_idx = int(np.argmin(coverage_counts))

    if interpolated_camera_poses is None:
        print(f"Using frame_idx={best_sparse_idx}")
        return best_sparse_idx

    N_sparse = len(sparse_poses)
    N_interp = len(interpolated_camera_poses)
    best_interp_idx = int(round(best_sparse_idx * (N_interp - 1) / max(N_sparse - 1, 1)))
    print(f"Using frame_idx={best_interp_idx}")
    return best_interp_idx


def prepare_dit360_inputs(
    pts3d: torch.Tensor,
    pts3d_conf: torch.Tensor,
    input_images: torch.Tensor,
    camera_poses: torch.Tensor,
    splats: Dict,
    frame_idx: int = 0,
    interpolated_camera_poses: Optional[np.ndarray] = None,
    H: int = 1024,
    W: int = 2048,
    num_views: int = 20,
    confidence_threshold: float = 0.1,
    subsample_ratio: float = 1.0,
    kernel_size: int = 6,
    min_region_size: int = 100,
    face_resolution: int = 512,
) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    
    if interpolated_camera_poses is not None:
        camera_poses = interpolated_camera_poses
    else:
        camera_poses = camera_poses.cpu().numpy()
    
    points, colors = create_pointcloud(
        pts3d=pts3d,
        pts3d_conf=pts3d_conf,
        input_images=input_images,
        num_views=num_views,
        confidence_threshold=confidence_threshold,
        subsample_ratio=subsample_ratio,
    )
    
    cam_c2w = camera_poses[frame_idx]
    w2c_pose = c2w_to_w2c(cam_c2w)
    
    _, rendered_depth, _ = render_equirectangular_torch(
        points, colors, H=H, W=W, camera_pose=w2c_pose
    )

    rendered_frames, rendered_alphas, _ = render_panoramic_video(
        splats=splats,
        w2c_poses=np.expand_dims(w2c_pose, axis=0),
        num_frames=1,
        render_mode="icosahedron",
        face_resolution=face_resolution,
        equi_height=H,
        equi_width=W,
        alpha_threshold=None,
        blend_mode="weighted",
        ico_convert_workers=12,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    rendered_3dgs = rendered_frames[0]
    gs_alpha = render_3dgs_alpha_with_radius_filter(
        splats=splats,
        w2c_pose=w2c_pose,
        equi_height=H,
        equi_width=W,
        face_resolution=face_resolution,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_gaussian_radius_px=max(48.0, face_resolution * 0.20),
        radius_percentile=99.5,
        scale_percentile=99.5,
    )
    
    refined_render_mask = build_mask_from_alpha(
        gs_alpha,
        low_threshold=0.5,
        high_threshold=0.9,
        kernel_size=kernel_size,
        min_region_size=max(min_region_size, 256),
        min_core_pixels=64,
        min_core_ratio=0.01,
        max_diffuse_area_ratio=0.35,
    )
    mask_np = (refined_render_mask * 255).astype(np.uint8)

    filled_rgb = fill_holes_with_alpha_blending(
        rgb=rendered_3dgs,
        alpha=gs_alpha,
        target_mask=refined_render_mask,
        alpha_threshold=0.8,
    ).astype(np.uint8)

    masked_rgb = filled_rgb.copy()
    masked_rgb[~refined_render_mask] = 0

    filled_rgb = masked_rgb

    init_image = Image.fromarray(filled_rgb.astype(np.uint8))

    return init_image, masked_rgb, mask_np, refined_render_mask, rendered_depth


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    other_output_dir = output_dir / "other_outputs"
    other_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Processing input video: {args.input_video}")
    hunyuan_model = HunyuanWorldMirror(
        fps=args.input_fps,
    )
    predictions = hunyuan_model(args.input_video)
    print("HunyuanWorld-Mirror prediction done.")

    input_images = predictions["input_images"][0]
    camera_poses = predictions["camera_poses"][0]
    pts3d = predictions["pts3d"][0]
    
    pts3d_conf = predictions.get("pts3d_conf")
    pts3d_conf = pts3d_conf[0] if pts3d_conf is not None else None


    splats = convert_hunyuanworld_to_gsplat(predictions["splats"])

    if args.save_splats:
        gsplat.export_splats(
            means=splats["means"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            format="ply",
            save_to=other_output_dir / "input_splats.ply",
        )
        print("saved HunyuanWorld splats to input_splats.ply")
    
    camera_poses_c2w_sparse = camera_poses.cpu().numpy()
    interpolated_c2w, interpolated_w2c, num_render_frames = get_interpolated_camera_poses(
        video_path=args.input_video,
        camera_poses_c2w=camera_poses_c2w_sparse,
        target_fps=args.render_fps,
    )
    stabilized_c2w = interpolated_c2w.copy()
    stabilized_c2w[:, :3, :3] = interpolated_c2w[0, :3, :3]
    print(f"Interpolated {len(camera_poses_c2w_sparse)} sparse poses to {num_render_frames} frames")
    
    anchor_fps = args.render_fps if args.anchor_fps is None else args.anchor_fps

    if anchor_fps == args.render_fps:
        anchor_stabilized_c2w = stabilized_c2w
    else:
        anchor_c2w, _, num_anchor_frames = get_interpolated_camera_poses(
            video_path=args.input_video,
            camera_poses_c2w=camera_poses_c2w_sparse,
            target_fps=anchor_fps,
        )
        anchor_stabilized_c2w = anchor_c2w.copy()
        anchor_stabilized_c2w[:, :3, :3] = anchor_c2w[0, :3, :3]
        print(f"Using {num_anchor_frames} poses at {anchor_fps} FPS for outpaint anchoring")

    if args.frame_idx == -1:
        frame_idx = find_least_covered_frame_idx(splats, camera_poses, anchor_stabilized_c2w)
    else:
        frame_idx = args.frame_idx
    
    init_image, masked_rgb, mask_np, rendered_mask, rendered_depth = prepare_dit360_inputs(
        pts3d=pts3d,
        pts3d_conf=pts3d_conf, 
        input_images=input_images,
        camera_poses=camera_poses,
        frame_idx=frame_idx, 
        splats=splats,
        interpolated_camera_poses=anchor_stabilized_c2w,
        confidence_threshold=args.confidence_threshold, 
        subsample_ratio=args.subsample_ratio,
    )
    init_image.save(other_output_dir / "refined_rgb_input.png")
    Image.fromarray(mask_np).save(other_output_dir / "refined_mask_input.png")
    Image.fromarray(masked_rgb.astype(np.uint8)).save(other_output_dir / "refined_rgb_input_masked.png")
    print("DiT360 inputs ready.")
    
    hunyuan_model.model = hunyuan_model.model.cpu()
    clear_gpu_memory()
    
    if args.outpainted_image_path is not None:
        print(f"Loading pre-existing outpainted image from: {args.outpainted_image_path}")
        outpainted_image = Image.open(args.outpainted_image_path)
    else:
        print("Generating outpainted image with DiT360...")
        dit360_model = DiT360()
        outpainted_image = dit360_model(
            init_image, 
            mask_np, 
            args.scene_prompt, 
            timestep=args.dit360_timestep, 
            seed=args.dit360_seed, 
            tau=args.dit360_tau, 
            guidance_scale=args.dit360_guidance_scale,
            use_rf_solver_invert=False,
            use_rf_solver_sample=False,
            source_key_scale=args.dit360_source_key_scale,
            source_value_scale=args.dit360_source_value_scale,
        )
        del dit360_model
        clear_gpu_memory()
    print("Outpainting complete!")

    
    outpainted_path = other_output_dir / "outpainted.png"
    outpainted_image.save(outpainted_path)
    print(f"Saved outpainted image to {outpainted_path}")
    
    print("Estimating panoramic depth using pager...")
    from models.pager import PaGeRDepth
    pager_model = PaGeRDepth(
        scene_mode="auto", 
        device=device, 
    )
    depth_map, depth_valid_mask = pager_model(outpainted_image)
    depth_vis_name = "pager_depth_vis.png"
    edge_keep = compute_edge_mask(depth_map)
    depth_map[~edge_keep] = 0.0

    valid_depth = depth_map[depth_valid_mask]
    _vis = cv2.applyColorMap(((depth_map - valid_depth.min()) / max(valid_depth.max() - valid_depth.min(), 1e-6) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    _vis[~depth_valid_mask] = _vis.max()
    cv2.imwrite(str(other_output_dir / depth_vis_name), _vis)
    print("Panoramic depth estimation done.")
    
    print("Updating 3D Gaussian Splats...")

    updated_splats = update_3dgs_with_outpaint(
        splats=splats,
        outpainted_image=outpainted_image,
        depth=depth_map,
        rendered_mask=rendered_mask,
        rendered_depth=rendered_depth,
        frame_idx=frame_idx,
        interpolated_camera_poses=anchor_stabilized_c2w,
        depth_valid_mask=depth_valid_mask,
        device=device,
        depth_align_percentile=args.depth_align_percentile,
        prune_floaters=True,
        prune_enclosing=True,
    )

    if args.save_splats:
        splats_path = other_output_dir / "updated_splats.ply"
        gsplat.export_splats(
            means=updated_splats["means"],
            sh0=updated_splats["sh0"],
            shN=updated_splats["shN"],
            scales=updated_splats["scales"],
            quats=updated_splats["quats"],
            opacities=updated_splats["opacities"],
            format="ply",
            save_to=str(splats_path),
        )
        print(f"Saved updated splats to {splats_path}")
    
    print(f"Rendering equirectangular video using {args.render_mode}...")
    equirectangular_frames, equirectangular_masks, equirectangular_depth = render_panoramic_video(
        splats=updated_splats,
        w2c_poses=interpolated_w2c,
        num_frames=num_render_frames,
        render_mode=args.render_mode,
        face_resolution=args.face_resolution,
        equi_height=args.equi_height,
        equi_width=args.equi_width,
        alpha_threshold=None,
        blend_mode=args.blend_mode,
        ico_convert_workers=args.ico_convert_workers,
        device=device,
    )
    
    save_equirectangular_outputs(
        output_dir=output_dir,
        other_output_dir=other_output_dir,
        prefix="",
        rgb_frames=equirectangular_frames,
        alpha_frames=equirectangular_masks,
        depth_frames=equirectangular_depth,
        fps=args.render_fps,
    )


    print(f"Rendering stabilized version of equirectangular video using {args.render_mode}...")
    stabilized_w2c = build_orientation_stabilized_w2c(interpolated_c2w)
    stabilized_equirectangular_frames, stabilized_equirectangular_masks, stabilized_equirectangular_depth = render_panoramic_video(
        splats=updated_splats,
        w2c_poses=stabilized_w2c,
        num_frames=num_render_frames,
        render_mode=args.render_mode,
        face_resolution=args.face_resolution,
        equi_height=args.equi_height,
        equi_width=args.equi_width,
        alpha_threshold=None,
        blend_mode=args.blend_mode,
        ico_convert_workers=args.ico_convert_workers,
        device=device,
    )

    save_equirectangular_outputs(
        output_dir=output_dir,
        other_output_dir=other_output_dir,
        prefix="stabilized",
        rgb_frames=stabilized_equirectangular_frames,
        alpha_frames=stabilized_equirectangular_masks,
        depth_frames=stabilized_equirectangular_depth,
        fps=args.render_fps,
    )

    if args.fill_video_telea:
        def _fill_frames(rgb_frames, alpha_frames, label):
            return fill_video_sequence_telea(
                rgb_frames,
                alpha_frames,
                num_workers=args.fill_workers,
                desc=f"Filling {label} frames (telea)",
            )


        filled_frames = _fill_frames(
            equirectangular_frames,
            equirectangular_masks,
            "equirectangular",
        )
        filled_video_path = output_dir / "equirectangular_filled.mp4"
        ImageSequenceClip(list(filled_frames), fps=args.render_fps).write_videofile(
            str(filled_video_path), codec="libx264", logger=None
        )
        print(f"Saved filled video to {filled_video_path}")

        stabilized_filled_frames = _fill_frames(
            stabilized_equirectangular_frames,
            stabilized_equirectangular_masks,
            "stabilized equirectangular",
        )
        stabilized_filled_video_path = output_dir / "stabilized_equirectangular_filled.mp4"
        ImageSequenceClip(list(stabilized_filled_frames), fps=args.render_fps).write_videofile(
            str(stabilized_filled_video_path), codec="libx264", logger=None
        )
        print(f"Saved filled video to {stabilized_filled_video_path}")
    


    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "camera_prior": None,
            "depth_model": 'pager',
            "device": args.device,
            "dit360_guidance_scale": args.dit360_guidance_scale,
            "dit360_postprocess_color_match": False,
            "dit360_seed": args.dit360_seed,
            "dit360_source_fill_mode": None,
            "dit360_source_key_scale": args.dit360_source_key_scale,
            "dit360_source_value_scale": args.dit360_source_value_scale,
            "dit360_tau": args.dit360_tau,
            "dit360_timestep": args.dit360_timestep,
            "equi_height": args.equi_height,
            "equi_width": args.equi_width,
            "fill_video_telea": args.fill_video_telea,
            "fill_video_xphoto": False,
            "fill_alpha_threshold": 0.1,
            "xphoto_method": 'shiftmap',
            "frame_idx": frame_idx,
            "anchor_fps": anchor_fps,
            "input_fps": args.input_fps,
            "input_video": args.input_video,
            "output_dir": args.output_dir,
            "render_fps": args.render_fps,
            "render_mode": args.render_mode,
            "scene_prompt": args.scene_prompt,
            "outpainted_image_path": args.outpainted_image_path,
        }, f, indent=2, sort_keys=True, default=str)
    print(f"Saved config to {config_path}") 

if __name__ == '__main__':
    main()
