from rendering.icosahedron import convert_icosahedron_video_to_equirectangular
from rendering.cubemap import cubemap_alpha_to_equirectangular_mask
from rendering.cubemap import cubemap_depth_to_equirectangular
from rendering.cubemap import cubemap_frames_to_equirectangular
import numpy as np
from rendering.cubemap import render_cubemap
from rendering.icosahedron import render_icosahedron


def render_panoramic_video(
    splats: dict,
    w2c_poses: np.ndarray,
    num_frames: int,
    render_mode: str = "cubemap",
    face_resolution: int = 512,
    equi_height: int = 1024,
    equi_width: int = 2048,
    alpha_threshold: float = 0.8,
    blend_mode: str = "weighted",
    ico_convert_workers: int = 1,
    device: str = None,
):
    if render_mode == "cubemap":
        print(f"Rendering {num_frames} frames using cubemap (6 faces)...")
        cubemap_frames, cubemap_alpha, cubemap_depth = render_cubemap(
            splats=splats,
            w2c_poses=w2c_poses,
            num_frames=num_frames,
            face_resolution=face_resolution,
            device=device,
        )
        
        equirectangular_frames = cubemap_frames_to_equirectangular(
            cubemap_frames=cubemap_frames,
            equi_height=equi_height,
            equi_width=equi_width,
        )
        
        equirectangular_masks = cubemap_alpha_to_equirectangular_mask(
            cubemap_alpha=cubemap_alpha,
            equi_height=equi_height,
            equi_width=equi_width,
            alpha_threshold=alpha_threshold,
        )
        
        equirectangular_depth = cubemap_depth_to_equirectangular(
            cubemap_depth=cubemap_depth,
            equi_height=equi_height,
            equi_width=equi_width,
        )
        
    elif render_mode == "icosahedron":
        print(f"Rendering {num_frames} frames using icosahedron...")
        ico_frames, ico_alpha, ico_depth = render_icosahedron(
            splats=splats,
            w2c_poses=w2c_poses,
            num_frames=num_frames,
            face_resolution=face_resolution,
            device=device,
        )
        
        print("converting icosahedron RGB to equirectangular RGB...")
        equirectangular_frames = convert_icosahedron_video_to_equirectangular(
            ico_views=ico_frames,
            equi_height=equi_height,
            equi_width=equi_width,
            blend_mode=blend_mode,
            num_workers=ico_convert_workers,
        )
        
        print("converting icosahedron alpha to equirectangular alpha...")
        ico_alpha_expanded = ico_alpha[..., None]  
        equi_alpha = convert_icosahedron_video_to_equirectangular(
            ico_views=ico_alpha_expanded,
            equi_height=equi_height,
            equi_width=equi_width,
            blend_mode=blend_mode,
            num_workers=ico_convert_workers,
        )
        equi_alpha = equi_alpha.squeeze(-1)  
        
        if alpha_threshold is not None:
            equirectangular_masks = equi_alpha > alpha_threshold
        else:
            equirectangular_masks = equi_alpha
        
        print("converting icosahedron depth to equirectangular depth...")
        ico_depth_expanded = ico_depth[..., None]  
        equirectangular_depth = convert_icosahedron_video_to_equirectangular(
            ico_views=ico_depth_expanded,
            equi_height=equi_height,
            equi_width=equi_width,
            blend_mode="nearest",
            num_workers=ico_convert_workers,
        )
        equirectangular_depth = equirectangular_depth.squeeze(-1)  
    else:
        raise ValueError(f"Invalid render_mode: {render_mode}. Must be 'cubemap' or 'icosahedron'")
    
    return equirectangular_frames, equirectangular_masks, equirectangular_depth
