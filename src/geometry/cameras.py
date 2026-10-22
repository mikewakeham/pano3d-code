from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import Tuple
from moviepy.editor import VideoFileClip
import numpy as np
import torch
import utils3d


CUBE_FACE_DIRS = [
    ("front", np.array([0, 0, -1], np.float32), np.array([0, 1, 0], np.float32)),
    ("right", np.array([1, 0, 0], np.float32),  np.array([0, 1, 0], np.float32)),
    ("back",  np.array([0, 0, 1], np.float32),  np.array([0, 1, 0], np.float32)),
    ("left",  np.array([-1, 0, 0], np.float32), np.array([0, 1, 0], np.float32)),
    ("up",    np.array([0, -1, 0], np.float32),  np.array([0, 0, -1], np.float32)),
    ("down",  np.array([0, 1, 0], np.float32), np.array([0, 0, 1], np.float32)),
]


def get_panorama_cameras_icosahedron():
    vertices, _ = utils3d.np.create_icosahedron_mesh()
    intrinsics = utils3d.np.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.np.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)


def build_orientation_stabilized_w2c(interpolated_c2w: np.ndarray) -> np.ndarray:
    stable_c2w = np.asarray(interpolated_c2w, dtype=np.float32).copy()
    first_rotation_c2w = stable_c2w[0, :3, :3].copy()

    stable_c2w[:, :3, :3] = first_rotation_c2w[None]
    stable_c2w[:, 3, :] = np.array([0, 0, 0, 1], dtype=np.float32)

    return np.stack([c2w_to_w2c(pose) for pose in stable_c2w], axis=0)


def c2w_to_w2c(c2w):
    is_torch = isinstance(c2w, torch.Tensor)
    if is_torch:
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        w2c = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        return w2c
    else:
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        w2c = np.eye(4, dtype=c2w.dtype)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        return w2c


def face_w2c_relative(face_forward, face_up):
    f = face_forward / np.linalg.norm(face_forward)
    u = face_up / np.linalg.norm(face_up)
    r = np.cross(u, f); r /= np.linalg.norm(r)
    u = np.cross(f, r); u /= np.linalg.norm(u)
    R_standard = np.stack([r, u, f], axis=1)

    R_flip = np.array([[-1,0,0],[0,1,0],[0,0,-1]], np.float32)
    R_face = R_standard @ R_flip

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_face
    return w2c


def interpolate_w2c_poses(w2c_poses, num_frames):
    w2c_poses = np.asarray(w2c_poses, dtype=np.float32)
    if len(w2c_poses) == num_frames or num_frames <= 1:
        return w2c_poses

    positions = w2c_poses[:, :3, 3]
    step_dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total = float(step_dists.sum())
    eps = max(total * 1e-6, 1e-9)
    step_dists = np.maximum(step_dists, eps)
    dists = np.concatenate([[0.0], np.cumsum(step_dists)]).astype(np.float32)
    t_orig = dists / dists[-1]

    t_new = np.linspace(0, 1, num_frames, dtype=np.float32)

    rots = R.from_matrix(w2c_poses[:, :3, :3])
    slerp = Slerp(t_orig, rots)
    R_interp = slerp(t_new).as_matrix()

    trans = np.vstack([
        np.interp(t_new, t_orig, positions[:, 0]),
        np.interp(t_new, t_orig, positions[:, 1]),
        np.interp(t_new, t_orig, positions[:, 2]),
    ]).T

    out = np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0)
    out[:, :3, :3] = R_interp
    out[:, :3, 3] = trans
    return out


def get_interpolated_camera_poses(
    video_path: str,
    camera_poses_c2w: np.ndarray,
    target_fps: int,
) -> Tuple[np.ndarray, np.ndarray, int]:

    with VideoFileClip(video_path) as clip:
        num_frames = int(clip.duration * target_fps)
    
    camera_poses_w2c = np.stack([c2w_to_w2c(p) for p in camera_poses_c2w], axis=0)
    interpolated_w2c = interpolate_w2c_poses(camera_poses_w2c, num_frames)
    interpolated_c2w = np.stack([c2w_to_w2c(p) for p in interpolated_w2c], axis=0)
    
    return interpolated_c2w, interpolated_w2c, num_frames
