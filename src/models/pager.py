import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")


from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from third_party.pager.pager import Pager
from third_party.pager.utils.geometry_utils import erp_to_cubemap, cubemap_to_erp


MODEL_TO_REPO = {
    "pager": "prs-eth/PaGeR",
    "pager-metric-depth": "prs-eth/PaGeR-metric-depth",
    "pager-normals": "prs-eth/PaGeR-normals",
}

SCALE_HEADS = {
    "indoor": "scale_indoor",
    "outdoor": "scale_outdoor",
}


class PaGeRDepth:
    def __init__(
        self,
        checkpoint: str = "pager",
        scene_mode: str = "auto",
        device: Optional[Union[str, torch.device]] = None,
        min_depth: float = 1e-2,
        max_depth: float = 75.0,
        sky_mask_threshold: float = 0.3,
        sky_mask_softness: float = 0.15,
        sky_mask_open_kernel: int = 5,
        face_size: Optional[int] = None,
        cube_fov: Optional[float] = None,
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scene_mode = scene_mode
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.sky_mask_threshold = float(sky_mask_threshold)
        self.sky_mask_softness = float(sky_mask_softness)
        self.sky_mask_open_kernel = int(sky_mask_open_kernel)
        self.inference_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        checkpoint_path = MODEL_TO_REPO.get(checkpoint, checkpoint)

        try:
            config_path = hf_hub_download(repo_id=checkpoint_path, filename="config.yaml")
        except Exception:
            config_path = Path(checkpoint_path) / "config.yaml"

        cfg = OmegaConf.load(config_path)
        self.modalities = list(cfg.modalities)
        if "depth" not in self.modalities:
            raise ValueError(f"PaGeR checkpoint {checkpoint!r} does not expose a depth head.")

        self.face_size = int(face_size or cfg.face_size)
        self.cube_fov = float(cube_fov if cube_fov is not None else getattr(cfg, "cube_fov", 90.0))
        self.has_dual_scale = all(m in self.modalities for m in SCALE_HEADS.values())
        self.scene_classifier = None

        self.pager = Pager(Path(checkpoint_path), cfg=cfg, device=self.device)
        self.pager.get_intrinsics_extrinsics(image_size=self.face_size, fov=self.cube_fov)
        self.pager.set_depth_range(self.min_depth, self.max_depth)
        self.pager.model.to(self.device, dtype=self.pager.weight_dtype)
        self.pager.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _skip_heads_for(self, rgb_cubemap_01: torch.Tensor):
        if not self.has_dual_scale:
            return None

        if self.scene_mode == "auto":
            from third_party.pager.utils.scene_classifier import get_classifier

            if self.scene_classifier is None:
                self.scene_classifier = get_classifier(device=self.device)
            label, _ = self.scene_classifier.classify(rgb_cubemap_01)
        else:
            label = self.scene_mode

        keep = SCALE_HEADS[label]
        return {h for h in SCALE_HEADS.values() if h != keep}

    @staticmethod
    def _to_rgb_uint8(image):
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return arr

    @torch.inference_mode()
    def __call__(self, image):
        image_np = self._to_rgb_uint8(image)
        height, width = image_np.shape[:2]

        rgb = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        rgb_cubemap_01 = erp_to_cubemap(rgb, face_w=self.face_size, fov=self.cube_fov)

        rgb_cubemap = ((rgb_cubemap_01 - self.mean) / self.std).unsqueeze(0).to(self.device)

        pred = self.pager(
            rgb_cubemap,
            dtype=self.inference_dtype,
            skip_heads=self._skip_heads_for(rgb_cubemap_01),
        )
        
        sky_pred = pred["sky"][0] if "sky" in pred else None
        log_scale = pred.get("scale", None)

        depth_erp, _ = self.pager.process_depth_output(
            pred["depth"][0],
            orig_size=(height, width),
            sky_mask=sky_pred,
            log_scale=log_scale,
            sky_mask_threshold=self.sky_mask_threshold,
            sky_mask_softness=self.sky_mask_softness,
            sky_mask_open_kernel=self.sky_mask_open_kernel,
        )

        depth = np.squeeze(depth_erp.float().cpu().numpy()).astype(np.float32)
        valid = np.isfinite(depth) & (depth > self.min_depth)

        if sky_pred is not None:
            sky_prob_erp = cubemap_to_erp(
                torch.sigmoid(sky_pred.float()),
                height,
                width,
                fov=self.cube_fov,
            )
            sky_prob = np.squeeze(sky_prob_erp.cpu().numpy())
            valid &= sky_prob < self.sky_mask_threshold

        if valid.sum() == 0:
            valid = np.isfinite(depth) & (depth > 0)

        conf_erp = cubemap_to_erp(
            pred["depth_conf"][0].float(),
            height,
            width,
            fov=self.cube_fov,
        )
        self.depth_confidence = np.squeeze(conf_erp.cpu().numpy()).astype(np.float32)

        return depth, valid.astype(bool)
