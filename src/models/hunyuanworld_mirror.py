import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union

from third_party.worldmirror.models.models.worldmirror import WorldMirror
from third_party.worldmirror.utils.inference_utils import extract_load_and_preprocess_images


class HunyuanWorldMirror:
    def __init__(
        self,
        device: Optional[str] = None,
        fps: int = 1,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fps = fps

        print("loading hunyuanworld-mirror...")
        self.target_size = 518
        self.model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(self.device)
        self.model.eval()
    
    def _run_worldmirror_v1(self, input_path):
        input_images = extract_load_and_preprocess_images(
            Path(input_path), fps=self.fps, target_size=self.target_size,
        ).to(self.device)

        inputs = {"img": input_images}
        cond_flags = [0, 0, 0]


        with torch.no_grad():
            predictions = self.model(views=inputs, cond_flags=cond_flags)

        predictions["input_images"] = input_images
        return predictions


    def __call__(
        self,
        input_path: Union[str, Path],
    ) -> Dict[str, Any]:
        return self._run_worldmirror_v1(input_path)
