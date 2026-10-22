import torch.nn.functional as F
from PIL import Image
from typing import Optional
import numpy as np
import torch


class DiT360:
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        lora_name: str = "Insta360-Research/DiT360-Panorama-Image-Generation",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        self.dtype = torch.float16
        self.cache_dir = cache_dir
        
        print(f"loading DiT360 model {model_name}...")
        from third_party.dit360.pa_src.pipeline import RFPanoInversionParallelFluxPipeline
        
        self.pipe = RFPanoInversionParallelFluxPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        ).to(self.device)
        self.pipe.load_lora_weights(lora_name)

        print("DiT360 model loaded")
    
    def __call__(
        self,
        init_image: Image.Image,
        mask_np: np.ndarray,
        prompt: str,
        height: int = 1024,
        width: int = 2048,
        timestep: int = 50,
        seed: int = 0,
        tau: int = 50,
        guidance_scale: float = 3.5,
        use_rf_solver_invert: bool = False,
        use_rf_solver_sample: bool = False,
        source_key_scale: float = 1.0,
        source_value_scale: float = 1.0,
    ) -> Image.Image:

        from third_party.dit360.pa_src.attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor
        
        latent_h = height // (self.pipe.vae_scale_factor * 2)
        latent_w = width // (self.pipe.vae_scale_factor * 2)
        img_dims = latent_h * (latent_w + 2)
        
        mask_full = torch.tensor(np.where(mask_np == 255, 1, 0), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        block_h = height // latent_h
        block_w = width // latent_w
        
        kernel = torch.ones(1, 1, block_h, block_w, device=mask_full.device)
        conv_result = F.conv2d(mask_full, kernel, stride=(block_h, block_w))
        mask_block = (conv_result == (block_h * block_w)).float()
        
        mask = mask_block[0, 0]
        mask = torch.cat([mask[:, 0:1], mask, mask[:, -1:]], dim=-1).view(-1, 1)

        inverted_latents, image_latents, latent_image_ids = self.pipe.invert(
            source_prompt="",
            image=init_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,
            use_rf_solver_invert=use_rf_solver_invert,
        )
        
        set_flux_transformer_attn_processor(
            self.pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name,
                tau=tau / 100,
                mask=mask,
                device=self.device,
                img_dims=img_dims,
                source_key_scale=source_key_scale,
                source_value_scale=source_value_scale,
            ),
        )
        
        image = self.pipe(
            [prompt, prompt],
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            start_timestep=0.0,
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            mask=mask,
            use_timestep=True,
            use_rf_solver_sample=use_rf_solver_sample,
        ).images[1]
        
        return image
