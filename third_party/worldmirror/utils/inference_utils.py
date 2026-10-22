import torch
from PIL import Image
from torchvision import transforms

import glob
import tempfile
import os
from third_party.worldmirror.utils.video_utils import video_to_image_frames

IMAGE_EXTS = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.webm', '.gif']
    




def _handle_alpha_channel(img_data):
    """Process RGBA images by blending with white background."""
    if img_data.mode == "RGBA":
        white_bg = Image.new("RGBA", img_data.size, (255, 255, 255, 255))
        img_data = Image.alpha_composite(white_bg, img_data)
    return img_data.convert("RGB")


def _calculate_resize_dims(orig_w, orig_h, max_dim, resize_strategy, patch_size=14):
    """Calculate new dimensions based on resize strategy."""
    if resize_strategy == "pad":
        if orig_w >= orig_h:
            new_w = max_dim
            new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
        else:
            new_h = max_dim
            new_w = round(orig_w * (new_h / orig_h) / patch_size) * patch_size
    else:  # crop strategy
        if orig_w >= orig_h:
            new_w = max_dim
            new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
        else:
            new_h = max_dim
            new_w = round(orig_w * (new_h / orig_h) / patch_size) * patch_size
    return new_w, new_h


def _apply_padding(tensor_img, target_dim):
    """Apply padding to make tensor square."""
    h_pad = target_dim - tensor_img.shape[1]
    w_pad = target_dim - tensor_img.shape[2]
    
    if h_pad > 0 or w_pad > 0:
        pad_top, pad_bottom = h_pad // 2, h_pad - h_pad // 2
        pad_left, pad_right = w_pad // 2, w_pad - w_pad // 2
        return torch.nn.functional.pad(
            tensor_img, (pad_left, pad_right, pad_top, pad_bottom), 
            mode="constant", value=1.0
        )
    return tensor_img


def prepare_images_to_tensor(file_paths, resize_strategy="crop", target_size=518):
    """
    Process image files into uniform tensor batch for model input.
    
    Args:
        file_paths (list): Paths to image files
        resize_strategy (str): "crop" or "pad" processing mode
        target_size (int): Target size for processing
        
    Returns:
        torch.Tensor: Processed image batch (1, N, 3, H, W)
    """
    if not file_paths:
        raise ValueError("At least 1 image is required")
    
    if resize_strategy not in ["crop", "pad"]:
        raise ValueError("Strategy must be 'crop' or 'pad'")
    
    tensor_list = []
    dimension_set = set()
    converter = transforms.ToTensor()
    
    # Process each image file
    for file_path in file_paths:
        img_data = Image.open(file_path)
        img_data = _handle_alpha_channel(img_data)
        
        orig_w, orig_h = img_data.size
        new_w, new_h = _calculate_resize_dims(orig_w, orig_h, target_size, resize_strategy)
        
        # Resize and convert to tensor
        img_data = img_data.resize((new_w, new_h), Image.Resampling.BICUBIC)
        tensor_img = converter(img_data)
        
        # Apply center crop for crop strategy
        if resize_strategy == "crop" and new_h > target_size:
            crop_start = (new_h - target_size) // 2
            tensor_img = tensor_img[:, crop_start:crop_start + target_size, :]
        
        # Apply padding for pad strategy
        if resize_strategy == "pad":
            tensor_img = _apply_padding(tensor_img, target_size)
        
        dimension_set.add((tensor_img.shape[1], tensor_img.shape[2]))
        tensor_list.append(tensor_img)
    
    # Handle mixed dimensions
    if len(dimension_set) > 1:
        print(f"Warning: Mixed image dimensions found: {dimension_set}")
        max_h = max(dims[0] for dims in dimension_set)
        max_w = max(dims[1] for dims in dimension_set)
        
        tensor_list = [_apply_padding(img, max(max_h, max_w)) if img.shape[1] != max_h or img.shape[2] != max_w 
                      else img for img in tensor_list]
    
    batch_tensor = torch.stack(tensor_list)
    
    # Ensure proper batch dimensions
    if batch_tensor.dim() == 3:
        batch_tensor = batch_tensor.unsqueeze(0)
    
    return batch_tensor.unsqueeze(0)


def extract_load_and_preprocess_images(image_folder_or_video_path, fps=1, target_size=518, mode="crop"):
    from pathlib import Path
    path = Path(image_folder_or_video_path)
    
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        temp_dir = Path(tempfile.mkdtemp(prefix="hunyuan_frames_"))
        frame_paths = video_to_image_frames(str(path), str(temp_dir), fps=fps)
        img_paths = sorted(frame_paths)
    else:
        img_paths = []
        for ext in IMAGE_EXTS:
            img_paths.extend(glob.glob(os.path.join(str(path), ext)))
        img_paths = sorted(img_paths)
    images = prepare_images_to_tensor(img_paths, resize_strategy=mode, target_size=target_size)
    return images
