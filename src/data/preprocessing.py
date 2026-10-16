"""
Preprocessing and augmentations for multimodal RGB and NIR produce imagery.
Handles dual-stream transforms, normalization, and spectral index calculations.
"""
from typing import Tuple, Optional, Union
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

NIR_MEAN = [0.5]
NIR_STD = [0.25]


def get_rgb_transforms(is_training: bool = False, img_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Return PyTorch transform pipeline for RGB imagery."""
    if is_training:
        return T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            T.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])


def get_nir_transforms(is_training: bool = False, img_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """Return PyTorch transform pipeline for single-channel NIR imagery."""
    if is_training:
        return T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=NIR_MEAN, std=NIR_STD),
        ])
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=NIR_MEAN, std=NIR_STD),
    ])


def preprocess_pair(
    rgb_img: Union[Image.Image, np.ndarray, str],
    nir_img: Union[Image.Image, np.ndarray, str],
    img_size: Tuple[int, int] = (224, 224),
    is_training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess an aligned pair of RGB and NIR images into model-ready tensors.
    
    Args:
        rgb_img: PIL Image, numpy array (H, W, 3), or file path
        nir_img: PIL Image, numpy array (H, W) or (H, W, 1), or file path
        img_size: target dimensions (H, W)
        is_training: whether to apply training augmentations
        
    Returns:
        rgb_tensor: (3, H, W) float32 tensor normalized
        nir_tensor: (1, H, W) float32 tensor normalized
    """
    if isinstance(rgb_img, str):
        rgb_pil = Image.open(rgb_img).convert("RGB")
    elif isinstance(rgb_img, np.ndarray):
        rgb_pil = Image.fromarray(rgb_img).convert("RGB")
    else:
        rgb_pil = rgb_img.convert("RGB")

    if isinstance(nir_img, str):
        nir_pil = Image.open(nir_img).convert("L")
    elif isinstance(nir_img, np.ndarray):
        if nir_img.ndim == 3 and nir_img.shape[-1] == 1:
            nir_img = nir_img.squeeze(-1)
        nir_pil = Image.fromarray(nir_img).convert("L")
    else:
        nir_pil = nir_img.convert("L")

    rgb_transform = get_rgb_transforms(is_training=is_training, img_size=img_size)
    nir_transform = get_nir_transforms(is_training=is_training, img_size=img_size)

    rgb_tensor = rgb_transform(rgb_pil)
    nir_tensor = nir_transform(nir_pil)

    return rgb_tensor, nir_tensor


def compute_spectral_contrast_index(rgb_tensor: torch.Tensor, nir_tensor: torch.Tensor) -> float:
    """
    Compute a normalized spectral differential index between Red channel and NIR channel.
    Analogous to NDVI: (NIR - Red) / (NIR + Red + 1e-6).
    In healthy fruit with intact cellular tissue, NIR reflectance is high; as cell walls
    degrade and water/sugar decomposure happens, internal scattering drops.
    """
    # Unnormalize tensors roughly to [0, 1] range for index calculation
    red_channel = (rgb_tensor[0] * RGB_STD[0]) + RGB_MEAN[0]
    nir_channel = (nir_tensor[0] * NIR_STD[0]) + NIR_MEAN[0]
    
    red_mean = torch.clamp(red_channel.mean(), 0.0, 1.0)
    nir_mean = torch.clamp(nir_channel.mean(), 0.0, 1.0)
    
    sci = (nir_mean - red_mean) / (nir_mean + red_mean + 1e-6)
    return float(sci.item())
