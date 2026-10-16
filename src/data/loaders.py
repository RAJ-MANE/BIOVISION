"""
Dataset loaders and sample generators for multimodal RGB + NIR produce monitoring.
Supports real file paths and synthetic dataset generation for development/testing.
"""
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.data.preprocessing import get_rgb_transforms, get_nir_transforms


class SpectralProduceDataset(Dataset):
    """
    Multimodal produce dataset loading paired RGB and NIR images along with
    ground truth ripeness stage, remaining shelf-life (days), and degradation score.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        is_training: bool = False,
        img_size: Tuple[int, int] = (224, 224),
    ):
        self.df = dataframe.reset_index(drop=True)
        self.is_training = is_training
        self.img_size = img_size
        self.rgb_transforms = get_rgb_transforms(is_training=is_training, img_size=img_size)
        self.nir_transforms = get_nir_transforms(is_training=is_training, img_size=img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        rgb_path = row.get("rgb_path")
        nir_path = row.get("nir_path")

        # Load RGB
        if rgb_path and os.path.exists(str(rgb_path)):
            rgb_img = Image.open(rgb_path).convert("RGB")
        else:
            # Fallback or synthetic in-memory generation
            stage = int(row.get("stage", 2))
            rgb_img = self._create_synthetic_avocado_rgb(stage)

        # Load NIR
        if nir_path and os.path.exists(str(nir_path)):
            nir_img = Image.open(nir_path).convert("L")
        else:
            stage = int(row.get("stage", 2))
            degradation = float(row.get("degradation_score", 0.3))
            nir_img = self._create_synthetic_avocado_nir(stage, degradation)

        rgb_tensor = self.rgb_transforms(rgb_img)
        nir_tensor = self.nir_transforms(nir_img)

        # Convert stage from 1-based (1-5) to 0-based (0-4) for classification
        raw_stage = int(row.get("stage", 3))
        stage_idx = max(0, min(4, raw_stage - 1))

        shelf_life_days = float(row.get("shelf_life_days", 5.0))
        degradation_score = float(row.get("degradation_score", 0.35))

        return {
            "rgb": rgb_tensor,
            "nir": nir_tensor,
            "stage_idx": torch.tensor(stage_idx, dtype=torch.long),
            "shelf_life_days": torch.tensor(shelf_life_days, dtype=torch.float32),
            "degradation_score": torch.tensor(degradation_score, dtype=torch.float32),
            "fruit_id": str(row.get("fruit_id", f"FRUIT_{idx:03d}")),
            "sample_id": str(row.get("sample_id", f"SMP_{idx:04d}")),
            "stage_name": str(row.get("stage_name", f"Stage {raw_stage}")),
        }

    @staticmethod
    def _create_synthetic_avocado_rgb(stage: int) -> Image.Image:
        """Generate synthetic avocado RGB image corresponding to ripeness stage."""
        img = Image.new("RGB", (224, 224), (245, 245, 247))
        draw = ImageDraw.Draw(img)

        skin_colors = [
            (52, 114, 43),    # Stage 1: Bright emerald green
            (46, 85, 38),     # Stage 2: Olive/dark green
            (38, 55, 32),     # Stage 3: Dark olive / brownish green
            (30, 32, 28),     # Stage 4: Dark purplish black
            (20, 18, 16),     # Stage 5: Dull bruised black
        ]
        color = skin_colors[max(0, min(4, stage - 1))]
        draw.ellipse([45, 30, 179, 200], fill=color, outline=(15, 20, 15))
        np_arr = np.array(img)
        noise = np.random.normal(0, 12, np_arr.shape).astype(np.int16)
        noisy = np.clip(np_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy).filter(ImageFilter.SMOOTH_MORE)

    @staticmethod
    def _create_synthetic_avocado_nir(stage: int, degradation: float) -> Image.Image:
        """Generate synthetic NIR image showing avocado internal vascular breakdown."""
        img = Image.new("L", (224, 224), 20)
        draw = ImageDraw.Draw(img)
        base_val = int(220 - (degradation * 140))
        base_val = max(50, min(230, base_val))
        draw.ellipse([45, 30, 179, 200], fill=base_val)

        if degradation > 0.3:
            num_spots = int(degradation * 8)
            for _ in range(num_spots):
                cx = np.random.randint(70, 150)
                cy = np.random.randint(60, 170)
                r = np.random.randint(8, 24)
                lesion_val = max(20, base_val - 60)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=lesion_val)

        return img.filter(ImageFilter.GaussianBlur(radius=3))

    @staticmethod
    def _create_synthetic_apple_rgb(stage: int) -> Image.Image:
        """Generate synthetic apple RGB image. Firm red skin with natural highlights."""
        img = Image.new("RGB", (224, 224), (245, 245, 247))
        draw = ImageDraw.Draw(img)
        apple_colors = [
            (140, 180, 50),   # Stage 1: Granny Smith green / yellow blush
            (195, 60, 45),    # Stage 2: Crisp Honeycrisp red with yellow streaks
            (180, 35, 35),    # Stage 3: Rich ruby red (Prime retail)
            (145, 28, 30),    # Stage 4: Deep crimson / softening
            (95, 35, 30),     # Stage 5: Dull bruised brownish red
        ]
        color = apple_colors[max(0, min(4, stage - 1))]
        draw.ellipse([40, 40, 184, 195], fill=color, outline=(80, 20, 20))
        # Apple top cleft indent
        draw.ellipse([100, 32, 124, 52], fill=(245, 245, 247))
        # Small stem
        draw.line([112, 36, 112, 22], fill=(90, 60, 30), width=4)
        np_arr = np.array(img)
        noise = np.random.normal(0, 8, np_arr.shape).astype(np.int16)
        noisy = np.clip(np_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy).filter(ImageFilter.SMOOTH)

    @staticmethod
    def _create_synthetic_apple_nir(stage: int, degradation: float) -> Image.Image:
        """Generate synthetic NIR image showing apple internal core browning and watercore."""
        img = Image.new("L", (224, 224), 20)
        draw = ImageDraw.Draw(img)
        base_val = int(225 - (degradation * 130))
        base_val = max(55, min(235, base_val))
        draw.ellipse([40, 40, 184, 195], fill=base_val)

        # Internal core browning: concentrated around center core
        if degradation > 0.25:
            core_radius = int(12 + degradation * 35)
            core_darkness = max(15, base_val - int(degradation * 110))
            draw.ellipse([112 - core_radius, 115 - core_radius, 112 + core_radius, 115 + core_radius], fill=core_darkness)
            # Watercore radiating streaks
            for angle in [0, 60, 120, 180, 240, 300]:
                rad = np.radians(angle)
                ex = int(112 + np.cos(rad) * (core_radius + 15))
                ey = int(115 + np.sin(rad) * (core_radius + 15))
                draw.line([112, 115, ex, ey], fill=core_darkness, width=5)

        return img.filter(ImageFilter.GaussianBlur(radius=4))

    @staticmethod
    def _create_synthetic_mango_rgb(stage: int) -> Image.Image:
        """Generate synthetic Alphonso mango RGB image. Vibrant golden yellow with sunset blush."""
        img = Image.new("RGB", (224, 224), (245, 245, 247))
        draw = ImageDraw.Draw(img)
        mango_colors = [
            (85, 145, 55),    # Stage 1: Unripe raw green
            (195, 175, 45),   # Stage 2: Breaking greenish yellow
            (245, 160, 25),   # Stage 3: Golden saffron with reddish blush (Prime)
            (225, 125, 20),   # Stage 4: Deep ripe orange-amber
            (130, 85, 30),    # Stage 5: Dull wrinkled brownish orange
        ]
        color = mango_colors[max(0, min(4, stage - 1))]
        # Asymmetrical kidney/mango shape
        draw.chord([42, 35, 182, 200], start=30, end=330, fill=color, outline=(70, 70, 20))
        draw.ellipse([60, 50, 175, 190], fill=color)
        np_arr = np.array(img)
        noise = np.random.normal(0, 10, np_arr.shape).astype(np.int16)
        noisy = np.clip(np_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy).filter(ImageFilter.SMOOTH)

    @staticmethod
    def _create_synthetic_mango_nir(stage: int, degradation: float) -> Image.Image:
        """Generate synthetic NIR image showing mango spongy tissue and stem-end rot."""
        img = Image.new("L", (224, 224), 20)
        draw = ImageDraw.Draw(img)
        base_val = int(230 - (degradation * 135))
        base_val = max(60, min(235, base_val))
        draw.chord([42, 35, 182, 200], start=30, end=330, fill=base_val)
        draw.ellipse([60, 50, 175, 190], fill=base_val)

        # Spongy tissue breakdown: internal cavity and stem-end browning
        if degradation > 0.28:
            # Stem cavity rot
            draw.ellipse([85, 45, 135, 95], fill=max(20, base_val - 85))
            # Internal spongy tissue spots
            num_cavities = int(degradation * 6)
            for _ in range(num_cavities):
                cx = np.random.randint(80, 150)
                cy = np.random.randint(90, 165)
                r = np.random.randint(10, 22)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=max(25, base_val - 70))

        return img.filter(ImageFilter.GaussianBlur(radius=4))

    @staticmethod
    def _create_synthetic_strawberry_rgb(stage: int) -> Image.Image:
        """Generate synthetic strawberry RGB image. Glossy ruby red with leafy green crown."""
        img = Image.new("RGB", (224, 224), (245, 245, 247))
        draw = ImageDraw.Draw(img)
        berry_colors = [
            (210, 225, 170),  # Stage 1: Green/white tip underripe
            (235, 115, 105),  # Stage 2: Light pinkish red
            (225, 25, 40),    # Stage 3: Vibrant glossy ruby red (Prime)
            (165, 15, 30),    # Stage 4: Dark soft burgundy
            (95, 30, 35),     # Stage 5: Soft senescent brownish purple
        ]
        color = berry_colors[max(0, min(4, stage - 1))]
        # Strawberry conical heart shape
        draw.polygon([(112, 195), (55, 75), (80, 50), (112, 60), (144, 50), (169, 75)], fill=color)
        draw.ellipse([55, 52, 169, 135], fill=color)
        # Green leafy calyx at top
        draw.polygon([(112, 60), (90, 30), (105, 45), (112, 25), (120, 45), (135, 30)], fill=(34, 139, 34))
        # Surface achenes (seeds)
        for sx, sy in [(90, 90), (120, 85), (105, 115), (135, 110), (95, 140), (125, 145), (112, 165)]:
            draw.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=(245, 230, 110))
        np_arr = np.array(img)
        noise = np.random.normal(0, 6, np_arr.shape).astype(np.int16)
        noisy = np.clip(np_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy).filter(ImageFilter.SMOOTH)

    @staticmethod
    def _create_synthetic_strawberry_nir(stage: int, degradation: float) -> Image.Image:
        """Generate synthetic NIR image showing latent sub-surface mycelia and cell-wall collapse."""
        img = Image.new("L", (224, 224), 20)
        draw = ImageDraw.Draw(img)
        base_val = int(220 - (degradation * 140))
        base_val = max(45, min(230, base_val))
        draw.polygon([(112, 195), (55, 75), (80, 50), (112, 60), (144, 50), (169, 75)], fill=base_val)
        draw.ellipse([55, 52, 169, 135], fill=base_val)

        # Latent fungal mycelia & cellular collapse: irregular dark moisture sink pockets
        if degradation > 0.22:
            num_macerations = int(degradation * 7)
            for _ in range(num_macerations):
                cx = np.random.randint(75, 145)
                cy = np.random.randint(75, 160)
                r = np.random.randint(8, 20)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=max(15, base_val - 80))

        return img.filter(ImageFilter.GaussianBlur(radius=3))

    @classmethod
    def get_synthetic_produce_pair(
        cls,
        produce_type: str,
        stage: int,
        degradation: float,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Unified factory generating paired RGB and NIR images for any supported produce category.
        """
        p_clean = produce_type.lower()
        if "apple" in p_clean:
            return cls._create_synthetic_apple_rgb(stage), cls._create_synthetic_apple_nir(stage, degradation)
        elif "mango" in p_clean:
            return cls._create_synthetic_mango_rgb(stage), cls._create_synthetic_mango_nir(stage, degradation)
        elif "berry" in p_clean or "strawberr" in p_clean:
            return cls._create_synthetic_strawberry_rgb(stage), cls._create_synthetic_strawberry_nir(stage, degradation)
        else:
            # Default to Avocado
            return cls._create_synthetic_avocado_rgb(stage), cls._create_synthetic_avocado_nir(stage, degradation)



def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: Tuple[int, int] = (224, 224),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating PyTorch DataLoaders for train, val, and test splits."""
    train_ds = SpectralProduceDataset(train_df, is_training=True, img_size=img_size)
    val_ds = SpectralProduceDataset(val_df, is_training=False, img_size=img_size)
    test_ds = SpectralProduceDataset(test_df, is_training=False, img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
