"""
BioVision Model Training & Fine-Tuning Script.
Trains RGB baseline, NIR spectral model, or Dual-Stream Fusion network
using multi-task loss (Ripeness Cross-Entropy + Shelf-Life Smooth L1 + Degradation BCE).
"""
import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.fusion_model import DualStreamSpectralFusionModel
from src.models.rgb_model import RGBProduceModel
from src.models.spectral_model import NIRSpectralModel
from src.data.loaders import SpectralProduceDataset, create_dataloaders
from src.data.splits import create_fruit_id_splits
from src.evaluation import evaluate_model_on_loader


def generate_mock_produce_metadata(num_fruits: int = 50, days_per_fruit: int = 5) -> pd.DataFrame:
    """Generate realistic synthetic specimen tracking metadata for training/demo."""
    rows = []
    np.random.seed(42)

    for f_idx in range(num_fruits):
        fruit_id = f"FRUIT_{f_idx:03d}"
        initial_stage = np.random.randint(1, 3)

        for day in range(days_per_fruit):
            current_stage = min(5, initial_stage + int(day * 0.8))
            shelf_life = max(0.5, 12.0 - (current_stage * 2.2) + np.random.normal(0, 0.4))
            degradation = min(1.0, max(0.05, (current_stage - 1) * 0.22 + np.random.normal(0, 0.03)))

            rows.append({
                "sample_id": f"SMP_{f_idx:03d}_{day:02d}",
                "fruit_id": fruit_id,
                "stage": current_stage,
                "stage_name": f"Stage {current_stage}",
                "shelf_life_days": round(float(shelf_life), 2),
                "degradation_score": round(float(degradation), 3),
                "rgb_path": None,
                "nir_path": None,
            })

    return pd.DataFrame(rows)


def train_model(
    model_type: str = "fusion",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    checkpoint_dir: str = "checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Starting training for '{model_type}' on device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Prepare Data with Leakage-Safe Fruit Splitting
    df = generate_mock_produce_metadata(num_fruits=40, days_per_fruit=5)
    train_df, val_df, test_df = create_fruit_id_splits(df, fruit_id_col="fruit_id", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"[Data] Total samples: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size=batch_size)

    # 2. Instantiate Model
    if model_type == "fusion":
        model = DualStreamSpectralFusionModel(num_classes=5, feature_dim=256, rgb_pretrained=False)
    elif model_type == "rgb_only":
        model = RGBProduceModel(num_classes=5, feature_dim=256, pretrained=False)
    elif model_type == "nir_only":
        model = NIRSpectralModel(in_channels=1, num_classes=5, feature_dim=256)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    # 3. Loss & Optimizer
    ce_loss_fn = nn.CrossEntropyLoss()
    sl_loss_fn = nn.SmoothL1Loss()
    bce_loss_fn = nn.BCELoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    save_path = os.path.join(checkpoint_dir, f"{model_type}_model.pt")

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)
            stage_idx = batch["stage_idx"].to(device)
            shelf_life = batch["shelf_life_days"].to(device)
            degradation = batch["degradation_score"].to(device)

            optimizer.zero_grad()

            if model_type == "fusion":
                outputs = model(rgb, nir)
            elif model_type == "rgb_only":
                outputs = model(rgb)
            elif model_type == "nir_only":
                outputs = model(nir)

            loss_stage = ce_loss_fn(outputs["logits"], stage_idx)
            loss_sl = sl_loss_fn(outputs["shelf_life_days"], shelf_life)
            loss_deg = bce_loss_fn(outputs["degradation_score"], degradation)

            loss = loss_stage + (0.5 * loss_sl) + (0.5 * loss_deg)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        val_metrics = evaluate_model_on_loader(model, val_loader, device, model_mode=model_type)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.1f}% | Val Shelf-Life MAE: {val_metrics['shelf_life_mae_days']:.2f}d"
        )

        # Save checkpoint
        if val_metrics["shelf_life_mae_days"] < best_val_loss:
            best_val_loss = val_metrics["shelf_life_mae_days"]
            torch.save(model.state_dict(), save_path)
            print(f"  [Checkpoint] Saved new best model to {save_path}")

    # Final test evaluation
    print(f"\n[Evaluation] Evaluating best model on held-out test split...")
    model.load_state_dict(torch.load(save_path))
    test_metrics = evaluate_model_on_loader(model, test_loader, device, model_mode=model_type)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}% | Test MAE: {test_metrics['shelf_life_mae_days']:.2f} days")
    print(f"Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BioVision models")
    parser.add_argument("--model", type=str, default="fusion", choices=["fusion", "rgb_only", "nir_only"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    train_model(model_type=args.model, epochs=args.epochs, batch_size=args.batch_size)
