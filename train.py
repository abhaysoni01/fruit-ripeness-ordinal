"""
train.py — Main training/validation engine for fruit-ripeness grading.

Supports both model variants (``"baseline"`` and ``"ordinal"``) through
a unified loop.  The script can be run directly::

    python train.py --variant baseline
    python train.py --variant ordinal

Key features
------------
* Deterministic seeding (NumPy, Python stdlib, PyTorch CPU+CUDA).
* AdamW optimiser with ``ReduceLROnPlateau`` scheduler keyed on val MAE.
* Best-checkpoint saving keyed on lowest validation MAE.
* Early stopping with configurable patience.
* CSV training-log output for downstream analysis/plotting.
* Epoch-level console progress with loss, accuracy, and MAE.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config
from dataset import build_dataloaders, get_class_names
from losses import build_loss_fn
from metrics import compute_all_metrics
from models import BaselineNet, CoralNet, build_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = config.SEED) -> None:
    """
    Set all relevant RNG seeds for full determinism.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA kernels (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Per-batch inference helpers
# ---------------------------------------------------------------------------

def _predict_from_logits(
    logits: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    """
    Convert raw model output to integer class predictions.

    Parameters
    ----------
    logits:
        * Baseline: shape ``(B, K)`` — argmax gives the class.
        * Ordinal:  shape ``(B, K-1)`` — threshold sum gives the class.
    variant:
        ``"baseline"`` or ``"ordinal"``.

    Returns
    -------
    Predicted integer labels, shape ``(B,)``.
    """
    if variant == config.MODEL_BASELINE:
        return torch.argmax(logits, dim=1)
    else:  # ordinal
        return CoralNet.logits_to_label(logits)


# ---------------------------------------------------------------------------
# Single epoch — train
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    variant: str,
    scaler: torch.amp.GradScaler,
    num_classes: int = config.NUM_CLASSES,
) -> Tuple[float, float, float]:
    """
    Run one full training epoch.

    Parameters
    ----------
    model:
        The model to train (moved to *device* by the caller).
    loader:
        Training DataLoader.
    optimizer:
        Optimiser (AdamW).
    loss_fn:
        Callable ``(logits, targets) → scalar``.
    device:
        Compute device.
    variant:
        Model variant string (``"baseline"`` or ``"ordinal"``).

    Returns
    -------
    (avg_loss, accuracy, mae)
        Average per-batch loss, top-1 accuracy, and MAE over the epoch.
    """
    model.train()
    if hasattr(model, "backbone") and not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    ):
        model.backbone.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=scaler.is_enabled(),
        ):
            logits = model(images)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = _predict_from_logits(logits, variant)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_all_metrics(all_targets, all_preds, num_classes=num_classes)
    return avg_loss, metrics["accuracy"], metrics["mae"]


# ---------------------------------------------------------------------------
# Single epoch — evaluate (val / test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    variant: str,
    use_amp: bool = False,
    num_classes: int = config.NUM_CLASSES,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on a DataLoader (no gradients computed).

    Parameters
    ----------
    model, loader, loss_fn, device, variant:
        Same as :func:`train_one_epoch`.

    Returns
    -------
    (avg_loss, accuracy, mae, off_by_1_acc)
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = loss_fn(logits, labels)

        total_loss += loss.item()
        preds = _predict_from_logits(logits, variant)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_all_metrics(all_targets, all_preds, num_classes=num_classes)
    return (
        avg_loss,
        metrics["accuracy"],
        metrics["mae"],
        metrics["off_by_1_acc"],
    )


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """
    Append epoch-level metrics to a CSV file for later analysis.

    Parameters
    ----------
    filepath:
        Destination CSV path (created on first write).
    """

    FIELDS = [
        "epoch", "variant",
        "train_loss", "train_acc", "train_mae",
        "val_loss", "val_acc", "val_mae", "val_off1_acc",
        "lr",
    ]

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self._file = open(filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: Dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    variant: str,
    data_root: Optional[Path] = None,
    num_epochs: int = config.NUM_EPOCHS,
    seed: int = config.SEED,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Full training pipeline for one model variant.

    Parameters
    ----------
    variant:
        ``"baseline"`` or ``"ordinal"``.
    data_root:
        Dataset root directory.  Defaults to ``config.DATA_ROOT``.
    num_epochs:
        Maximum training epochs.
    seed:
        Global RNG seed for reproducibility.
    device:
        Compute device.  Auto-detected when ``None``.

    Returns
    -------
    model:
        Best model (lowest val MAE) with weights loaded from checkpoint.
    """
    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this PyTorch installation. "
            "Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available()."
        )
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"CUDA device {device.index} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )

    use_amp = device.type == "cuda"
    effective_batch_size = (
        batch_size if batch_size is not None else (8 if use_amp else config.BATCH_SIZE)
    )
    if effective_batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {effective_batch_size}.")
    class_names = get_class_names(data_root)
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("Ordinal training requires at least two ordered classes.")

    print(f"\n{'='*60}")
    print(f"  Training variant: [{variant.upper()}]")
    print(f"  Device          : {device}")
    if use_amp:
        print(f"  GPU             : {torch.cuda.get_device_name(device)}")
    print(f"  Batch size      : {effective_batch_size}")
    print(f"  Mixed precision : {'enabled' if use_amp else 'disabled'}")
    print(f"  Classes         : {' -> '.join(class_names)}")
    print(f"  Seed            : {seed}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ Data
    train_loader, val_loader, _ = build_dataloaders(
        data_root=data_root,
        seed=seed,
        batch_size=effective_batch_size,
        pin_memory=use_amp,
        balance_classes=True,
    )

    # ----------------------------------------------------------------- Model
    model = build_model(variant, device, num_classes=num_classes)
    if freeze_backbone:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise ValueError(f"Model variant '{variant}' does not have a backbone to freeze.")
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    # --------------------------------------------------------------- Loss fn
    loss_fn = build_loss_fn(variant, num_classes=num_classes)

    # ------------------------------------------------------------ Optimiser
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ----------------------------------------------------------- Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",           # monitor val MAE
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.LR_MIN,
    )

    # ------------------------------------------------------- Logging / state
    log_path = config.LOG_DIR / f"training_{variant}.csv"
    logger = CSVLogger(log_path)

    best_val_mae: float = float("inf")
    ckpt_path = config.checkpoint_path(variant)
    epochs_no_improve: int = 0

    # ------------------------------------------------------------- Epoch loop
    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()

        # Train
        train_loss, train_acc, train_mae = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            variant,
            scaler,
            num_classes=num_classes,
        )

        # Validate
        val_loss, val_acc, val_mae, val_off1 = evaluate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            variant,
            use_amp=use_amp,
            num_classes=num_classes,
        )

        elapsed = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Console summary
        print(
            f"Epoch [{epoch:3d}/{num_epochs}] "
            f"| {elapsed:5.1f}s "
            f"| LR={current_lr:.2e} "
            f"| Train  loss={train_loss:.4f}  acc={train_acc:.4f}  mae={train_mae:.4f} "
            f"| Val    loss={val_loss:.4f}  acc={val_acc:.4f}  mae={val_mae:.4f}  "
            f"off1={val_off1:.4f}"
        )

        # CSV log
        logger.log({
            "epoch": epoch, "variant": variant,
            "train_loss": f"{train_loss:.6f}",
            "train_acc":  f"{train_acc:.6f}",
            "train_mae":  f"{train_mae:.6f}",
            "val_loss":   f"{val_loss:.6f}",
            "val_acc":    f"{val_acc:.6f}",
            "val_mae":    f"{val_mae:.6f}",
            "val_off1_acc": f"{val_off1:.6f}",
            "lr": f"{current_lr:.8f}",
        })

        # LR schedule step
        scheduler.step(val_mae)

        # Checkpointing — save on best val MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "variant": variant,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "dataset_root": str((data_root or config.DATA_ROOT).resolve()),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(f"  [+] Checkpoint saved  (val_mae={val_mae:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(
                f"\n[Early Stopping] No improvement for "
                f"{config.EARLY_STOP_PATIENCE} epochs.  Stopping."
            )
            break

    logger.close()

    # ---------------------------------------- Load best weights before return
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"\n[Done] Best val MAE = {best_val_mae:.4f} "
        f"(epoch {checkpoint['epoch']})"
    )
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fruit-ripeness grading model (baseline or ordinal)."
    )
    parser.add_argument(
        "--variant",
        choices=[config.MODEL_BASELINE, config.MODEL_ORDINAL],
        default=config.MODEL_BASELINE,
        help="Model variant to train (default: baseline).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help=f"Number of training epochs (default: {config.NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help=f"Random seed (default: {config.SEED}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g. 'cuda:0' or 'cpu' (auto-detected by default).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (defaults to 8 on CUDA and the configured size on CPU).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the classifier head; faster on CPU, but limits fine-tuning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device) if args.device else None
    train(
        variant=args.variant,
        data_root=args.data_root,
        num_epochs=args.epochs,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        freeze_backbone=args.freeze_backbone,
    )
