"""
evaluate.py — Post-training evaluation, comparative table, and Grad-CAM visualisations.

Usage::

    # Evaluate both variants on the held-out test set
    python evaluate.py

    # Evaluate a single variant
    python evaluate.py --variant ordinal

    # Skip Grad-CAM generation
    python evaluate.py --no-gradcam

Outputs
-------
* Console: side-by-side comparison table (MAE, Accuracy, Off-by-1).
* Console: per-class confusion matrices.
* Files:   ``gradcam_outputs/<variant>_class<k>_sample<n>.png``
* File:    ``logs/evaluation_results.csv``

Grad-CAM implementation
-----------------------
The standard Gradient-weighted Class Activation Mapping (Grad-CAM) approach
is used (Selvaraju et al., 2017).  For the ordinal model the gradient is
taken with respect to the *cumulative predicted rank* (sum of sigmoid logits),
which reflects the full ordinal score rather than any single binary task.

References
----------
Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep
    networks via gradient-based localization. ICCV.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import config
from dataset import build_dataloaders, build_eval_transform, get_class_names
from fruit_crop import crop_fruit
from losses import build_loss_fn
from metrics import (
    compute_all_metrics,
    format_metrics_table,
    print_confusion_matrix,
)
from models import BaselineNet, CoralNet, build_model
from train import _predict_from_logits, evaluate_one_epoch, seed_everything


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet-18.

    Parameters
    ----------
    model:
        The model to explain.
    target_layer_name:
        Name of the convolutional layer to hook (e.g. ``"layer4"``).
    variant:
        ``"baseline"`` or ``"ordinal"`` — controls how the scalar
        target output is computed for backprop.
    device:
        Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str,
        variant: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.variant = variant
        self.device = device

        # Register hooks on the target layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    def _save_activations(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    def _scalar_target(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the scalar used for backprop.

        * Baseline : sum of logits for the predicted class (equivalent
          to the pre-softmax score of the top class).
        * Ordinal  : cumulative sigmoid sum (the predicted ordinal rank
          as a continuous score).
        """
        if self.variant == config.MODEL_BASELINE:
            pred_class = logits.argmax(dim=1)
            return logits[0, pred_class]
        else:
            return torch.sigmoid(logits).sum()

    # ------------------------------------------------------------------
    def generate(
        self,
        image_tensor: torch.Tensor,
    ) -> Tuple[np.ndarray, int]:
        """
        Produce a Grad-CAM heat-map for a single image.

        Parameters
        ----------
        image_tensor:
            Pre-processed image tensor of shape ``(1, 3, H, W)``.

        Returns
        -------
        heatmap:
            Normalised heat-map array of shape ``(H, W)`` in [0, 1].
        pred_label:
            Predicted class index.
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)

        # Forward
        logits = self.model(image_tensor)
        pred_label = _predict_from_logits(logits, self.variant).item()

        # Backward through scalar target
        self.model.zero_grad()
        scalar = self._scalar_target(logits)
        scalar.backward()

        # Pooled gradients × activations
        gradients = self._gradients       # (1, C, H', W')
        activations = self._activations   # (1, C, H', W')

        if gradients is None or activations is None:
            raise RuntimeError("Hooks did not fire — check target_layer_name.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Resize to input resolution and normalise
        cam = F.interpolate(
            cam,
            size=config.IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, int(pred_label)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Blend a jet-coloured heatmap over the original image.

    Parameters
    ----------
    original_image:
        PIL RGB image.
    heatmap:
        Float array in [0, 1] of shape ``(H, W)``.
    alpha:
        Heatmap blend weight (0 = image only, 1 = heatmap only).

    Returns
    -------
    Blended PIL image.
    """
    import matplotlib as mpl
    import matplotlib.cm as cm  # lazy import to keep hard deps optional

    try:
        colormap = mpl.colormaps["jet"]
    except (AttributeError, KeyError):
        colormap = cm.get_cmap("jet")
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb).resize(original_image.size)

    blended = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=alpha)
    return blended


def save_gradcam_grid(
    images: List[Image.Image],
    heatmaps: List[np.ndarray],
    true_labels: List[int],
    pred_labels: List[int],
    out_path: Path,
    variant: str,
    class_names: List[str] = config.CLASS_NAMES,
) -> None:
    """
    Save a grid of (original | overlay) image pairs.

    Parameters
    ----------
    images:
        List of PIL images (un-normalised originals).
    heatmaps:
        Corresponding Grad-CAM heat-maps.
    true_labels, pred_labels:
        Ground-truth and predicted class indices.
    out_path:
        Output PNG path.
    variant:
        Model variant label for the title.
    class_names:
        Human-readable class names.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed — skipping Grad-CAM grid save.")
        return

    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = [axes]

    for i, (img, cam, t, p) in enumerate(
        zip(images, heatmaps, true_labels, pred_labels)
    ):
        overlay = _overlay_heatmap(img, cam)
        correct = "[OK]" if t == p else "[X]"
        axes[i][0].imshow(img)
        axes[i][0].set_title(f"True: {class_names[t]}", fontsize=8)
        axes[i][0].axis("off")
        axes[i][1].imshow(overlay)
        axes[i][1].set_title(
            f"Pred: {class_names[p]}  {correct}", fontsize=8
        )
        axes[i][1].axis("off")

    fig.suptitle(f"Grad-CAM - {variant}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Grad-CAM grid -> {out_path}")


# ---------------------------------------------------------------------------
# Grad-CAM collection loop
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: nn.Module,
    test_loader: DataLoader,
    variant: str,
    device: torch.device,
    num_samples: int = config.NUM_GRADCAM_SAMPLES,
    out_dir: Path = config.GRADCAM_OUT_DIR,
    class_names: List[str] = config.CLASS_NAMES,
) -> None:
    """
    Collect Grad-CAM visualisations for a few samples per class.

    Parameters
    ----------
    model:
        Trained model (best checkpoint already loaded).
    test_loader:
        Test-set DataLoader (not shuffled).
    variant:
        ``"baseline"`` or ``"ordinal"``.
    device:
        Compute device.
    num_samples:
        Maximum samples to visualise *per class*.
    out_dir:
        Directory where PNG grids are written.
    """
    gradcam = GradCAM(
        model,
        target_layer_name=f"backbone.{config.GRADCAM_TARGET_LAYER}",
        variant=variant,
        device=device,
    )

    eval_tf = build_eval_transform()

    # Collect per-class buffers
    num_classes = len(class_names)
    class_images: Dict[int, List] = {k: [] for k in range(num_classes)}
    class_heatmaps: Dict[int, List] = {k: [] for k in range(num_classes)}
    class_preds: Dict[int, List] = {k: [] for k in range(num_classes)}

    dataset = test_loader.dataset

    for idx in range(len(dataset)):
        # Check if all classes have enough samples
        if all(len(v) >= num_samples for v in class_images.values()):
            break

        img_path, true_label = dataset.samples[idx]

        if len(class_images[true_label]) >= num_samples:
            continue

        # Load original (un-normalised) for visualisation
        pil_img = crop_fruit(Image.open(img_path).convert("RGB")).resize(config.IMAGE_SIZE)

        # Pre-process for model
        tensor = eval_tf(pil_img).unsqueeze(0)  # (1, 3, H, W)

        try:
            heatmap, pred = gradcam.generate(tensor)
        except Exception as exc:
            warnings.warn(f"Grad-CAM failed for {img_path}: {exc}")
            continue

        class_images[true_label].append(pil_img)
        class_heatmaps[true_label].append(heatmap)
        class_preds[true_label].append(pred)

    # Save one grid per class
    for cls_id in range(num_classes):
        imgs = class_images[cls_id]
        if not imgs:
            continue
        hms = class_heatmaps[cls_id]
        preds = class_preds[cls_id]
        trues = [cls_id] * len(imgs)

        out_path = out_dir / f"{variant}_class{cls_id}_{config.CLASS_NAMES[cls_id].replace(' ', '_')}.png"
        save_gradcam_grid(
            imgs, hms, trues, preds, out_path, variant, class_names=class_names
        )


# ---------------------------------------------------------------------------
# Full test-set evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    variant: str,
    device: torch.device,
    num_classes: int = config.NUM_CLASSES,
) -> Dict[str, object]:
    """
    Run the full evaluation suite on the test set.

    Parameters
    ----------
    model:
        Trained model (best checkpoint).
    test_loader:
        Test-set DataLoader.
    variant:
        Model variant.
    device:
        Compute device.

    Returns
    -------
    Full metrics dict (see :func:`metrics.compute_all_metrics`).
    """
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = _predict_from_logits(logits, variant)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    return compute_all_metrics(all_targets, all_preds, num_classes=num_classes)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(
    results: Dict[str, Dict[str, object]],
    out_path: Path,
    class_names: List[str] = config.CLASS_NAMES,
) -> None:
    """
    Write the comparison table to a CSV file.

    Parameters
    ----------
    results:
        ``{variant: metrics_dict}`` mapping.
    out_path:
        Destination CSV path.
    """
    fieldnames = ["metric"] + list(results.keys())
    scalar_keys = [
        ("Accuracy",       "accuracy"),
        ("MAE",            "mae"),
        ("Off-by-1 Acc",   "off_by_1_acc"),
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label, key in scalar_keys:
            row = {"metric": label}
            for v, m in results.items():
                row[v] = f"{m[key]:.6f}"
            writer.writerow(row)

        # Per-class MAE
        for cls_id, cls_name in enumerate(class_names):
            row = {"metric": f"MAE [{cls_name}]"}
            for v, m in results.items():
                val = m["per_class_mae"].get(cls_id, float("nan"))
                row[v] = f"{val:.6f}"
            writer.writerow(row)

    print(f"\n[Results CSV] -> {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    variants: Optional[List[str]] = None,
    data_root: Optional[Path] = None,
    run_gradcam: bool = True,
    seed: int = config.SEED,
    device: Optional[torch.device] = None,
) -> None:
    """
    Load best checkpoints and evaluate the selected variants.

    Parameters
    ----------
    variants:
        List of variant names to evaluate. Defaults to ``["ordinal"]``.
    data_root:
        Dataset root.  Defaults to ``config.DATA_ROOT``.
    run_gradcam:
        Whether to generate Grad-CAM visualisations.
    seed:
        Random seed for DataLoader reproducibility.
    device:
        Compute device (auto-detected when ``None``).
    """
    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if variants is None:
        variants = [config.MODEL_ORDINAL]

    # ------------------------------------------------------------------ Data
    _, _, test_loader = build_dataloaders(data_root=data_root, seed=seed)
    class_names = get_class_names(data_root)
    num_classes = len(class_names)

    # ---------------------------------------------------------- Evaluate loop
    all_results: Dict[str, Dict[str, object]] = {}

    for variant in variants:
        ckpt_path = config.checkpoint_path(variant)
        if not ckpt_path.exists():
            warnings.warn(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run `python train.py --variant {variant}` first."
            )
            continue

        print(f"\n{'-'*50}")
        print(f"  Evaluating: {variant.upper()}")
        print(f"{'-'*50}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_classes = checkpoint.get("class_names", class_names)
        if checkpoint_classes != class_names:
            raise ValueError(
                f"Checkpoint classes {checkpoint_classes} do not match dataset classes "
                f"{class_names}. Train this model with the selected dataset."
            )
        model = build_model(variant, device, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(
            f"  Loaded checkpoint from epoch {checkpoint['epoch']}  "
            f"(val_mae={checkpoint['val_mae']:.4f})"
        )

        metrics = evaluate_on_test(
            model, test_loader, variant, device, num_classes=num_classes
        )
        all_results[variant] = metrics

        print_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=class_names,
            title=f"Confusion Matrix — {variant}",
        )

        if run_gradcam:
            print(f"\n  Generating Grad-CAM for [{variant}] ...")
            generate_gradcam(
                model, test_loader, variant, device, class_names=class_names
            )

    # ------------------------------------------------------- Comparison table
    if all_results:
        print("\n" + "=" * 60)
        print("  COMPARATIVE EVALUATION RESULTS")
        print("=" * 60)
        print(format_metrics_table(all_results, class_names=class_names))

        csv_path = config.LOG_DIR / "evaluation_results.csv"
        save_results_csv(all_results, csv_path, class_names=class_names)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fruit-ripeness models and generate Grad-CAM maps."
    )
    parser.add_argument(
        "--variant",
        choices=[config.MODEL_BASELINE, config.MODEL_ORDINAL],
        default=None,
        help="Evaluate a single variant (default: ordinal only).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip Grad-CAM visualisation (faster).",
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
        help="Compute device, e.g. 'cuda:0' or 'cpu'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    variants = [args.variant] if args.variant else None
    device = torch.device(args.device) if args.device else None
    evaluate(
        variants=variants,
        data_root=args.data_root,
        run_gradcam=not args.no_gradcam,
        seed=args.seed,
        device=device,
    )
