"""
app.py — Flask backend for interactive Fruit Ripeness Ordinal Classification & Grad-CAM explanations.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

import config
from dataset import build_eval_transform
from evaluate import GradCAM, _overlay_heatmap
from fruit_crop import crop_fruit
from models import BaselineNet, CoralNet, build_model

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model singletons
_baseline_model: Optional[BaselineNet] = None
_ordinal_model: Optional[CoralNet] = None
_baseline_class_names: Optional[List[str]] = None
_ordinal_class_names: Optional[List[str]] = None
_ordinal_dataset_root: Optional[Path] = None
_eval_transform = build_eval_transform()


def get_models() -> Tuple[Optional[BaselineNet], CoralNet]:
    """Load trained checkpoints and their dataset-specific ordinal labels."""
    global _baseline_model, _ordinal_model
    global _baseline_class_names, _ordinal_class_names, _ordinal_dataset_root

    ckpt_path_o = config.checkpoint_path(config.MODEL_ORDINAL)
    if _ordinal_model is None:
        if not ckpt_path_o.exists():
            raise FileNotFoundError(
                f"Ordinal checkpoint not found: {ckpt_path_o}. "
                "Train the ordinal model before running image analysis."
            )
        ckpt_o = torch.load(ckpt_path_o, map_location=device, weights_only=False)
        class_names = ckpt_o.get("class_names", config.CLASS_NAMES)
        if not isinstance(class_names, list) or len(class_names) < 2:
            raise ValueError(f"Invalid class_names in {ckpt_path_o}.")
        model_o = build_model(
            config.MODEL_ORDINAL, device, num_classes=len(class_names)
        )
        model_o.load_state_dict(ckpt_o["model_state_dict"])
        model_o.eval()
        _ordinal_model = model_o
        _ordinal_class_names = class_names
        dataset_root = ckpt_o.get("dataset_root")
        if dataset_root:
            _ordinal_dataset_root = Path(dataset_root)
        print(f"[Flask] Loaded ordinal checkpoint from {ckpt_path_o}")

    ckpt_path_b = config.checkpoint_path(config.MODEL_BASELINE)
    if _baseline_model is None and ckpt_path_b.exists():
        ckpt_b = torch.load(ckpt_path_b, map_location=device, weights_only=False)
        class_names = ckpt_b.get("class_names", config.CLASS_NAMES)
        model_b = build_model(
            config.MODEL_BASELINE, device, num_classes=len(class_names)
        )
        model_b.load_state_dict(ckpt_b["model_state_dict"])
        model_b.eval()
        _baseline_model = model_b
        _baseline_class_names = class_names
        print(f"[Flask] Loaded baseline checkpoint from {ckpt_path_b}")

    return _baseline_model, _ordinal_model


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def run_inference(image: Image.Image) -> Dict:
    """Run trained models on a fruit crop and produce ordinal Grad-CAM."""
    model_b, model_o = get_models()

    # Preprocess
    orig_rgb = image.convert("RGB")
    fruit_crop = crop_fruit(orig_rgb)
    classifier_image = fruit_crop.resize(config.IMAGE_SIZE)
    tensor = _eval_transform(classifier_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_o = model_o(tensor)
        thresh_probs = torch.sigmoid(logits_o).squeeze(0).cpu().tolist()
        pred_o_idx = int(CoralNet.logits_to_label(logits_o).item())

        probs_o = [1.0 - thresh_probs[0]]
        probs_o.extend(
            max(0.0, lower - upper)
            for lower, upper in zip(thresh_probs, thresh_probs[1:])
        )
        probs_o.append(thresh_probs[-1])
        total_p = sum(probs_o) or 1.0
        probs_o = [p / total_p for p in probs_o]

    gradcam_o = GradCAM(
        model_o,
        target_layer_name=f"backbone.{config.GRADCAM_TARGET_LAYER}",
        variant=config.MODEL_ORDINAL,
        device=device,
    )
    cam_o, _ = gradcam_o.generate(tensor.clone())
    blended_o = _overlay_heatmap(classifier_image, cam_o)

    stage_names = _ordinal_class_names or config.CLASS_NAMES
    baseline = None
    if model_b is not None:
        baseline_names = _baseline_class_names or stage_names
        with torch.no_grad():
            logits_b = model_b(tensor)
            probs_b = F.softmax(logits_b, dim=1).squeeze(0).cpu().tolist()
            pred_b_idx = int(torch.argmax(logits_b, dim=1).item())
        gradcam_b = GradCAM(
            model_b,
            target_layer_name=f"backbone.{config.GRADCAM_TARGET_LAYER}",
            variant=config.MODEL_BASELINE,
            device=device,
        )
        cam_b, _ = gradcam_b.generate(tensor.clone())
        baseline = {
            "predicted_index": pred_b_idx,
            "predicted_label": baseline_names[pred_b_idx],
            "confidence": probs_b[pred_b_idx],
            "class_probabilities": [
                {"label": baseline_names[i], "probability": round(p, 4)}
                for i, p in enumerate(probs_b)
            ],
            "gradcam_image": pil_to_base64(_overlay_heatmap(classifier_image, cam_b)),
            "explanation": (
                "Standard Cross-Entropy treats classes as distinct nominal boxes. "
                f"It predicted '{baseline_names[pred_b_idx]}' "
                f"with {probs_b[pred_b_idx] * 100:.1f}% confidence."
            ),
        }

    return {
        "original_image": pil_to_base64(orig_rgb.resize(config.IMAGE_SIZE)),
        "fruit_crop_image": pil_to_base64(classifier_image),
        "baseline": baseline,
        "ordinal": {
            "class_names": stage_names,
            "predicted_index": pred_o_idx,
            "predicted_label": stage_names[pred_o_idx],
            "threshold_probabilities": [
                {
                    "threshold": f"Rank > {stage_names[i]} (>= {stage_names[i + 1]})",
                    "probability": round(probability, 4),
                    "active": probability > 0.5,
                }
                for i, probability in enumerate(thresh_probs)
            ],
            "derived_probabilities": [
                {"label": stage_names[i], "probability": round(p, 4)}
                for i, p in enumerate(probs_o)
            ],
            "gradcam_image": pil_to_base64(blended_o),
            "explanation": (
                "CORAL evaluates cumulative rank consistency across binary boundaries. "
                f"Passed {pred_o_idx} of {len(thresh_probs)} threshold gates, "
                f"yielding Stage {pred_o_idx}: '{stage_names[pred_o_idx]}'."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    """Return backend status, device, and checkpoint availability."""
    global _ordinal_class_names, _ordinal_dataset_root
    ckpt_b = config.checkpoint_path(config.MODEL_BASELINE).exists()
    ckpt_o_path = config.checkpoint_path(config.MODEL_ORDINAL)
    ckpt_o = ckpt_o_path.exists()
    class_names = config.CLASS_NAMES
    if ckpt_o:
        checkpoint = torch.load(ckpt_o_path, map_location="cpu", weights_only=False)
        class_names = checkpoint.get("class_names", class_names)
        _ordinal_class_names = class_names
        dataset_root = checkpoint.get("dataset_root")
        if dataset_root:
            _ordinal_dataset_root = Path(dataset_root)
    return jsonify({
        "status": "online",
        "device": str(device),
        "checkpoints": {
            "baseline": ckpt_b,
            "ordinal": ckpt_o,
        },
        "num_classes": len(class_names),
        "class_names": class_names,
    })


@app.route("/api/samples")
def get_sample_images():
    """Return sample images from the active model's training dataset."""
    class_names = _ordinal_class_names or config.CLASS_NAMES
    data_dir = _ordinal_dataset_root or config.DATA_ROOT
    split_dir = data_dir / "train"
    if not split_dir.is_dir():
        split_dir = data_dir

    samples_by_stage: Dict[str, List[Dict[str, str]]] = {}
    for class_name in class_names:
        stage_name = class_name.casefold()
        samples_by_stage[stage_name] = []
        stage_dir = split_dir / stage_name
        if not stage_dir.is_dir():
            continue
        for image_path in sorted(stage_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                samples_by_stage[stage_name].append({
                    "name": image_path.name,
                    "url": f"/dataset-image/{stage_name}/{image_path.name}",
                    "path": str(image_path.resolve()),
                })
                if len(samples_by_stage[stage_name]) >= 6:
                    break

    return jsonify(samples_by_stage)


@app.route("/dataset-image/<stage>/<filename>")
def serve_dataset_image(stage: str, filename: str):
    """Serve a curated image from the active model's dataset."""
    class_names = _ordinal_class_names or config.CLASS_NAMES
    stage_names = {name.casefold() for name in class_names}
    if stage.casefold() not in stage_names:
        return jsonify({"error": "Unknown ripeness stage"}), 404
    data_dir = _ordinal_dataset_root or config.DATA_ROOT
    split_dir = data_dir / "train"
    if not split_dir.is_dir():
        split_dir = data_dir
    stage_dir = split_dir / stage.casefold()
    return send_from_directory(stage_dir, filename)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Run model prediction and Grad-CAM on uploaded or sample image."""
    try:
        # Check for file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "Empty filename provided"}), 400
            image = Image.open(file.stream).convert("RGB")
        # Check for sample image path
        elif request.is_json:
            data = request.get_json()
            image_path = data.get("sample_path")
            if not image_path or not Path(image_path).exists():
                return jsonify({"error": "Sample image path not found"}), 400
            image = Image.open(image_path).convert("RGB")
        else:
            return jsonify({"error": "No image file or sample path supplied"}), 400

        results = run_inference(image)
        return jsonify(results)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/metrics")
def get_metrics():
    """Return model evaluation metrics and comparison table."""
    results_csv = config.LOG_DIR / "evaluation_results.csv"
    if results_csv.exists():
        import csv
        rows = []
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return jsonify({"source": "evaluation_results.csv", "rows": rows})

    return jsonify({"source": "not_evaluated", "rows": []})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Fruit Ripeness Ordinal AI — Flask Web Dashboard")
    print("  Local URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
