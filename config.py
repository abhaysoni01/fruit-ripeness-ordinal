"""
config.py — Global hyperparameters, paths, and seed configuration.

All tunable knobs live here so that train.py and evaluate.py remain
clean and re-runnable without touching source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Ordinal label map
# ---------------------------------------------------------------------------
LABEL_MAP: Dict[str, int] = {
    "unripe": 0,
    "slightly_ripe": 1,
    "ripe": 2,
    "overripe": 3,
}
CLASS_NAMES: List[str] = ["Unripe", "Slightly Ripe", "Ripe", "Overripe"]
NUM_CLASSES: int = len(CLASS_NAMES)          # 4
NUM_THRESHOLDS: int = NUM_CLASSES - 1        # 3  (CORAL binary tasks)

# ---------------------------------------------------------------------------
# Fruits-360 Ordinal Ripeness Mapping
# Maps Fruits-360 varieties and maturation states to the 4 ordinal stages
# ---------------------------------------------------------------------------
FRUITS360_CLASS_MAP: Dict[str, int] = {
    # Stage 0: Unripe
    "tomato not ripened": 0,
    "tomato 1": 0,
    "tomato 2": 0,
    "apple granny smith": 0,
    "apple granny smith 1": 0,
    "avocado": 0,
    "avocado 1": 0,
    "mango": 0,
    "mango 1": 0,

    # Stage 1: Slightly Ripe
    "tomato yellow": 1,
    "tomato yellow 1": 1,
    "tomato cherry yellow 1": 1,
    "apple golden 1": 1,
    "banana lady finger": 1,
    "banana lady finger 1": 1,

    # Stage 2: Ripe
    "tomato": 2,
    "tomato 3": 2,
    "tomato cherry red": 2,
    "tomato cherry red 1": 2,
    "tomato cherry red 2": 2,
    "apple golden 2": 2,
    "apple golden 3": 2,
    "banana": 2,
    "banana 1": 2,
    "banana 2": 2,
    "avocado ripe": 2,
    "mango red": 2,

    # Stage 3: Overripe / Late Maturity
    "tomato maroon": 3,
    "tomato maroon 1": 3,
    "tomato maroon 2": 3,
    "tomato heart": 3,
    "tomato heart 1": 3,
    "apple red 1": 3,
    "apple crimson snow": 3,
    "apple braeburn": 3,
    "banana red": 3,
    "banana red 1": 3,
    "banana 3": 3,
    "banana 4": 3,
}

# Focused single-fruit presets within Fruits-360
FRUITS360_PRESETS: Dict[str, Dict[str, int]] = {
    "tomato": {
        "Tomato 1": 0,
        "Tomato Yellow 1": 1,
        "Tomato Cherry Red 1": 2,
        "Tomato Maroon 1": 3,
    },
    "apple": {
        "Apple Granny Smith 1": 0,
        "Apple Golden 1": 1,
        "Apple Golden 2": 2,
        "Apple Red 1": 3,
    },
    "banana": {
        "Banana Lady Finger 1": 1,
        "Banana 1": 2,
        "Banana Red 1": 3,
    },
    "mixed": FRUITS360_CLASS_MAP,
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent.resolve()
DATA_ROOT: Path = PROJECT_ROOT / "data"          # root of the raw image tree
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
LOG_DIR: Path = PROJECT_ROOT / "logs"
GRADCAM_OUT_DIR: Path = PROJECT_ROOT / "gradcam_outputs"

# Expected sub-tree under DATA_ROOT:
#   data/
#     unripe/        <-- class folder
#     slightly_ripe/
#     ripe/
#     overripe/
# (mirrors the LABEL_MAP keys)

# Create output dirs at import time so callers never have to mkdir manually
for _d in (CHECKPOINT_DIR, LOG_DIR, GRADCAM_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15   # remainder; must sum to 1.0


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
IMAGE_SIZE: Tuple[int, int] = (224, 224)

# ImageNet statistics
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Augmentation strengths (kept subtle to avoid distorting ripeness cues)
COLOR_JITTER_BRIGHTNESS: float = 0.2
COLOR_JITTER_CONTRAST: float = 0.2
COLOR_JITTER_SATURATION: float = 0.1
COLOR_JITTER_HUE: float = 0.05


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
NUM_WORKERS: int = min(4, os.cpu_count() or 1)
PIN_MEMORY: bool = True


# ---------------------------------------------------------------------------
# Optimiser & Scheduler
# ---------------------------------------------------------------------------
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4

# ReduceLROnPlateau
LR_PATIENCE: int = 5
LR_FACTOR: float = 0.5
LR_MIN: float = 1e-7


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_EPOCHS: int = 30
EARLY_STOP_PATIENCE: int = 10   # epochs without val-MAE improvement


# ---------------------------------------------------------------------------
# Model variants (used as keys throughout the project)
# ---------------------------------------------------------------------------
MODEL_BASELINE: str = "baseline"   # cross-entropy, 4-class head
MODEL_ORDINAL: str = "ordinal"     # CORAL, 3 binary logits


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
GRADCAM_TARGET_LAYER: str = "layer4"   # ResNet-18 last conv block
NUM_GRADCAM_SAMPLES: int = 8           # images per class to visualise


# ---------------------------------------------------------------------------
# Checkpoint filenames  (one per model variant)
# ---------------------------------------------------------------------------
def checkpoint_path(variant: str) -> Path:
    """Return the best-checkpoint path for a given model variant."""
    return CHECKPOINT_DIR / f"best_{variant}.pth"
