"""
download_dataset.py — Download Fruits-360 dataset using kagglehub and
optionally link/prepare it for the fruit ripeness project.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Step 1: Ensure kagglehub is installed
try:
    import kagglehub
except ImportError:
    print("[1/3] 'kagglehub' not found. Installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

import config
from prepare_fruits360 import PRESETS, prepare_from_local

parser = argparse.ArgumentParser(
    description="Download Fruits-360 and prepare ordinal ripeness images."
)
parser.add_argument(
    "--preset",
    choices=sorted(PRESETS),
    default="tomato",
    help="Fruits-360 fruit mapping to prepare (default: tomato).",
)
parser.add_argument(
    "--target",
    type=Path,
    default=config.PROJECT_ROOT / "data_fruits360",
    help="Prepared dataset destination (default: data_fruits360).",
)
args = parser.parse_args()

# Step 2: Download dataset
print("\n[2/3] Downloading latest version of 'moltean/fruits' via kagglehub...")
print("      (This may take a few minutes depending on your internet connection)")

dataset_path_str = kagglehub.dataset_download("moltean/fruits")
dataset_path = Path(dataset_path_str)

print("\n" + "=" * 60)
print("Path to dataset files:", dataset_path)
print("=" * 60)

# Search for the Training directory inside the downloaded files
training_dir = None
for candidate in [
    dataset_path,
    dataset_path / "fruits-360_dataset" / "fruits-360",
    dataset_path / "fruits-360",
    dataset_path / "fruits-360-original-size",
]:
    if (candidate / "Training").is_dir():
        training_dir = candidate
        break

if training_dir is None:
    # Check if direct class folders exist
    training_dir = dataset_path

print(f"\n[3/3] Found Fruits-360 dataset root at: {training_dir}")
print(f"      Organizing ripeness classes into '{args.target}' using the {args.preset} preset...")

try:
    prepare_from_local(
        source_dir=training_dir,
        target_dir=args.target,
        preset_name=args.preset,
        max_per_class=0,   # take all images
        symlink=False,
    )
    print("\n[Done] Dataset downloaded and prepared at:", args.target)
    print(
        "    You can now run: python train.py --variant ordinal "
        f"--data-root \"{args.target}\" --device cpu --epochs 30"
    )
except Exception as exc:
    print(f"\n[!] Note: You can prepare the dataset manually with:")
    print(
        f"    python prepare_fruits360.py --source \"{training_dir}\" "
        f"--target \"{args.target}\" --preset {args.preset}"
    )
    raise
