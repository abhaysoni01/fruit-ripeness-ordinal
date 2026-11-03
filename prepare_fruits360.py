"""
prepare_fruits360.py — Tool for mapping or fetching Fruits-360 dataset
for fruit ripeness ordinal regression.

Supports two modes:
1. Local extraction: Map an existing local Fruits-360 dataset into data/
   python prepare_fruits360.py --source "C:/path/to/fruits-360" --preset tomato

2. Auto-download: Fetch authentic sample images from official Fruits-360 GitHub
   python prepare_fruits360.py --download --preset tomato --samples-per-class 50
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import config

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/Training"
)

# Standard index names under config.LABEL_MAP:
# 0 -> unripe, 1 -> slightly_ripe, 2 -> ripe, 3 -> overripe
TARGET_FOLDER_MAP = {
    0: "unripe",
    1: "slightly_ripe",
    2: "ripe",
    3: "overripe",
}

# Fruits-360 presets for ripeness grading (supports both classic and v112 folder names)
PRESETS: Dict[str, Dict[str, int]] = {
    "tomato": {
        "Tomato 1": 0,            # Stage 0: Unripe green tomato
        "Tomato not Ripened": 0,  # Stage 0: Classic name
        "Tomato Yellow 1": 1,     # Stage 1: Slightly Ripe yellow tomato
        "Tomato Yellow": 1,       # Stage 1: Classic name
        "Tomato Cherry Red 1": 2, # Stage 2: Ripe red tomato
        "Tomato Cherry Red": 2,   # Stage 2: Classic name
        "Tomato Maroon 1": 3,     # Stage 3: Overripe deep maroon tomato
        "Tomato Maroon": 3,       # Stage 3: Classic name
    },
    "apple": {
        "Apple Granny Smith 1": 0,# Stage 0: Green
        "Apple Granny Smith": 0,
        "Apple Golden 1": 1,      # Stage 1: Yellow-green
        "Apple Golden 2": 2,      # Stage 2: Golden ripe
        "Apple Red 1": 3,         # Stage 3: Deep red ripe
    },
    "banana": {
        "Banana Lady Finger 1": 1,
        "Banana 1": 2,
        "Banana Red 1": 3,
    },
    "mixed": {
        "Tomato 1": 0,
        "Apple Granny Smith 1": 0,
        "Tomato Yellow 1": 1,
        "Banana Lady Finger 1": 1,
        "Tomato Cherry Red 1": 2,
        "Banana 1": 2,
        "Tomato Maroon 1": 3,
        "Banana Red 1": 3,
    },
}


def prepare_from_local(
    source_dir: Path,
    target_dir: Path,
    preset_name: str = "tomato",
    max_per_class: int = 0,
    symlink: bool = False,
) -> None:
    """
    Scan local Fruits-360 directory and organize selected classes into
    data/unripe, data/slightly_ripe, data/ripe, data/overripe.
    """
    preset = PRESETS.get(preset_name, PRESETS["tomato"])
    print(f"[Prepare] Source: {source_dir}")
    print(f"[Prepare] Target: {target_dir}")
    print(f"[Prepare] Preset: {preset_name.upper()}")

    # Search candidates: nested 100x100, Training, or direct subfolders
    search_dirs = [
        source_dir / "fruits-360_100x100" / "fruits-360" / "Training",
        source_dir / "fruits-360_100x100" / "fruits-360",
        source_dir / "fruits-360" / "Training",
        source_dir / "fruits-360",
        source_dir / "Training",
        source_dir,
    ]
    valid_search_dirs = [d for d in search_dirs if d.is_dir()]

    if not valid_search_dirs:
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Prepare target directories
    for stage_folder in TARGET_FOLDER_MAP.values():
        (target_dir / stage_folder).mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts: Dict[int, int] = {k: 0 for k in TARGET_FOLDER_MAP}

    for source_class_name, stage_idx in preset.items():
        stage_folder = TARGET_FOLDER_MAP[stage_idx]
        dest_dir = target_dir / stage_folder

        # Locate folder (case-insensitive)
        matched_class_dir: Path | None = None
        for s_dir in valid_search_dirs:
            for sub in s_dir.iterdir():
                if sub.is_dir() and sub.name.lower() == source_class_name.lower():
                    matched_class_dir = sub
                    break
            if matched_class_dir:
                break

        if not matched_class_dir:
            print(f"  [!] Subfolder not found: '{source_class_name}' (skipping)")
            continue

        images = [
            p for p in sorted(matched_class_dir.iterdir())
            if p.suffix.lower() in image_extensions
        ]

        if max_per_class > 0:
            images = images[:max_per_class]

        print(f"  [+] Mapping '{source_class_name}' -> '{stage_folder}' ({len(images)} images)")

        for img in images:
            out_name = f"{matched_class_dir.name.replace(' ', '_')}_{img.name}"
            out_path = dest_dir / out_name
            if symlink:
                try:
                    if out_path.exists():
                        out_path.unlink()
                    os.symlink(img, out_path)
                except OSError:
                    shutil.copy2(img, out_path)
            else:
                shutil.copy2(img, out_path)
            counts[stage_idx] += 1

    print("\n[Complete] Fruits-360 data organized:")
    for stage_idx, folder_name in TARGET_FOLDER_MAP.items():
        print(f"  Stage {stage_idx} [{folder_name}]: {counts[stage_idx]} images")


def download_sample_fruits360(
    target_dir: Path,
    preset_name: str = "tomato",
    samples_per_class: int = 50,
) -> None:
    """
    Download authentic Fruits-360 images directly from the official GitHub repo.
    """
    preset = PRESETS.get(preset_name, PRESETS["tomato"])
    print(f"\n[Download] Fetching Fruits-360 ({preset_name.upper()} preset) from official GitHub repo...")
    print(f"[Download] Destination: {target_dir}")
    print(f"[Download] Samples per stage: {samples_per_class}")

    for stage_folder in TARGET_FOLDER_MAP.values():
        (target_dir / stage_folder).mkdir(parents=True, exist_ok=True)

    counts: Dict[int, int] = {k: 0 for k in TARGET_FOLDER_MAP}

    for source_class_name, stage_idx in preset.items():
        stage_folder = TARGET_FOLDER_MAP[stage_idx]
        dest_dir = target_dir / stage_folder
        encoded_class = urllib.parse.quote(source_class_name)

        print(f"\n  Downloading stage {stage_idx} [{stage_folder}] from '{source_class_name}'...")
        successful = 0
        idx = 0
        attempts = 0
        max_attempts = samples_per_class * 4  # allow for gaps in index numbering

        while successful < samples_per_class and attempts < max_attempts:
            filename = f"{idx}_100.jpg"
            img_url = f"{GITHUB_RAW_BASE}/{encoded_class}/{filename}"
            dest_file = dest_dir / f"{source_class_name.replace(' ', '_')}_{filename}"

            try:
                urllib.request.urlretrieve(img_url, dest_file)
                successful += 1
                counts[stage_idx] += 1
                if successful % 10 == 0 or successful == samples_per_class:
                    print(f"    Downloaded {successful}/{samples_per_class} images...")
            except Exception:
                pass  # index might not exist in Fruits-360 numbering

            idx += 1
            attempts += 1

    print("\n[Complete] Fruits-360 images downloaded and organized:")
    for stage_idx, folder_name in TARGET_FOLDER_MAP.items():
        print(f"  Stage {stage_idx} [{folder_name}]: {counts[stage_idx]} images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or download Fruits-360 dataset for ordinal fruit ripeness classification."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to extracted local Fruits-360 directory.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=config.DATA_ROOT,
        help=f"Target directory for processed dataset (default: {config.DATA_ROOT}).",
    )
    parser.add_argument(
        "--preset",
        choices=["tomato", "apple", "mixed"],
        default="tomato",
        help="Ripeness preset from Fruits-360 (default: tomato).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Maximum images per class to copy (0 = all).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download sample Fruits-360 images directly from official GitHub repository.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Number of images per class to download with --download (default: 50).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying files (saves disk space).",
    )

    args = parser.parse_args()

    if args.download:
        download_sample_fruits360(
            target_dir=args.target,
            preset_name=args.preset,
            samples_per_class=args.samples_per_class,
        )
    elif args.source:
        prepare_from_local(
            source_dir=args.source,
            target_dir=args.target,
            preset_name=args.preset,
            max_per_class=args.max_per_class,
            symlink=args.symlink,
        )
    else:
        print(
            "Please specify either --source <path_to_fruits360> or --download.\n"
            "Examples:\n"
            "  python prepare_fruits360.py --source 'D:/datasets/fruits-360' --preset tomato\n"
            "  python prepare_fruits360.py --download --preset tomato\n"
        )


if __name__ == "__main__":
    main()
