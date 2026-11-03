"""Download and prepare public mango/banana ripeness images from Kaggle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

try:
    import kagglehub
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub


DATASET_HANDLE = "asadullahprl/fruits-ripeness-classification-dataset"
CLASS_NAMES = ["Unripe", "Ripe", "Overripe"]
FRUIT_NAMES = ("banana", "mango")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_split_root(dataset_root: Path, split_name: str) -> Path:
    """Find a Train/Test directory regardless of capitalization."""
    for path in dataset_root.iterdir():
        if path.is_dir() and path.name.casefold() == split_name.casefold():
            return path
    raise FileNotFoundError(f"Dataset split '{split_name}' was not found in {dataset_root}.")


def prepare_dataset(target_dir: Path) -> Dict[str, Dict[str, int]]:
    """Download the source dataset and copy mango/banana images into splits."""
    source_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    stage_aliases = {
        "unripe": ("unripe",),
        "ripe": ("ripe",),
        "overripe": ("overripe", "overipe"),
    }
    counts: Dict[str, Dict[str, int]] = {}

    for split_name in ("Train", "Test"):
        source_split = _find_split_root(source_dir, split_name)
        split_counts: Dict[str, int] = {}
        for class_name in CLASS_NAMES:
            class_key = class_name.casefold()
            source_stage = next(
                (
                    candidate
                    for candidate in source_split.iterdir()
                    if candidate.is_dir()
                    and candidate.name.casefold() in stage_aliases[class_key]
                ),
                None,
            )
            if source_stage is None:
                raise FileNotFoundError(
                    f"Ripeness folder '{class_name}' is missing from {source_split}."
                )

            destination = target_dir / split_name.lower() / class_key
            destination.mkdir(parents=True, exist_ok=True)
            copied = 0
            for image_path in sorted(source_stage.iterdir()):
                if (
                    image_path.suffix.lower() not in IMAGE_EXTENSIONS
                    or not any(fruit in image_path.stem.casefold() for fruit in FRUIT_NAMES)
                ):
                    continue
                shutil.copy2(image_path, destination / image_path.name)
                copied += 1
            split_counts[class_name] = copied
        counts[split_name.lower()] = split_counts

    if any(count == 0 for split in counts.values() for count in split.values()):
        raise ValueError(f"Dataset preparation produced an empty class: {counts}")

    metadata = {
        "dataset": DATASET_HANDLE,
        "license": "CC0: Public Domain (per Kaggle dataset metadata)",
        "fruit_names": list(FRUIT_NAMES),
        "class_names": CLASS_NAMES,
        "source_splits": ["Train", "Test"],
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Prepared images in: {target_dir.resolve()}")
    for split_name, class_counts in counts.items():
        print(f"{split_name}: {class_counts} (total={sum(class_counts.values())})")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public mango/banana photos with unripe, ripe, and overripe labels."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).parent / "data_mango_banana",
        help="Output folder (default: data_mango_banana).",
    )
    args = parser.parse_args()
    prepare_dataset(args.target)


if __name__ == "__main__":
    main()
