"""
dataset.py — Custom PyTorch Dataset, augmentation transforms, and
             deterministic stratified train/val/test splitting.

Expected on-disk layout::

    data/
    ├── unripe/
    │   ├── img001.jpg
    │   └── ...
    ├── slightly_ripe/
    ├── ripe/
    └── overripe/

Any JPEG / PNG / BMP image format is supported.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

import config
from fruit_crop import crop_fruit


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------
SampleList = List[Tuple[Path, int]]   # [(image_path, label), ...]
DATASET_METADATA_FILE = "dataset_metadata.json"


def get_class_names(data_root: Optional[Path] = None) -> List[str]:
    """Return class names declared by a dataset, or the project defaults."""
    root = data_root or config.DATA_ROOT
    metadata_path = root / DATASET_METADATA_FILE
    if not metadata_path.is_file():
        return config.CLASS_NAMES

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError(f"Invalid class_names in {metadata_path}.")
    if not all(isinstance(name, str) and name for name in class_names):
        raise ValueError(f"All class_names in {metadata_path} must be non-empty strings.")
    return class_names


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def build_train_transform() -> transforms.Compose:
    """
    Training augmentation pipeline.

    Applies spatial flips and subtle colour jitter so the model learns
    ripeness cues (colour, texture) rather than orientation artefacts,
    then normalises with ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST,
            saturation=config.COLOR_JITTER_SATURATION,
            hue=config.COLOR_JITTER_HUE,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def build_eval_transform() -> transforms.Compose:
    """
    Deterministic evaluation / test transform — no random ops.
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------

class FruitRipenessDataset(Dataset):
    """
    PyTorch Dataset for fruit ripeness grading.

    Parameters
    ----------
    samples:
        Pre-built list of ``(image_path, ordinal_label)`` tuples.
    transform:
        Optional torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        samples: SampleList,
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image; convert to RGB to handle greyscale / RGBA inputs
        image = crop_fruit(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    # ------------------------------------------------------------------
    @property
    def labels(self) -> List[int]:
        """Return all labels (useful for class-frequency computation)."""
        return [lbl for _, lbl in self.samples]

    # ------------------------------------------------------------------
    def class_counts(self) -> Dict[int, int]:
        """Return a mapping ``{class_id: sample_count}``."""
        counts: Dict[int, int] = defaultdict(int)
        for _, lbl in self.samples:
            counts[lbl] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

def _scan_directory(
    root: Path,
    class_names: Optional[List[str]] = None,
) -> SampleList:
    """
    Walk *root* and collect all image files.

    Directory names must match :data:`config.LABEL_MAP` keys exactly.

    Parameters
    ----------
    root:
        Path to the dataset root containing one sub-folder per class.

    Returns
    -------
    List of ``(image_path, label)`` tuples.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist.
    ValueError
        If no recognised class sub-directories are found.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Create the directory and populate it according to the "
            "layout described in dataset.py."
        )

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples: SampleList = []

    metadata_path = root / DATASET_METADATA_FILE
    if class_names is not None or metadata_path.is_file():
        class_names = class_names or get_class_names(root)
        class_roots = [root]
        if (root / "train").is_dir():
            class_roots = [root / "train"]
        for class_root in class_roots:
            for label, class_name in enumerate(class_names):
                class_dir = class_root / class_name.casefold()
                if not class_dir.is_dir():
                    raise ValueError(
                        f"Expected class folder '{class_dir}' for dataset label '{class_name}'."
                    )
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() in image_extensions:
                        samples.append((img_path, label))
        if not samples:
            raise ValueError(f"No images found in the declared classes under {root}.")
        return samples

    # 1. First attempt: standard 4-class subfolders (unripe, slightly_ripe, ripe, overripe)
    for class_name, label in config.LABEL_MAP.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                samples.append((img_path, label))

    if samples:
        return samples

    # 2. Second attempt: Fruits-360 layout (direct class subfolders or inside Training/Test)
    fruits360_dirs = []
    for candidate in [root / "Training", root / "Test", root]:
        if candidate.is_dir() and candidate not in fruits360_dirs:
            fruits360_dirs.append(candidate)

    found_f360_classes = set()
    for search_dir in fruits360_dirs:
        for sub_dir in search_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            key = sub_dir.name.lower().strip()
            if key in config.FRUITS360_CLASS_MAP:
                label = config.FRUITS360_CLASS_MAP[key]
                found_f360_classes.add(sub_dir.name)
                for img_path in sorted(sub_dir.iterdir()):
                    if img_path.suffix.lower() in image_extensions:
                        samples.append((img_path, label))

    if samples:
        print(f"[Dataset] Detected Fruits-360 structure! Mapped classes: {sorted(found_f360_classes)}")
        return samples

    raise ValueError(
        f"No images found under {root}.\n"
        f"Expected either:\n"
        f"  1) Standard sub-folders: {list(config.LABEL_MAP.keys())}, or\n"
        f"  2) Fruits-360 class folders (e.g. 'Tomato not Ripened', 'Tomato Yellow', 'Tomato', 'Tomato Maroon')."
    )

    return samples


# ---------------------------------------------------------------------------
# Stratified splitter
# ---------------------------------------------------------------------------

def stratified_split(
    samples: SampleList,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.SEED,
) -> Tuple[SampleList, SampleList, SampleList]:
    """
    Deterministic, per-class stratified split into train / val / test.

    Each class is shuffled independently with *seed* and divided
    proportionally, preserving class balance across splits.

    Parameters
    ----------
    samples:
        Full dataset as ``[(path, label), ...]``.
    train_ratio:
        Fraction of data reserved for training.
    val_ratio:
        Fraction of data reserved for validation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Tuple of (train_samples, val_samples, test_samples).
    """
    assert abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) < 1e-9

    rng = random.Random(seed)

    # Group by class
    by_class: Dict[int, SampleList] = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    train_s: SampleList = []
    val_s: SampleList = []
    test_s: SampleList = []

    for label in sorted(by_class):
        class_samples = by_class[label][:]
        rng.shuffle(class_samples)
        n = len(class_samples)

        if n >= 3:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = n - n_train - n_val
            if n_test < 1:
                n_val = max(1, n_val - 1)
                n_test = n - n_train - n_val
                if n_test < 1 and n_train > 1:
                    n_train -= 1
                    n_test = n - n_train - n_val
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train, n_val, n_test = 1, 0, 0

        train_s.extend(class_samples[:n_train])
        val_s.extend(class_samples[n_train: n_train + n_val])
        test_s.extend(class_samples[n_train + n_val:])

    return train_s, val_s, test_s


def stratified_train_val_split(
    samples: SampleList,
    train_ratio: float = 0.85,
    seed: int = config.SEED,
) -> Tuple[SampleList, SampleList]:
    """Split an existing training partition into train and validation only."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}.")

    rng = random.Random(seed)
    by_class: Dict[int, SampleList] = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    train_samples: SampleList = []
    val_samples: SampleList = []
    for class_samples in by_class.values():
        class_samples = class_samples[:]
        rng.shuffle(class_samples)
        if len(class_samples) < 2:
            raise ValueError(
                "Each class needs at least two training images for a stratified split."
            )
        n_train = min(len(class_samples) - 1, max(1, int(len(class_samples) * train_ratio)))
        train_samples.extend(class_samples[:n_train])
        val_samples.extend(class_samples[n_train:])
    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_datasets(
    data_root: Optional[Path] = None,
    seed: int = config.SEED,
) -> Tuple[FruitRipenessDataset, FruitRipenessDataset, FruitRipenessDataset]:
    """
    Scan *data_root*, split, and return three ``FruitRipenessDataset`` objects.

    Parameters
    ----------
    data_root:
        Root directory of the dataset.  Defaults to ``config.DATA_ROOT``.
    seed:
        Controls shuffle order for reproducible splits.

    Returns
    -------
    ``(train_dataset, val_dataset, test_dataset)``
    """
    root = data_root or config.DATA_ROOT
    if (root / DATASET_METADATA_FILE).is_file() and (root / "train").is_dir():
        class_names = get_class_names(root)
        train_samples = _scan_directory(root / "train", class_names)
        train_s, val_s = stratified_train_val_split(train_samples, seed=seed)
        test_s = _scan_directory(root / "test", class_names)
        total_samples = len(train_samples) + len(test_s)
    else:
        all_samples = _scan_directory(root)
        train_s, val_s, test_s = stratified_split(all_samples, seed=seed)
        total_samples = len(all_samples)

    train_ds = FruitRipenessDataset(train_s, transform=build_train_transform())
    val_ds   = FruitRipenessDataset(val_s,   transform=build_eval_transform())
    test_ds  = FruitRipenessDataset(test_s,  transform=build_eval_transform())

    print(
        f"[Dataset] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}  "
        f"(total={total_samples})"
    )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Convenience: build DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: Optional[Path] = None,
    seed: int = config.SEED,
    batch_size: int = config.BATCH_SIZE,
    pin_memory: Optional[bool] = None,
    balance_classes: bool = False,
    num_workers: Optional[int] = None,
) -> Tuple[
    "torch.utils.data.DataLoader",
    "torch.utils.data.DataLoader",
    "torch.utils.data.DataLoader",
]:
    """
    Build ``(train_loader, val_loader, test_loader)`` ready for training.

    A ``torch.Generator`` seeded with *seed* is used for the training
    loader's shuffle so that epoch order is deterministic.
    """
    from torch.utils.data import DataLoader

    train_ds, val_ds, test_ds = build_datasets(data_root=data_root, seed=seed)

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if pin_memory is None:
        pin_memory = config.PIN_MEMORY and torch.cuda.is_available()
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    g = torch.Generator()
    g.manual_seed(seed)

    drop_last = len(train_ds) >= batch_size
    sampler = None
    if balance_classes:
        class_counts = train_ds.class_counts()
        sample_weights = [1.0 / class_counts[label] for label in train_ds.labels]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_ds),
            replacement=True,
            generator=g,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
