"""
metrics.py — Evaluation metrics for ordinal fruit-ripeness grading.

All functions operate on **plain Python lists or NumPy arrays** (not
tensors) so they can be called independently of any deep-learning
framework.  A single :func:`compute_all_metrics` convenience wrapper
collects every metric into one dictionary.

Metrics implemented
-------------------
* **Accuracy** — top-1 exact-match accuracy.
* **MAE** — Mean Absolute Error (primary ordinal metric).
* **Off-by-1 Accuracy** — fraction of samples where |y_pred - y_true| ≤ 1.
* **Confusion Matrix** — raw N×N count matrix (not normalised).
* **Per-class MAE** — MAE broken down by true class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Labels = Sequence[int]   # any sequence of integer class indices


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: Labels, y_pred: Labels) -> float:
    """
    Top-1 exact-match classification accuracy.

    Parameters
    ----------
    y_true, y_pred:
        Integer class label sequences of equal length.

    Returns
    -------
    Float in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def mean_absolute_error(y_true: Labels, y_pred: Labels) -> float:
    """
    Mean Absolute Error — primary ordinal regression metric.

    .. math::

        \\text{MAE} = \\frac{1}{N} \\sum_{i=1}^{N} |y_i - \\hat{y}_i|

    Parameters
    ----------
    y_true, y_pred:
        Integer ordinal label sequences of equal length.

    Returns
    -------
    Float ≥ 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def off_by_one_accuracy(y_true: Labels, y_pred: Labels, margin: int = 1) -> float:
    """
    Fraction of samples whose prediction is within *margin* ranks of the truth.

    Parameters
    ----------
    y_true, y_pred:
        Integer ordinal label sequences.
    margin:
        Allowed absolute deviation (default 1 → off-by-1).

    Returns
    -------
    Float in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred) <= margin))


def confusion_matrix(
    y_true: Labels,
    y_pred: Labels,
    num_classes: int = config.NUM_CLASSES,
) -> np.ndarray:
    """
    Compute the raw (unnormalised) confusion matrix.

    Parameters
    ----------
    y_true, y_pred:
        Integer label sequences in ``[0, num_classes)``.
    num_classes:
        Matrix dimension.

    Returns
    -------
    ``cm`` of shape ``(num_classes, num_classes)`` where
    ``cm[i, j]`` is the count of samples with true class *i*
    predicted as class *j*.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_mae(
    y_true: Labels,
    y_pred: Labels,
    num_classes: int = config.NUM_CLASSES,
) -> Dict[int, float]:
    """
    Compute MAE separately for each true class.

    Parameters
    ----------
    y_true, y_pred:
        Integer label sequences.
    num_classes:
        Number of ordinal classes.

    Returns
    -------
    Dictionary ``{class_id: mae_value}``.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    result: Dict[int, float] = {}
    for cls in range(num_classes):
        mask = y_true_arr == cls
        if mask.sum() == 0:
            result[cls] = float("nan")
        else:
            result[cls] = float(np.mean(np.abs(y_true_arr[mask] - y_pred_arr[mask])))
    return result


# ---------------------------------------------------------------------------
# Combined wrapper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: Labels,
    y_pred: Labels,
    num_classes: int = config.NUM_CLASSES,
) -> Dict[str, object]:
    """
    Compute the full suite of evaluation metrics in one call.

    Parameters
    ----------
    y_true, y_pred:
        Integer ordinal label sequences.
    num_classes:
        Number of ordinal classes.

    Returns
    -------
    Dictionary with keys::

        "accuracy"         → float
        "mae"              → float
        "off_by_1_acc"     → float
        "confusion_matrix" → np.ndarray (num_classes, num_classes)
        "per_class_mae"    → Dict[int, float]
    """
    return {
        "accuracy":         accuracy(y_true, y_pred),
        "mae":              mean_absolute_error(y_true, y_pred),
        "off_by_1_acc":     off_by_one_accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes),
        "per_class_mae":    per_class_mae(y_true, y_pred, num_classes),
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def format_metrics_table(
    results: Dict[str, Dict[str, object]],
    class_names: List[str] = config.CLASS_NAMES,
) -> str:
    """
    Render a side-by-side comparison table for multiple model variants.

    Parameters
    ----------
    results:
        Mapping ``{variant_name: metrics_dict}`` where each
        ``metrics_dict`` is the output of :func:`compute_all_metrics`.
    class_names:
        Human-readable class labels for per-class MAE rows.

    Returns
    -------
    Formatted multi-line string suitable for printing to stdout or a log.
    """
    variants = list(results.keys())
    col_w = 18

    header = f"{'Metric':<25}" + "".join(f"{v:>{col_w}}" for v in variants)
    sep = "-" * len(header)

    rows = [sep, header, sep]

    def _add_row(label: str, key: str, fmt: str = ".4f") -> None:
        row = f"{label:<25}"
        for v in variants:
            val = results[v].get(key, float("nan"))
            row += f"{val:>{col_w}{fmt}}"
        rows.append(row)

    _add_row("Accuracy",                   "accuracy")
    _add_row("MAE (lower is better)",       "mae")
    _add_row("Off-by-1 Acc (higher better)", "off_by_1_acc")

    # Per-class MAE
    rows.append(sep)
    for cls_id, cls_name in enumerate(class_names):
        label = f"  MAE [{cls_name}]"
        row = f"{label:<25}"
        for v in variants:
            pc_mae = results[v].get("per_class_mae", {})
            val = pc_mae.get(cls_id, float("nan"))
            row += f"{val:>{col_w}.4f}"
        rows.append(row)

    rows.append(sep)
    return "\n".join(rows)


def print_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = config.CLASS_NAMES,
    title: str = "Confusion Matrix",
) -> None:
    """Pretty-print a confusion matrix to stdout."""
    print(f"\n{title}")
    col_w = max(len(n) for n in class_names) + 2
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)
    print()
