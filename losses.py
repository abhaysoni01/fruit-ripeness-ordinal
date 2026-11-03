"""
losses.py — Loss functions for the fruit-ripeness grading experiment.

Two losses are implemented:

1. **cross_entropy_loss** — thin wrapper around ``nn.CrossEntropyLoss``
   that accepts the same signature as ``coral_loss`` for drop-in
   interoperability in the training loop.

2. **coral_loss** — CORAL ordinal regression loss.
   For each of the K-1 binary threshold tasks the loss is a standard
   binary cross-entropy.  The total loss is the mean over all K-1
   tasks and all N samples::

       L_CORAL = -1/(N·(K-1)) Σ_i Σ_k [
           y_ik · log σ(logit_ik) + (1 - y_ik) · log(1 - σ(logit_ik))
       ]

   where the binary label ``y_ik = 𝟙[y_i > k]`` encodes whether
   sample *i*'s true rank exceeds threshold *k*.

References
----------
Cao, W., Mirjalili, V., & Raschka, S. (2020).
    Rank Consistent Ordinal Regression for Neural Networks.
    Pattern Recognition Letters.
    https://doi.org/10.1016/j.patrec.2020.11.006
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# 1. Standard Cross-Entropy (baseline)
# ---------------------------------------------------------------------------

_ce_criterion = nn.CrossEntropyLoss()


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Standard multi-class cross-entropy loss.

    Parameters
    ----------
    logits:
        Raw unnormalised scores, shape ``(B, num_classes)``.
    targets:
        Ground-truth integer class indices, shape ``(B,)``.

    Returns
    -------
    loss:
        Scalar tensor.
    """
    return _ce_criterion(logits, targets)


# ---------------------------------------------------------------------------
# 2. CORAL ordinal loss
# ---------------------------------------------------------------------------

def _labels_to_binary_targets(
    labels: torch.Tensor,
    num_thresholds: int = config.NUM_THRESHOLDS,
) -> torch.Tensor:
    """
    Convert ordinal integer labels to K-1 binary indicator vectors.

    For each sample *i* and threshold *k*:

        binary_target[i, k] = 1  if  label[i] > k  else 0

    Parameters
    ----------
    labels:
        Integer class indices, shape ``(B,)``.
    num_thresholds:
        K-1 thresholds (one per adjacent-class boundary).

    Returns
    -------
    binary_targets:
        Float tensor of shape ``(B, K-1)`` in ``{0.0, 1.0}``.

    Examples
    --------
    >>> _labels_to_binary_targets(torch.tensor([0, 1, 2, 3]), num_thresholds=3)
    tensor([[0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    thresholds = torch.arange(num_thresholds, device=labels.device)  # (K-1,)
    # (B, 1) > (1, K-1) → (B, K-1)
    binary = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
    return binary


def coral_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = config.NUM_THRESHOLDS,
) -> torch.Tensor:
    """
    CORAL ordinal regression loss.

    Parameters
    ----------
    logits:
        Raw binary logits from :class:`models.CoralNet`, shape ``(B, K-1)``.
    targets:
        Ground-truth integer ordinal labels, shape ``(B,)``.
    num_thresholds:
        K-1 threshold tasks (default from ``config.NUM_THRESHOLDS``).

    Returns
    -------
    loss:
        Scalar mean binary cross-entropy over all (sample, threshold) pairs.

    Notes
    -----
    ``F.binary_cross_entropy_with_logits`` is numerically more stable
    than computing ``sigmoid`` then ``log`` separately.
    """
    binary_targets = _labels_to_binary_targets(targets, num_thresholds)
    # Element-wise BCE; mean over both batch and threshold dimensions
    loss = F.binary_cross_entropy_with_logits(
        logits,
        binary_targets,
        reduction="mean",
    )
    return loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss_fn(variant: str, num_classes: int = config.NUM_CLASSES):
    """
    Return the appropriate loss function for a model variant.

    Parameters
    ----------
    variant:
        ``"baseline"`` → :func:`cross_entropy_loss`.
        ``"ordinal"``  → :func:`coral_loss`.

    Returns
    -------
    Callable with signature ``(logits, targets) → scalar_tensor``.
    """
    if variant == config.MODEL_BASELINE:
        return cross_entropy_loss
    elif variant == config.MODEL_ORDINAL:
        return partial(coral_loss, num_thresholds=num_classes - 1)
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose '{config.MODEL_BASELINE}' or '{config.MODEL_ORDINAL}'."
        )
