"""
models.py — ResNet-18 model variants for fruit ripeness grading.

Two architectures are provided:

1. **BaselineNet**  — standard pretrained ResNet-18 with a 4-logit head
   trained with ``nn.CrossEntropyLoss``.

2. **CoralNet** — pretrained ResNet-18 with a CORAL (COnsistent RAnk
   Logits) ordinal head.  The head produces K-1 binary logits that
   share a single linear weight vector but learn independent bias
   (threshold) scalars, ensuring rank consistency by construction
   (Cao et al., 2020).

References
----------
Cao, W., Mirjalili, V., & Raschka, S. (2020).
    Rank Consistent Ordinal Regression for Neural Networks with
    Application to Age Estimation. Pattern Recognition Letters.
    https://doi.org/10.1016/j.patrec.2020.11.006
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

import config


# ---------------------------------------------------------------------------
# 1. Baseline — standard 4-class softmax head
# ---------------------------------------------------------------------------

class BaselineNet(nn.Module):
    """
    Pretrained ResNet-18 with a fully-connected 4-class output head.

    The backbone is initialised from ImageNet weights; only the final
    ``fc`` layer is replaced.  The new head is *not* frozen so that
    end-to-end fine-tuning can adapt both body and head.

    Parameters
    ----------
    num_classes:
        Number of ordinal categories (default 4).
    pretrained:
        Load ImageNet-pretrained weights for the backbone.
    freeze_backbone:
        If ``True``, freeze all backbone parameters during the first
        few warm-up epochs (useful for transfer learning warm-up).
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Save the feature dimension before replacing the head
        in_features: int = backbone.fc.in_features

        # Replace classification head
        backbone.fc = nn.Identity()   # feature extractor
        self.backbone = backbone

        # New classification head
        self.classifier = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input batch of shape ``(B, 3, H, W)``.

        Returns
        -------
        logits:
            Raw unnormalised scores of shape ``(B, num_classes)``.
            Pass through ``nn.CrossEntropyLoss`` during training or
            ``torch.argmax`` during inference.
        """
        features = self.backbone(x)           # (B, 512)
        logits = self.classifier(features)    # (B, 4)
        return logits

    # ------------------------------------------------------------------
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer class predictions ``(B,)``."""
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)

    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimension of the penultimate feature vector."""
        return self.classifier.in_features


# ---------------------------------------------------------------------------
# 2. CORAL ordinal head
# ---------------------------------------------------------------------------

class _CoralHead(nn.Module):
    """
    Rank-consistent ordinal head for K-1 binary classification tasks.

    A *single* weight vector ``w`` (shape ``[d]``) projects the feature
    vector; K-1 ordered bias scalars act as rank thresholds. The output is::

        logit_k = x @ w.T + b_k, with b_0 > b_1 > ... > b_{K-2}

    Ordered thresholds ensure that the cumulative probabilities decrease
    with rank, as required for ordinal predictions.

    Parameters
    ----------
    in_features:
        Dimension of the input feature vector (backbone output).
    num_thresholds:
        K-1 binary output logits (one per adjacent-class boundary).
    """

    def __init__(self, in_features: int, num_thresholds: int = config.NUM_THRESHOLDS) -> None:
        super().__init__()
        self.num_thresholds = num_thresholds

        # Shared weight (no bias — biases are the threshold parameters)
        self.weight = nn.Linear(in_features, 1, bias=False)

        self.base_bias = nn.Parameter(torch.zeros(1))
        self.raw_threshold_gaps = nn.Parameter(torch.zeros(max(0, num_thresholds - 1)))

    def threshold_biases(self) -> torch.Tensor:
        """Return monotonically decreasing threshold biases."""
        if self.num_thresholds == 1:
            return self.base_bias
        gaps = F.softplus(self.raw_threshold_gaps)
        return torch.cat((self.base_bias, self.base_bias - torch.cumsum(gaps, dim=0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Feature tensor of shape ``(B, in_features)``.

        Returns
        -------
        logits:
            Shape ``(B, K-1)`` — raw binary logits before sigmoid.
        """
        shared = self.weight(x)                         # (B, 1)
        logits = shared + self.threshold_biases().unsqueeze(0)
        return logits


class CoralNet(nn.Module):
    """
    Pretrained ResNet-18 backbone + CORAL ordinal regression head.

    Inference converts the K-1 binary sigmoid probabilities into an
    ordinal class label by summing the rounded indicators::

        ŷ = ∑_{k=0}^{K-2}  𝟙[σ(logit_k) > 0.5]

    Parameters
    ----------
    num_classes:
        Number of ordinal categories (default 4).
    pretrained:
        Load ImageNet-pretrained weights for the backbone.
    freeze_backbone:
        If ``True``, freeze backbone parameters initially.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features: int = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.ordinal_head = _CoralHead(in_features, self.num_thresholds)

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input batch ``(B, 3, H, W)``.

        Returns
        -------
        logits:
            Shape ``(B, K-1)`` — raw binary logits.
            Apply ``torch.sigmoid`` for probabilities or pass directly
            to :func:`losses.coral_loss`.
        """
        features = self.backbone(x)             # (B, 512)
        logits = self.ordinal_head(features)    # (B, K-1)
        return logits

    # ------------------------------------------------------------------
    @staticmethod
    def logits_to_label(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert CORAL binary logits to ordinal class labels.

        Parameters
        ----------
        logits:
            Shape ``(B, K-1)``.

        Returns
        -------
        labels:
            Integer predictions in ``[0, K-1]``, shape ``(B,)``.
        """
        probs = torch.sigmoid(logits)           # (B, K-1)
        return probs.gt(0.5).sum(dim=1).long()  # (B,)

    # ------------------------------------------------------------------
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer class predictions ``(B,)``."""
        with torch.no_grad():
            return self.logits_to_label(self.forward(x))

    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    variant: str,
    device: torch.device,
    num_classes: int = config.NUM_CLASSES,
) -> nn.Module:
    """
    Instantiate and move a model variant to *device*.

    Parameters
    ----------
    variant:
        ``"baseline"`` or ``"ordinal"``.
    device:
        Target compute device.

    Returns
    -------
    model:
        ``BaselineNet`` or ``CoralNet`` instance on *device*.
    """
    if variant == config.MODEL_BASELINE:
        model: nn.Module = BaselineNet(num_classes=num_classes)
    elif variant == config.MODEL_ORDINAL:
        model = CoralNet(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model variant '{variant}'. "
            f"Choose from '{config.MODEL_BASELINE}' or '{config.MODEL_ORDINAL}'."
        )

    return model.to(device)
