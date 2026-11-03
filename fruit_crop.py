"""Foreground-based fruit cropping for plain or nearly uniform backgrounds."""

from __future__ import annotations

import numpy as np
from PIL import Image


def crop_fruit(image: Image.Image, padding: float = 0.08) -> Image.Image:
    """Crop around foreground pixels that contrast with the image border.

    The border color estimates the background, making this suitable for
    isolated-fruit images such as Fruits-360. If a distinct foreground cannot
    be identified, the input image is returned unchanged.
    """
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    if width < 3 or height < 3:
        return rgb_image

    scale = min(1.0, 512 / max(width, height))
    sample_size = (max(3, round(width * scale)), max(3, round(height * scale)))
    sample = rgb_image.resize(sample_size, Image.Resampling.BILINEAR)
    pixels = np.asarray(sample, dtype=np.float32)
    sample_height, sample_width = pixels.shape[:2]

    border_width = max(1, round(min(sample_width, sample_height) * 0.03))
    border_pixels = np.concatenate(
        (
            pixels[:border_width, :, :].reshape(-1, 3),
            pixels[-border_width:, :, :].reshape(-1, 3),
            pixels[:, :border_width, :].reshape(-1, 3),
            pixels[:, -border_width:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(border_pixels, axis=0)
    border_distances = np.linalg.norm(border_pixels - background, axis=1)
    threshold = max(24.0, float(np.percentile(border_distances, 95)) + 8.0)
    foreground_y, foreground_x = np.where(
        np.linalg.norm(pixels - background, axis=2) > threshold
    )

    if foreground_x.size < max(16, round(sample_width * sample_height * 0.005)):
        return rgb_image

    x0, x1 = np.percentile(foreground_x, (1, 99))
    y0, y1 = np.percentile(foreground_y, (1, 99))
    x0 = int(x0 * width / sample_width)
    x1 = int((x1 + 1) * width / sample_width)
    y0 = int(y0 * height / sample_height)
    y1 = int((y1 + 1) * height / sample_height)

    if x1 - x0 >= width * 0.98 or y1 - y0 >= height * 0.98:
        return rgb_image

    pad_x = round((x1 - x0) * padding)
    pad_y = round((y1 - y0) * padding)
    crop_box = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )
    return rgb_image.crop(crop_box)
