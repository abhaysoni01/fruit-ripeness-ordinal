"""
create_dummy_data.py — Generate a minimal synthetic image dataset for testing
and verification of the fruit ripeness ordinal regression pipeline.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import config

CLASS_COLORS = {
    "unripe": (34, 139, 34),          # Forest green
    "slightly_ripe": (154, 205, 50),   # Yellow green
    "ripe": (255, 215, 0),            # Gold / yellow
    "overripe": (139, 69, 19),         # Saddle brown
}

def generate_synthetic_dataset(num_samples_per_class: int = 12) -> None:
    data_root = config.DATA_ROOT
    print(f"Creating synthetic dataset at: {data_root}")

    for class_name, base_rgb in CLASS_COLORS.items():
        class_dir = data_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_class):
            # Create a 224x224 synthetic image with fruit-like circular patch and color variation
            img_arr = np.full((224, 224, 3), 240, dtype=np.uint8)  # light background

            # Circular fruit patch in center
            y, x = np.ogrid[:224, :224]
            center_y, center_x = 112, 112
            radius = 80
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (radius ** 2)

            # Add subtle noise/variation to base color
            noise = np.random.randint(-15, 15, (224, 224, 3))
            fruit_color = np.clip(np.array(base_rgb) + noise, 0, 255).astype(np.uint8)

            img_arr[mask] = fruit_color[mask]

            img = Image.fromarray(img_arr)
            img_path = class_dir / f"{class_name}_{i:03d}.jpg"
            img.save(img_path, quality=90)

    print(f"Generated {num_samples_per_class * len(CLASS_COLORS)} synthetic images across {len(CLASS_COLORS)} classes.")

if __name__ == "__main__":
    generate_synthetic_dataset()
