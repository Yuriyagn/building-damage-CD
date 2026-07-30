from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


MASK_COLORS = np.array(
    [
        [0, 0, 0],
        [76, 175, 80],
        [255, 193, 7],
        [244, 67, 54],
    ],
    dtype=np.uint8,
)


def overlay_png(image_path: Path, mask_path: Path, alpha: float = 0.42) -> bytes:
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    colors = MASK_COLORS[np.clip(mask.astype(np.int64), 0, len(MASK_COLORS) - 1)]
    color_image = Image.fromarray(colors).resize(image.size, Image.Resampling.NEAREST)
    blended = Image.blend(image, color_image, alpha)
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return buffer.getvalue()

