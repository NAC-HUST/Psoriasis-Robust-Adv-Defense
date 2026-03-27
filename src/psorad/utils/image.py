from __future__ import annotations

from PIL import Image, ImageOps


def center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    rgb_image = image.convert("RGB")
    return ImageOps.fit(
        rgb_image,
        (image_size, image_size),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )
