from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from psorad.utils.image import center_crop_resize

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


@dataclass
class PreprocessConfig:
    raw_dataset_dir: Path
    processed_dataset_dir: Path
    manifest_csv: Path
    image_size: int = 224


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix in IMAGE_EXTENSIONS


def _resize_and_save(src: Path, dst: Path, image_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        processed = center_crop_resize(img, image_size=image_size)
        processed.save(dst, format="JPEG", quality=95)


def _discover_class_dirs(raw_dataset_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(raw_dataset_dir.iterdir())
        if p.is_dir() and any(_is_image_file(x) for x in p.rglob("*"))
    ]


def preprocess_dataset_and_build_manifest(config: PreprocessConfig) -> pd.DataFrame:
    if not config.raw_dataset_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {config.raw_dataset_dir}")

    class_dirs = _discover_class_dirs(config.raw_dataset_dir)
    if not class_dirs:
        raise RuntimeError(
            f"未在 {config.raw_dataset_dir} 发现包含图像的类别子目录，"
            "请按 raw_data/<datadir>/<class_name>/*.jpg 组织数据。"
        )

    rows: list[dict[str, str | int]] = []
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_images = [p for p in sorted(class_dir.rglob("*")) if _is_image_file(p)]

        for image_idx, source_path in enumerate(class_images, start=1):
            target_name = f"{class_name}_{image_idx:06d}.jpg"
            target_path = config.processed_dataset_dir / class_name / target_name
            _resize_and_save(source_path, target_path, config.image_size)

            rows.append(
                {
                    "file_path": str(target_path),
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "label_name": class_name,
                    "subtype_label": class_name,
                    "source": f"raw_data/{config.raw_dataset_dir.name}",
                    "original_file": str(source_path),
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("构建后的 manifest 为空，请检查原始数据路径和图像文件。")

    manifest = manifest.sample(frac=1.0, random_state=42).reset_index(drop=True)
    config.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.manifest_csv, index=False)
    return manifest


def run_preprocess(
    datadir: str,
    dataset_root: str = "dataset",
    raw_data_root: str | None = None,
    processed_data_root: str | None = None,
    image_size: int = 224,
) -> tuple[int, int, Path]:
    dataset_root_path = Path(dataset_root)
    raw_root = Path(raw_data_root) if raw_data_root is not None else dataset_root_path / "raw_data"
    processed_root = Path(processed_data_root) if processed_data_root is not None else dataset_root_path / "processed_data"
    raw_dataset_dir = raw_root / datadir
    processed_dataset_dir = processed_root / datadir
    manifest_csv = processed_dataset_dir / "class_manifest.csv"

    config = PreprocessConfig(
        raw_dataset_dir=raw_dataset_dir,
        processed_dataset_dir=processed_dataset_dir,
        manifest_csv=manifest_csv,
        image_size=image_size,
    )

    manifest = preprocess_dataset_and_build_manifest(config)
    return len(manifest), len(manifest), manifest_csv
