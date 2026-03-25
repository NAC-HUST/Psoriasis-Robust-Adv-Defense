from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


@dataclass
class PreprocessConfig:
    dataset_root: Path
    psoriasis_metadata_csv: Path
    normal_raw_dir: Path
    normal_processed_dir: Path
    manifest_csv: Path
    image_size: int = 224


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix in IMAGE_EXTENSIONS


def _resize_and_save(src: Path, dst: Path, image_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        image = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        image.save(dst, format="JPEG", quality=95)


def preprocess_normal_images(config: PreprocessConfig) -> int:
    raw_images = [p for p in sorted(config.normal_raw_dir.rglob("*")) if _is_image_file(p)]
    processed_count = 0

    for idx, source_path in enumerate(raw_images, start=1):
        target_name = f"normal_{idx:06d}.jpg"
        target_path = config.normal_processed_dir / target_name
        _resize_and_save(source_path, target_path, config.image_size)
        processed_count += 1

    return processed_count


def build_binary_manifest(config: PreprocessConfig) -> pd.DataFrame:
    metadata = pd.read_csv(config.psoriasis_metadata_csv)
    required_columns = {"file_name", "subtype_label"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"psoriasis metadata 缺少列: {sorted(missing_columns)}")

    psoriasis_rows: list[dict[str, str | int]] = []
    for _, row in metadata.iterrows():
        rel = str(row["file_name"])
        image_path = config.dataset_root / rel
        if image_path.exists():
            psoriasis_rows.append(
                {
                    "file_path": str(image_path),
                    "label_binary": 1,
                    "label_name": "psoriasis",
                    "subtype_label": str(row["subtype_label"]),
                    "source": "psoriasis_metadata",
                }
            )

    normal_rows: list[dict[str, str | int]] = []
    for normal_img in sorted(config.normal_processed_dir.glob("*.jpg")):
        normal_rows.append(
            {
                "file_path": str(normal_img),
                "label_binary": 0,
                "label_name": "normal",
                "subtype_label": "normal",
                "source": "normal_preprocessed",
            }
        )

    manifest = pd.DataFrame(psoriasis_rows + normal_rows)
    if manifest.empty:
        raise RuntimeError("构建后的 manifest 为空，请检查数据路径。")

    manifest = manifest.sample(frac=1.0, random_state=42).reset_index(drop=True)
    config.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.manifest_csv, index=False)
    return manifest


def run_preprocess(
    dataset_root: str = "dataset",
    metadata_csv: str = "dataset/psoriasis_metadata.csv",
    normal_raw_dir: str = "dataset/normal",
    normal_processed_dir: str = "dataset/normal_224",
    manifest_csv: str = "dataset/binary_manifest.csv",
    image_size: int = 224,
) -> tuple[int, int]:
    config = PreprocessConfig(
        dataset_root=Path(dataset_root),
        psoriasis_metadata_csv=Path(metadata_csv),
        normal_raw_dir=Path(normal_raw_dir),
        normal_processed_dir=Path(normal_processed_dir),
        manifest_csv=Path(manifest_csv),
        image_size=image_size,
    )

    processed_count = preprocess_normal_images(config)
    manifest = build_binary_manifest(config)
    return processed_count, len(manifest)
