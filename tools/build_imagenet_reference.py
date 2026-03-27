from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


@dataclass
class BuildConfig:
    imagenet_dir: Path
    output_dir: Path
    manifest_csv: Path
    image_size: int
    max_per_class: int
    max_total: int
    seed: int


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix in IMAGE_EXTENSIONS


def _discover_class_dirs(imagenet_dir: Path) -> list[Path]:
    class_dirs = [p for p in sorted(imagenet_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(
            "未发现类别子目录，请传入类似 ImageNet val/train 的目录（一级子目录为类别）。"
        )
    return class_dirs


def _resize_and_save(src: Path, dst: Path, image_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        image = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        image.save(dst, format="JPEG", quality=95)


def build_reference_set(config: BuildConfig) -> tuple[int, int]:
    rng = random.Random(config.seed)
    class_dirs = _discover_class_dirs(config.imagenet_dir)

    picked: list[tuple[str, Path]] = []
    for class_dir in class_dirs:
        images = [p for p in sorted(class_dir.rglob("*")) if _is_image_file(p)]
        if not images:
            continue
        rng.shuffle(images)
        for image_path in images[: max(0, config.max_per_class)]:
            picked.append((class_dir.name, image_path))

    rng.shuffle(picked)
    if config.max_total > 0:
        picked = picked[: config.max_total]

    if not picked:
        raise RuntimeError("未采样到任何图像，请检查 imagenet 目录结构或参数设置。")

    rows: list[dict[str, str | int]] = []
    class_name_to_idx = {name: idx for idx, name in enumerate(sorted({c for c, _ in picked}))}

    for i, (class_name, image_path) in enumerate(picked, start=1):
        dst_name = f"imagenet_ref_{i:06d}.jpg"
        dst_path = config.output_dir / class_name / dst_name
        _resize_and_save(image_path, dst_path, config.image_size)

        rows.append(
            {
                "file_path": str(dst_path),
                "label_binary": -1,
                "label_name": "imagenet_reference",
                "subtype_label": class_name,
                "source": "imagenet_reference",
                "imagenet_class": class_name,
                "imagenet_class_idx": class_name_to_idx[class_name],
                "original_path": str(image_path),
            }
        )

    config.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), len(class_name_to_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 ImageNet 对照参考子集（独立 manifest，不影响现有二分类训练）")
    parser.add_argument(
        "--imagenet-dir",
        required=True,
        help="ImageNet 目录，要求一级子目录为类别（例如 val/n01440764/...）",
    )
    parser.add_argument("--output-dir", default="dataset/imagenet_ref_224")
    parser.add_argument("--manifest-csv", default="dataset/imagenet_ref_manifest.csv")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-per-class", type=int, default=5, help="每个类别最多采样图像数")
    parser.add_argument("--max-total", type=int, default=500, help="最终总图像上限，<=0 表示不限制")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        imagenet_dir=Path(args.imagenet_dir),
        output_dir=Path(args.output_dir),
        manifest_csv=Path(args.manifest_csv),
        image_size=args.image_size,
        max_per_class=args.max_per_class,
        max_total=args.max_total,
        seed=args.seed,
    )

    count, class_count = build_reference_set(config)
    print(f"ImageNet 对照集构建完成: {count} 张, 类别数: {class_count}")
    print(f"输出目录: {config.output_dir}")
    print(f"manifest: {config.manifest_csv}")
    print("注意: 该 manifest 的 label_binary=-1，仅用于对照分析，不应直接用于当前二分类训练。")


if __name__ == "__main__":
    main()
