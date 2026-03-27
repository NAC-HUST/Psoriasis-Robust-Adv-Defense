from __future__ import annotations

import argparse
import csv
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


@dataclass
class DownloadConfig:
    source: str
    output_dir: Path
    manifest_csv: Path
    cache_dir: Path
    image_size: int
    max_total: int
    max_per_class: int
    seed: int


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix in IMAGE_EXTENSIONS


def _resize_to_jpg(image: Image.Image, dst: Path, image_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    image.save(dst, format="JPEG", quality=95)


def _write_manifest(rows: list[dict[str, str | int]], manifest_csv: Path) -> None:
    if not rows:
        raise RuntimeError("没有可写入 manifest 的样本。")

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_indices(total: int, max_total: int, seed: int) -> list[int]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_total > 0:
        return indices[: max_total]
    return indices


def _build_from_cifar10(config: DownloadConfig) -> tuple[int, int]:
    try:
        from torchvision.datasets import CIFAR10
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("未安装 torchvision，无法下载 CIFAR-10。") from exc

    dataset = CIFAR10(root=str(config.cache_dir / "cifar10"), train=False, download=True)
    selected = _sample_indices(len(dataset), config.max_total, config.seed)

    rows: list[dict[str, str | int]] = []
    for out_idx, source_idx in enumerate(selected, start=1):
        image, label = dataset[source_idx]
        class_name = dataset.classes[label]

        dst_name = f"cifar10_ref_{out_idx:06d}.jpg"
        dst_path = config.output_dir / "cifar10" / class_name / dst_name
        _resize_to_jpg(image, dst_path, config.image_size)

        rows.append(
            {
                "file_path": str(dst_path),
                "label_binary": -1,
                "label_name": "reference",
                "subtype_label": class_name,
                "source": "cifar10_reference",
                "class_name": class_name,
                "class_idx": int(label),
                "original_id": int(source_idx),
            }
        )

    _write_manifest(rows, config.manifest_csv)
    return len(rows), len({row["class_name"] for row in rows})


def _download_imagenette(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "imagenette2-160.tgz"
    extract_root = cache_dir / "imagenette2-160"

    if not archive.exists():
        print(f"下载 Imagenette 到: {archive}")
        urllib.request.urlretrieve(IMAGENETTE_URL, archive)

    if not extract_root.exists():
        print(f"解压 Imagenette 到: {extract_root}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=cache_dir)

    return extract_root


def _build_from_imagenette(config: DownloadConfig) -> tuple[int, int]:
    extract_root = _download_imagenette(config.cache_dir / "imagenette")

    candidates: list[tuple[str, Path]] = []
    for split in ("train", "val"):
        split_dir = extract_root / split
        if not split_dir.exists():
            continue
        class_dirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
        for class_dir in class_dirs:
            imgs = [p for p in sorted(class_dir.rglob("*")) if _is_image_file(p)]
            if config.max_per_class > 0:
                imgs = imgs[: config.max_per_class]
            for img_path in imgs:
                candidates.append((class_dir.name, img_path))

    if not candidates:
        raise RuntimeError("Imagenette 数据为空，下载或解压可能失败。")

    rng = random.Random(config.seed)
    rng.shuffle(candidates)
    if config.max_total > 0:
        candidates = candidates[: config.max_total]

    class_to_idx = {name: i for i, name in enumerate(sorted({name for name, _ in candidates}))}

    rows: list[dict[str, str | int]] = []
    for out_idx, (class_name, src_path) in enumerate(candidates, start=1):
        with Image.open(src_path) as img:
            dst_name = f"imagenette_ref_{out_idx:06d}.jpg"
            dst_path = config.output_dir / "imagenette" / class_name / dst_name
            _resize_to_jpg(img, dst_path, config.image_size)

        rows.append(
            {
                "file_path": str(dst_path),
                "label_binary": -1,
                "label_name": "reference",
                "subtype_label": class_name,
                "source": "imagenette_reference",
                "class_name": class_name,
                "class_idx": class_to_idx[class_name],
                "original_id": str(src_path),
            }
        )

    _write_manifest(rows, config.manifest_csv)
    return len(rows), len(class_to_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载小规模对照数据集（CIFAR-10 / Imagenette）并生成独立 manifest")
    parser.add_argument("--source", choices=["cifar10", "imagenette"], default="cifar10")
    parser.add_argument("--output-dir", default="dataset/reference_224")
    parser.add_argument("--manifest-csv", default="dataset/reference_manifest.csv")
    parser.add_argument("--cache-dir", default="tmp/reference_cache")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-total", type=int, default=500)
    parser.add_argument("--max-per-class", type=int, default=50, help="仅对 imagenette 生效")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DownloadConfig(
        source=args.source,
        output_dir=Path(args.output_dir),
        manifest_csv=Path(args.manifest_csv),
        cache_dir=Path(args.cache_dir),
        image_size=args.image_size,
        max_total=args.max_total,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    if config.source == "cifar10":
        count, class_count = _build_from_cifar10(config)
    elif config.source == "imagenette":
        count, class_count = _build_from_imagenette(config)
    else:  # pragma: no cover
        raise ValueError(f"不支持的数据源: {config.source}")

    print(f"构建完成: source={config.source}, images={count}, classes={class_count}")
    print(f"输出目录: {config.output_dir}")
    print(f"manifest: {config.manifest_csv}")
    print("注意: 该 manifest 使用 label_binary=-1，仅用于对照分析，不用于当前二分类训练。")


if __name__ == "__main__":
    main()
