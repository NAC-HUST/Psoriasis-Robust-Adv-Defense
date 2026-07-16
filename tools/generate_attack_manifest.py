#!/usr/bin/env python3
"""Generate attack manifests from split_data label CSVs.

For each dataset in baseline_dataset.toml, reads train_labels.csv and
val_labels.csv (one-hot label_* columns) and writes a unified manifest CSV
with columns: file_path,class_idx,class_name,split.

Usage:
    .venv/bin/python tools/generate_attack_manifest.py

Output:
    output/manifests/{dataset_name}_manifest.csv  (gitignored)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from psorad.config import load_dataset_config

DATASET_NAMES = [
    "psoriasis224_2c",
    "cifar32_10c",
    "imagenette224_10c",
    "dermamnist224_7c",
    "milk10k_11c",
]

DATASET_CONFIG = "configs/dataset_config/baseline_dataset.toml"
OUTPUT_DIR = Path("output/manifests")


def _build_manifest(
    dataset_name: str,
    class_names: tuple[str, ...],
    label_columns: tuple[str, ...],
    image_column: str,
    dataset_dir: Path,
    split: str,
) -> pd.DataFrame:
    label_csv = dataset_dir / "labels" / f"{split}_labels.csv"
    if not label_csv.exists():
        print(f"  [skip] {label_csv} not found")
        return pd.DataFrame()

    df = pd.read_csv(label_csv)

    if image_column not in df.columns:
        raise ValueError(f"{label_csv} 缺少 image_column='{image_column}'")

    # resolve relative paths to absolute
    df["file_path"] = df[image_column].apply(lambda p: str((dataset_dir / p).resolve()))

    # class_idx: argmax of label_* columns matching class_names order
    # first, map label_columns to class_names order (they should already match)
    label_values = df[list(label_columns)].values.astype(np.float32)
    df["class_idx"] = np.argmax(label_values, axis=1).astype(np.int64)

    # class_name
    name_map = {i: name for i, name in enumerate(class_names)}
    df["class_name"] = df["class_idx"].map(name_map)

    df["split"] = split

    # keep only needed columns
    result = df[["file_path", "class_idx", "class_name", "split"]].copy()
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for d_name in DATASET_NAMES:
        print(f"Processing {d_name} ...")
        cfg = load_dataset_config(DATASET_CONFIG, name=d_name)

        parts: list[pd.DataFrame] = []
        for split in ("train", "val"):
            part = _build_manifest(
                dataset_name=d_name,
                class_names=cfg.class_names,
                label_columns=cfg.label_columns,
                image_column=cfg.image_column,
                dataset_dir=cfg.dataset_dir,
                split=split,
            )
            if not part.empty:
                parts.append(part)

        if not parts:
            print(f"  ✗ {d_name}: no data found, skipping")
            continue

        manifest = pd.concat(parts, axis=0, ignore_index=True)
        out_path = OUTPUT_DIR / f"{d_name}_manifest.csv"
        manifest.to_csv(out_path, index=False)
        print(f"  ✓ {d_name}: {len(manifest)} samples -> {out_path}")


if __name__ == "__main__":
    main()
