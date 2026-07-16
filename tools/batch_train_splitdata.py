#!/usr/bin/env python3
"""Batch train ResNet50 on all 5 split_data datasets with uniform params.

Usage:
    .venv/bin/python tools/batch_train_splitdata.py

Uniform training params (user-approved):
    epochs=5, batch_size=32, lr=1e-4, seed=42, image_size=224.
"""

from __future__ import annotations

from pathlib import Path

import torch

from psorad.config import DatasetConfig, ExperimentConfig, ModelConfig, TrainSettings, load_dataset_config
from psorad.models.train import train_experiment

DATASET_NAMES = [
    "psoriasis224_2c",
    "cifar32_10c",
    "imagenette224_10c",
    "dermamnist224_7c",
    "milk10k_11c",
]

DATASET_CONFIG = "configs/dataset_config/baseline_dataset.toml"
PRETRAINED_PATH = "model/pretrained_model/resnet/resnet50_imagenet1k_v2.pth"
OUTPUT_DIR = "model/trained_classifier"

TRAIN_KWARGS = {
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "seed": 42,
    "num_workers": 4,
    "image_size": 224,
}


def main() -> None:
    results: dict[str, dict[str, object]] = {}

    for d_name in DATASET_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Training {d_name} ...")
        print(f"{'=' * 60}")

        dataset_cfg: DatasetConfig = load_dataset_config(  # type: ignore[no-untyped-call]
            DATASET_CONFIG, name=d_name
        )
        dataset_cfg.image_size = TRAIN_KWARGS["image_size"]

        num_classes = len(dataset_cfg.class_names)
        print(f"  classes={num_classes}, image_size={dataset_cfg.image_size}")

        model_cfg = ModelConfig(
            backbone="resnet50",
            pretrained_path=Path(PRETRAINED_PATH),
            num_classes=num_classes,
        )

        train_cfg = TrainSettings(
            epochs=TRAIN_KWARGS["epochs"],
            batch_size=TRAIN_KWARGS["batch_size"],
            learning_rate=TRAIN_KWARGS["learning_rate"],
            seed=TRAIN_KWARGS["seed"],
            num_workers=TRAIN_KWARGS["num_workers"],
            output_dir=Path(OUTPUT_DIR),
            model_name=f"resnet50_{d_name}.pt",
        )

        exp_cfg = ExperimentConfig(
            dataset=dataset_cfg,
            model=model_cfg,
            train=train_cfg,
        )

        try:
            ckpt_path = train_experiment(exp_cfg)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            val_acc = float(ckpt.get("val_acc", -1.0))
            results[d_name] = {"checkpoint": str(ckpt_path), "val_acc": val_acc}
            print(f"  ✓ {d_name}: val_acc={val_acc:.4f} -> {ckpt_path}")
        except Exception as e:
            results[d_name] = {"error": str(e)}
            print(f"  ✗ {d_name} FAILED: {e}")

    print(f"\n{'=' * 60}")
    print("Batch training summary:")
    print(f"{'=' * 60}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name}: FAILED - {r['error']}")
        else:
            print(f"  {name}: val_acc={r['val_acc']:.4f}  ({r['checkpoint']})")


if __name__ == "__main__":
    main()
