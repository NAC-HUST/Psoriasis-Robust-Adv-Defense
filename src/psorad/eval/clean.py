from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from psorad.config import ModelConfig
from psorad.data.dataset import SkinDataset, build_transforms
from psorad.eval.metrics import clean_metrics
from psorad.models.factory import build_model


def load_classifier(backbone: str, checkpoint_path: str, device: torch.device) -> tuple[nn.Module, int]:
    # 加载分类器
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    num_classes = 2
    for key, tensor in state_dict.items():
        if key.endswith("fc.weight") or key.endswith("classifier.weight"):
            num_classes = int(tensor.shape[0])
            break

    model_cfg = ModelConfig(backbone=backbone, pretrained_path=None, freeze_backbone=False, num_classes=num_classes)
    model = build_model(model_cfg, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, num_classes


def _select_split_manifest(manifest_csv: str, split: str, val_ratio: float, split_seed: int) -> pd.DataFrame:
    # 按 split 选取子集
    split_name = split.lower().strip()
    if split_name not in {"all", "train", "val"}:
        raise ValueError("split 仅支持 all/train/val")

    manifest = pd.read_csv(manifest_csv)
    if split_name == "all":
        return manifest.reset_index(drop=True)

    if "split" in manifest.columns:
        subset = manifest[manifest["split"].astype(str).str.lower() == split_name].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"manifest 中未找到 split={split_name} 的样本")
        return subset

    total = len(manifest)
    val_len = max(int(total * val_ratio), 1)
    train_len = total - val_len
    if train_len <= 0:
        raise ValueError("训练集为空，请增大数据量或减小 val_ratio")

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(total, generator=generator).tolist()
    selected = indices[:train_len] if split_name == "train" else indices[train_len:]
    return manifest.iloc[selected].reset_index(drop=True)


@torch.no_grad()
def evaluate_clean(
    *,
    backbone: str,
    checkpoint_path: str,
    manifest_csv: str,
    split: str = "val",
    val_ratio: float = 0.2,
    split_seed: int = 42,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict[str, Any]:
    # 前向推理并计算指标
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, num_classes = load_classifier(backbone, checkpoint_path, device)

    subset = _select_split_manifest(manifest_csv, split, val_ratio, split_seed)
    dataset = SkinDataset(manifest_csv=None, transform=build_transforms(image_size=image_size, for_siglip=backbone == "siglip", train=False))
    dataset.set_manifest(subset)
    loader: DataLoader[Any] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        scores.append(probs.detach().cpu().numpy())
        labels.append(targets.numpy())

    if not scores:
        raise RuntimeError("评估子集为空，无法计算 clean 指标")

    y_score = np.concatenate(scores, axis=0).astype(np.float64)
    y_true = np.concatenate(labels, axis=0).astype(np.int64)

    metrics = clean_metrics(y_true, y_score, num_classes)
    metrics["split"] = split
    metrics["checkpoint"] = checkpoint_path
    metrics["backbone"] = backbone
    metrics["manifest_csv"] = str(Path(manifest_csv))
    return metrics
