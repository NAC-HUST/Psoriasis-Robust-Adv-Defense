from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from psorad.config import DatasetConfig


class SkinDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, manifest_csv: str | None, transform: transforms.Compose):
        if manifest_csv is not None:
            self.manifest = pd.read_csv(manifest_csv)
            if "file_path" not in self.manifest.columns or "class_idx" not in self.manifest.columns:
                raise ValueError("manifest 必须包含 file_path 和 class_idx 列")
        else:
            self.manifest = None
        self.transform = transform

    def set_manifest(self, manifest: pd.DataFrame) -> None:
        """支持外部设置 manifest（用于分层抽样）"""
        if "file_path" not in manifest.columns or "class_idx" not in manifest.columns:
            raise ValueError("manifest 必须包含 file_path 和 class_idx 列")
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.manifest.iloc[index]
        image_path = Path(str(row["file_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image_tensor = self.transform(image)

        label = torch.tensor(int(row["class_idx"]), dtype=torch.int64)
        return image_tensor, label


class SplitDataDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        label_csv: str | Path,
        *,
        image_column: str = "image_path",
        label_columns: Sequence[str] | None = None,
        label_mode: str = "single_label",
        transform: transforms.Compose,
    ) -> None:
        self.label_csv = Path(label_csv)
        self.dataset_root = self.label_csv.parent.parent
        self.manifest = pd.read_csv(self.label_csv)
        self.image_column = image_column
        self.label_columns = tuple(label_columns or ())
        self.label_mode = label_mode
        self.transform = transform

        if self.image_column not in self.manifest.columns:
            raise ValueError(f"label CSV 必须包含 {self.image_column} 列")

        if self.label_columns:
            missing = [column for column in self.label_columns if column not in self.manifest.columns]
            if missing:
                raise ValueError(f"label CSV 缺少标签列: {missing}")
        elif "class_idx" not in self.manifest.columns:
            inferred_columns = [column for column in self.manifest.columns if column.startswith("label_")]
            if not inferred_columns:
                raise ValueError("label CSV 需要 class_idx 或 label_* 列")
            self.label_columns = tuple(inferred_columns)

    def __len__(self) -> int:
        return len(self.manifest)

    def _resolve_image_path(self, value: str) -> Path:
        image_path = Path(value)
        if image_path.is_absolute():
            return image_path
        return self.dataset_root / image_path

    def _encode_label(self, row: pd.Series) -> Tensor:
        if "class_idx" in self.manifest.columns and not self.label_columns:
            return torch.tensor(int(row["class_idx"]), dtype=torch.int64)

        values = np.asarray([float(row[column]) for column in self.label_columns], dtype=np.float32)
        if self.label_mode == "multi_label":
            return torch.tensor(values, dtype=torch.float32)

        if values.size == 1:
            return torch.tensor(int(values.item()), dtype=torch.int64)
        return torch.tensor(int(values.argmax()), dtype=torch.int64)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.manifest.iloc[index]
        image_path = self._resolve_image_path(str(row[self.image_column]))
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image_tensor = self.transform(image)
        label = self._encode_label(row)
        return image_tensor, label


def _resolve_normalize(normalize: str) -> transforms.Normalize:
    normalize_key = normalize.strip().lower()
    if normalize_key == "siglip":
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if normalize_key in {"imagenet", "resnet", "resnet50"}:
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    raise ValueError(f"不支持的 normalize 策略: {normalize}")


def build_transforms(image_size: int, for_siglip: bool, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if for_siglip else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augments: list[object] = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
    ]
    if train:
        augments.append(transforms.RandomHorizontalFlip(p=0.5))
    augments.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(augments)


def build_split_transforms(image_size: int, normalize: str, train: bool) -> transforms.Compose:
    augments: list[object] = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
    ]
    if train:
        augments.append(transforms.RandomHorizontalFlip(p=0.5))
    augments.extend([transforms.ToTensor(), _resolve_normalize(normalize)])
    return transforms.Compose(augments)


def build_split_loaders(
    config: DatasetConfig,
    batch_size: int,
    num_workers: int,
    splits: Sequence[str] | None = None,
) -> dict[str, DataLoader[tuple[Tensor, Tensor]]]:
    requested_splits = tuple(splits or config.splits)
    loaders: dict[str, DataLoader[tuple[Tensor, Tensor]]] = {}

    for split in requested_splits:
        if split == "test" and not config.has_test_split:
            continue

        label_csv = config.label_file(split)
        if not label_csv.exists():
            if split == "test" and not config.has_test_split:
                continue
            raise FileNotFoundError(f"未找到 split={split} 的标签文件: {label_csv}")

        dataset = SplitDataDataset(
            label_csv,
            image_column=config.image_column,
            label_columns=config.label_columns or None,
            label_mode=config.label_mode,
            transform=build_split_transforms(config.image_size, config.normalize, train=split == "train"),
        )
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers, pin_memory=True)

    return loaders


def build_loaders(
    manifest_csv: str,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    image_size: int,
    for_siglip: bool,
    seed: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    train_manifest, val_manifest = split_manifest(manifest_csv=manifest_csv, val_ratio=val_ratio, seed=seed)

    train_dataset = SkinDataset(manifest_csv=None, transform=build_transforms(image_size=image_size, for_siglip=for_siglip, train=True))
    train_dataset.set_manifest(train_manifest)
    val_dataset = SkinDataset(manifest_csv=None, transform=build_transforms(image_size=image_size, for_siglip=for_siglip, train=False))
    val_dataset.set_manifest(val_manifest)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def split_manifest(manifest_csv: str, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_csv)
    total = len(manifest)
    val_len = max(int(total * val_ratio), 1)
    train_len = total - val_len
    if train_len <= 0:
        raise ValueError("训练集为空，请增大数据量或减小 val_ratio")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    train_manifest = manifest.iloc[train_indices].reset_index(drop=True)
    val_manifest = manifest.iloc[val_indices].reset_index(drop=True)
    return train_manifest, val_manifest
