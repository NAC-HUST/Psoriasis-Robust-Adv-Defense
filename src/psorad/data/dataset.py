from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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


def build_loaders(
    manifest_csv: str,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    image_size: int,
    for_siglip: bool,
    seed: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
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

    train_dataset = SkinDataset(manifest_csv=None, transform=build_transforms(image_size=image_size, for_siglip=for_siglip, train=True))
    train_dataset.set_manifest(train_manifest)
    val_dataset = SkinDataset(manifest_csv=None, transform=build_transforms(image_size=image_size, for_siglip=for_siglip, train=False))
    val_dataset.set_manifest(val_manifest)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
