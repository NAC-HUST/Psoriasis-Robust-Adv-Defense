from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class BinarySkinDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, manifest_csv: str, transform: transforms.Compose):
        self.manifest = pd.read_csv(manifest_csv)
        if "file_path" not in self.manifest.columns or "label_binary" not in self.manifest.columns:
            raise ValueError("manifest 必须包含 file_path 和 label_binary 列")
        self.transform = transform

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

        label = torch.tensor(float(row["label_binary"]), dtype=torch.float32)
        return image_tensor, label


def build_transforms(image_size: int, for_siglip: bool, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if for_siglip else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augments: list[object] = [transforms.Resize((image_size, image_size))]
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
    base_dataset = BinarySkinDataset(manifest_csv=manifest_csv, transform=build_transforms(image_size=image_size, for_siglip=for_siglip, train=False))

    total = len(base_dataset)
    val_len = max(int(total * val_ratio), 1)
    train_len = total - val_len
    if train_len <= 0:
        raise ValueError("训练集为空，请增大数据量或减小 val_ratio")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_dataset, [train_len, val_len], generator=generator)

    train_subset.dataset.transform = build_transforms(image_size=image_size, for_siglip=for_siglip, train=True)  # type: ignore[attr-defined]
    val_subset.dataset.transform = build_transforms(image_size=image_size, for_siglip=for_siglip, train=False)  # type: ignore[attr-defined]

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
