from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from tqdm import tqdm

from psorad.data.dataset import build_loaders
from psorad.models.classifier import build_resnet50_classifier
from psorad.utils.seed import set_seed


@dataclass
class TrainConfig:
    backbone: str
    num_classes: int = 2
    manifest_csv: str = "dataset/processed_data/psoriasis_normal/class_manifest.csv"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 2
    image_size: int = 224
    pretrained_resnet_path: str = "model/pretrained_model/resnet/resnet50_imagenet1k_v2.pth"
    pretrained_siglip_dir: str = "model/pretrained_model/siglip"
    output_dir: str = "model/trained_classifier"
    model_name: str = "best_classifier.pt"
    freeze_siglip_backbone: bool = True


def _resolve_model_name(model_name: str) -> str:
    filename = Path(model_name).name.strip()
    if not filename:
        raise ValueError("model_name 不能为空")
    if Path(filename).suffix == "":
        filename = f"{filename}.pt"
    return filename


def _multiclass_accuracy(logits: Tensor, targets: Tensor) -> float:
    """多分类准确率计算"""
    preds = torch.argmax(logits, dim=-1)
    targets = targets.reshape(-1)
    correct = (preds == targets).float().mean()
    return float(correct.item())


def _evaluate(model: nn.Module, loader: torch.utils.data.DataLoader[tuple[Tensor, Tensor]], criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    accs: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            accs.append(_multiclass_accuracy(logits, labels))

    return sum(losses) / max(len(losses), 1), sum(accs) / max(len(accs), 1)


def train_classifier(config: TrainConfig) -> Path:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动检测 num_classes
    import pandas as pd
    manifest = pd.read_csv(config.manifest_csv)
    if "class_idx" in manifest.columns:
        detected_num_classes = int(manifest["class_idx"].max()) + 1
        config.num_classes = detected_num_classes

    is_siglip = config.backbone == "siglip"
    train_loader, val_loader = build_loaders(
        manifest_csv=config.manifest_csv,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        num_workers=config.num_workers,
        image_size=config.image_size,
        for_siglip=is_siglip,
        seed=config.seed,
    )

    if config.backbone == "resnet50":
        model = build_resnet50_classifier(config.pretrained_resnet_path, num_classes=config.num_classes)
    elif config.backbone == "siglip":
        from psorad.models.classifier import SiglipClassifier

        model = SiglipClassifier(
            pretrained_dir_or_id=config.pretrained_siglip_dir,
            num_classes=config.num_classes,
            freeze_backbone=config.freeze_siglip_backbone,
        )
    else:
        raise ValueError("backbone 仅支持 resnet50 或 siglip")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)

    save_dir = Path(config.output_dir) / config.backbone
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / _resolve_model_name(config.model_name)

    best_val_acc = -1.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"[{config.backbone}] epoch {epoch}/{config.epochs}", leave=False)

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "backbone": config.backbone,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "image_size": config.image_size,
                },
                checkpoint_path,
            )

        print(f"epoch={epoch} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    return checkpoint_path
