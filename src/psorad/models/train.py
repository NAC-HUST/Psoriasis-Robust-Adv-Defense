from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from tqdm import tqdm

from psorad.config import ExperimentConfig, ModelConfig
from psorad.data.dataset import build_loaders, build_split_loaders, split_manifest
from psorad.models.factory import build_model
from psorad.utils.seed import set_seed


@dataclass
class TrainConfig:
    backbone: str
    num_classes: int = 2
    manifest_csv: str = ""
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


def _export_split_csv(checkpoint_path: Path, train_manifest: pd.DataFrame, val_manifest: pd.DataFrame) -> Path:
    split_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_train-val-split.csv")

    train_part = train_manifest.copy()
    train_part["split"] = "train"
    val_part = val_manifest.copy()
    val_part["split"] = "val"

    split_manifest_df = pd.concat([train_part, val_part], axis=0, ignore_index=True)
    split_manifest_df.to_csv(split_path, index=False)
    return split_path


def train_classifier(config: TrainConfig) -> Path:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动检测 num_classes
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

    # Build model via factory to support multiple backbones
    if config.backbone == "resnet50":
        pretrained = config.pretrained_resnet_path
    elif config.backbone == "siglip":
        pretrained = config.pretrained_siglip_dir
    else:
        pretrained = None

    model_cfg = ModelConfig(backbone=config.backbone, pretrained_path=Path(pretrained) if pretrained else None, freeze_backbone=config.freeze_siglip_backbone, num_classes=config.num_classes)
    model = build_model(model_cfg, num_classes=config.num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)

    save_dir = Path(config.output_dir) / config.backbone
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / _resolve_model_name(config.model_name)

    train_manifest, val_manifest = split_manifest(
        manifest_csv=config.manifest_csv,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    split_csv_path = _export_split_csv(checkpoint_path=checkpoint_path, train_manifest=train_manifest, val_manifest=val_manifest)
    print(f"数据集划分已保存: {split_csv_path}")

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


def train_experiment(exp_cfg: ExperimentConfig) -> Path:
    """Run training using an ExperimentConfig loaded from TOML (split_data style).

    This function keeps the same high-level behavior as train_classifier but reads data
    from split_data and uses the model config from the experiment.
    """
    set_seed(exp_cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine num_classes
    if exp_cfg.model.num_classes is not None:
        num_classes = int(exp_cfg.model.num_classes)
    elif exp_cfg.dataset.class_names:
        num_classes = len(exp_cfg.dataset.class_names)
    else:
        # try to read metadata.json
        meta = exp_cfg.dataset.dataset_dir / "metadata.json"
        if meta.exists():
            import json

            with meta.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
            num_classes = int(m.get("global_stats", {}).get("num_classes", 2))
        else:
            num_classes = 2

    # build loaders from split_data
    loaders = build_split_loaders(exp_cfg.dataset, batch_size=exp_cfg.train.batch_size, num_workers=exp_cfg.train.num_workers)
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")
    if train_loader is None or val_loader is None:
        raise RuntimeError("train或val loader 未能构建，请检查 split_data 与配置")

    # build model
    model = build_model(exp_cfg.model, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=exp_cfg.train.learning_rate)

    save_dir = Path(exp_cfg.train.output_dir) / exp_cfg.model.backbone
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / Path(exp_cfg.train.model_name)

    best_val_acc = -1.0
    for epoch in range(1, exp_cfg.train.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"[{exp_cfg.model.backbone}] epoch {epoch}/{exp_cfg.train.epochs}", leave=False)

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
                    "backbone": exp_cfg.model.backbone,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "image_size": exp_cfg.dataset.image_size,
                    "normalization": exp_cfg.dataset.normalize,
                    "label_columns": list(exp_cfg.dataset.label_columns),
                    "class_names": list(exp_cfg.dataset.class_names),
                },
                checkpoint_path,
            )

        print(f"epoch={epoch} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    return checkpoint_path
