from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from psorad.attack.losses import UnTargeted
from psorad.attack.samoo_core.attack import Attack
from psorad.models.binary_classifier import build_resnet50_binary


class BinaryModelAdapter:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device, dtype=torch.float32)
        logits_binary = self.model(x)
        logits_2class = torch.stack([-logits_binary, logits_binary], dim=1)
        return logits_2class


def _load_checkpoint(backbone: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if backbone == "resnet50":
        model = build_resnet50_binary()
    elif backbone == "siglip":
        from psorad.models.binary_classifier import SiglipBinaryClassifier

        model = SiglipBinaryClassifier(pretrained_dir_or_id="model/pretrained_model/siglip", freeze_backbone=False)
    else:
        raise ValueError("backbone 仅支持 resnet50 或 siglip")

    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def _load_sample_from_manifest(manifest_csv: str, sample_index: int) -> tuple[np.ndarray, int]:
    manifest = pd.read_csv(manifest_csv)
    if sample_index < 0 or sample_index >= len(manifest):
        raise IndexError(f"sample_index 越界: {sample_index}, 数据总量: {len(manifest)}")

    row = manifest.iloc[sample_index]
    image_path = Path(str(row["file_path"]))
    label = int(row["label_binary"])

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        x = np.asarray(image, dtype=np.float32) / 255.0

    return x, label


def run_samoo_attack(
    backbone: str,
    checkpoint_path: str,
    manifest_csv: str = "dataset/binary_manifest.csv",
    sample_index: int = 0,
    save_path: str | None = None,
    seed: int = 42,
) -> Path:
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_checkpoint(backbone=backbone, checkpoint_path=checkpoint_path, device=device)
    adapter = BinaryModelAdapter(model=model, device=device)

    x, y_true = _load_sample_from_manifest(manifest_csv=manifest_csv, sample_index=sample_index)
    loss = UnTargeted(model=adapter, true=y_true, to_pytorch_input=True)

    if save_path is None:
        save_path = f"model/trained_classifier/{backbone}/samoo_result.npy"
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "x": x,
        "eps": 24,
        "iterations": 200,
        "pc": 0.1,
        "pm": 0.4,
        "pop_size": 4,
        "zero_probability": 0.3,
        "include_dist": True,
        "max_dist": 1e-5,
        "p_size": 2.0 / 255.0,
        "tournament_size": 2,
        "save_directory": str(save_file),
    }

    attacker = Attack(params)
    attacker.attack(loss)
    return save_file
