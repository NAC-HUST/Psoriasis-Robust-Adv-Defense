from __future__ import annotations

import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torchvision import models


def download_resnet50(save_dir: str = "model/pretrained_model/resnet") -> Path:
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / "resnet50_imagenet1k_v2.pth"

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    torch.save(model.state_dict(), output_file)
    return output_file


def download_siglip(
    save_dir: str = "model/pretrained_model/siglip",
    repo_id: str = "google/siglip-base-patch16-224",
    use_hf_mirror: bool = True,
) -> Path:
    if use_hf_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(repo_id=repo_id, local_dir=str(target_dir), local_dir_use_symlinks=False)  # type: ignore[call-overload]
    return Path(snapshot_path)


def download_all_models() -> tuple[Path, Path]:
    resnet_path = download_resnet50()
    siglip_path = download_siglip()
    return resnet_path, siglip_path
