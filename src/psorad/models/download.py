from __future__ import annotations

from pathlib import Path

import paddle
from paddle.vision import models


def download_resnet50(save_dir: str = "model/pretrained_model/resnet") -> Path:
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / "resnet50_imagenet1k_v1.pdparams"

    model = models.resnet50(pretrained=True)
    paddle.save(model.state_dict(), str(output_file))
    return output_file


def download_all_models() -> tuple[Path]:
    return (download_resnet50(),)
