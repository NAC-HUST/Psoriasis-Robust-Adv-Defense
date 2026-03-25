from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision import models


class SiglipBinaryClassifier(nn.Module):
    def __init__(self, pretrained_dir_or_id: str, freeze_backbone: bool = True):
        super().__init__()
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("SigLIP 训练需要安装 transformers。请先安装项目依赖后再运行。") from exc

        self.vision_model = AutoModel.from_pretrained(pretrained_dir_or_id)
        hidden_size = int(self.vision_model.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: Tensor) -> Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return self.classifier(pooled).squeeze(1)


def build_resnet50_binary(pretrained_weight_path: str | None = None) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    if pretrained_weight_path is not None and Path(pretrained_weight_path).exists():
        state_dict = torch.load(pretrained_weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model
