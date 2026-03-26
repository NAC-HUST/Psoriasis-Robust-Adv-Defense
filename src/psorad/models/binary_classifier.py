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

        config = self.vision_model.config
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            feature_dim = int(config.vision_config.hidden_size)
        elif hasattr(config, "hidden_size"):
            feature_dim = int(config.hidden_size)
        elif hasattr(config, "projection_dim"):
            feature_dim = int(config.projection_dim)
        else:
            raise ValueError("无法从 SigLIP 配置中推断图像特征维度。")

        self.classifier = nn.Linear(feature_dim, 1)

        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: Tensor) -> Tensor:
        if not hasattr(self.vision_model, "get_image_features"):
            raise RuntimeError("当前 SigLIP 模型不支持 get_image_features。")

        image_features = self.vision_model.get_image_features(pixel_values=pixel_values)
        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            elif hasattr(image_features, "last_hidden_state"):
                image_features = image_features.last_hidden_state.mean(dim=1)
            else:
                raise RuntimeError("无法从 SigLIP 输出中提取图像特征。")

        return self.classifier(image_features).squeeze(1)  # type: ignore[no-any-return]


def build_resnet50_binary(pretrained_weight_path: str | None = None) -> nn.Module:
    model = models.resnet50(weights=None)  # type: ignore[assignment]

    if pretrained_weight_path is not None and Path(pretrained_weight_path).exists():
        state_dict = torch.load(pretrained_weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    return model  # type: ignore[no-any-return]
