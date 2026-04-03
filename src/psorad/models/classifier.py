from __future__ import annotations

from pathlib import Path

import paddle
from paddle import nn
from paddle.vision import models


class ResNet50Classifier(nn.Layer):
    def __init__(self, num_classes: int = 2, pretrained_weight_path: str | None = None):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)

        if pretrained_weight_path is not None and Path(pretrained_weight_path).exists():
            state_dict = paddle.load(pretrained_weight_path)
            self.backbone.set_state_dict(state_dict)

        in_features = int(self.backbone.fc.weight.shape[0])
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, images: paddle.Tensor) -> paddle.Tensor:
        return self.backbone(images)


def build_resnet50_classifier(pretrained_weight_path: str | None = None, num_classes: int = 2) -> nn.Layer:
    return ResNet50Classifier(num_classes=num_classes, pretrained_weight_path=pretrained_weight_path)
