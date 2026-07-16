from __future__ import annotations

from typing import Callable, Dict, Optional
from torch import nn

from psorad.models.classifier import build_resnet50_classifier, SiglipClassifier
from psorad.config import ModelConfig


Registry = Dict[str, Callable[[ModelConfig, int], nn.Module]]
_REGISTRY: Registry = {}


def register_backbone(name: str) -> Callable[[Callable[[ModelConfig, int], nn.Module]], Callable[[ModelConfig, int], nn.Module]]:
    def _decorator(fn: Callable[[ModelConfig, int], nn.Module]) -> Callable[[ModelConfig, int], nn.Module]:
        _REGISTRY[name] = fn
        return fn

    return _decorator


@register_backbone("resnet50")
def _build_resnet50(cfg: ModelConfig, num_classes: int) -> nn.Module:
    pretrained = str(cfg.pretrained_path) if cfg.pretrained_path is not None else None
    return build_resnet50_classifier(pretrained_weight_path=pretrained, num_classes=num_classes)


@register_backbone("siglip")
def _build_siglip(cfg: ModelConfig, num_classes: int) -> nn.Module:
    pretrained = str(cfg.pretrained_path) if cfg.pretrained_path is not None else "model/pretrained_model/siglip"
    return SiglipClassifier(pretrained_dir_or_id=pretrained, num_classes=num_classes, freeze_backbone=cfg.freeze_backbone)


def build_model(cfg: ModelConfig, num_classes: int) -> nn.Module:
    name = cfg.backbone.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(f"未知 backbone: {cfg.backbone}")
    return _REGISTRY[name](cfg, num_classes)


def available_backbones() -> list[str]:
    return sorted(list(_REGISTRY.keys()))
