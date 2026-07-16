from __future__ import annotations

from pathlib import Path

from psorad.config import (
    DatasetConfig,
    ModelConfig,
    TrainSettings,
    load_dataset_config,
    load_experiment_from_configs,
    load_model_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DATASET_TOML = CONFIGS / "dataset_config" / "baseline_dataset.toml"
MODEL_TOML = CONFIGS / "model_config" / "baseline_model.toml"


def test_load_dataset_config() -> None:
    cfg = load_dataset_config(DATASET_TOML)
    assert isinstance(cfg, DatasetConfig)
    assert cfg.name
    assert isinstance(cfg.image_size, int) and cfg.image_size > 0
    assert cfg.label_mode in {"single_label", "multi_label"}


def test_load_model_config() -> None:
    cfg = load_model_config(MODEL_TOML)
    assert isinstance(cfg, ModelConfig)
    assert cfg.backbone
    assert isinstance(cfg.image_size, int) and cfg.image_size > 0
    # [Backbone.<backbone>] 专属参数应被保留到 extra
    assert "backbone_params" in cfg.extra


def test_load_experiment_from_configs() -> None:
    exp = load_experiment_from_configs(DATASET_TOML, MODEL_TOML)
    assert isinstance(exp.dataset, DatasetConfig)
    assert isinstance(exp.model, ModelConfig)
    assert isinstance(exp.train, TrainSettings)
    assert exp.train.epochs > 0
