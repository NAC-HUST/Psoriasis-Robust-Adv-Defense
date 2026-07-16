from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LabelMode = Literal["single_label", "multi_label"]


def _to_path(value: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return path

    return path if base_dir is None else path


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _require_mapping(section: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{section} 必须是 TOML table")
    return value


def _coerce_image_size(value: Any, *, default: int = 224) -> int:
    """将 image_size 归一化为方形边长 int。

    兼容用户 TOML 中的多种写法：int（224）、字符串（"224x224" / "640x400" / "224"）。
    非方形时取首个维度，原始值由上层保留在 extra 中。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().lower()
    if not text:
        return default
    for sep in ("x", "*", ",", "×"):
        if sep in text:
            first = text.split(sep)[0].strip()
            return int(first)
    return int(text)


@dataclass(slots=True)
class DatasetConfig:
    name: str
    split_data_root: Path = Path("dataset/split_data")
    splits: tuple[str, ...] = ("train", "val", "test")
    image_column: str = "image_path"
    label_columns: tuple[str, ...] = ()
    label_mode: LabelMode = "single_label"
    class_names: tuple[str, ...] = ()
    image_size: int = 224
    normalize: str = "imagenet"
    has_test_split: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dataset_dir(self) -> Path:
        return self.split_data_root / self.name

    def image_dir(self, split: str) -> Path:
        return self.dataset_dir / "images" / split

    def label_file(self, split: str) -> Path:
        return self.dataset_dir / "labels" / f"{split}_labels.csv"


@dataclass(slots=True)
class ModelConfig:
    backbone: str
    pretrained_path: Path | None = None
    freeze_backbone: bool = False
    num_classes: int | None = None
    normalization: str | None = None
    image_size: int = 224
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainSettings:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    output_dir: Path = Path("model/trained_classifier")
    model_name: str = "best_classifier.pt"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainSettings = field(default_factory=TrainSettings)
    source_path: Path | None = None


# 被 loader 直接消费的字段；其余字段统一挂到 extra 保留。
_DATASET_KNOWN_KEYS = {
    "dataset_name",
    "name",
    "split_data_root",
    "splits",
    "image_column",
    "label_columns",
    "label_mode",
    "class_names",
    "image_size",
    "normalize",
    "has_test_split",
}
_MODEL_KNOWN_KEYS = {
    "backbone",
    "pretrained_path",
    "freeze_backbone",
    "num_classes",
    "normalization",
    "image_size",
}


def _select_named_section(data: dict[str, Any], table_name: str) -> tuple[dict[str, Any], str | None]:
    """从 [<table_name>.<name>] 嵌套结构中取出一个子表。

    返回 (section, selected_name)。若不存在嵌套结构，返回 ({}, None)。
    """
    table = data.get(table_name)
    if not isinstance(table, dict) or not table:
        return {}, None
    # 仅当值本身也是 table 时才视为 [<table_name>.<name>] 嵌套
    nested = {key: value for key, value in table.items() if isinstance(value, dict)}
    if not nested:
        return {}, None
    return nested, None


def _build_dataset_config(section: dict[str, Any], *, base_dir: Path, fallback_name: str) -> DatasetConfig:
    dataset_name = str(section.get("dataset_name", section.get("name", fallback_name))).strip()
    if not dataset_name:
        raise ValueError("dataset config 缺少 dataset_name 或 name")

    split_data_root = _to_path(section.get("split_data_root", "dataset/split_data"), base_dir=base_dir)
    assert split_data_root is not None

    label_mode = str(section.get("label_mode", "single_label"))
    if label_mode not in {"single_label", "multi_label"}:
        raise ValueError("label_mode 仅支持 single_label 或 multi_label")

    extra = {key: value for key, value in section.items() if key not in _DATASET_KNOWN_KEYS}

    return DatasetConfig(
        name=dataset_name,
        split_data_root=split_data_root,
        splits=tuple(str(item) for item in _as_tuple(section.get("splits", ("train", "val", "test")))),
        image_column=str(section.get("image_column", "image_path")),
        label_columns=tuple(str(item) for item in _as_tuple(section.get("label_columns"))),
        label_mode=label_mode,  # type: ignore[arg-type]
        class_names=tuple(str(item) for item in _as_tuple(section.get("class_names"))),
        image_size=_coerce_image_size(section.get("image_size", 224)),
        normalize=str(section.get("normalize", "imagenet")),
        has_test_split=bool(section.get("has_test_split", True)),
        extra=extra,
    )


def load_dataset_config(path: str | Path, name: str | None = None) -> DatasetConfig:
    """加载数据集配置。

    同时兼容两种写法：
    - 扁平：顶层直接写 dataset_name / split_data_root / ...
    - 嵌套：[Dataset.<name>]（catalog 目录式，可含多个数据集）。
      未指定 name 时取第一个。
    """
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    nested, _ = _select_named_section(data, "Dataset")
    if not nested:
        # 兼容小写 [dataset.<name>]
        nested, _ = _select_named_section(data, "dataset")

    if nested:
        if name is not None:
            if name not in nested:
                raise KeyError(f"数据集配置中未找到 [Dataset.{name}]，可选: {sorted(nested)}")
            selected_name, section = name, nested[name]
        else:
            selected_name, section = next(iter(nested.items()))
        return _build_dataset_config(section, base_dir=config_path.parent, fallback_name=selected_name)

    # 扁平写法
    return _build_dataset_config(data, base_dir=config_path.parent, fallback_name="")


def _extract_model_section(data: dict[str, Any]) -> dict[str, Any]:
    """从模型 TOML 中定位承载模型定义的 section。

    优先使用 [Train] / [model] / [Model]；否则回退到顶层。
    """
    for key in ("Train", "train", "Model", "model"):
        candidate = data.get(key)
        if isinstance(candidate, dict) and candidate.get("backbone"):
            return candidate
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    """加载模型配置。

    兼容用户 catalog 写法：模型定义位于 [Train] 表（含 backbone/pretrained_path/... 及训练超参），
    backbone 专属项位于 [Backbone.<backbone>]。被消费的字段进入 ModelConfig，其余进入 extra。
    """
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    section = _extract_model_section(data)
    backbone = str(section.get("backbone", "")).strip()
    if not backbone:
        raise ValueError("model config 缺少 backbone")

    pretrained_path = _to_path(section.get("pretrained_path"), base_dir=config_path.parent)

    extra: dict[str, Any] = {key: value for key, value in section.items() if key not in _MODEL_KNOWN_KEYS}

    # 合并 [Backbone.<backbone>] 专属参数
    backbone_table = data.get("Backbone", data.get("backbone_params"))
    if isinstance(backbone_table, dict):
        specific = backbone_table.get(backbone)
        if isinstance(specific, dict):
            extra["backbone_params"] = specific

    # 保留其它顶层配置块（Optimizer / Augmentation / Output / Download 等）
    for block in ("Optimizer", "Augmentation", "Output", "Download"):
        if isinstance(data.get(block), dict):
            extra[block.lower()] = data[block]

    return ModelConfig(
        backbone=backbone,
        pretrained_path=pretrained_path,
        freeze_backbone=bool(section.get("freeze_backbone", False)),
        num_classes=int(section["num_classes"]) if section.get("num_classes") is not None else None,
        normalization=str(section.get("normalization")) if section.get("normalization") is not None else None,
        image_size=_coerce_image_size(section.get("image_size", 224)),
        extra=extra,
    )


def load_train_settings(path: str | Path) -> TrainSettings:
    """从模型 TOML 的 [Train] 表中读取训练超参数（缺省走默认值）。"""
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = data.get("Train", data.get("train", {}))
    if not isinstance(section, dict):
        section = {}

    known = {"epochs", "batch_size", "learning_rate", "num_workers", "seed", "output_dir", "model_name"}
    extra = {key: value for key, value in section.items() if key not in known and key not in _MODEL_KNOWN_KEYS}

    return TrainSettings(
        epochs=int(section.get("epochs", 3)),
        batch_size=int(section.get("batch_size", 16)),
        learning_rate=float(section.get("learning_rate", 1e-4)),
        num_workers=int(section.get("num_workers", 2)),
        seed=int(section.get("seed", 42)),
        output_dir=_to_path(section.get("output_dir", "model/trained_classifier"), base_dir=config_path.parent) or Path("model/trained_classifier"),
        model_name=str(section.get("model_name", "best_classifier.pt")),
        extra=extra,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """加载单文件实验配置（包含 [dataset]/[model]/[train] 三个表）。

    面向"一个实验一个 TOML"的写法；分离的 dataset/model 目录式配置请用
    load_experiment_from_configs。
    """
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    dataset_section = _require_mapping("dataset", data.get("dataset"))
    model_section = _require_mapping("model", data.get("model"))
    train_section = data.get("train", {})
    if not isinstance(train_section, dict):
        raise TypeError("train 必须是 TOML table")

    dataset = _build_dataset_config(dataset_section, base_dir=config_path.parent, fallback_name="")

    pretrained_path = _to_path(model_section.get("pretrained_path"), base_dir=config_path.parent)
    backbone = str(model_section.get("backbone", "")).strip()
    if not backbone:
        raise ValueError("[model] 缺少 backbone")

    model_extra = {key: value for key, value in model_section.items() if key not in _MODEL_KNOWN_KEYS}
    model = ModelConfig(
        backbone=backbone,
        pretrained_path=pretrained_path,
        freeze_backbone=bool(model_section.get("freeze_backbone", False)),
        num_classes=int(model_section["num_classes"]) if model_section.get("num_classes") is not None else None,
        normalization=str(model_section.get("normalization")) if model_section.get("normalization") is not None else None,
        image_size=_coerce_image_size(model_section.get("image_size", 224)),
        extra=model_extra,
    )

    train_known = {"epochs", "batch_size", "learning_rate", "num_workers", "seed", "output_dir", "model_name"}
    train_extra = {key: value for key, value in train_section.items() if key not in train_known}
    train = TrainSettings(
        epochs=int(train_section.get("epochs", 3)),
        batch_size=int(train_section.get("batch_size", 16)),
        learning_rate=float(train_section.get("learning_rate", 1e-4)),
        num_workers=int(train_section.get("num_workers", 2)),
        seed=int(train_section.get("seed", 42)),
        output_dir=_to_path(train_section.get("output_dir", "model/trained_classifier"), base_dir=config_path.parent) or Path("model/trained_classifier"),
        model_name=str(train_section.get("model_name", "best_classifier.pt")),
        extra=train_extra,
    )

    return ExperimentConfig(dataset=dataset, model=model, train=train, source_path=config_path)


def load_experiment_from_configs(
    dataset_path: str | Path,
    model_path: str | Path,
    *,
    dataset_name: str | None = None,
) -> ExperimentConfig:
    """由分离的数据集 TOML + 模型 TOML 组合出实验配置（catalog 目录式）。"""
    dataset = load_dataset_config(dataset_path, name=dataset_name)
    model = load_model_config(model_path)
    train = load_train_settings(model_path)
    return ExperimentConfig(dataset=dataset, model=model, train=train, source_path=Path(model_path))
