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


@dataclass(slots=True)
class EvalConfig:
    backbone: str
    checkpoint: str
    manifest_csv: str
    split: str = "val"
    val_ratio: float = 0.2
    split_seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    metrics: tuple[str, ...] = ("acc", "f1", "auc", "asr")
    batch_report: str | None = None
    grid_size: int = 7
    high_freq_cutoff: float = 0.25
    save_visuals: bool = False
    output_dir: Path = Path("output/eval")
    report_name: str = "eval_report.json"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DefenseConfig:
    backbone: str
    checkpoint: str
    batch_report: str
    window: int = 3
    threshold: float = 0.08
    dilation: int = 1
    low_freq_cutoff: float = 0.25
    low_freq: str = "fft"
    output_dir: Path = Path("output/defense")
    report_name: str = "defense_report.json"
    extra: dict[str, Any] = field(default_factory=dict)


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

    # 路径重复修复：若 split_data_root 最末段已等于 dataset_name，
    # 取父目录避免 dataset_dir = split_data_root / name 时翻倍
    if split_data_root.name == dataset_name:
        split_data_root = split_data_root.parent

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
    # 优先从 class_names 推断 num_classes，覆盖 model config 的固定默认值
    if dataset.class_names and model.num_classes != len(dataset.class_names):
        model.num_classes = len(dataset.class_names)
    return ExperimentConfig(dataset=dataset, model=model, train=train, source_path=Path(model_path))


_EVAL_KNOWN_KEYS = {
    "backbone",
    "checkpoint",
    "manifest_csv",
    "datadir",
    "dataset_root",
    "split",
    "val_ratio",
    "split_seed",
    "image_size",
    "batch_size",
    "num_workers",
    "metrics",
    "grid_size",
    "high_freq_cutoff",
    "save_visuals",
}


def load_eval_config(path: str | Path) -> EvalConfig:
    """加载评估配置（configs/eval_config/*.toml）。

    读取 [Eval]（模型/数据/指标）、[Eval.robust]（batch_report 入口）、[Output]（报告输出）。
    未消费的键保留在 extra，沿用配置驱动的宽容策略。
    """
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    section = data.get("Eval", data.get("eval"))
    if not isinstance(section, dict):
        raise ValueError("eval config 缺少 [Eval] 表")

    backbone = str(section.get("backbone", "")).strip()
    if not backbone:
        raise ValueError("[Eval] 缺少 backbone")
    checkpoint = str(section.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("[Eval] 缺少 checkpoint")
    manifest_csv = str(section.get("manifest_csv", "")).strip()
    if not manifest_csv:
        raise ValueError("[Eval] 缺少 manifest_csv")

    robust_section = section.get("robust", {})
    if not isinstance(robust_section, dict):
        robust_section = {}
    batch_report = robust_section.get("batch_report")

    vuln_section = section.get("vulnerability", {})
    if not isinstance(vuln_section, dict):
        vuln_section = {}

    output_section = data.get("Output", data.get("output", {}))
    if not isinstance(output_section, dict):
        output_section = {}

    extra: dict[str, Any] = {key: value for key, value in section.items() if key not in _EVAL_KNOWN_KEYS and key not in ("robust", "vulnerability")}
    if robust_section:
        extra["robust"] = robust_section
    if vuln_section:
        extra["vulnerability"] = vuln_section
    if output_section:
        extra["output"] = output_section

    return EvalConfig(
        backbone=backbone,
        checkpoint=checkpoint,
        manifest_csv=manifest_csv,
        split=str(section.get("split", "val")),
        val_ratio=float(section.get("val_ratio", 0.2)),
        split_seed=int(section.get("split_seed", 42)),
        image_size=_coerce_image_size(section.get("image_size", 224)),
        batch_size=int(section.get("batch_size", 32)),
        num_workers=int(section.get("num_workers", 2)),
        metrics=tuple(str(item) for item in _as_tuple(section.get("metrics", ("acc", "f1", "auc", "asr")))),
        batch_report=str(batch_report) if batch_report is not None else None,
        grid_size=int(vuln_section.get("grid_size", 7)) if vuln_section.get("grid_size") is not None else 7,
        high_freq_cutoff=float(vuln_section.get("high_freq_cutoff", 0.25)),
        save_visuals=bool(vuln_section.get("save_visuals", False)),
        output_dir=_to_path(output_section.get("output_dir", "output/eval"), base_dir=config_path.parent) or Path("output/eval"),
        report_name=str(output_section.get("report_name", "eval_report.json")),
        extra=extra,
    )


_DEFENSE_KNOWN_KEYS = {
    "method",
    "backbone",
    "checkpoint",
    "batch_report",
    "window",
    "threshold",
    "dilation",
    "low_freq_cutoff",
    "low_freq",
}


def load_defense_config(path: str | Path) -> DefenseConfig:
    """加载防御配置（configs/defence_config/*.toml）。

    读取 [Defense]（backbone/checkpoint/batch_report）、[Defense.purify]（净化超参）、[Output]。
    """
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    section = data.get("Defense", data.get("defense"))
    if not isinstance(section, dict):
        raise ValueError("defense config 缺少 [Defense] 表")

    backbone = str(section.get("backbone", "")).strip()
    if not backbone:
        raise ValueError("[Defense] 缺少 backbone")
    checkpoint = str(section.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("[Defense] 缺少 checkpoint")
    batch_report = str(section.get("batch_report", "")).strip()
    if not batch_report:
        raise ValueError("[Defense] 缺少 batch_report")

    purify_section = section.get("purify", {})
    if not isinstance(purify_section, dict):
        purify_section = {}

    output_section = data.get("Output", data.get("output", {}))
    if not isinstance(output_section, dict):
        output_section = {}

    extra: dict[str, Any] = {key: value for key, value in section.items() if key not in _DEFENSE_KNOWN_KEYS and key != "purify"}
    if purify_section:
        extra["purify"] = purify_section
    if output_section:
        extra["output"] = output_section

    return DefenseConfig(
        backbone=backbone,
        checkpoint=checkpoint,
        batch_report=batch_report,
        window=int(purify_section.get("window", 3)) if purify_section.get("window") is not None else 3,
        threshold=float(purify_section.get("threshold", 0.08)),
        dilation=int(purify_section.get("dilation", 1)) if purify_section.get("dilation") is not None else 1,
        low_freq_cutoff=float(purify_section.get("low_freq_cutoff", 0.25)),
        low_freq=str(purify_section.get("low_freq", "fft")),
        output_dir=_to_path(output_section.get("output_dir", "output/defense"), base_dir=config_path.parent) or Path("output/defense"),
        report_name=str(output_section.get("report_name", "defense_report.json")),
        extra=extra,
    )
