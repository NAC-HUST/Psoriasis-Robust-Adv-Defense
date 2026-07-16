from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import tomllib


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


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    output_dir: Path = Path("model/trained_classifier")
    model_name: str = "best_classifier.pt"


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    source_path: Path | None = None


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    dataset_name = str(data.get("dataset_name", data.get("name", ""))).strip()
    if not dataset_name:
        raise ValueError("dataset config 缺少 dataset_name 或 name")

    split_data_root = _to_path(data.get("split_data_root", "dataset/split_data"), base_dir=config_path.parent)
    assert split_data_root is not None

    label_columns = tuple(str(item) for item in _as_tuple(data.get("label_columns")))
    class_names = tuple(str(item) for item in _as_tuple(data.get("class_names")))
    splits = tuple(str(item) for item in _as_tuple(data.get("splits", ("train", "val", "test"))))

    label_mode = str(data.get("label_mode", "single_label"))
    if label_mode not in {"single_label", "multi_label"}:
        raise ValueError("label_mode 仅支持 single_label 或 multi_label")

    return DatasetConfig(
        name=dataset_name,
        split_data_root=split_data_root,
        splits=splits,
        image_column=str(data.get("image_column", "image_path")),
        label_columns=label_columns,
        label_mode=label_mode,  # type: ignore[arg-type]
        class_names=class_names,
        image_size=int(data.get("image_size", 224)),
        normalize=str(data.get("normalize", "imagenet")),
        has_test_split=bool(data.get("has_test_split", True)),
    )


def load_model_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    backbone = str(data.get("backbone", "")).strip()
    if not backbone:
        raise ValueError("model config 缺少 backbone")

    pretrained_path = _to_path(data.get("pretrained_path"), base_dir=config_path.parent)
    return ModelConfig(
        backbone=backbone,
        pretrained_path=pretrained_path,
        freeze_backbone=bool(data.get("freeze_backbone", False)),
        num_classes=int(data["num_classes"]) if data.get("num_classes") is not None else None,
        normalization=str(data.get("normalization")) if data.get("normalization") is not None else None,
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    dataset_section = _require_mapping("dataset", data.get("dataset"))
    model_section = _require_mapping("model", data.get("model"))
    train_section = data.get("train", {})
    if not isinstance(train_section, dict):
        raise TypeError("train 必须是 TOML table")

    dataset_name = str(dataset_section.get("dataset_name", dataset_section.get("name", ""))).strip()
    if not dataset_name:
        raise ValueError("[dataset] 缺少 dataset_name 或 name")

    split_data_root = _to_path(dataset_section.get("split_data_root", "dataset/split_data"), base_dir=config_path.parent)
    assert split_data_root is not None

    label_mode = str(dataset_section.get("label_mode", "single_label"))
    if label_mode not in {"single_label", "multi_label"}:
        raise ValueError("label_mode 仅支持 single_label 或 multi_label")

    dataset = DatasetConfig(
        name=dataset_name,
        split_data_root=split_data_root,
        splits=tuple(str(item) for item in _as_tuple(dataset_section.get("splits", ("train", "val", "test")))),
        image_column=str(dataset_section.get("image_column", "image_path")),
        label_columns=tuple(str(item) for item in _as_tuple(dataset_section.get("label_columns"))),
        label_mode=label_mode,  # type: ignore[arg-type]
        class_names=tuple(str(item) for item in _as_tuple(dataset_section.get("class_names"))),
        image_size=int(dataset_section.get("image_size", 224)),
        normalize=str(dataset_section.get("normalize", "imagenet")),
        has_test_split=bool(dataset_section.get("has_test_split", True)),
    )

    pretrained_path = _to_path(model_section.get("pretrained_path"), base_dir=config_path.parent)
    backbone = str(model_section.get("backbone", "")).strip()
    if not backbone:
        raise ValueError("[model] 缺少 backbone")

    model = ModelConfig(
        backbone=backbone,
        pretrained_path=pretrained_path,
        freeze_backbone=bool(model_section.get("freeze_backbone", False)),
        num_classes=int(model_section["num_classes"]) if model_section.get("num_classes") is not None else None,
        normalization=str(model_section.get("normalization")) if model_section.get("normalization") is not None else None,
    )

    train = TrainConfig(
        epochs=int(train_section.get("epochs", 3)),
        batch_size=int(train_section.get("batch_size", 16)),
        learning_rate=float(train_section.get("learning_rate", 1e-4)),
        num_workers=int(train_section.get("num_workers", 2)),
        seed=int(train_section.get("seed", 42)),
        output_dir=_to_path(train_section.get("output_dir", "model/trained_classifier"), base_dir=config_path.parent) or Path("model/trained_classifier"),
        model_name=str(train_section.get("model_name", "best_classifier.pt")),
    )

    return ExperimentConfig(dataset=dataset, model=model, train=train, source_path=config_path)
