from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from psorad.config import load_eval_config
from psorad.eval.metrics import accuracy, clean_metrics, precision_recall_f1, roc_auc
from psorad.eval.robust import load_robust_from_report

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EVAL_TOML = CONFIGS / "eval_config" / "baseline_eval.toml"


def test_accuracy_and_prf1() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert accuracy(y_true, y_pred) == 0.75

    prf1 = precision_recall_f1(y_true, y_pred, num_classes=2)
    # 类1: tp=2, pred_sum=3 -> precision=2/3, recall=2/2=1 -> f1=0.8
    per_class = prf1["per_class"]
    assert abs(per_class["precision"][1] - (2.0 / 3.0)) < 1e-9
    assert abs(per_class["recall"][1] - 1.0) < 1e-9
    assert abs(per_class["f1"][1] - 0.8) < 1e-9


def test_roc_auc_binary_perfect_and_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    perfect_scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    assert abs(roc_auc(y_true, perfect_scores, num_classes=2) - 1.0) < 1e-9

    # 0.5 AUC：正负样本分数完全交错且成对相等
    tied_scores = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    assert abs(roc_auc(y_true, tied_scores, num_classes=2) - 0.5) < 1e-9


def test_clean_metrics_shape() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6], [0.7, 0.3]])
    metrics = clean_metrics(y_true, y_score, num_classes=2)
    assert metrics["num_samples"] == 4
    assert metrics["accuracy"] == 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_load_robust_from_report(tmp_path: Path) -> None:
    report = {
        "meta": {"backbone": "resnet50"},
        "hparams": {"seed": 42},
        "summary": {"completed_count": 10, "success_count": 3, "success_rate": 0.3},
        "stats": {
            "queries": {"mean": 100.0, "median": 90.0, "p90": 200.0},
            "l2": {"mean": 1.0, "median": 1.0, "p90": 2.0},
            "linf": {"mean": 0.5, "median": 0.5, "p90": 0.9},
            "modified_pixels": {"mean": 10.0, "median": 9.0, "p90": 20.0},
        },
        "class_stats": {"0": {"count": 5, "success": 1, "success_rate": 0.2}},
    }
    report_path = tmp_path / "batch_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    robust = load_robust_from_report(report_path)
    assert robust["available"] is True
    assert robust["asr"] == 0.3
    assert abs(robust["robust_acc"] - 0.7) < 1e-9
    assert robust["stats"]["queries"]["mean"] == 100.0


def test_load_robust_missing_file() -> None:
    robust = load_robust_from_report("does/not/exist.json")
    assert robust["available"] is False


def test_load_eval_config() -> None:
    cfg = load_eval_config(EVAL_TOML)
    assert cfg.backbone
    assert cfg.checkpoint
    assert cfg.manifest_csv
    assert cfg.split in {"all", "train", "val"}
