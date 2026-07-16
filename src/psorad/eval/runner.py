from __future__ import annotations

from pathlib import Path
from typing import Any

from psorad.config import EvalConfig
from psorad.eval.clean import evaluate_clean
from psorad.eval.report import build_eval_report, write_eval_report
from psorad.eval.robust import load_robust_from_report


def run_evaluate(cfg: EvalConfig, *, skip_clean: bool = False) -> Path:
    """执行评估：clean 指标 + robust 指标（读 batch_report），写出报告。

    skip_clean=True 时仅汇总 robust 指标（无权重/数据时的降级路径）。
    """
    clean: dict[str, Any] | None = None
    if not skip_clean:
        clean = evaluate_clean(
            backbone=cfg.backbone,
            checkpoint_path=cfg.checkpoint,
            manifest_csv=cfg.manifest_csv,
            split=cfg.split,
            val_ratio=cfg.val_ratio,
            split_seed=cfg.split_seed,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    robust = load_robust_from_report(cfg.batch_report)

    meta = {
        "backbone": cfg.backbone,
        "checkpoint": cfg.checkpoint,
        "manifest_csv": cfg.manifest_csv,
        "split": cfg.split,
        "metrics": list(cfg.metrics),
    }
    report = build_eval_report(clean=clean, robust=robust, meta=meta)
    return write_eval_report(report, output_dir=cfg.output_dir, report_name=cfg.report_name)
