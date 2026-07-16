from __future__ import annotations

from pathlib import Path
from typing import Any

from psorad.config import EvalConfig
from psorad.eval.clean import evaluate_clean
from psorad.eval.report import build_eval_report, write_eval_report
from psorad.eval.robust import load_robust_from_report
from psorad.eval.vulnerability import analyze_vulnerability, save_vulnerability_visuals


def run_evaluate(cfg: EvalConfig, *, skip_clean: bool = False) -> Path:
    """执行评估：clean 指标 + robust 指标 + 脆弱性分析，写出报告。"""
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

    vulnerability: dict[str, Any] | None = None
    if cfg.batch_report is not None:
        vuln_kwargs: dict[str, Any] = {}
        if hasattr(cfg, "grid_size") and cfg.grid_size is not None:
            vuln_kwargs["grid_size"] = cfg.grid_size
        if hasattr(cfg, "high_freq_cutoff") and cfg.high_freq_cutoff is not None:
            vuln_kwargs["high_freq_cutoff"] = cfg.high_freq_cutoff
        vulnerability = analyze_vulnerability(cfg.batch_report, **vuln_kwargs)
        if vulnerability.get("available") and cfg.save_visuals:
            save_vulnerability_visuals(vulnerability, cfg.output_dir)

    meta = {
        "backbone": cfg.backbone,
        "checkpoint": cfg.checkpoint,
        "manifest_csv": cfg.manifest_csv,
        "split": cfg.split,
        "metrics": list(cfg.metrics),
    }
    report = build_eval_report(clean=clean, robust=robust, meta=meta, vulnerability=vulnerability)
    return write_eval_report(report, output_dir=cfg.output_dir, report_name=cfg.report_name)
