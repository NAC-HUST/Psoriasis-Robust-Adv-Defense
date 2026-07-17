from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _stat_block(block: Any) -> dict[str, float]:
    # 提取统计量
    if not isinstance(block, dict):
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(block.get("mean", float("nan"))),
        "median": float(block.get("median", float("nan"))),
        "p90": float(block.get("p90", float("nan"))),
    }


def load_robust_from_report(batch_report: str | Path | None) -> dict[str, Any]:
    # 解析批量攻击报告
    if batch_report is None:
        return {"available": False, "reason": "no_batch_report_configured"}

    report_path = Path(batch_report)
    if not report_path.exists():
        return {"available": False, "reason": f"batch_report_not_found: {report_path}"}

    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    stats = data.get("stats", {})

    completed = int(summary.get("completed_count", 0))
    success = int(summary.get("success_count", 0))
    asr = float(summary.get("success_rate", 0.0))

    return {
        "available": True,
        "batch_report": str(report_path),
        "attack_meta": data.get("meta", {}),
        "hparams": data.get("hparams", {}),
        "attacked_count": completed,
        "success_count": success,
        "asr": asr,
        "robust_acc": float(1.0 - asr) if completed > 0 else float("nan"),
        "stats": {
            "queries": _stat_block(stats.get("queries")),
            "l2": _stat_block(stats.get("l2")),
            "linf": _stat_block(stats.get("linf")),
            "modified_pixels": _stat_block(stats.get("modified_pixels")),
        },
        "per_class": data.get("class_stats", {}),
    }
