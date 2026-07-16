from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_eval_report(clean: dict[str, Any] | None, robust: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """合并 clean 与 robust 指标为统一评估报告。"""
    return {
        "meta": {"created_at": datetime.now().isoformat(timespec="seconds"), **meta},
        "clean": clean if clean is not None else {"available": False, "reason": "clean_eval_skipped"},
        "robust": robust,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.4f}"
    return str(value)


def _render_markdown(report: dict[str, Any]) -> str:
    meta = report.get("meta", {})
    clean = report.get("clean", {})
    robust = report.get("robust", {})

    lines: list[str] = ["# 评估报告", ""]
    lines.append(f"- 生成时间: {meta.get('created_at', '')}")
    lines.append(f"- backbone: {meta.get('backbone', '')}")
    lines.append(f"- checkpoint: {meta.get('checkpoint', '')}")
    lines.append(f"- split: {meta.get('split', '')}")
    lines.append("")

    lines.append("## Clean 指标（干净样本）")
    if clean.get("available") is False:
        lines.append(f"- 不可用: {clean.get('reason', '')}")
    else:
        lines.append(f"- 样本数: {clean.get('num_samples', '')}")
        lines.append(f"- accuracy: {_fmt(clean.get('accuracy'))}")
        lines.append(f"- macro_precision: {_fmt(clean.get('macro_precision'))}")
        lines.append(f"- macro_recall: {_fmt(clean.get('macro_recall'))}")
        lines.append(f"- macro_f1: {_fmt(clean.get('macro_f1'))}")
        lines.append(f"- auc: {_fmt(clean.get('auc'))}")
        lines.append(f"- confusion_matrix: {clean.get('confusion_matrix')}")
    lines.append("")

    lines.append("## Robust 指标（对抗鲁棒性）")
    if not robust.get("available"):
        lines.append(f"- 不可用: {robust.get('reason', '')}")
    else:
        lines.append(f"- 攻击样本数: {robust.get('attacked_count', '')}")
        lines.append(f"- 攻击成功数: {robust.get('success_count', '')}")
        lines.append(f"- ASR(攻击成功率): {_fmt(robust.get('asr'))}")
        lines.append(f"- robust_acc(子集内): {_fmt(robust.get('robust_acc'))}")
        stats = robust.get("stats", {})
        for name in ("queries", "l2", "linf", "modified_pixels"):
            block = stats.get(name, {})
            lines.append(f"- {name}: mean={_fmt(block.get('mean'))}, median={_fmt(block.get('median'))}, p90={_fmt(block.get('p90'))}")
    lines.append("")
    return "\n".join(lines)


def write_eval_report(report: dict[str, Any], output_dir: str | Path, report_name: str = "eval_report.json") -> Path:
    """写出 json 与同名 .md 报告，返回 json 路径。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_name = report_name if report_name.endswith(".json") else f"{report_name}.json"
    json_path = out_dir / json_name
    md_path = json_path.with_suffix(".md")

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path
