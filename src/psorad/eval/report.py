from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_eval_report(clean: dict[str, Any] | None, robust: dict[str, Any], meta: dict[str, Any], *, vulnerability: dict[str, Any] | None = None) -> dict[str, Any]:
    """合并 clean、robust 与 vulnerability 指标为统一评估报告。"""
    report: dict[str, Any] = {
        "meta": {"created_at": datetime.now().isoformat(timespec="seconds"), **meta},
        "clean": clean if clean is not None else {"available": False, "reason": "clean_eval_skipped"},
        "robust": robust,
    }
    if vulnerability is not None:
        report["vulnerability"] = vulnerability
    return report


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

    vuln = report.get("vulnerability", {})
    if vuln.get("available"):
        lines.append("## 脆弱性分析")
        bs = vuln.get("by_success", {})
        for outcome in ("success", "failure"):
            group = bs.get(outcome)
            if group is not None:
                lines.append(f"### {'攻击成功' if outcome == 'success' else '攻击失败'}组")
                lines.append(f"- 样本数: {group.get('count')}")
                lines.append(f"- prob_true_before: mean={_fmt(group.get('prob_true_before_mean'))}, median={_fmt(group.get('prob_true_before_median'))}")
                lines.append(f"- queries: mean={_fmt(group.get('queries_mean'))}, median={_fmt(group.get('queries_median'))}")
                lines.append(f"- linf: mean={_fmt(group.get('linf_mean'))}, median={_fmt(group.get('linf_median'))}")
                lines.append(f"- modified_pixels: mean={_fmt(group.get('modified_pixels_mean'))}, median={_fmt(group.get('modified_pixels_median'))}")
                if outcome == "success":
                    lines.append(f"- confidence_drop: mean={_fmt(group.get('confidence_drop_mean'))}")

        cc = vuln.get("confidence_correlation", [])
        if cc:
            lines.append("### 置信度与 ASR 相关性")
            lines.append("| 置信度区间 | 样本数 | ASR |")
            lines.append("|-----------|--------|-----|")
            for entry in cc:
                lines.append(f"| {entry['bucket']} | {entry['count']} | {_fmt(entry['asr'])} |")

        freq = vuln.get("frequency", {})
        if freq:
            lines.append("### 频域分析")
            lines.append(f"- 高频能量占比: mean={_fmt(freq.get('high_freq_ratio_mean'))}, median={_fmt(freq.get('high_freq_ratio_median'))}")
            lines.append(f"- 成功组平均高频占比: {_fmt(freq.get('success_high_freq_mean'))}")
            lines.append(f"- 失败组平均高频占比: {_fmt(freq.get('failure_high_freq_mean'))}")

        spat = vuln.get("spatial", {})
        if spat:
            lines.append("### 空间脆弱性")
            lines.append(f"- 最脆弱区域(网格): {spat.get('most_vulnerable_region')}")
            lines.append(f"- 中心能量占比: {_fmt(spat.get('center_energy_ratio'))}")
            lines.append(f"- 边缘能量占比: {_fmt(spat.get('edge_energy_ratio'))}")
            if spat.get("heatmap_path"):
                lines.append(f"- 热图: {spat['heatmap_path']}")
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
