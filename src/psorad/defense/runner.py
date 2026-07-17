from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from psorad.config import DefenseConfig
from psorad.defense.purify import purify
from psorad.eval.clean import load_classifier


def _load_image(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None


def _model_forward(
    model: torch.nn.Module,
    img: np.ndarray,
    device: torch.device,
    *,
    backbone: str = "resnet50",
) -> tuple[int, float]:
    # 模型前向，返回 (pred_label, max_prob)
    import torchvision.transforms as transforms  # fmt: skip

    if backbone in {"resnet50", "resnet"}:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # HWC 转 CHW 并归一化
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)
    tensor = normalize(tensor).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)
        pred = int(torch.argmax(probs, dim=-1).item())
        prob = float(probs[0, pred].item())
    return pred, prob


def run_defense(cfg: DefenseConfig) -> Path:
    # 区域感知净化防御评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, num_classes = load_classifier(cfg.backbone, cfg.checkpoint, device)

    report_path = Path(cfg.batch_report)
    if not report_path.exists():
        raise FileNotFoundError(f"batch_report not found: {report_path}")

    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    dataset_name = report_path.parent.name  # 如 psoriasis224_2c

    ok_results = [row for row in results if row.get("status") == "ok"]
    if not ok_results:
        raise RuntimeError("no ok results in batch report")

    artifact_root = report_path.parent

    # 指标累积
    clean_correct = 0
    clean_purified_correct = 0
    recovered = 0
    attack_success_total = 0
    robust_after_correct = 0
    total = 0

    # 各类恢复率
    class_recovery: dict[str, dict[str, int]] = {}

    for row in ok_results:
        summary = row.get("summary", {})
        true_label = int(summary.get("true_label", row.get("true_label", -1)))
        success = bool(summary.get("success", False))
        class_name = str(row.get("class_name", str(true_label)))
        total += 1

        artifact_dir = artifact_root / str(row.get("artifact_dir", ""))
        if not artifact_dir.exists():
            artifact_dir = Path(str(row.get("artifact_dir", "")))

        before = _load_image(artifact_dir / "before.png")
        after = _load_image(artifact_dir / "after.png")
        if before is None or after is None:
            continue

        # 干净精度
        pred_b, _ = _model_forward(model, before, device, backbone=cfg.backbone)
        if pred_b == true_label:
            clean_correct += 1

        # 净化后干净精度
        purified_before = purify(
            before,
            window=cfg.window,
            threshold=cfg.threshold,
            dilation=cfg.dilation,
            low_freq_cutoff=cfg.low_freq_cutoff,
            low_freq=cfg.low_freq,
        )
        pred_pb, _ = _model_forward(model, purified_before, device, backbone=cfg.backbone)
        if pred_pb == true_label:
            clean_purified_correct += 1

        # 防御后鲁棒精度
        purified_after = purify(
            after,
            window=cfg.window,
            threshold=cfg.threshold,
            dilation=cfg.dilation,
            low_freq_cutoff=cfg.low_freq_cutoff,
            low_freq=cfg.low_freq,
        )
        pred_pa, _ = _model_forward(model, purified_after, device, backbone=cfg.backbone)
        if pred_pa == true_label:
            robust_after_correct += 1

        # 恢复：仅对攻击成功的样本
        if success:
            attack_success_total += 1
            if pred_pa == true_label:
                recovered += 1
                cls_key = class_name
                if cls_key not in class_recovery:
                    class_recovery[cls_key] = {"attacked": 0, "recovered": 0}
                class_recovery[cls_key]["attacked"] += 1
                class_recovery[cls_key]["recovered"] += 1
        elif success is False:
            # 各类统计
            cls_key = class_name
            if cls_key not in class_recovery:
                class_recovery[cls_key] = {"attacked": 0, "recovered": 0}
            if success:
                class_recovery[cls_key]["attacked"] += 1

    # 计算指标
    clean_acc = clean_correct / total if total > 0 else float("nan")
    clean_acc_purified = clean_purified_correct / total if total > 0 else float("nan")
    clean_acc_drop = clean_acc - clean_acc_purified
    robust_acc_after = robust_after_correct / total if total > 0 else float("nan")
    recovery_rate = recovered / attack_success_total if attack_success_total > 0 else float("nan")

    # 各类恢复率
    per_class_recovery: dict[str, dict[str, Any]] = {}
    for cls_name, cr in class_recovery.items():
        attacked = cr["attacked"]
        rec = cr["recovered"]
        per_class_recovery[cls_name] = {
            "attacked_count": attacked,
            "recovered_count": rec,
            "recovery_rate": float(rec / attacked) if attacked > 0 else float("nan"),
        }

    report: dict[str, Any] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": "freq_region_purify",
            "backbone": cfg.backbone,
            "checkpoint": cfg.checkpoint,
            "batch_report": cfg.batch_report,
            "dataset": dataset_name,
        },
        "config": {
            "window": cfg.window,
            "threshold": cfg.threshold,
            "dilation": cfg.dilation,
            "low_freq_cutoff": cfg.low_freq_cutoff,
            "low_freq": cfg.low_freq,
        },
        "summary": {
            "total_samples": total,
            "clean_acc": clean_acc,
            "clean_acc_purified": clean_acc_purified,
            "clean_acc_drop": clean_acc_drop,
            "robust_acc_after_purification": robust_acc_after,
            "attack_success_count": attack_success_total,
            "recovered_count": recovered,
            "recovery_rate": recovery_rate,
        },
        "per_class_recovery": per_class_recovery,
    }

    out_dir = Path(cfg.output_dir) / cfg.backbone / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_name = cfg.report_name if cfg.report_name.endswith(".json") else f"{cfg.report_name}.json"
    json_path = out_dir / json_name
    md_path = json_path.with_suffix(".md")

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown 报告
    md_lines = [
        "# 净化防御报告",
        "",
        f"- 生成时间: {report['meta']['created_at']}",
        f"- 方法: {report['meta']['method']}",
        f"- backbone: {report['meta']['backbone']}",
        f"- 数据集: {report['meta']['dataset']}",
        f"- checkpoint: {report['meta']['checkpoint']}",
        "",
        "## 配置参数",
        f"- 局部中值窗口: {cfg.window}",
        f"- 可疑像素阈值: {cfg.threshold}",
        f"- 区域膨胀: {cfg.dilation}",
        f"- 低频截止: {cfg.low_freq_cutoff}",
        f"- 低频重建方式: {cfg.low_freq}",
        "",
        "## 指标",
        f"- 总样本数: {report['summary']['total_samples']}",
        f"- 干净精度(原图): {_fmt(report['summary']['clean_acc'])}",
        f"- 干净精度(净化后): {_fmt(report['summary']['clean_acc_purified'])}",
        f"- 干净精度下降: {_fmt(report['summary']['clean_acc_drop'])}",
        f"- 鲁棒精度(净化后): {_fmt(report['summary']['robust_acc_after_purification'])}",
        "",
        "## 防御恢复率",
        f"- 被成功攻击样本数: {report['summary']['attack_success_count']}",
        f"- 恢复样本数: {report['summary']['recovered_count']}",
        f"- 恢复率(recovery_rate): {_fmt(report['summary']['recovery_rate'])}",
        "",
        "## 各类恢复率",
    ]
    for cls_name, cr in per_class_recovery.items():
        md_lines.append(f"- {cls_name}: attacked={cr['attacked_count']}, recovered={cr['recovered_count']}, recovery_rate={_fmt(cr['recovery_rate'])}")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[Defense] report: {json_path}")
    print(f"[Defense] md:     {md_path}")
    print(f"[Defense] clean_acc={_fmt(clean_acc)}, clean_acc_drop={_fmt(clean_acc_drop)}, recovery_rate={_fmt(recovery_rate)}, robust_acc_after={_fmt(robust_acc_after)}")

    return json_path


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.4f}"
    return str(v)
