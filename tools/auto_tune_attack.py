from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psorad.attack.runner import run_samoo_attack


def build_param_candidates() -> list[dict[str, Any]]:
    return [
        {"iterations": 300, "pop_size": 12, "eps": 128, "p_size": 0.25, "pm": 0.65, "pc": 0.3},
        {"iterations": 400, "pop_size": 16, "eps": 192, "p_size": 0.35, "pm": 0.7, "pc": 0.35},
        {"iterations": 500, "pop_size": 20, "eps": 256, "p_size": 0.5, "pm": 0.75, "pc": 0.4},
        {"iterations": 700, "pop_size": 24, "eps": 384, "p_size": 1.0, "pm": 0.8, "pc": 0.45},
        {"iterations": 900, "pop_size": 28, "eps": 512, "p_size": 2.0, "pm": 0.85, "pc": 0.5},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAMOO 自动调参并搜索可成功攻击的参数组合")
    parser.add_argument("--backbone", choices=["resnet50", "siglip"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--datadir", default="psoriasis_normal")
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-root", default="output/autotune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--seeds-per-trial", type=int, default=2, help="每组参数重复不同seed次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_csv = args.manifest_csv or str(Path(args.dataset_root) / "processed_data" / args.datadir / "class_manifest.csv")
    candidates = build_param_candidates()[: max(1, args.max_trials)]
    sample_index = args.sample_index

    output_root = Path(args.output_root) / args.backbone / f"samoo_autotune_sample_{sample_index}"
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    best_success = False

    for idx, params in enumerate(candidates, start=1):
        for seed_offset in range(max(1, args.seeds_per_trial)):
            trial_id = f"{idx:02d}_s{seed_offset:02d}"
            trial_dir = output_root / f"trial_{trial_id}"
            export_dir = trial_dir / "artifacts"
            raw_path = trial_dir / "samoo_result.npy"

            result_dir = run_samoo_attack(
                backbone=args.backbone,
                checkpoint_path=args.checkpoint,
                datadir=args.datadir,
                manifest_csv=manifest_csv,
                sample_index=sample_index,
                image_size=args.image_size,
                save_path=str(raw_path),
                export_dir=str(export_dir),
                keep_raw_npy=True,
                iterations=int(params["iterations"]),
                pop_size=int(params["pop_size"]),
                eps=int(params["eps"]),
                p_size=float(params["p_size"]),
                pm=float(params["pm"]),
                pc=float(params["pc"]),
                include_dist=False,
                max_dist=1e9,
                seed=args.seed + idx * 100 + seed_offset,
            )

            summary_path = result_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            record = {
                "trial": idx,
                "seed_offset": seed_offset,
                "success": bool(summary.get("success", False)),
                "pred_before": int(summary.get("pred_before", -1)),
                "pred_after": int(summary.get("pred_after", -1)),
                "logit_before": float(summary.get("logit_before", 0.0)),
                "logit_after": float(summary.get("logit_after", 0.0)),
                "l2": float(summary.get("l2", 0.0)),
                "linf": float(summary.get("linf", 0.0)),
                "queries": int(summary.get("queries", -1)),
                "params": params,
                "artifact_dir": str(result_dir),
            }
            records.append(record)
            print(
                f"[trial {idx} seed {seed_offset}] success={record['success']} "
                f"pred {record['pred_before']} -> {record['pred_after']} "
                f"logit {record['logit_before']:.4f}->{record['logit_after']:.4f} params={params}"
            )

            if record["success"]:
                best_success = True
                break

        if best_success:
            break

    report = {
        "backbone": args.backbone,
        "checkpoint": args.checkpoint,
        "sample_index": sample_index,
        "success_found": best_success,
        "total_trials": len(records),
        "records": records,
    }

    report_path = output_root / "autotune_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n自动调参完成，报告: {report_path}")


if __name__ == "__main__":
    main()
