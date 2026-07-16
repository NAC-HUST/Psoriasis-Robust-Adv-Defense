#!/usr/bin/env python3
"""Run batch attacks on all 5 trained models and produce baseline comparison.

Usage:
    .venv/bin/python tools/batch_run_attacks.py [--max-samples 50] [--workers 2] [--dry-run]

Runs batch_parallel_attack.py for each of the 5 datasets with uniform SAMOO params.
Output: output/batch_attack/resnet50/<dataset>/batch_report.json (gitignored).
Summary: output/eval/baseline_comparison.{json,md} (gitignored).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = [
    {
        "name": "psoriasis224_2c",
        "checkpoint": "model/trained_classifier/resnet50/resnet50_psoriasis224_2c.pt",
        "manifest": "output/manifests/psoriasis224_2c_manifest.csv",
    },
    {
        "name": "cifar32_10c",
        "checkpoint": "model/trained_classifier/resnet50/resnet50_cifar32_10c.pt",
        "manifest": "output/manifests/cifar32_10c_manifest.csv",
    },
    {
        "name": "imagenette224_10c",
        "checkpoint": "model/trained_classifier/resnet50/resnet50_imagenette224_10c.pt",
        "manifest": "output/manifests/imagenette224_10c_manifest.csv",
    },
    {
        "name": "dermamnist224_7c",
        "checkpoint": "model/trained_classifier/resnet50/resnet50_dermamnist224_7c.pt",
        "manifest": "output/manifests/dermamnist224_7c_manifest.csv",
    },
    {
        "name": "milk10k_11c",
        "checkpoint": "model/trained_classifier/resnet50/resnet50_milk10k_11c.pt",
        "manifest": "output/manifests/milk10k_11c_manifest.csv",
    },
]

# Uniform SAMOO params (None = runner defaults)
SAMOO_PARAMS = {
    "eps": None,
    "iterations": None,
    "pc": None,
    "pm": None,
    "pm_end": None,
    "pop_size": None,
    "query_budget": None,
    "zero_probability": None,
    "include_dist": None,
    "max_dist": None,
    "p_size": None,
    "tournament_size": None,
}

ATTACK_SPLIT = "val"
ATTACK_WORKERS = 2
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch attack all 5 datasets")
    parser.add_argument("--max-samples", type=int, default=50, help="samples per dataset")
    parser.add_argument("--workers", type=int, default=ATTACK_WORKERS)
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    return parser.parse_args()


def build_cmd(dataset: dict, max_samples: int, workers: int) -> list[str]:
    cmd = [
        str(BASE_DIR / ".venv/bin/python"),
        str(BASE_DIR / "tools/batch_parallel_attack.py"),
        "--backbone",
        "resnet50",
        "--checkpoint",
        str(BASE_DIR / dataset["checkpoint"]),
        "--manifest-csv",
        str(BASE_DIR / dataset["manifest"]),
        "--datadir",
        dataset["name"],
        "--attack-split",
        ATTACK_SPLIT,
        "--seed",
        str(SEED),
        "--workers",
        str(workers),
        "--max-samples",
        str(max_samples),
        "--output-root",
        str(BASE_DIR / "output/batch_attack"),
    ]

    for key, val in SAMOO_PARAMS.items():
        if val is not None:
            flag = "--" + key.replace("_", "-")
            cmd.append(str(flag))
            cmd.append(str(val))

    return cmd


def run() -> None:
    args = parse_args()
    comparison: dict[str, dict] = {}

    for ds in DATASETS:
        cmd = build_cmd(ds, args.max_samples, args.workers)
        print(f"\n{'=' * 60}")
        print(f"Attacking {ds['name']} ({args.max_samples} samples) ...")
        print(f"{'=' * 60}")

        if args.dry_run:
            print(" ".join(str(c) for c in cmd))
            continue

        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=7200)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}", file=sys.stderr)
            comparison[ds["name"]] = {"error": result.stderr.strip()}
            continue

        # Parse batch_report.json for summary
        report_dir = BASE_DIR / "output" / "batch_attack" / "resnet50" / ds["name"]
        report_path = report_dir / "batch_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = report.get("summary", {})
            stats = report.get("stats", {})
            comparison[ds["name"]] = {
                "success_rate": summary.get("success_rate"),
                "completed": summary.get("completed_count"),
                "errors": summary.get("error_count"),
                "robust_acc": 1.0 - float(summary.get("success_rate", 0.0)),
                "avg_queries": stats.get("queries", {}).get("mean"),
                "avg_linf": stats.get("linf", {}).get("mean"),
                "avg_modified_pixels": stats.get("modified_pixels", {}).get("mean"),
                "class_stats": report.get("class_stats", {}),
            }
        else:
            comparison[ds["name"]] = {"error": "batch_report.json not found"}

    # Write comparison
    if not args.dry_run:
        out_dir = BASE_DIR / "output" / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "baseline_comparison.json"
        json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

        md_lines = [
            "# Baseline Attack Comparison",
            "",
            "| Dataset | Classes | ASR | Robust Acc | Samples | Avg Queries | Avg Linf | Avg Pixels |",
            "|---------|---------|-----|------------|---------|-------------|----------|------------|",
        ]
        for name, data in comparison.items():
            if "error" in data:
                md_lines.append(f"| {name} | - | ERROR: {data['error']} | - | - | - | - | - |")
            else:
                md_lines.append(
                    f"| {name} | - | {float(data['success_rate']):.4f} | {float(data['robust_acc']):.4f} | {data['completed']} | {float(data['avg_queries']):.1f} | {float(data['avg_linf']):.4f} | {float(data['avg_modified_pixels']):.1f} |"
                )

        md_path = out_dir / "baseline_comparison.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        print(f"\n{'=' * 60}")
        print("Comparison written to:")
        print(f"  {json_path}")
        print(f"  {md_path}")

        # Print summary table
        print(f"\n{'=' * 60}")
        print("Baseline Comparison Summary:")
        print(f"{'=' * 60}")
        for name, data in comparison.items():
            if "error" in data:
                print(f"  {name}: ERROR - {data['error']}")
            else:
                print(f"  {name}: ASR={float(data['success_rate']):.4f}  RobustAcc={float(data['robust_acc']):.4f}  queries={float(data['avg_queries']):.1f}")


if __name__ == "__main__":
    run()
