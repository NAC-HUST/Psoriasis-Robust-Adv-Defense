from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import statistics
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from psorad.attack.runner import run_samoo_attack


@dataclass
class AttackTask:
    subset_index: int
    global_index: int
    true_label: int
    class_name: str


def _resolve_manifest_csv(manifest_csv: str | None, dataset_root: str, datadir: str) -> str:
    if manifest_csv is not None:
        return manifest_csv
    return str(Path(dataset_root) / "processed_data" / datadir / "class_manifest.csv")


def _build_subset_dataframe(
    manifest_csv: str,
    attack_split: str,
    val_ratio: float,
    split_seed: int,
) -> pd.DataFrame:
    split_name = attack_split.lower().strip()
    if split_name not in {"all", "train", "val"}:
        raise ValueError("attack_split 仅支持 all/train/val")

    manifest = pd.read_csv(manifest_csv)
    manifest = manifest.reset_index().rename(columns={"index": "global_index"})

    if split_name == "all":
        return manifest.reset_index(drop=True)

    if "split" in manifest.columns:
        subset = manifest[manifest["split"].astype(str).str.lower() == split_name].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"在 manifest 中未找到 split={split_name} 的样本")
        return subset

    total = len(manifest)
    val_len = max(int(total * val_ratio), 1)
    train_len = total - val_len
    if train_len <= 0:
        raise ValueError("训练集为空，请增大数据量或减小 val_ratio")

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    selected_indices = train_indices if split_name == "train" else val_indices
    subset = manifest.iloc[selected_indices].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"在 manifest 中未找到 split={split_name} 的样本")
    return subset


def _select_tasks(
    subset_df: pd.DataFrame,
    start_index: int,
    max_samples: int | None,
    sample_indices: list[int] | None,
) -> list[AttackTask]:
    if sample_indices is not None and len(sample_indices) > 0:
        chosen = []
        for subset_idx in sample_indices:
            if subset_idx < 0 or subset_idx >= len(subset_df):
                raise IndexError(f"sample-index 越界: {subset_idx}, split样本总量: {len(subset_df)}")
            row = subset_df.iloc[subset_idx]
            chosen.append(
                AttackTask(
                    subset_index=int(subset_idx),
                    global_index=int(row["global_index"]),
                    true_label=int(row["class_idx"]),
                    class_name=str(row.get("class_name", row.get("label_name", row.get("class_idx", "unknown")))),
                )
            )
        return chosen

    if start_index < 0:
        raise ValueError("start-index 不能小于 0")
    if start_index >= len(subset_df):
        return []

    if max_samples is None or max_samples <= 0:
        end = len(subset_df)
    else:
        end = min(len(subset_df), start_index + max_samples)

    tasks: list[AttackTask] = []
    for subset_idx in range(start_index, end):
        row = subset_df.iloc[subset_idx]
        tasks.append(
            AttackTask(
                subset_index=int(subset_idx),
                global_index=int(row["global_index"]),
                true_label=int(row["class_idx"]),
                class_name=str(row.get("class_name", row.get("label_name", row.get("class_idx", "unknown")))),
            )
        )
    return tasks


def _run_single_attack(task: AttackTask, kwargs: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(str(kwargs["run_root"]))
    sample_dir = run_root / f"sample_{task.subset_index:06d}"
    raw_path = sample_dir / "samoo_result.npy"

    seed_base = int(kwargs["seed"])
    task_seed = seed_base + task.subset_index

    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        export_dir = run_samoo_attack(
            backbone=str(kwargs["backbone"]),
            checkpoint_path=str(kwargs["checkpoint"]),
            datadir=str(kwargs["datadir"]),
            manifest_csv=str(kwargs["manifest_csv"]),
            sample_index=int(task.subset_index),
            attack_split=str(kwargs["attack_split"]),
            val_ratio=float(kwargs["val_ratio"]),
            split_seed=int(kwargs["split_seed"]),
            image_size=int(kwargs["image_size"]),
            save_path=str(raw_path),
            export_dir=str(sample_dir),
            keep_raw_npy=bool(kwargs["keep_raw_npy"]),
            eps=kwargs["eps"],
            iterations=kwargs["iterations"],
            pc=kwargs["pc"],
            pm=kwargs["pm"],
            pm_end=kwargs["pm_end"],
            pop_size=kwargs["pop_size"],
            query_budget=kwargs["query_budget"],
            zero_probability=kwargs["zero_probability"],
            include_dist=bool(kwargs["include_dist"]),
            max_dist=float(kwargs["max_dist"]),
            p_size=kwargs["p_size"],
            tournament_size=kwargs["tournament_size"],
            seed=task_seed,
        )

    duration_sec = time.time() - start
    summary_path = export_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "status": "ok",
        "subset_index": task.subset_index,
        "global_index": task.global_index,
        "true_label": task.true_label,
        "class_name": task.class_name,
        "seed": task_seed,
        "duration_sec": float(duration_sec),
        "artifact_dir": str(export_dir),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _safe_median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def _safe_p90(values: list[float]) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    rank = max(0, min(len(values_sorted) - 1, int(math.ceil(0.9 * len(values_sorted)) - 1)))
    return float(values_sorted[rank])


def _build_aggregate_report(
    *,
    backbone: str,
    checkpoint: str,
    datadir: str,
    manifest_csv: str,
    attack_split: str,
    split_total: int,
    scheduled_tasks: list[AttackTask],
    run_root: Path,
    workers: int,
    started_at: float,
    ended_at: float,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    hparams: dict[str, Any],
) -> dict[str, Any]:
    ok_rows = [row for row in results if row.get("status") == "ok"]
    summaries = [row["summary"] for row in ok_rows]
    success_rows = [s for s in summaries if bool(s.get("success", False))]
    failure_rows = [s for s in summaries if not bool(s.get("success", False))]

    queries = [float(s.get("queries", -1)) for s in summaries if float(s.get("queries", -1)) >= 0]
    l2_values = [float(s.get("l2", float("nan"))) for s in summaries if not math.isnan(float(s.get("l2", float("nan"))))]
    linf_values = [float(s.get("linf", float("nan"))) for s in summaries if not math.isnan(float(s.get("linf", float("nan"))))]
    modified_pixels = [
        float(s.get("modified_pixel_count", float("nan")))
        for s in summaries
        if not math.isnan(float(s.get("modified_pixel_count", float("nan"))))
    ]
    durations = [float(row.get("duration_sec", float("nan"))) for row in ok_rows if not math.isnan(float(row.get("duration_sec", float("nan"))))]

    class_stats: dict[str, dict[str, int | float]] = {}
    for row in ok_rows:
        summary = row["summary"]
        key = str(summary.get("true_label", row.get("true_label", "unknown")))
        if key not in class_stats:
            class_stats[key] = {"count": 0, "success": 0, "success_rate": 0.0}
        class_stats[key]["count"] = int(class_stats[key]["count"]) + 1
        if bool(summary.get("success", False)):
            class_stats[key]["success"] = int(class_stats[key]["success"]) + 1

    for stats in class_stats.values():
        count = int(stats["count"])
        success = int(stats["success"])
        stats["success_rate"] = float(success / count) if count > 0 else 0.0

    report = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_root": str(run_root),
            "backbone": backbone,
            "checkpoint": checkpoint,
            "datadir": datadir,
            "manifest_csv": manifest_csv,
            "attack_split": attack_split,
            "workers": workers,
            "split_total": split_total,
            "scheduled_count": len(scheduled_tasks),
            "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
            "ended_at": datetime.fromtimestamp(ended_at).isoformat(timespec="seconds"),
            "elapsed_sec": float(ended_at - started_at),
        },
        "hparams": hparams,
        "summary": {
            "completed_count": len(ok_rows),
            "error_count": len(errors),
            "success_count": len(success_rows),
            "failure_count": len(failure_rows),
            "success_rate": float(len(success_rows) / len(ok_rows)) if ok_rows else 0.0,
        },
        "stats": {
            "queries": {
                "mean": _safe_mean(queries),
                "median": _safe_median(queries),
                "p90": _safe_p90(queries),
                "min": float(min(queries)) if queries else float("nan"),
                "max": float(max(queries)) if queries else float("nan"),
            },
            "l2": {
                "mean": _safe_mean(l2_values),
                "median": _safe_median(l2_values),
                "p90": _safe_p90(l2_values),
            },
            "linf": {
                "mean": _safe_mean(linf_values),
                "median": _safe_median(linf_values),
                "p90": _safe_p90(linf_values),
            },
            "modified_pixels": {
                "mean": _safe_mean(modified_pixels),
                "median": _safe_median(modified_pixels),
                "p90": _safe_p90(modified_pixels),
            },
            "duration_sec": {
                "mean": _safe_mean(durations),
                "median": _safe_median(durations),
                "p90": _safe_p90(durations),
                "min": float(min(durations)) if durations else float("nan"),
                "max": float(max(durations)) if durations else float("nan"),
            },
        },
        "class_stats": class_stats,
        "results": results,
        "errors": errors,
    }
    return report


def _write_report_files(report: dict[str, Any], run_root: Path) -> tuple[Path, Path, Path]:
    report_json = run_root / "batch_report.json"
    report_csv = run_root / "batch_report.csv"
    report_md = run_root / "batch_report.md"

    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    headers = [
        "subset_index",
        "global_index",
        "true_label",
        "pred_before",
        "pred_after",
        "success",
        "queries",
        "l2",
        "linf",
        "modified_pixel_count",
        "modified_channel_count",
        "duration_sec",
        "artifact_dir",
        "summary_path",
    ]
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in report.get("results", []):
            if row.get("status") != "ok":
                continue
            summary = row.get("summary", {})
            writer.writerow(
                {
                    "subset_index": row.get("subset_index"),
                    "global_index": row.get("global_index"),
                    "true_label": summary.get("true_label", row.get("true_label")),
                    "pred_before": summary.get("pred_before"),
                    "pred_after": summary.get("pred_after"),
                    "success": summary.get("success"),
                    "queries": summary.get("queries"),
                    "l2": summary.get("l2"),
                    "linf": summary.get("linf"),
                    "modified_pixel_count": summary.get("modified_pixel_count"),
                    "modified_channel_count": summary.get("modified_channel_count"),
                    "duration_sec": row.get("duration_sec"),
                    "artifact_dir": row.get("artifact_dir"),
                    "summary_path": row.get("summary_path"),
                }
            )

    meta = report.get("meta", {})
    summary = report.get("summary", {})
    stats = report.get("stats", {})

    md_lines = [
        "# Batch Attack Report",
        "",
        "## Meta",
        f"- created_at: {meta.get('created_at')}",
        f"- run_root: {meta.get('run_root')}",
        f"- backbone: {meta.get('backbone')}",
        f"- datadir: {meta.get('datadir')}",
        f"- manifest_csv: {meta.get('manifest_csv')}",
        f"- attack_split: {meta.get('attack_split')}",
        f"- workers: {meta.get('workers')}",
        f"- elapsed_sec: {meta.get('elapsed_sec'):.3f}",
        "",
        "## Summary",
        f"- scheduled_count: {meta.get('scheduled_count')}",
        f"- completed_count: {summary.get('completed_count')}",
        f"- error_count: {summary.get('error_count')}",
        f"- success_count: {summary.get('success_count')}",
        f"- failure_count: {summary.get('failure_count')}",
        f"- success_rate: {float(summary.get('success_rate', 0.0)):.4f}",
        "",
        "## Key Stats",
        f"- queries(mean/median/p90): {stats.get('queries', {}).get('mean')} / {stats.get('queries', {}).get('median')} / {stats.get('queries', {}).get('p90')}",
        f"- l2(mean/median/p90): {stats.get('l2', {}).get('mean')} / {stats.get('l2', {}).get('median')} / {stats.get('l2', {}).get('p90')}",
        f"- linf(mean/median/p90): {stats.get('linf', {}).get('mean')} / {stats.get('linf', {}).get('median')} / {stats.get('linf', {}).get('p90')}",
        f"- modified_pixels(mean/median/p90): {stats.get('modified_pixels', {}).get('mean')} / {stats.get('modified_pixels', {}).get('median')} / {stats.get('modified_pixels', {}).get('p90')}",
        f"- duration_sec(mean/median/p90): {stats.get('duration_sec', {}).get('mean')} / {stats.get('duration_sec', {}).get('median')} / {stats.get('duration_sec', {}).get('p90')}",
        "",
        "## Class Stats",
    ]

    class_stats = report.get("class_stats", {})
    if class_stats:
        for class_key in sorted(class_stats.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            row = class_stats[class_key]
            md_lines.append(
                f"- class_{class_key}: count={row.get('count')}, success={row.get('success')}, success_rate={float(row.get('success_rate', 0.0)):.4f}"
            )
    else:
        md_lines.append("- 无可用样本")

    md_lines.extend(["", "## Files", f"- JSON: {report_json}", f"- CSV: {report_csv}", f"- Markdown: {report_md}"])
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return report_json, report_csv, report_md


def _parse_sample_indices(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    items = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        items.append(int(token))
    return sorted(set(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量并行执行 SAMOO 攻击，并生成完整统计报告")
    parser.add_argument("--backbone", choices=["resnet50", "siglip"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--datadir", default="psoriasis_normal")
    parser.add_argument("--manifest-csv", default=None)

    parser.add_argument("--attack-split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--start-index", type=int, default=0, help="从子集内哪个 sample_index 开始")
    parser.add_argument("--max-samples", type=int, default=None, help="最大攻击样本数；不传或<=0表示跑到子集末尾")
    parser.add_argument("--sample-indices", default=None, help="逗号分隔的子集内 sample_index 列表，优先级高于 start/max")

    parser.add_argument("--workers", type=int, default=min(4, (os.cpu_count() or 1)), help="并行 worker 数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-raw-npy", action="store_true", help="保留每样本原始 npy（默认删除）")

    parser.add_argument("--eps", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--pc", type=float, default=None)
    parser.add_argument("--pm", type=float, default=None)
    parser.add_argument("--pm-end", type=float, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--query-budget", type=int, default=None)
    parser.add_argument("--zero-probability", type=float, default=None)
    parser.add_argument("--include-dist", action="store_true")
    parser.add_argument("--max-dist", type=float, default=1e9)
    parser.add_argument("--p-size", type=float, default=None)
    parser.add_argument("--tournament-size", type=int, default=None)

    parser.add_argument("--output-root", default="output/batch_attack", help="输出根目录")
    parser.add_argument("--run-name", default=None, help="可选运行子目录名；不传则直接输出到 <output-root>/<backbone>/<datadir>/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_csv = _resolve_manifest_csv(args.manifest_csv, dataset_root=args.dataset_root, datadir=args.datadir)
    sample_indices = _parse_sample_indices(args.sample_indices)
    subset_df = _build_subset_dataframe(
        manifest_csv=manifest_csv,
        attack_split=args.attack_split,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    tasks = _select_tasks(
        subset_df=subset_df,
        start_index=args.start_index,
        max_samples=args.max_samples,
        sample_indices=sample_indices,
    )

    if not tasks:
        raise RuntimeError("没有可执行的攻击任务，请检查 start-index/max-samples/sample-indices")

    base_root = Path(args.output_root) / args.backbone / args.datadir
    run_root = base_root if args.run_name in (None, "", ".") else base_root / str(args.run_name)
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("[BatchAttack] 启动批量并行攻击")
    print(f"[BatchAttack] 输出目录: {run_root}")
    print(f"[BatchAttack] manifest: {manifest_csv}")
    print(f"[BatchAttack] split={args.attack_split}, split_total={len(subset_df)}, scheduled={len(tasks)}")
    print(f"[BatchAttack] workers={args.workers}, seed={args.seed}, keep_raw_npy={args.keep_raw_npy}")
    print("=" * 100)

    started_at = time.time()
    worker_kwargs = {
        "run_root": str(run_root),
        "backbone": args.backbone,
        "checkpoint": args.checkpoint,
        "datadir": args.datadir,
        "manifest_csv": manifest_csv,
        "attack_split": args.attack_split,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "image_size": args.image_size,
        "seed": args.seed,
        "keep_raw_npy": args.keep_raw_npy,
        "eps": args.eps,
        "iterations": args.iterations,
        "pc": args.pc,
        "pm": args.pm,
        "pm_end": args.pm_end,
        "pop_size": args.pop_size,
        "query_budget": args.query_budget,
        "zero_probability": args.zero_probability,
        "include_dist": args.include_dist,
        "max_dist": args.max_dist,
        "p_size": args.p_size,
        "tournament_size": args.tournament_size,
    }

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    success_count = 0
    completed_count = 0
    queries_sum = 0.0
    queries_count = 0
    modified_pixels_sum = 0.0
    modified_pixels_count = 0
    duration_sum = 0.0

    futures: dict[Future[dict[str, Any]], AttackTask] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for task in tasks:
            future = executor.submit(_run_single_attack, task, worker_kwargs)
            futures[future] = task

        progress_bar = tqdm(total=len(futures), desc="Batch Attack", unit="sample", dynamic_ncols=True)
        for future in as_completed(futures):
            task = futures[future]
            try:
                row = future.result()
                results.append(row)
                completed_count += 1
                duration_sec = float(row.get("duration_sec", 0.0))
                duration_sum += max(0.0, duration_sec)

                summary = row.get("summary", {})
                query_value = float(summary.get("queries", -1.0))
                if query_value >= 0:
                    queries_sum += query_value
                    queries_count += 1

                modified_pixels_value = float(summary.get("modified_pixel_count", float("nan")))
                if not math.isnan(modified_pixels_value):
                    modified_pixels_sum += modified_pixels_value
                    modified_pixels_count += 1

                if bool(row.get("summary", {}).get("success", False)):
                    success_count += 1
            except Exception as exc:
                errors.append(
                    {
                        "subset_index": task.subset_index,
                        "global_index": task.global_index,
                        "true_label": task.true_label,
                        "class_name": task.class_name,
                        "error": repr(exc),
                    }
                )

            elapsed_sec = max(0.0, time.time() - started_at)
            processed = completed_count + len(errors)
            current_success_rate = (float(success_count) / float(completed_count)) if completed_count > 0 else 0.0
            avg_queries = (queries_sum / queries_count) if queries_count > 0 else float("nan")
            avg_modified_pixels = (modified_pixels_sum / modified_pixels_count) if modified_pixels_count > 0 else float("nan")
            avg_duration = (duration_sum / completed_count) if completed_count > 0 else float("nan")
            throughput = (processed / elapsed_sec) if elapsed_sec > 0 else float("nan")

            progress_bar.set_postfix(
                {
                    "done": processed,
                    "success": success_count,
                    "error": len(errors),
                    "succ_rate": f"{current_success_rate:.2%}",
                    "avg_q": f"{avg_queries:.1f}" if not math.isnan(avg_queries) else "-",
                    "avg_px": f"{avg_modified_pixels:.1f}" if not math.isnan(avg_modified_pixels) else "-",
                    "elapsed_s": f"{elapsed_sec:.1f}",
                    "avg_dur_s": f"{avg_duration:.1f}" if not math.isnan(avg_duration) else "-",
                    "spd": f"{throughput:.2f}/s" if not math.isnan(throughput) else "-",
                }
            )
            progress_bar.update(1)
        progress_bar.close()

    ended_at = time.time()
    results.sort(key=lambda row: int(row.get("subset_index", 1 << 30)))
    errors.sort(key=lambda row: int(row.get("subset_index", 1 << 30)))

    hparams = {
        "eps": args.eps,
        "iterations": args.iterations,
        "pc": args.pc,
        "pm": args.pm,
        "pm_end": args.pm_end,
        "pop_size": args.pop_size,
        "query_budget": args.query_budget,
        "zero_probability": args.zero_probability,
        "include_dist": args.include_dist,
        "max_dist": args.max_dist,
        "p_size": args.p_size,
        "tournament_size": args.tournament_size,
        "seed": args.seed,
    }

    report = _build_aggregate_report(
        backbone=args.backbone,
        checkpoint=args.checkpoint,
        datadir=args.datadir,
        manifest_csv=manifest_csv,
        attack_split=args.attack_split,
        split_total=len(subset_df),
        scheduled_tasks=tasks,
        run_root=run_root,
        workers=max(1, int(args.workers)),
        started_at=started_at,
        ended_at=ended_at,
        results=results,
        errors=errors,
        hparams=hparams,
    )
    report_json, report_csv, report_md = _write_report_files(report, run_root)

    summary = report["summary"]
    print("=" * 100)
    print("[BatchAttack] 全部任务完成")
    print(
        "[BatchAttack] 汇总: "
        f"scheduled={len(tasks)}, completed={summary['completed_count']}, errors={summary['error_count']}, "
        f"success={summary['success_count']}, failure={summary['failure_count']}, "
        f"success_rate={float(summary['success_rate']):.4f}"
    )
    print(f"[BatchAttack] 报告文件: {report_json}")
    print(f"[BatchAttack] 报告文件: {report_csv}")
    print(f"[BatchAttack] 报告文件: {report_md}")
    print("=" * 100)


if __name__ == "__main__":
    main()
