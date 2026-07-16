from __future__ import annotations

import argparse
from pathlib import Path


def _resolve_manifest_csv(manifest_csv: str | None, datadir: str, dataset_root: str) -> str:
    if manifest_csv is not None:
        return manifest_csv
    return str(Path(dataset_root) / "processed_data" / datadir / "class_manifest.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Psoriasis Robust Adv&Defense CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="预处理 raw_data/<datadir> 并构建分类 manifest")
    preprocess_parser.add_argument("--dataset-root", default="dataset")
    preprocess_parser.add_argument("--datadir", required=True, help="raw_data 下的数据集目录名")
    preprocess_parser.add_argument("--raw-data-root", default=None, help="原始数据根目录，默认 <dataset-root>/raw_data")
    preprocess_parser.add_argument("--processed-data-root", default=None, help="处理后数据根目录，默认 <dataset-root>/processed_data")
    preprocess_parser.add_argument("--image-size", type=int, default=224)

    subparsers.add_parser("download-models", help="下载resnet50与siglip预训练模型")

    train_parser = subparsers.add_parser("train", help="训练分类模型")
    train_parser.add_argument("--backbone", choices=["resnet50", "siglip"], required=True)
    train_parser.add_argument("--dataset-root", default="dataset")
    train_parser.add_argument("--datadir", default="psoriasis_normal")
    train_parser.add_argument("--manifest-csv", default=None)
    train_parser.add_argument("--experiment-config", default=None, help="可选：实验配置 TOML 路径（优先于 manifest/dataset-root 等参数）")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--modelname", default="best_classifier.pt", help="训练输出模型文件名")
    train_parser.add_argument("--freeze-siglip-backbone", action="store_true")

    attack_parser = subparsers.add_parser("attack", help="运行SAMOO攻击")
    attack_parser.add_argument("--backbone", choices=["resnet50", "siglip"], required=True)
    attack_parser.add_argument("--checkpoint", required=True)
    attack_parser.add_argument("--dataset-root", default="dataset")
    attack_parser.add_argument("--datadir", default="psoriasis_normal")
    attack_parser.add_argument("--manifest-csv", default=None)
    attack_parser.add_argument("--sample-index", type=int, default=0)
    attack_parser.add_argument("--attack-split", choices=["all", "train", "val"], default="val", help="默认仅从验证集(val)采样攻击")
    attack_parser.add_argument("--val-ratio", type=float, default=0.2, help="当manifest不含split列时，用于重建train/val划分")
    attack_parser.add_argument("--split-seed", type=int, default=42, help="当manifest不含split列时，划分随机种子")
    attack_parser.add_argument("--image-size", type=int, default=224)
    attack_parser.add_argument("--save-path", default=None)
    attack_parser.add_argument("--export-dir", default=None, help="攻击文本与图像结果输出目录")
    attack_parser.add_argument("--no-raw-npy", action="store_true", help="不保留原始npy结果文件")
    attack_parser.add_argument("--eps", type=int, default=None, help="扰动像素数；不传时按分辨率预设")
    attack_parser.add_argument("--iterations", type=int, default=None, help="进化迭代次数；不传时由分辨率预设或query-budget推导")
    attack_parser.add_argument("--pc", type=float, default=None, help="交叉概率；不传时按分辨率预设")
    attack_parser.add_argument("--pm", type=float, default=None, help="初始变异概率；不传时按分辨率预设")
    attack_parser.add_argument("--pm-end", type=float, default=None, help="末期变异概率；不传时按分辨率预设")
    attack_parser.add_argument("--pop-size", type=int, default=None, help="种群大小；不传时按分辨率预设")
    attack_parser.add_argument("--query-budget", type=int, default=None, help="查询预算上限；不传时按分辨率预设")
    attack_parser.add_argument("--zero-probability", type=float, default=None, help="像素扰动为0的概率；不传时按分辨率预设")
    attack_parser.add_argument("--include-dist", action="store_true", help="在可行解筛选时加入距离约束")
    attack_parser.add_argument("--max-dist", type=float, default=1e9, help="可行解最大距离阈值")
    attack_parser.add_argument("--p-size", type=float, default=None, help="单步扰动幅度；不传时按分辨率预设")
    attack_parser.add_argument("--tournament-size", type=int, default=None, help="锦标赛选择规模；不传时按分辨率预设")
    attack_parser.add_argument("--seed", type=int, default=42)

    evaluate_parser = subparsers.add_parser("evaluate", help="评估分类器：clean 指标 + 从批量攻击报告读取 robust/ASR")
    evaluate_parser.add_argument("--config", default=None, help="评估配置 TOML 路径（优先于以下显式参数）")
    evaluate_parser.add_argument("--backbone", choices=["resnet50", "siglip"], default="resnet50")
    evaluate_parser.add_argument("--checkpoint", default=None, help="待评估分类器权重")
    evaluate_parser.add_argument("--dataset-root", default="dataset")
    evaluate_parser.add_argument("--datadir", default="psoriasis_normal")
    evaluate_parser.add_argument("--manifest-csv", default=None)
    evaluate_parser.add_argument("--split", choices=["all", "train", "val"], default="val", help="评估所用子集")
    evaluate_parser.add_argument("--val-ratio", type=float, default=0.2)
    evaluate_parser.add_argument("--split-seed", type=int, default=42)
    evaluate_parser.add_argument("--image-size", type=int, default=224)
    evaluate_parser.add_argument("--batch-size", type=int, default=32)
    evaluate_parser.add_argument("--num-workers", type=int, default=2)
    evaluate_parser.add_argument("--batch-report", default=None, help="批量攻击报告 batch_report.json 路径（robust 指标来源）")
    evaluate_parser.add_argument("--grid-size", type=int, default=7, help="空间脆弱性网格大小")
    evaluate_parser.add_argument("--high-freq-cutoff", type=float, default=0.25, help="频域分析高频截止比例")
    evaluate_parser.add_argument("--save-visuals", action="store_true", help="保存脆弱性可视化热图")
    evaluate_parser.add_argument("--output-dir", default="output/eval", help="评估报告输出目录")
    evaluate_parser.add_argument("--report-name", default="eval_report.json", help="评估报告文件名")
    evaluate_parser.add_argument("--skip-clean", action="store_true", help="跳过 clean 前向推理，仅汇总 robust 指标")

    defend_parser = subparsers.add_parser("defend", help="高低频+区域感知净化防御")
    defend_parser.add_argument("--config", default=None, help="防御配置 TOML 路径（优先）")
    defend_parser.add_argument("--backbone", choices=["resnet50", "siglip"], default="resnet50")
    defend_parser.add_argument("--checkpoint", required=True)
    defend_parser.add_argument("--batch-report", required=True)
    defend_parser.add_argument("--window", type=int, default=3, help="局部中值窗口")
    defend_parser.add_argument("--threshold", type=float, default=0.08, help="可疑像素阈值")
    defend_parser.add_argument("--dilation", type=int, default=1, help="区域膨胀半径")
    defend_parser.add_argument("--low-freq-cutoff", type=float, default=0.25, help="FFT低频截止")
    defend_parser.add_argument("--low-freq", choices=["fft", "median"], default="fft", help="低频重建方式")
    defend_parser.add_argument("--output-dir", default="output/defense")
    defend_parser.add_argument("--report-name", default="defense_report.json")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        from psorad.preprocess.pipeline import run_preprocess

        processed_count, manifest_rows, manifest_path = run_preprocess(
            datadir=args.datadir,
            dataset_root=args.dataset_root,
            raw_data_root=args.raw_data_root,
            processed_data_root=args.processed_data_root,
            image_size=args.image_size,
        )
        print(f"预处理完成: {processed_count} 张, manifest样本数: {manifest_rows}, manifest路径: {manifest_path}")
        return

    if args.command == "download-models":
        from psorad.utils.download import download_all_models

        resnet_path, siglip_path = download_all_models()
        print(f"resnet50已下载到: {resnet_path}")
        print(f"siglip已下载到: {siglip_path}")
        return

    if args.command == "train":
        if args.experiment_config is not None:
            from psorad.config import load_experiment_config
            from psorad.models.train import train_experiment

            exp = load_experiment_config(args.experiment_config)
            checkpoint_path = train_experiment(exp)
            print(f"训练完成，最佳模型保存至: {checkpoint_path}")
            return

        from psorad.models.train import TrainConfig, train_classifier

        manifest_csv = _resolve_manifest_csv(args.manifest_csv, datadir=args.datadir, dataset_root=args.dataset_root)
        config = TrainConfig(
            backbone=args.backbone,
            manifest_csv=manifest_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            image_size=args.image_size,
            model_name=args.modelname,
            freeze_siglip_backbone=args.freeze_siglip_backbone,
        )
        checkpoint_path = train_classifier(config)
        print(f"训练完成，最佳模型保存至: {checkpoint_path}")
        return

    if args.command == "attack":
        from psorad.attack.runner import run_samoo_attack

        manifest_csv = _resolve_manifest_csv(args.manifest_csv, datadir=args.datadir, dataset_root=args.dataset_root)
        output = run_samoo_attack(
            backbone=args.backbone,
            checkpoint_path=args.checkpoint,
            datadir=args.datadir,
            manifest_csv=manifest_csv,
            sample_index=args.sample_index,
            attack_split=args.attack_split,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            image_size=args.image_size,
            save_path=args.save_path,
            export_dir=args.export_dir,
            keep_raw_npy=not args.no_raw_npy,
            eps=args.eps,
            iterations=args.iterations,
            pc=args.pc,
            pm=args.pm,
            pm_end=args.pm_end,
            pop_size=args.pop_size,
            query_budget=args.query_budget,
            zero_probability=args.zero_probability,
            include_dist=args.include_dist,
            max_dist=args.max_dist,
            p_size=args.p_size,
            tournament_size=args.tournament_size,
            seed=args.seed,
        )
        print(f"SAMOO攻击完成，文本与图像结果保存至: {output}")
        return

    if args.command == "evaluate":
        from psorad.eval import run_evaluate

        if args.config is not None:
            from psorad.config import load_eval_config

            eval_cfg = load_eval_config(args.config)
        else:
            from psorad.config import EvalConfig

            if args.checkpoint is None:
                parser.error("evaluate 需要 --checkpoint（或改用 --config）")
            manifest_csv = _resolve_manifest_csv(args.manifest_csv, datadir=args.datadir, dataset_root=args.dataset_root)
            eval_cfg = EvalConfig(
                backbone=args.backbone,
                checkpoint=args.checkpoint,
                manifest_csv=manifest_csv,
                split=args.split,
                val_ratio=args.val_ratio,
                split_seed=args.split_seed,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                batch_report=args.batch_report,
                grid_size=args.grid_size,
                high_freq_cutoff=args.high_freq_cutoff,
                save_visuals=args.save_visuals,
                output_dir=Path(args.output_dir),
                report_name=args.report_name,
            )

        report_path = run_evaluate(eval_cfg, skip_clean=args.skip_clean)
        print(f"评估完成，报告已保存至: {report_path}")
        return

    if args.command == "defend":
        from psorad.defense.runner import run_defense

        if args.config is not None:
            from psorad.config import load_defense_config

            cfg = load_defense_config(args.config)
        else:
            from psorad.config import DefenseConfig

            cfg = DefenseConfig(
                backbone=args.backbone,
                checkpoint=args.checkpoint,
                batch_report=args.batch_report,
                window=args.window,
                threshold=args.threshold,
                dilation=args.dilation,
                low_freq_cutoff=args.low_freq_cutoff,
                low_freq=args.low_freq,
                output_dir=Path(args.output_dir),
                report_name=args.report_name,
            )

        report_path = run_defense(cfg)
        print(f"防御完成，报告已保存至: {report_path}")
        return


if __name__ == "__main__":
    main()
