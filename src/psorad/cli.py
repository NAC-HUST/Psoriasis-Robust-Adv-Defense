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
    attack_parser.add_argument("--image-size", type=int, default=224)
    attack_parser.add_argument("--save-path", default=None)
    attack_parser.add_argument("--export-dir", default=None, help="攻击文本与图像结果输出目录")
    attack_parser.add_argument("--no-raw-npy", action="store_true", help="不保留原始npy结果文件")
    attack_parser.add_argument("--eps", type=int, default=None, help="扰动像素数；不传时按分辨率自适应")
    attack_parser.add_argument("--iterations", type=int, default=400, help="进化迭代次数")
    attack_parser.add_argument("--pc", type=float, default=0.3, help="交叉概率")
    attack_parser.add_argument("--pm", type=float, default=0.6, help="变异概率")
    attack_parser.add_argument("--pop-size", type=int, default=12, help="种群大小")
    attack_parser.add_argument("--zero-probability", type=float, default=0.2, help="像素扰动为0的概率")
    attack_parser.add_argument("--include-dist", action="store_true", help="在可行解筛选时加入距离约束")
    attack_parser.add_argument("--max-dist", type=float, default=1e9, help="可行解最大距离阈值")
    attack_parser.add_argument("--p-size", type=float, default=0.25, help="单步扰动幅度")
    attack_parser.add_argument("--tournament-size", type=int, default=2, help="锦标赛选择规模")
    attack_parser.add_argument("--seed", type=int, default=42)

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
        from psorad.models.download import download_all_models

        resnet_path, siglip_path = download_all_models()
        print(f"resnet50已下载到: {resnet_path}")
        print(f"siglip已下载到: {siglip_path}")
        return

    if args.command == "train":
        from psorad.trainers.train import TrainConfig, train_classifier

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
            image_size=args.image_size,
            save_path=args.save_path,
            export_dir=args.export_dir,
            keep_raw_npy=not args.no_raw_npy,
            eps=args.eps,
            iterations=args.iterations,
            pc=args.pc,
            pm=args.pm,
            pop_size=args.pop_size,
            zero_probability=args.zero_probability,
            include_dist=args.include_dist,
            max_dist=args.max_dist,
            p_size=args.p_size,
            tournament_size=args.tournament_size,
            seed=args.seed,
        )
        print(f"SAMOO攻击完成，文本与图像结果保存至: {output}")
        return
