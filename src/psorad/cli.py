from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Psoriasis Robust Adv&Defense CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="预处理 normal 数据并构建二分类 manifest")
    preprocess_parser.add_argument("--dataset-root", default="dataset")
    preprocess_parser.add_argument("--metadata-csv", default="dataset/psoriasis_metadata.csv")
    preprocess_parser.add_argument("--normal-raw-dir", default="dataset/normal")
    preprocess_parser.add_argument("--normal-processed-dir", default="dataset/normal_224")
    preprocess_parser.add_argument("--manifest-csv", default="dataset/binary_manifest.csv")
    preprocess_parser.add_argument("--image-size", type=int, default=224)

    subparsers.add_parser("download-models", help="下载resnet50与siglip预训练模型")

    train_parser = subparsers.add_parser("train", help="训练二分类模型")
    train_parser.add_argument("--backbone", choices=["resnet50", "siglip"], required=True)
    train_parser.add_argument("--manifest-csv", default="dataset/binary_manifest.csv")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--freeze-siglip-backbone", action="store_true")

    attack_parser = subparsers.add_parser("attack", help="运行SAMOO攻击")
    attack_parser.add_argument("--backbone", choices=["resnet50", "siglip"], required=True)
    attack_parser.add_argument("--checkpoint", required=True)
    attack_parser.add_argument("--manifest-csv", default="dataset/binary_manifest.csv")
    attack_parser.add_argument("--sample-index", type=int, default=0)
    attack_parser.add_argument("--save-path", default=None)
    attack_parser.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        from psorad.preprocess.pipeline import run_preprocess

        processed_count, manifest_rows = run_preprocess(
            dataset_root=args.dataset_root,
            metadata_csv=args.metadata_csv,
            normal_raw_dir=args.normal_raw_dir,
            normal_processed_dir=args.normal_processed_dir,
            manifest_csv=args.manifest_csv,
            image_size=args.image_size,
        )
        print(f"normal预处理完成: {processed_count} 张, manifest样本数: {manifest_rows}")
        return

    if args.command == "download-models":
        from psorad.models.download import download_all_models

        resnet_path, siglip_path = download_all_models()
        print(f"resnet50已下载到: {resnet_path}")
        print(f"siglip已下载到: {siglip_path}")
        return

    if args.command == "train":
        from psorad.trainers.train_binary import TrainConfig, train_binary_classifier

        config = TrainConfig(
            backbone=args.backbone,
            manifest_csv=args.manifest_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            freeze_siglip_backbone=args.freeze_siglip_backbone,
        )
        checkpoint_path = train_binary_classifier(config)
        print(f"训练完成，最佳模型保存至: {checkpoint_path}")
        return

    if args.command == "attack":
        from psorad.attack.runner import run_samoo_attack

        output = run_samoo_attack(
            backbone=args.backbone,
            checkpoint_path=args.checkpoint,
            manifest_csv=args.manifest_csv,
            sample_index=args.sample_index,
            save_path=args.save_path,
            seed=args.seed,
        )
        print(f"SAMOO攻击完成，结果保存至: {output}")
        return
