# Psoriasis Robust Adv&Defense

面向银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究。

[![Status](https://img.shields.io/badge/status-in%20progress-yellow)](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
[![University](https://img.shields.io/badge/university-HUST-red)](https://www.hust.edu.cn/)

## 项目简介

项目当前聚焦一个可执行的最小闭环：

1. 预处理 `dataset/normal` 并统一为 `224x224`。
2. 构建银屑病 vs 正常的二分类清单 `dataset/binary_manifest.csv`。
3. 下载两类预训练模型（ResNet50、SigLIP）。
4. 训练基础二分类分类器（ResNet50 / SigLIP）。
5. 在单样本上运行 SAMOO 对抗攻击并保存结果。

> 说明：当前代码以 `src/psorad` 为主实现，基于[phoenixwilliams/Black-Box-Sparse-Adversarial-Attack-via-Multi-Objective-Optimisation](https://github.com/phoenixwilliams/Black-Box-Sparse-Adversarial-Attack-via-Multi-Objective-Optimisation)项目进行重构。

## 目录结构（关键部分）

```text
dataset/
	normal/                    # 原始 normal 图像
	normal_224/                # 预处理后 normal 图像（运行 preprocess 后生成）
	psoriasis/                 # 银屑病图像
	psoriasis_metadata.csv     # 银屑病元数据（含 subtype_label）
	binary_manifest.csv        # 二分类清单（运行 preprocess 后生成）

model/
	pretrained_model/
		resnet/
		siglip/
	trained_classifier/
		resnet50/
		siglip/

src/psorad/
	preprocess/                # 数据预处理
	data/                      # Dataset / DataLoader
	models/                    # 下载与模型构建
	trainers/                  # 分类器训练
	attack/                    # SAMOO 攻击
```

## 环境准备

### 要求

- Python >= 3.12
- Linux / Windows（WSL）均可
- 建议有可用 NVIDIA GPU

### 安装（推荐 uv）

```bash
uv sync
```

## 代码运行流程

你可以用 `uv run main.py ...` 或 `psorad ...` 两种方式执行。

### 1) 预处理与 manifest 生成

注意，在运行之前，你需要先将数据集文件下载到相应位置。

```bash
uv run main.py preprocess \
	--dataset-root dataset \
	--metadata-csv dataset/psoriasis_metadata.csv \
	--normal-raw-dir dataset/normal \
	--normal-processed-dir dataset/normal_224 \
	--manifest-csv dataset/binary_manifest.csv \
	--image-size 224
```

输出：

- `dataset/normal_224/*.jpg`
- `dataset/binary_manifest.csv`

### 2) 下载预训练模型

```bash
uv run main.py download-models
```

默认下载到：

- `model/pretrained_model/resnet/resnet50_imagenet1k_v2.pth`
- `model/pretrained_model/siglip/`

目前已经默认使用hf-mirror镜像站进行模型的下载。

### 3) 训练二分类模型

#### ResNet50

```bash
uv run main.py train \
	--backbone resnet50 \
	--manifest-csv dataset/binary_manifest.csv \
	--epochs 3 \
	--batch-size 16 \
	--learning-rate 1e-4
```

#### SigLIP

```bash
uv run main.py train \
	--backbone siglip \
	--manifest-csv dataset/binary_manifest.csv \
	--epochs 3 \
	--batch-size 16 \
	--learning-rate 1e-4 \
	--freeze-siglip-backbone
```

输出 checkpoint：

- `model/trained_classifier/resnet50/best_binary_classifier.pt`
- `model/trained_classifier/siglip/best_binary_classifier.pt`

### 4) 运行 SAMOO 攻击

```bash
uv run main.py attack \
	--backbone resnet50 \
	--checkpoint model/trained_classifier/resnet50/best_binary_classifier.pt \
	--manifest-csv dataset/binary_manifest.csv \
	--sample-index 0
```

默认输出：

- `model/trained_classifier/<backbone>/samoo_result.npy`

## CLI 参数速查

### preprocess

- `--dataset-root`：数据集根目录（默认 `dataset`）
- `--metadata-csv`：银屑病元数据 CSV
- `--normal-raw-dir`：原始 normal 目录
- `--normal-processed-dir`：预处理输出目录
- `--manifest-csv`：清单输出 CSV
- `--image-size`：统一尺寸（默认 224）

### download-models

- 无必填参数，默认下载 ResNet50 与 SigLIP。

### train

- `--backbone`：`resnet50` 或 `siglip`（必填）
- `--manifest-csv`：训练清单
- `--epochs` / `--batch-size` / `--learning-rate`
- `--val-ratio`：验证集比例
- `--seed`：随机种子
- `--num-workers`：DataLoader worker 数
- `--freeze-siglip-backbone`：仅对 SigLIP 生效

### attack

- `--backbone`：`resnet50` 或 `siglip`（必填）
- `--checkpoint`：训练好的 checkpoint（必填）
- `--manifest-csv`：样本来源清单
- `--sample-index`：攻击样本索引
- `--save-path`：自定义结果路径
- `--seed`：随机种子

## 常见问题

### 1) `ModuleNotFoundError: transformers`

SigLIP 路径依赖 `transformers`。请确认环境安装了项目依赖，或先使用 ResNet50 流程。

### 2) `manifest 为空`

检查：

- `dataset/psoriasis_metadata.csv` 是否存在并含 `file_name`、`subtype_label` 两列。
- `file_name` 对应图像路径是否真实存在于 `dataset/` 下。
- `dataset/normal` 是否包含可识别图像文件。

### 3) checkpoint 加载失败

请确保 `--backbone` 与 checkpoint 对应一致（例如 ResNet50 checkpoint 不可用于 SigLIP）。

## 开发与贡献

- 贡献指南：见 [CONTRIBUTING.md](CONTRIBUTING.md)
- 项目入口：`main.py`
- 包入口：`psorad`（`pyproject.toml` 中 `project.scripts`）

## 整体规划

### 1. 面向医学影像的定向对抗攻击方法
- 研究生成人眼不可察觉且具有高迁移性的对抗扰动方法。
- 设计多目标攻击策略，在保证扰动微小的同时最大化分类器误判率。
- 通过分析攻击成功区域，定位模型决策中的脆弱环节，挖掘对诊断敏感的关键区域。

### 2. 基于对抗样本反馈的鲁棒分类器优化
- 提取模型在医学影像中的决策脆弱区域，作为空间注意力引导信号。
- 构建融合区域感知的对抗训练与特征一致性正则化机制。
- 在关键诊断区域增强特征稳定性，在非敏感区域适度松弛约束。

### 3. 攻击 - 防御协同演化的联合优化框架
- 设计动态交互的闭环优化系统，使攻击模块与防御模块相互驱动、协同演化。
- 引入多目标攻防机制，驱动攻击器与防御器在训练中协同演化。
- 最终形成既能精准识别病灶、又对抗扰具有强内在鲁棒性的诊断系统。

