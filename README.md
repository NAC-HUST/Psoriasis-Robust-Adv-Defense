# Psoriasis Robust Adv&Defense

面向银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究。

[![Status](https://img.shields.io/badge/status-in%20progress-yellow)](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
[![University](https://img.shields.io/badge/university-HUST-red)](https://www.hust.edu.cn/)

## 项目简介

项目当前聚焦一个可执行的最小闭环：

1. 预处理 `dataset/raw_data/<datadir>/` 并自动识别子目录类别。
2. 以“等比缩放 + 中心裁剪”统一为 `224x224`（不拉伸）。
3. 输出到 `dataset/processed_data/<datadir>/` 并生成 `class_manifest.csv`。
4. 下载两类预训练模型（ResNet50、SigLIP）。
5. 训练多分类分类器（ResNet50 / SigLIP）。
6. 在单样本上运行 SAMOO 对抗攻击并保存结果。

> 说明：当前代码以 `src/psorad` 为主实现，基于[phoenixwilliams/Black-Box-Sparse-Adversarial-Attack-via-Multi-Objective-Optimisation](https://github.com/phoenixwilliams/Black-Box-Sparse-Adversarial-Attack-via-Multi-Objective-Optimisation)项目进行重构。

## 目录结构（关键部分）

```text
dataset/
	raw_data/
		<datadir>/
			<class_a>/
			<class_b>/
	processed_data/
		<datadir>/               # 预处理输出目录
			<class_a>/
			<class_b>/
			class_manifest.csv

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
	--datadir psoriasis_normal \
	--image-size 224
```

要求原始数据目录结构如下：

```text
dataset/raw_data/psoriasis_normal/
	normal/
		img1.jpg
	psoriasis/
		img2.jpg
```

类别名直接取子目录名（如 `normal`、`psoriasis`）。

输出：

- `dataset/processed_data/psoriasis_normal/<class_name>/*.jpg`
- `dataset/processed_data/psoriasis_normal/class_manifest.csv`

### 2) 下载预训练模型

```bash
uv run main.py download-models
```

默认下载到：

- `model/pretrained_model/resnet/resnet50_imagenet1k_v2.pth`
- `model/pretrained_model/siglip/`

目前已经默认使用hf-mirror镜像站进行模型的下载。

### 3) 训练分类模型

#### ResNet50

```bash
uv run main.py train \
	--backbone resnet50 \
	--datadir psoriasis_normal \
	--modelname resnet50_psoriasis_v1.pt \
	--epochs 3 \
	--batch-size 16 \
	--learning-rate 1e-4
```

#### SigLIP

```bash
uv run main.py train \
	--backbone siglip \
	--datadir psoriasis_normal \
	--modelname siglip_psoriasis_v1.pt \
	--epochs 3 \
	--batch-size 16 \
	--learning-rate 1e-4 \
	--freeze-siglip-backbone
```

输出 checkpoint：

- 默认：`model/trained_classifier/<backbone>/best_classifier.pt`
- 若设置 `--modelname`：`model/trained_classifier/<backbone>/<modelname>`

### 4) 运行 SAMOO 攻击

```bash
uv run main.py attack \
	--backbone resnet50 \
	--checkpoint model/trained_classifier/resnet50/best_classifier.pt \
	--datadir psoriasis_normal \
	--sample-index 0
```

默认输出：

- `output/attack/<backbone>/<datadir>-<model_name>-<sample_index>/summary.txt`
- `output/attack/<backbone>/<datadir>-<model_name>-<sample_index>/summary.json`
- `output/attack/<backbone>/<datadir>-<model_name>-<sample_index>/attack_log.txt`
- `output/attack/<backbone>/<datadir>-<model_name>-<sample_index>/before_after_diff.png`
- `output/attack/<backbone>/<datadir>-<model_name>-<sample_index>/samoo_result.npy`（原始结果，默认保留）

运行 `attack` 时，终端会输出更详细的过程信息，包括：

- 当前使用的数据清单与 datadir、采样图像路径与原始标签
- 被攻击模型（backbone + checkpoint）与原图预测置信度（含 Top-K）
- SAMOO 进化过程（初始化种群、迭代进度、可行解数量、选择/交叉/变异后的查询量）
- 最终结果（是否攻击成功、查询次数、修改像素数、对抗样本前后置信度）

上述终端内容会同步写入 `attack_log.txt` 便于复盘。

如需指定导出目录并删除原始 `npy`：

```bash
uv run main.py attack \
	--backbone resnet50 \
	--checkpoint model/trained_classifier/resnet50/best_classifier.pt \
	--export-dir output/attack/attack_case_001 \
	--no-raw-npy
```

## CLI 参数速查

### preprocess

- `--dataset-root`：数据集根目录（默认 `dataset`）
- `--datadir`：数据集目录名，对应 `raw_data/<datadir>`（必填）
- `--raw-data-root`：原始数据根目录（默认 `<dataset-root>/raw_data`）
- `--processed-data-root`：处理后数据根目录（默认 `<dataset-root>/processed_data`）
- `--image-size`：统一尺寸（默认 224）

### download-models

- 无必填参数，默认下载 ResNet50 与 SigLIP。

### train

- `--backbone`：`resnet50` 或 `siglip`（必填）
- `--datadir`：默认读取 `dataset/processed_data/<datadir>/class_manifest.csv`
- `--manifest-csv`：手动指定清单（优先级高于 `--datadir`）
- `--modelname`：训练输出模型文件名（默认 `best_classifier.pt`）
- `--epochs` / `--batch-size` / `--learning-rate`
- `--val-ratio`：验证集比例
- `--seed`：随机种子
- `--num-workers`：DataLoader worker 数
- `--image-size`：输入尺寸（默认 224）
- `--freeze-siglip-backbone`：仅对 SigLIP 生效

### attack

- `--backbone`：`resnet50` 或 `siglip`（必填）
- `--checkpoint`：训练好的 checkpoint（必填）
- `--datadir`：默认读取 `dataset/processed_data/<datadir>/class_manifest.csv`
- `--manifest-csv`：手动指定清单（优先级高于 `--datadir`）
- `--sample-index`：攻击样本索引
- `--image-size`：攻击样本读取尺寸（默认 224）
- `--save-path`：自定义原始 `npy` 结果路径（默认在 `output/attack/<backbone>/<datadir>-<model_name>-<sample_index>/samoo_result.npy`）
- `--export-dir`：攻击文本与图像结果目录（默认在 `output/attack/<backbone>/<datadir>-<model_name>-<sample_index>/`）
- `--no-raw-npy`：不保留原始 `npy`
- `--eps` / `--iterations` / `--pop-size`：攻击强度核心参数（`--eps` 不传时按分辨率自适应，默认约为总像素数 2%，下限 128）
- `--p-size`：单步扰动幅度（默认 `0.25`）
- `--include-dist` + `--max-dist`：是否启用距离约束筛选
- `--seed`：随机种子

说明：在多分类任务中，攻击摘要会输出通用置信度字段（`prob_true_before/after`、`prob_pred_before/after`）。
`prob_class1_*` 仅为兼容旧版字段，不代表“psoriasis 概率”。

## 常见问题

### 1) `ModuleNotFoundError: transformers`

SigLIP 路径依赖 `transformers`。请确认环境安装了项目依赖，或先使用 ResNet50 流程。

### 2) `manifest 为空`

检查：

- `dataset/raw_data/<datadir>/` 下是否有至少一个类别子目录。
- 每个类别子目录内是否包含可识别图像文件（jpg/png/webp 等）。
- 是否正确传入了 `--datadir`。

### 3) checkpoint 加载失败

请确保 `--backbone` 与 checkpoint 对应一致（例如 ResNet50 checkpoint 不可用于 SigLIP）。

### 4) 攻击结果 `success=False`

这不代表代码错误，通常表示“在当前预算与约束下未找到可行扰动”。

建议：

- 增加搜索预算：`--iterations`、`--pop-size`
- 增大扰动空间：`--eps`、`--p-size`
- 关闭距离约束筛选（默认已关闭）：不要加 `--include-dist`
- 更换样本索引：有些样本本身更难攻击

## 攻击超参数解读（SAMOO）

- `eps`：允许修改的像素位置数量；越大越容易成功，但扰动更明显。
- `p-size`：每次像素扰动步长（输入空间，单位约等于像素归一化值）；越大越容易翻转。
- 说明：`p-size` 不是必须按 `1/255` 量级设置；在黑盒稀疏攻击里常用更大步长（如 `0.25~2.0`）以提高成功率。
- `iterations`：进化迭代轮数；越大搜索更充分但更慢。
- `pop-size`：每代候选解数量；越大多样性更好但计算更重。
- `pm`：变异概率；高一点有助于跳出局部最优。
- `pc`：交叉概率；调节父代信息重组强度。
- `zero-probability`：每个像素通道扰动为 0 的概率；越低表示扰动更“密”。
- `include-dist` + `max-dist`：是否用距离阈值筛可行解；用于“更隐蔽攻击”，但会降低成功率。

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

