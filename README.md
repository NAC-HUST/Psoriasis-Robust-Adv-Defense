# Psoriasis Robust Adv&Defense

面向银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究。

[![Status](https://img.shields.io/badge/status-in%20progress-yellow)](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
[![University](https://img.shields.io/badge/university-HUST-red)](https://www.hust.edu.cn/)

## 项目简介

项目实现了一个完整的对抗攻防闭环：

1. 在 `dataset/split_data/` 格式的数据集上**训练**多分类分类器（ResNet50/SigLIP）。
2. 使用 **SAMOO** 稀疏黑盒进化攻击评估模型鲁棒性。
3. 从频域、空间、置信度三个维度进行**脆弱性分析**。
4. 使用**区域感知高低频净化防御**（freq_region_purify）抵御稀疏攻击。

> 说明：当前代码以 `src/psorad` 为主实现，基于[phoenixwilliams/Black-Box-Sparse-Adversarial-Attack-via-Multi-Objective-Optimisation](https://github.com/phoenixwilliams/Black-Box-Sparse-Adversarial-Attack-via-Multi-Objective-Optimisation)项目进行重构。

## 目录结构（关键部分）

```text
dataset/
    split_data/
        <dataset>/
            images/
                train/
                val/
                test/
            labels/
                train_labels.csv       # one-hot 标签格式
                val_labels.csv
                test_labels.csv

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
    models/                    # 模型构建、工厂、训练
    attack/                    # SAMOO 稀疏攻击
    eval/                      # 评估与脆弱性分析
    defense/                   # 区域感知净化防御
    config.py                  # 配置 dataclass 与 TOML loader
    cli.py                     # CLI 入口（6 个子命令）

tools/
    batch_parallel_attack.py   # 批量并行攻击
    batch_train_splitdata.py   # 统一训练 5 个 split_data 数据集
    generate_attack_manifest.py# split_data → 攻击 manifest 桥接
    batch_run_attacks.py       # 批量攻击编排与基线汇总

output/                        # 实验产物（gitignored）
    batch_attack/
    batch_attack_minpx/
    eval/
    defense/
    attack_results/
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

类别名直接取子目录名（如 `normal`、`psoriasis`）。输出到 `dataset/processed_data/`。

### 2) 下载预训练模型

```bash
uv run main.py download-models
```

默认下载到：

- `model/pretrained_model/resnet/resnet50_imagenet1k_v2.pth`
- `model/pretrained_model/siglip/`

目前已经默认使用hf-mirror镜像站进行模型的下载。

### 3) 训练分类模型

#### 单数据集训练（manifest 方式）

```bash
uv run main.py train \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --modelname resnet50_psoriasis_v1.pt \
    --epochs 3 \
    --batch-size 16 \
    --learning-rate 1e-4
```

#### 实验配置方式（split_data 格式）

```bash
uv run main.py train \
    --experiment-config configs/dataset_config/baseline_dataset.toml \
    --backbone resnet50 \
    --datadir psoriasis_normal
```

#### 批量训练 5 个 split_data 数据集

```bash
.venv/bin/python tools/batch_train_splitdata.py
```

统一参数：epochs=5, batch_size=32, lr=1e-4, seed=42, image_size=224。

输出 checkpoint 到 `model/trained_classifier/resnet50/resnet50_<dataset>.pt`。

### 4) 运行 SAMOO 攻击

#### 单样本攻击

```bash
uv run main.py attack \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
    --manifest-csv output/manifests/<dataset>_manifest.csv \
    --datadir <dataset> \
    --sample-index 0
```

#### 生成攻击 manifest（split_data → 攻击格式）

```bash
.venv/bin/python tools/generate_attack_manifest.py
```

输出 `output/manifests/<dataset>_manifest.csv`（含 `file_path, class_idx, class_name, split`）。

#### 批量并行攻击

```bash
.venv/bin/python tools/batch_parallel_attack.py \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/resnet50_<dataset>.pt \
    --manifest-csv output/manifests/<dataset>_manifest.csv \
    --datadir <dataset> \
    --attack-split val \
    --max-samples 20 \
    --workers 2
```

默认输出根目录为 `output/batch_attack/<backbone>/<datadir>/`。报告含：

- `batch_report.json`（完整结构化报告）
- `batch_report.csv`（逐样本统计）
- `batch_report.md`（人类可读总结）

#### 批量攻击编排（5 数据集基线）

```bash
.venv/bin/python tools/batch_run_attacks.py --max-samples 20 --workers 2
```

输出 `output/eval/baseline_comparison.md`（ASR / RobustAcc 对比表）。

### 5) 评估与脆弱性分析

`evaluate` 子命令对某个划分做干净前向推理并计算指标，读取批量攻击报告汇总鲁棒性（ASR / robust_acc），并自动进行脆弱性分析。

```bash
uv run main.py evaluate \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/resnet50_<dataset>.pt \
    --manifest-csv output/manifests/<dataset>_manifest.csv \
    --split val \
    --batch-report output/batch_attack/resnet50/<dataset>/batch_report.json \
    --grid-size 7 \
    --high-freq-cutoff 0.25 \
    --save-visuals \
    --output-dir output/eval/<dataset>
```

也可用配置文件驱动：

```bash
uv run main.py evaluate --config configs/eval_config/baseline_eval.toml
```

说明：

- Clean 指标：accuracy、macro precision/recall/f1、per-class f1、AUC、混淆矩阵。
- Robust 指标：ASR、robust_acc、queries/l2/linf/modified_pixels 的 mean/median/p90、per-class 成功率。
- **脆弱性分析**：成功率×代价关联、空间脆弱性热图（7×7 网格）、频域高频能量占比（2D FFT）、中心vs边缘扰动能量比。
- 输出 `output/eval/<dataset>/eval_report.{json,md}` + `vuln_spatial_heatmap.png`。

### 6) 净化防御

高低频 + 区域感知输入净化防御（freq_region_purify），无需重训练。

```bash
uv run main.py defend \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/resnet50_<dataset>.pt \
    --batch-report output/batch_attack/resnet50/<dataset>/batch_report.json \
    --window 3 \
    --threshold 0.08 \
    --dilation 1 \
    --low-freq-cutoff 0.25 \
    --low-freq fft \
    --output-dir output/defense
```

也可用配置文件驱动：

```bash
uv run main.py defend --config configs/defence_config/example_defence.toml
```

说明：

- **区域感知掩码**：像素偏离局部中值 > threshold 即标记为可疑。
- **低频重建**：FFT 低通滤波或局部中值重建可疑区域。
- **只替换可疑像素**，保留干净区域 → clean acc 下降为 0。
- 输出 `output/defense/<backbone>/<dataset>/defense_report.{json,md}`。
- 指标：clean_acc_drop、recovery_rate（攻击成功样本恢复率）、robust_acc_after。

## CLI 参数速查

### preprocess

- `--dataset-root`：数据集根目录（默认 `dataset`）
- `--datadir`：数据集目录名，对应 `raw_data/<datadir>`（必填）
- `--image-size`：统一尺寸（默认 224）

### download-models

- 无必填参数，默认下载 ResNet50 与 SigLIP。

### train

- `--backbone`：`resnet50` 或 `siglip`（必填）
- `--datadir` / `--manifest-csv` / `--experiment-config`：数据来源（`--experiment-config` 优先）
- `--epochs` / `--batch-size` / `--learning-rate` / `--seed`
- `--val-ratio` / `--num-workers` / `--image-size`
- `--modelname`：输出模型文件名
- `--freeze-siglip-backbone`

### attack

- `--backbone` / `--checkpoint`：必填
- `--datadir` / `--manifest-csv`：数据来源
- `--attack-split`：`all` / `train` / `val`（默认 `val`）
- `--sample-index`：攻击样本索引
- `--eps` / `--iterations` / `--pop-size` / `--query-budget`：攻击强度（不传时自动预设）
- `--export-dir`：输出目录
- 其余超参数：`--pc` / `--pm` / `--pm-end` / `--p-size` / `--zero-probability` / `--seed`

### evaluate

- `--config`：TOML 配置（优先于显式参数）
- `--backbone` / `--checkpoint` / `--manifest-csv`
- `--split`：`all` / `train` / `val`
- `--batch-report`：批量攻击报告路径（鲁棒性 + 脆弱性来源）
- `--grid-size`：空间网格划分（默认 7）
- `--high-freq-cutoff`：高频截止比例（默认 0.25）
- `--save-visuals`：保存空间热图
- `--output-dir` / `--report-name`
- `--skip-clean`：跳过前向推理，仅汇总

### defend

- `--config`：TOML 配置（优先于显式参数）
- `--backbone` / `--checkpoint` / `--batch-report`：必填
- `--window`：局部中值窗口（默认 3）
- `--threshold`：可疑像素阈值（默认 0.08）
- `--dilation`：区域膨胀（默认 1）
- `--low-freq-cutoff`：FFT 截止比例（默认 0.25）
- `--low-freq`：重建方式，`fft` / `median`
- `--output-dir` / `--report-name`

### tools/batch_parallel_attack.py

- `--backbone` / `--checkpoint` / `--manifest-csv`：必填
- `--attack-split` / `--sample-indices` / `--max-samples`
- `--workers`：并行进程数
- `--output-root` / `--run-name`
- 其余 SAMOO 超参数同 `attack` 子命令。

## 实验基线（参考）

5 个 `dataset/split_data/` 数据集统一训练 + 攻击基线：

| 数据集 | 类别 | Val Acc | ASR | Robust Acc |
|--------|------|---------|-----|------------|
| psoriasis224_2c | 2 | 99.82% | 15% | 85% |
| cifar32_10c | 10 | 88.13% | 100% | 0% |
| imagenette224_10c | 10 | 98.80% | 65% | 35% |
| dermamnist224_7c | 7 | 83.22% | 90% | 10% |
| milk10k_11c | 11 | 65.72% | 95% | 5% |

脆弱性分析表明：SAMOO 稀疏攻击扰动约 **95% 能量位于高频带**、**~73% 在图像边缘区域**。

净化防御（freq_region_purify）恢复率 68–100%，**clean acc 下降为 0**。

详情见 `output/eval/<dataset>/eval_report.md` 和 `output/defense/<backbone>/<dataset>/defense_report.md`。

## 攻击超参数解读（SAMOO）

- `eps`：允许修改的像素位置数量；越大越容易成功，但扰动更明显。
- `query-budget`：总查询预算；推荐优先固定预算，再调 `pop-size` 与 `iterations`。
- `pm` / `pm-end`：变异率起止值；算法会从 `pm` 逐步退火到 `pm-end`。
- `p-size`：每次像素扰动步长；不一定要 1/255 量级，黑盒稀疏攻击常用 `0.25~2.0`。
- `iterations` / `pop-size`：进化迭代轮数和种群大小。
- `zero-probability`：每个像素通道扰动为 0 的概率；越低表示扰动更"密"。
- `include-dist` + `max-dist`：是否用距离阈值筛可行解；会降低成功率。

## 常见问题

### 1) `ModuleNotFoundError: transformers`

SigLIP 路径依赖 `transformers`。请确认环境安装了项目依赖，或先使用 ResNet50 流程。

### 2) checkpoint 加载失败

请确保 `--backbone` 与 checkpoint 对应一致（如 ResNet50 checkpoint 不可用于 SigLIP）。

### 3) 攻击结果 `success=False`

这不代表代码错误，通常表示"在当前预算与约束下未找到可行扰动"。建议：

- 增加搜索预算：`--iterations`、`--pop-size`
- 增大扰动空间：`--eps`、`--p-size`
- 关闭距离约束筛选：不要加 `--include-dist`
- 更换样本索引：有些样本本身更难攻击

## 开发与贡献

- 贡献指南：见 [CONTRIBUTING.md](CONTRIBUTING.md)
- 项目入口：`main.py`
- 包入口：`psorad`（`pyproject.toml` 中 `project.scripts`）

## 整体规划 · WIP

### 1. 面向医学影像的定向对抗攻击方法
研究生成人眼不可察觉且具有高迁移性的对抗扰动方法。设计多目标攻击策略，在保证扰动微小的同时最大化分类器误判率。通过分析攻击成功区域，定位模型决策中的脆弱环节，挖掘对诊断敏感的关键区域。

### 2. 基于对抗样本反馈的鲁棒分类器优化
提取模型在医学影像中的决策脆弱区域，作为空间注意力引导信号。构建融合区域感知的对抗训练与特征一致性正则化机制。

### 3. 攻击-防御协同演化的联合优化框架
设计动态交互的闭环优化系统，使攻击模块与防御模块相互驱动、协同演化。引入多目标攻防机制，驱动攻击器与防御器在训练中协同演化。
