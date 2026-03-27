# 快速开始

⏱️ **预计时间**: 5-10 分钟

## 🎯 5分钟快速体验

假设已完成 [安装指南](installation.md)，按以下步骤快速体验：

### 步骤 1️⃣ 激活环境

```bash
# 进入项目目录
cd Psoriasis-Robust-Adv-Defense

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows PowerShell
```

### 步骤 2️⃣ 准备数据

如果已有数据集，将其放在 `dataset/raw_data/<datadir>/` 目录，目录结构如下：

```
dataset/raw_data/psoriasis_normal/
├── normal/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── psoriasis/
    ├── img3.jpg
    ├── img4.jpg
    └── ...
```

然后预处理数据：

```bash
uv run main.py preprocess \
    --dataset-root dataset \
    --datadir psoriasis_normal \
    --image-size 224
```

✅ **输出**: `dataset/processed_data/psoriasis_normal/`

### 步骤 3️⃣ 下载预训练模型

```bash
uv run main.py download-models
```

!!! note "首次下载可能需要几分钟"
    - ResNet50: ~100MB
    - SigLIP: ~500MB
    
    由于网络原因，如果下载超时，可设置镜像源：
    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    uv run main.py download-models
    ```

✅ **输出**: `model/pretrained_model/`

### 步骤 4️⃣ 训练分类器

=== "ResNet50 (推荐)"

    ```bash
    uv run main.py train \
        --backbone resnet50 \
        --datadir psoriasis_normal \
        --epochs 3 \
        --batch-size 16 \
        --learning-rate 1e-4
    ```
    
    ⏱️ **预计时间**: 5-10 分钟（带 GPU）

=== "SigLIP (Vision-Language)"

    ```bash
    uv run main.py train \
        --backbone siglip \
        --datadir psoriasis_normal \
        --epochs 3 \
        --batch-size 16 \
        --learning-rate 1e-4 \
        --freeze-siglip-backbone
    ```
    
    ⏱️ **预计时间**: 10-15 分钟（带 GPU）

✅ **输出**: `model/trained_classifier/<backbone>/best_classifier.pt`

### 步骤 5️⃣ 执行对抗攻击

```bash
uv run main.py attack \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
    --datadir psoriasis_normal \
    --sample-index 0
```

✅ **输出**: `output/attack/resnet50/`

## 📊 理解输出结果

运行完对抗攻击后，输出目录结构如下：

```
output/attack/resnet50/
├── sample_0/
│   ├── original_image.png       # 原始图像
│   ├── adversarial_image.png    # 对抗样本
│   ├── perturbation.png         # 扰动可视化
│   ├── predictions.json         # 预测结果
│   └── metrics.json             # 攻击指标
└── summary.json                 # 整体汇总
```

### 关键指标说明

=== "成功率 (Success Rate)"
    模型在对抗样本上做出错误预测的比例。值越高攻击效果越好。
    $$\text{SR} = \frac{\text{错误预测数}}{\text{总样本数}} \times 100\%$$

=== "稀疏性 (Sparsity)"
    被修改的像素比例。值越低攻击越稀疏。
    $$\text{稀疏性} = \frac{\text{修改像素数}}{H \times W \times C} \times 100\%$$

=== "扰动强度 (Perturbation)"
    修改的平均幅度，用 $\ell_2$ 范数或 $\ell_\infty$ 范数衡量。

## 🔧 常见命令参考

### 查看所有命令

```bash
uv run main.py --help
```

### 查看子命令帮助

```bash
uv run main.py preprocess --help
uv run main.py download-models --help
uv run main.py train --help
uv run main.py attack --help
```

### 训练多个模型

```bash
# 顺序训练 ResNet50 和 SigLIP
uv run main.py train --backbone resnet50 --datadir psoriasis_normal
uv run main.py train --backbone siglip --datadir psoriasis_normal
```

### 对比不同模型

```bash
# ResNet50 对抗攻击
uv run main.py attack --backbone resnet50 --datadir psoriasis_normal

# SigLIP 对抗攻击（对标测试）
uv run main.py attack --backbone siglip --datadir psoriasis_normal
```

### 批量处理多个样本

```bash
for i in {0..9}; do
    uv run main.py attack \
        --backbone resnet50 \
        --datadir psoriasis_normal \
        --sample-index $i
done
```

## 📚 完整参数说明

### preprocess 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--dataset-root` | str | `dataset` | 数据集根目录 |
| `--datadir` | str | - | 数据集名称（必需） |
| `--image-size` | int | 224 | 目标图像尺寸 |

### train 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--backbone` | str | - | 骨干网络：`resnet50` 或 `siglip`（必需） |
| `--datadir` | str | - | 数据集名称（必需） |
| `--epochs` | int | 10 | 训练轮数 |
| `--batch-size` | int | 32 | 批大小 |
| `--learning-rate` | float | 1e-4 | 学习率 |
| `--device` | str | `cuda` | 计算设备：`cpu` 或 `cuda` |
| `--freeze-siglip-backbone` | - | False | 冻结 SigLIP 骨干（SigLIP 特有） |

### attack 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--backbone` | str | - | 骨干网络（必需） |
| `--checkpoint` | str | - | 分类器权重路径（必需） |
| `--datadir` | str | - | 数据集名称（必需） |
| `--sample-index` | int | 0 | 目标样本索引 |
| `--device` | str | `cuda` | 计算设备 |

详见 [CLI 参考](../documentation/cli-reference.md)

## 🧪 运行示例

### 示例 1: 完整工作流（已有数据）

```bash
#!/bin/bash

# 激活环境
source .venv/bin/activate

# 数据预处理
uv run main.py preprocess --datadir cifar10

# 下载模型
uv run main.py download-models

# 训练 ResNet50
uv run main.py train \
    --backbone resnet50 \
    --datadir cifar10 \
    --epochs 5

# 对抗攻击
uv run main.py attack \
    --backbone resnet50 \
    --datadir cifar10
```

### 示例 2: 对比两个模型

```bash
#!/bin/bash

source .venv/bin/activate

# 准备数据
uv run main.py preprocess --datadir imagenette
uv run main.py download-models

# 训练两个模型
echo "Training ResNet50..."
uv run main.py train --backbone resnet50 --datadir imagenette

echo "Training SigLIP..."
uv run main.py train --backbone siglip --datadir imagenette

# 对比攻击
echo "Testing ResNet50..."
uv run main.py attack --backbone resnet50 --datadir imagenette

echo "Testing SigLIP..."
uv run main.py attack --backbone siglip --datadir imagenette

# 分析结果
echo "Results saved in output/attack/"
```

## 🐛 快速故障排查

| 问题 | 解决方案 |
|------|--------|
| `ModuleNotFoundError` | 检查虚拟环境是否激活 |
| `CUDA out of memory` | 减小 `--batch-size` |
| 数据加载失败 | 检查 `--datadir` 和数据格式 |
| 下载超时 | 设置 `export HF_ENDPOINT=https://hf-mirror.com` |
| GPU 未检测到 | 运行 `nvidia-smi` 检查驱动 |

## 📖 后续阅读

- 📚 [完整 CLI 参考](../documentation/cli-reference.md)
- 🔬 [技术方案详解](../research/methodology.md)
- 🤝 [贡献指南](../contributing/guide.md)

---

❓ **有问题？** 检查 [安装指南](installation.md) 或 [提交 Issue](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/issues)

✅ **准备好了？** 开始体验吧！
