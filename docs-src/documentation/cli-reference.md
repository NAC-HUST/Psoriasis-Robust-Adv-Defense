# CLI 参考

完整的命令行界面参考文档。

## 总体命令

```bash
uv run main.py [COMMAND] [OPTIONS]
```

或使用已安装的 CLI：

```bash
psorad [COMMAND] [OPTIONS]
```

## 子命令列表

| 命令 | 说明 | 优先级 |
|------|------|--------|
| `preprocess` | 数据预处理 | ⭐ 必需 |
| `download-models` | 下载预训练模型 | ⭐ 必需 |
| `train` | 训练分类器 | ⭐ 必需 |
| `attack` | 执行对抗攻击 | ⭐ 必需 |

---

## 📥 preprocess

数据预处理和规范化。

### 用法

```bash
uv run main.py preprocess [OPTIONS]
```

### 选项

```
--dataset-root TEXT      数据集根目录 [默认: dataset]
--datadir TEXT           数据集名称 [必需]
--image-size INTEGER     目标图像尺寸 (像素) [默认: 224]
--help                   显示帮助信息
```

### 示例

```bash
# 基本用法
uv run main.py preprocess --datadir psoriasis_normal

# 自定义图像大小
uv run main.py preprocess \
    --datadir imagenette \
    --image-size 256

# 自定义数据集根目录
uv run main.py preprocess \
    --dataset-root /path/to/data \
    --datadir my_dataset
```

### 输出

```
dataset/processed_data/<datadir>/
├── <class_1>/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── <class_2>/
│   └── ...
└── class_manifest.csv
```

### 理解 class_manifest.csv

| 列 | 类型 | 说明 |
|----|------|------|
| `class_id` | int | 类别 ID |
| `class_name` | str | 类别名称 |
| `num_samples` | int | 该类样本数 |
| `split` | str | 数据集分割 (train/val/test) |

---

## 📦 download-models

下载预训练模型。

### 用法

```bash
uv run main.py download-models [OPTIONS]
```

### 选项

```
--model TEXT    指定模型 (resnet50/siglip/all) [默认: all]
--force         强制重新下载 [默认: False]
--help          显示帮助信息
```

### 示例

```bash
# 下载所有模型
uv run main.py download-models

# 仅下载 ResNet50
uv run main.py download-models --model resnet50

# 仅下载 SigLIP
uv run main.py download-models --model siglip

# 强制重新下载
uv run main.py download-models --force
```

### 输出

```
model/pretrained_model/
├── resnet/
│   └── resnet50_imagenet1k_v2.pth         (~100MB)
└── siglip/
    ├── config.json
    ├── model.safetensors                  (~500MB)
    ├── preprocessor_config.json
    └── ...
```

::: tip 网络问题排查
如果下载缓慢或超时：

```bash
# 使用 HF Mirror (中国地区)
export HF_ENDPOINT=https://hf-mirror.com
uv run main.py download-models
```
:::

---

## 🎓 train

训练分类器。

### 用法

```bash
uv run main.py train [OPTIONS]
```

### 核心选项

```
--backbone TEXT              骨干网络 [必需]
                            可选值: resnet50, siglip
--datadir TEXT               数据集名称 [必需]
--epochs INTEGER             训练轮数 [默认: 10]
--batch-size INTEGER         批大小 [默认: 32]
--learning-rate FLOAT        学习率 [默认: 1e-4]
--device TEXT                计算设备 [默认: cuda]
                            可选值: cpu, cuda, mps
--modelname TEXT             保存模型文件名 [可选]
--help                       显示帮助信息
```

### SigLIP 特殊选项

```
--freeze-siglip-backbone     冻结 SigLIP 骨干网络
                            只训练分类头 [默认: False]
```

### 示例

#### ResNet50 基本训练

```bash
uv run main.py train \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --epochs 10 \
    --batch-size 16
```

#### ResNet50 高级配置

```bash
uv run main.py train \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --modelname resnet50_v2.pt \
    --device cuda
```

#### SigLIP (冻结骨干)

```bash
uv run main.py train \
    --backbone siglip \
    --datadir psoriasis_normal \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --freeze-siglip-backbone
```

#### SigLIP (微调骨干)

```bash
uv run main.py train \
    --backbone siglip \
    --datadir psoriasis_normal \
    --epochs 10 \
    --batch-size 4 \
    --learning-rate 1e-5
```

### 输出

```
model/trained_classifier/
├── resnet50/
│   ├── best_classifier.pt        # 最佳检查点
│   ├── latest_classifier.pt      # 最新检查点
│   └── training_log.csv          # 训练日志
└── siglip/
    └── ...
```

### 训练日志

训练过程中会输出日志，包括：

```
Epoch [1/10], Step [10/100]
Loss: 0.5234, Train Acc: 0.7542
Val Loss: 0.4123, Val Acc: 0.8123
Saved best checkpoint: model/trained_classifier/resnet50/best_classifier.pt
```

::: tip 显存不足?
```bash
# 减小 batch-size
uv run main.py train \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --batch-size 8
```
:::

---

## ⚔️ attack

执行对抗攻击（SAMOO）。

### 用法

```bash
uv run main.py attack [OPTIONS]
```

### 核心选项

```
--backbone TEXT              骨干网络 [必需]
--checkpoint TEXT            分类器权重路径 [必需]
--datadir TEXT               数据集名称 [必需]
--sample-index INTEGER       目标样本索引 [默认: 0]
--device TEXT                计算设备 [默认: cuda]
--help                       显示帮助信息
```

### SAMOO 特殊选项

```
--num-objectives INTEGER     目标数量 [默认: 2]
--population-size INTEGER    种群大小 [默认: 100]
--generations INTEGER        代数 [默认: 100]
--target-class INTEGER       目标类别 [可选]
--targeted                   有目标攻击模式 [默认: False]
```

### 示例

#### 基本对抗攻击（无目标）

```bash
uv run main.py attack \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
    --datadir psoriasis_normal \
    --sample-index 0
```

#### 多样本攻击

```bash
# 攻击前 5 个样本
for i in {0..4}; do
    uv run main.py attack \
        --backbone resnet50 \
        --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
        --datadir psoriasis_normal \
        --sample-index $i
done
```

#### 有目标攻击

```bash
uv run main.py attack \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
    --datadir psoriasis_normal \
    --sample-index 0 \
    --targeted \
    --target-class 1  # 攻击目标类别
```

#### 自定义 SAMOO 参数

```bash
uv run main.py attack \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
    --datadir psoriasis_normal \
    --population-size 200 \
    --generations 150 \
    --sample-index 0
```

#### SigLIP 攻击

```bash
uv run main.py attack \
    --backbone siglip \
    --checkpoint model/trained_classifier/siglip/best_classifier.pt \
    --datadir psoriasis_normal \
    --sample-index 0
```

### 输出

```
output/attack/resnet50/
└── sample_0/
    ├── original_image.png          # 原始图像
    ├── adversarial_image.png       # 对抗样本
    ├── perturbation.png            # 扰动可视化
    ├── predictions.json            # 预测结果
    ├── metrics.json                # 攻击指标
    └── optimization_history.csv    # 优化历史
```

#### predictions.json 格式

```json
{
    "original": {
        "predicted_class": 0,
        "confidence": 0.95,
        "all_probs": [0.95, 0.05]
    },
    "adversarial": {
        "predicted_class": 1,
        "confidence": 0.87,
        "all_probs": [0.13, 0.87]
    }
}
```

#### metrics.json 格式

```json
{
    "success": true,
    "sparsity": 0.02,
    "perturbation_l2": 15.3,
    "perturbation_linf": 25,
    "num_iterations": 87,
    "execution_time_seconds": 45.2
}
```

---

## 🔍 调试选项

所有命令都支持以下调试选项：

```bash
uv run main.py [COMMAND] --verbose    # 详细输出
uv run main.py [COMMAND] --seed 42    # 固定随机种子
```

---

## 📊 性能优化建议

### GPU 加速

```bash
# 确保 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"

# 使用 GPU
uv run main.py train --device cuda --batch-size 32
```

### 批处理

```bash
# 批量处理 10 个样本
for i in {0..9}; do
    uv run main.py attack \
        --backbone resnet50 \
        --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
        --datadir psoriasis_normal \
        --sample-index $i &
done
wait
```

### 内存优化

```bash
# 减小 batch-size
uv run main.py train --batch-size 8

# 使用梯度累积
uv run main.py train --batch-size 8 --gradient-accumulation-steps 4
```

---

## 🆘 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `Checkpoint not found` | 模型路径不对 | 检查 `--checkpoint` 参数 |
| `CUDA out of memory` | 显存不足 | 减小 `--batch-size` |
| `datadir not found` | 数据集不存在 | 运行 `preprocess` 或检查 `--datadir` |
| `Model download failed` | 网络问题 | 使用 HF Mirror 或手动下载 |

---

## 📚 完整参考链接

- [安装指南](../getting-started/installation.md)
- [快速开始](../getting-started/quickstart.md)
- [技术方案](../research/methodology.md)
- [算法详解](../research/algorithm.md)
