# 项目结构

详细的项目代码组织说明。

## 🏗️ 目录树

```
Psoriasis-Robust-Adv-Defense/
│
├── src/psorad/                          # 核心源代码
│   ├── __init__.py
│   ├── cli.py                           # CLI 入口点
│   │
│   ├── preprocess/                      # 数据预处理模块
│   │   ├── __init__.py
│   │   └── processor.py                 # 图像处理、缩放、裁剪
│   │
│   ├── data/                            # 数据加载模块
│   │   ├── __init__.py
│   │   ├── dataset.py                   # PyTorch Dataset 类
│   │   └── dataloader.py                # 数据加载器配置
│   │
│   ├── models/                          # 模型构建与下载
│   │   ├── __init__.py
│   │   ├── downloader.py                # 模型下载逻辑
│   │   ├── resnet.py                    # ResNet50 包装
│   │   └── siglip.py                    # SigLIP 包装
│   │
│   ├── trainers/                        # 训练模块
│   │   ├── __init__.py
│   │   ├── base.py                      # 基础训练类
│   │   ├── resnet_trainer.py            # ResNet50 训练器
│   │   └── siglip_trainer.py            # SigLIP 训练器
│   │
│   └── attack/                          # 对抗攻击模块
│       ├── __init__.py
│       ├── samoo.py                     # SAMOO 算法核心
│       ├── objectives.py                # 多目标函数
│       └── runner.py                    # 攻击执行器
│
├── dataset/                             # 数据集目录
│   ├── raw_data/                        # 原始数据
│   │   ├── cifar10/
│   │   ├── imagenette/
│   │   └── psoriasis_normal/
│   │
│   └── processed_data/                  # 预处理后数据
│       ├── cifar10/
│       ├── imagenette/
│       └── psoriasis_normal/
│
├── model/                               # 模型存储目录
│   ├── pretrained_model/                # 预训练模型
│   │   ├── resnet/
│   │   │   └── resnet50_imagenet1k_v2.pth
│   │   └── siglip/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       └── preprocessor_config.json
│   │
│   └── trained_classifier/              # 训练好的分类器
│       ├── resnet50/
│       │   ├── best_classifier.pt
│       │   ├── latest_classifier.pt
│       │   └── training_log.csv
│       └── siglip/
│           └── ...
│
├── output/                              # 输出结果
│   ├── attack/                          # 攻击结果
│   │   ├── resnet50/
│   │   │   ├── sample_0/
│   │   │   │   ├── original_image.png
│   │   │   │   ├── adversarial_image.png
│   │   │   │   ├── perturbation.png
│   │   │   │   ├── predictions.json
│   │   │   │   └── metrics.json
│   │   │   └── summary.json
│   │   └── siglip/
│   │       └── ...
│   │
│   └── defence/                         # 防御结果（扩展）
│       └── ...
│
├── docs-src/                            # 文档源码（VitePress）
│   ├── .vitepress/
│   │   ├── config.ts
│   │   └── theme/
│   ├── index.md
│   ├── getting-started/
│   ├── documentation/
│   ├── research/
│   └── contributing/
├── docs/                                # 文档构建产物（GitHub Pages 发布目录）
│   ├── index.html
│   ├── assets/
│   └── .nojekyll
│
├── tests/                               # 单元测试
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_models.py
│   ├── test_trainers.py
│   └── test_attack.py
│
├── reb/                                 # 参考实现（来自原 repo）
│   ├── main.py
│   ├── ImageNetModels.py
│   ├── LossFunctions.py
│   ├── MODULE_RESPONSIBILITY_AND_CALL_FLOW.md
│   ├── moaa_core/
│   ├── losses/
│   └── models/
│
├── tools/                               # 工具脚本
│   ├── bak/                             # 备份脚本
│   └── ...
│
├── pyproject.toml                       # 项目配置
├── package.json                         # VitePress 构建配置
├── README.md                            # 项目首页
├── LICENSE                              # 许可证 (GPLv3)
├── CONTRIBUTING.md                      # 贡献指南
└── main.py                              # 主入口（开发用）
```

## 📦 模块说明

### src/psorad/cli.py

**职责**: CLI 命令行接口

```python
# 需要实现的命令
- preprocess(dataset_root, datadir, image_size)
- download_models(model, force)
- train(backbone, datadir, epochs, batch_size, ...)
- attack(backbone, checkpoint, datadir, sample_index, ...)
```

### src/psorad/preprocess/

**职责**: 数据预处理

**关键功能**:
- 图像加载（支持 JPG, PNG, etc）
- 等比缩放 + 中心裁剪
- 统一至 224×224 分辨率
- 自动生成 class_manifest.csv

### src/psorad/data/

**职责**: 数据加载和处理

**关键类**:
```python
class PsoriasisDataset(torch.utils.data.Dataset):
    """加载预处理后的图像"""
    
class DataModule:
    """统一的数据加载管理"""
```

### src/psorad/models/

**职责**: 模型构建和下载

**支持的模型**:
- ResNet50 (CNN) - 来自 torchvision
- SigLIP (Vision-Language) - 来自 HuggingFace

**关键函数**:
```python
def download_pretrained_models(model: str, force: bool = False)
def load_resnet50(pretrained: bool = True) -> torch.nn.Module
def load_siglip(pretrained: bool = True) -> torch.nn.Module
```

### src/psorad/trainers/

**职责**: 模型训练

**基类**:
```python
class BaseTrainer:
    def train_epoch(self)
    def validate(self)
    def save_checkpoint(self)
```

**具体实现**:
- `ResNet50Trainer` - ResNet50 训练器
- `SigLIPTrainer` - SigLIP 训练器（支持冻结/微调）

### src/psorad/attack/

**职责**: 对抗攻击

**核心算法**: SAMOO (Sparse Multi-Objective Optimization)

**关键类**:
```python
class SAMOOAttacker:
    """SAMOO 攻击实现"""
    
    def attack(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        label: int,
    ) -> tuple[torch.Tensor, dict]:
        """执行攻击，返回对抗样本和度量"""
```

## 🔄 数据流向

```
原始数据
    ↓
┌───────────────────┐
│ 数据预处理 (CLI) │
└───────────────────┘
    ↓
处理后数据
    ↓
┌───────────────────┐
│ 下载预训练模型   │ → 预训练权重
└───────────────────┘
    ↓
┌───────────────────┐
│ 训练分类器      │
└───────────────────┘ → 训练好的分类器
    ↓
┌───────────────────┐
│ SAMOO 对抗攻击    │
└───────────────────┘ → 对抗样本 + 度量
    ↓
分析与评估
```

## 🧪 测试层级

```
tests/
├── test_preprocess.py       # 数据预处理测试
├── test_models.py           # 模型加载和下载测试
├── test_trainers.py         # 训练器功能测试
└── test_attack.py           # 攻击算法测试
```

**测试范围**:
- ✅ 单元测试 (Unit Test)
- ✅ 集成测试 (Integration Test)
- ⏳ 端到端测试 (E2E Test) - 待完善

## 📊 配置文件

### pyproject.toml

定义项目依赖和元数据：

```toml
[project]
name = "psoriasis-robust-adv-defense"
version = "0.1.0"
requires-python = ">=3.12"

[project.scripts]
psorad = "psorad.cli:main"  # 定义 CLI 入口
```

### docs-src/.vitepress/config.ts

文档网站配置：

```ts
import { defineConfig } from 'vitepress'

export default defineConfig({
    lang: 'zh-CN',
    base: '/Psoriasis-Robust-Adv-Defense/'
})
```

## 🔧 开发规范

### 代码组织原则

1. **模块化**: 功能独立，接口清晰
2. **可扩展**: 易于添加新模型或攻击算法
3. **可测试**: 依赖注入，容易 mock
4. **文档化**: 添加 docstring 和类型注解

### 命名规范

- **类**: PascalCase (如 `ResNet50Trainer`)
- **函数**: snake_case (如 `preprocess_image`)
- **常量**: UPPER_SNAKE_CASE (如 `DEFAULT_IMAGE_SIZE = 224`)
- **私有**: 前缀下划线 (如 `_internal_helper`)

### 导入规则

```python
# 标准库
import os
from pathlib import Path

# 第三方
import torch
import numpy as np

# 项目内
from psorad.models import load_resnet50
```

---

**下一步**: 📚 [数据预处理](preprocessing.md) | 🎓 [模型训练](training.md)
