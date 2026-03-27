---
layout: home

hero:
    name: "Psoriasis Robust Adv&Defense"
    text: "银屑病医学影像的对抗鲁棒性研究"
    tagline: "基于 SAMOO 的攻防协同框架，支持 ResNet50 与 SigLIP 的统一评估与训练。"
    actions:
        - theme: brand
            text: 快速开始
            link: /getting-started/quickstart
        - theme: alt
            text: GitHub
            link: https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense

features:
    - title: 端到端闭环
        details: 从数据预处理、模型下载到分类训练与对抗攻击，形成完整实验链路。
    - title: 多模型统一接口
        details: 同时支持 ResNet50 与 SigLIP，在同一 CLI 下完成训练和攻击对比。
    - title: 医学场景导向
        details: 面向银屑病影像诊断鲁棒性问题，强调可复现和工程可落地。
---

# 🏥 Psoriasis Robust Adv&Defense

> 银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究

[![Status](https://img.shields.io/badge/status-in%20progress-yellow)](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/blob/main/LICENSE)
[![University](https://img.shields.io/badge/university-HUST-red)](https://www.hust.edu.cn/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

## 🎯 项目概览

本项目致力于研究**对抗攻击**对医学影像分类模型的影响，并提出系统的**防御机制**，以提高银屑病诊断的鲁棒性和可信度。

### 核心研究内容

- **对抗攻击**: 采用 SAMOO（Sparse Multi-Objective Optimization）算法生成稀疏对抗样本
- **多模态学习**: 集成 ResNet50 (CNN) 与 SigLIP (Vision-Language) 模型
- **医学应用**: 专注银屑病医学影像的可靠诊断
- **鲁棒性增强**: 开发防御策略提升模型对对抗样本的抵抗力

## ✨ 主要特性

::: details 🚀 端到端流程
从数据预处理、模型下载，到训练分类器，再到对抗攻击，**一键完成全流程**。

```bash
uv run main.py preprocess --datadir psoriasis_normal
uv run main.py download-models
uv run main.py train --backbone resnet50 --epochs 3
uv run main.py attack --backbone resnet50
```
:::

::: details 🤖 多架构支持
灵活支持 **ResNet50** 与 **SigLIP** 骨干网络，可任意组合训练和攻击。

- ResNet50: CNNs，快速高效
- SigLIP: Vision-Language，0-shot 能力
:::

::: details 📊 完整数据集
支持 **CIFAR-10**、**ImageNette**、**银屑病患者数据**，提供自动预处理管道。

- 等比缩放 + 中心裁剪
- 统一 224×224 分辨率
- 自动类别识别
:::

::: details 🎓 学术输出
推进医学影像诊断的对抗鲁棒性研究，支持学术发表和开源贡献。
:::

## 🏗️ 项目架构

```
Psoriasis-Robust-Adv-Defense/
├── src/psorad/              # 核心实现模块
│   ├── preprocess/          # 数据预处理
│   ├── data/                # 数据加载器
│   ├── models/              # 模型构建与下载
│   ├── trainers/            # 分类器训练
│   └── attack/              # SAMOO 对抗攻击
├── dataset/                 # 数据目录
│   ├── raw_data/            # 原始数据
│   └── processed_data/      # 处理后数据
├── model/                   # 模型存储
│   ├── pretrained_model/    # 预训练模型
│   └── trained_classifier/  # 训练好的分类器
├── docs/                    # 文档（本网站）
└── tests/                   # 单元测试
```

## 📖 快速导航

| 文档 | 描述 |
|------|------|
| [👋 安装指南](getting-started/installation.md) | 环境配置和依赖安装 |
| [⚡ 快速开始](getting-started/quickstart.md) | 5分钟快速上手 |
| [📚 API 参考](documentation/cli-reference.md) | 完整命令行参考 |
| [🔬 技术方案](research/methodology.md) | 研究方法与算法详解 |
| [🤝 贡献指南](contributing/guide.md) | 参与开发与贡献 |

## 🛠️ 技术栈

| 组件 | 用途 | 版本 |
|------|------|------|
| **PyTorch** | 深度学习框架 | 2.9.0 |
| **Transformers** | 预训练模型库 | ≥4.57.0 |
| **TorchVision** | 计算机视觉工具 | 0.24.0 |
| **NumPy** | 数值计算 | ≥2.4.3 |
| **Pandas** | 数据处理 | ≥3.0.1 |
| **Pillow** | 图像处理 | ≥12.1.1 |

## 🚀 立即开始

### 安装（推荐使用 uv）

```bash
git clone https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense.git
cd Psoriasis-Robust-Adv-Defense
uv sync
```

### 预处理数据

```bash
uv run main.py preprocess \
    --dataset-root dataset \
    --datadir psoriasis_normal \
    --image-size 224
```

### 训练分类器

```bash
uv run main.py train \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --epochs 3 \
    --batch-size 16
```

### 执行对抗攻击

```bash
uv run main.py attack \
    --backbone resnet50 \
    --datadir psoriasis_normal
```

👉 **更多详情** 请参考 [快速开始指南](getting-started/quickstart.md)

## 📚 学术信息

### 研究成果

**论文标题**: 银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法

**作者**: Cao WJ, Cheng Q, Li YX, Feng ZY, Li YX

**机构**: 华中科技大学 (HUST)

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request！请参考 [贡献指南](contributing/guide.md) 了解详情。

### 代码质量检查

提交前请运行：

```bash
ruff check .           # 代码风格检查
mypy src              # 类型检查
pytest                # 单元测试
```

## 📄 许可证

本项目采用 [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) 开源许可证。

## 📧 联系方式

- **GitHub Issues**: [报告问题或建议](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/issues)
- **机构**: 华中科技大学 (HUST)
- **官网**: [www.hust.edu.cn](https://www.hust.edu.cn/)

---

<div align="center">

**Made with ❤️ by HUST Researchers**

⭐ 如果有帮助，请给我们一个 Star ~

</div>
