---
layout: home

hero:
  name: "Psoriasis Robust Adv&Defense"
  text: "银屑病医学影像对抗鲁棒性研究"
  tagline: "数据预处理、模型训练与对抗攻击的一体化研究框架。"
  image:
    src: "https://vitepress.dev/vitepress-logo-large.webp"
    alt: "PsoraDefense Docs"
  actions:
    - theme: brand
      text: "快速开始"
      link: "/getting-started/quickstart"
    - theme: alt
      text: "方法说明"
      link: "/research/methodology"
    - theme: alt
      text: "GitHub"
      link: "https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense"

features:
  - title: "研究导向"
    details: "聚焦医学影像模型在对抗场景下的稳健性分析与改进。"
  - title: "可复现实验"
    details: "统一命令入口覆盖数据、训练、攻击与结果导出。"
  - title: "双模型路线"
    details: "支持 ResNet50 与 SigLIP，便于横向对比与迭代。"
  - title: "发布规范"
    details: "docs-src 维护源码，docs 保存静态产物用于 Pages 发布。"
---

# Psoriasis Robust Adv&Defense

> 银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究

[![Status](https://img.shields.io/badge/status-in%20progress-yellow)](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/blob/main/LICENSE)
[![University](https://img.shields.io/badge/university-HUST-red)](https://www.hust.edu.cn/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

## 项目概览

本项目致力于研究**对抗攻击**对医学影像分类模型的影响，并提出系统的**防御机制**，以提高银屑病诊断的鲁棒性和可信度。

### 核心研究内容

- **对抗攻击**: 采用 SAMOO（Sparse Multi-Objective Optimization）算法生成稀疏对抗样本
- **多模态学习**: 集成 ResNet50 (CNN) 与 SigLIP (Vision-Language) 模型
- **医学应用**: 专注银屑病医学影像的可靠诊断
- **鲁棒性增强**: 开发防御策略提升模型对对抗样本的抵抗力

## 关键能力

- 端到端流程：预处理、训练、攻击、评估
- 双模型支持：ResNet50 与 SigLIP
- 多数据集实验：CIFAR-10、ImageNette、银屑病数据
- 手动发布：本地构建后提交 `doc` 分支 `docs/`

## 项目架构

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
├── docs-src/                # 文档源码（VitePress）
├── docs/                    # 文档发布目录（静态产物）
└── tests/                   # 单元测试
```

## 快速导航

| 文档 | 描述 |
|------|------|
| [安装指南](getting-started/installation.md) | 环境配置和依赖安装 |
| [快速开始](getting-started/quickstart.md) | 快速上手 |
| [CLI 参考](documentation/cli-reference.md) | 命令行参数与示例 |
| [技术方案](research/methodology.md) | 方法与评估指标 |
| [贡献指南](contributing/guide.md) | 协作规范与提交流程 |

## 技术栈

| 组件 | 用途 | 版本 |
|------|------|------|
| **PyTorch** | 深度学习框架 | 2.9.0 |
| **Transformers** | 预训练模型库 | ≥4.57.0 |
| **TorchVision** | 计算机视觉工具 | 0.24.0 |
| **NumPy** | 数值计算 | ≥2.4.3 |
| **Pandas** | 数据处理 | ≥3.0.1 |
| **Pillow** | 图像处理 | ≥12.1.1 |

## 立即开始

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

更多详情见 [快速开始指南](getting-started/quickstart.md)

## 学术信息

### 研究成果

**论文标题**: 银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法

**作者**: Cao WJ, Cheng Q, Li YX, Feng ZY, Li YX

**机构**: 华中科技大学 (HUST)

## 参与贡献

欢迎提交 Issue 和 Pull Request！请参考 [贡献指南](contributing/guide.md) 了解详情。

### 代码质量检查

提交前请运行：

```bash
ruff check .           # 代码风格检查
mypy src              # 类型检查
pytest                # 单元测试
```

## 许可证

本项目采用 [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) 开源许可证。

## 联系方式

- **GitHub Issues**: [报告问题或建议](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/issues)
- **机构**: 华中科技大学 (HUST)
- **官网**: [www.hust.edu.cn](https://www.hust.edu.cn/)

---

<div align="center">

Maintained by HUST Researchers

</div>
