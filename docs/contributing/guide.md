# 贡献指南

感谢你对本项目的兴趣！本文档说明如何参与项目开发。

## 🎯 贡献类型

我们欢迎以下类型的贡献：

| 类型 | 示例 | 难度 |
|------|------|------|
| 🐛 **问题报告** | 发现 bug、报告错误 | ⭐ 简单 |
| 📚 **文档改进** | 补充文档、改进示例 | ⭐ 简单 |
| 🚀 **功能增强** | 新功能、性能优化 | ⭐⭐⭐ 困难 |
| 🔧 **代码重构** | 改进代码质量、整理结构 | ⭐⭐ 中等 |
| 🧪 **测试补充** | 增加单元测试、集成测试 | ⭐⭐ 中等 |

## 📋 开发前提

- **技术要求**:
  - Python ≥ 3.12
  - Git 基础知识
  - 熟悉 PyTorch / Transformers
  
- **工具要求**:
  - VS Code 或其他编辑器
  - Git 客户端
  - 虚拟环境管理工具（uv / conda）

## 🚀 开发工作流

### 1️⃣ Fork 仓库

在 GitHub 上 Fork 项目：

```bash
# 访问 https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense
# 点击 Fork 按钮
```

### 2️⃣ 克隆本地

```bash
# 克隆你的 fork
git clone https://github.com/<YOUR-USERNAME>/Psoriasis-Robust-Adv-Defense.git
cd Psoriasis-Robust-Adv-Defense

# 添加上游仓库
git remote add upstream https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense.git
```

### 3️⃣ 创建功能分支

```bash
# 从 main 分支创建
git checkout main
git pull upstream main

# 创建功能分支（推荐命名）
git checkout -b feat/your-feature-name
# 或
git checkout -b fix/bug-description
# 或
git checkout -b docs/doc-update
```

### 4️⃣ 配置开发环境

```bash
# 安装依赖（包括开发工具）
uv sync

# 或激活虚拟环境
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows
```

### 5️⃣ 进行修改

在本地进行代码修改和测试...

### 6️⃣ 运行检查

修改完成前，必须运行以下命令：

```bash
# 代码风格检查
ruff check .

# 类型检查
mypy src

# 单元测试
pytest

# 或一次性运行所有检查
./scripts/check-all.sh  # 如果存在此脚本
```

### 7️⃣ 提交更改

遵循提交规范：

```bash
git add <files>
git commit -m "<type>: <description>"
```

**提交前缀规范**:

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feat:` | 新功能 | `feat: add targeted attack mode` |
| `fix:` | 缺陷修复 | `fix: resolve GPU memory leak` |
| `docs:` | 文档修改 | `docs: update installation guide` |
| `refactor:` | 代码重构 | `refactor: optimize model loading` |
| `perf:` | 性能优化 | `perf: improve data loading speed` |
| `test:` | 测试相关 | `test: add unit tests for attack module` |
| `chore:` | 构建或工具 | `chore: update dependencies` |

### 8️⃣ 推送并创建 PR

```bash
# 推送到你的 fork
git push origin feat/your-feature-name

# 在 GitHub 创建 Pull Request
# 填写 PR 描述和相关信息
```

## ✅ Pull Request 清单

提交 PR 时，请确保：

- [ ] 代码符合项目风格指南
- [ ] 已运行 `ruff check .` 并修复问题
- [ ] 已运行 `mypy src` 并解决类型错误
- [ ] 已运行 `pytest` 并通过所有测试
- [ ] 添加了必要的测试用例
- [ ] 更新了相关文档
- [ ] PR 描述清晰说明了修改内容
- [ ] 合并冲突已解决

## 📝 PR 描述模板

```markdown
## 修改描述

简要描述你的修改内容。

## 修改类型

- [ ] 🐛 缺陷修复
- [ ] ✨ 新功能
- [ ] 📚 文档更新
- [ ] ♻️ 代码重构
- [ ] ⚡ 性能优化

## 相关 Issue

关闭 #Issue_Number

## 修改清单

- [ ] 代码检查通过
- [ ] 类型检查通过
- [ ] 单元测试通过
- [ ] 文档已更新

## 测试说明

描述你进行的手动测试：

```bash
# 运行的命令
uv run main.py train --backbone resnet50 --datadir cifar10
```

输出结果：...

## 截图（如适用）

如果有 UI 变化或结果输出，请附加截图。
```

## 🔍 代码规范

### 代码风格

遵循 PEP 8，项目已配置 ruff：

```bash
# 自动修复格式问题
ruff check . --fix

# 检查格式
ruff check .
```

### 类型注解

添加类型提示以支持静态检查：

```python
# ❌ 不推荐
def preprocess_image(image, size):
    return image.resize(size)

# ✅ 推荐
from PIL import Image

def preprocess_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Preprocess image to target size.
    
    Args:
        image: Input PIL Image
        size: Target size as (width, height)
        
    Returns:
        Resized image
    """
    return image.resize(size)
```

### 文档字符串

使用 Google 风格的 docstring：

```python
def compute_attack_metrics(
    original: torch.Tensor,
    adversarial: torch.Tensor,
) -> dict[str, float]:
    """Compute attack evaluation metrics.
    
    Args:
        original: Original input tensor of shape (C, H, W)
        adversarial: Adversarial input tensor of shape (C, H, W)
        
    Returns:
        Dictionary containing:
        - sparsity: Fraction of modified pixels (0-1)
        - perturbation_l2: L2 norm of perturbation
        - perturbation_linf: L∞ norm of perturbation
        
    Raises:
        ValueError: If tensor shapes don't match
    """
    ...
```

### 导入规范

按以下顺序组织导入：

```python
# 1. 标准库
import os
import sys
from pathlib import Path

# 2. 第三方库
import numpy as np
import torch
from PIL import Image

# 3. 项目内部
from psorad.data import DataLoader
from psorad.models import ResNet50Classifier
```

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_preprocess.py

# 运行特定测试函数
pytest tests/test_preprocess.py::test_image_resize

# 显示详细输出
pytest -v

# 显示覆盖率
pytest --cov=src
```

### 编写测试

```python
# tests/test_attack.py
import pytest
import torch
from psorad.attack import SAMOOAttacker

class TestSAMOOAttacker:
    @pytest.fixture
    def attacker(self):
        """Create attacker instance."""
        return SAMOOAttacker(population_size=50, generations=50)
    
    def test_attack_success(self, attacker):
        """Test that attack produces valid adversarial sample."""
        # 准备测试数据
        model = create_dummy_model()
        image = create_dummy_image()
        
        # 执行攻击
        adversarial, metrics = attacker.attack(model, image)
        
        # 验证结果
        assert adversarial.shape == image.shape
        assert metrics["success"] is True
        assert 0 <= metrics["sparsity"] <= 1
```

## 📚 文档贡献

### 修改文档

项目文档使用 VitePress，位于 `docs/` 目录：

```
docs/
├── .vitepress/
│   ├── config.ts
│   └── theme/
├── index.md                    # 首页
├── getting-started/
│   ├── overview.md
│   ├── installation.md
│   └── quickstart.md
├── documentation/
├── research/
└── contributing/
```

### 本地预览

```bash
# 安装前端依赖
npm install

# 启动本地服务器
npm run docs:dev

# 访问 http://localhost:8000
```

### 修改建议

- ✅ 补充缺失的说明
- ✅ 改进代码示例
- ✅ 修正错别字或语法错误
- ✅ 添加常见问题解答
- ✅ 包含更多实际用例

## 🏆 最佳实践

### 提交信息

```bash
# ✅ 好的提交信息：清晰、描述性
git commit -m "feat: implement SAMOO attack with multi-objective optimization

- Add population-based variant algorithm
- Implement Pareto front selection
- Add comprehensive metrics computation
- Update documentation with examples"

# ❌ 不好的提交信息：模糊、无意义
git commit -m "fix: stuff"
git commit -m "update"
```

### 分支管理

```bash
# ✅ 好的分支名：清晰、有意义
git checkout -b feat/sparse-adversarial-attack
git checkout -b fix/cuda-memory-leak
git checkout -b docs/add-architecture-diagram

# ❌ 不好的分支名：模糊、无意义
git checkout -b feature
git checkout -b fix
git checkout -b temp
```

### 代码审查

参与复审时，请：

- 💬 提出建设性意见
- 🎯 指出具体问题
- ✅ 建议改进方案
- 🤝 保持友好、尊重的态度

## 📞 沟通渠道

- **Issues**: [GitHub Issues](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/issues) - 问题报告和讨论
- **Discussions**: [GitHub Discussions](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/discussions) - 功能建议和一般讨论
- **Email**: 通过 Issue 联系项目维护者

## 🎓 行为准则

本项目采用 [Contributor Covenant](https://www.contributor-covenant.org/) 行为准则。参与者应该：

- ✅ 尊重并欢迎多样性
- ✅ 使用包容性语言
- ✅ 接受建设性批评
- ✅ 专注于对项目最好的内容

不能容忍：

- ❌ 骚扰、歧视、骚扰性语言
- ❌ 人身攻击
- ❌ 发布他人私人信息

严重违反行为准则的贡献者将被移除项目。

## 🎉 致谢

非常感谢所有贡献者！你们的工作使这个项目更好。

---

**还有疑问？** [提交 Issue](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/issues) 或 [进行讨论](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/discussions)

