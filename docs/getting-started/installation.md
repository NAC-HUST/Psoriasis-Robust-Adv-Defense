# 安装指南

## 📋 系统要求

### 硬件要求

- **CPU**: 多核处理器（i5 或更好）
- **内存**: ≥8GB（推荐 16GB+）
- **GPU**: 可选但推荐（NVIDIA GPU，支持 CUDA）
  - CUDA 计算能力 ≥7.0
  - 显存 ≥4GB（推荐 8GB+）
- **硬盘**: ≥50GB（用于数据集和模型）

### 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| **Python** | ≥3.12 | 编程语言 |
| **Git** | ≥2.20 | 版本控制 |
| **CUDA** | 13.0 (可选) | GPU 加速 |
| **cuDNN** | 9.0 (可选) | GPU 库 |

### 操作系统

- ✅ Linux (Ubuntu 20.04 LTS 或更新)
- ✅ Windows (WSL2 推荐)
- ✅ macOS (M1/M2 可用，但 GPU 支持有限)

## 🚀 快速安装（推荐）

### 1. 克隆仓库

```bash
git clone https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense.git
cd Psoriasis-Robust-Adv-Defense
```

### 2. 使用 uv 安装（推荐）

!!! note "什么是 uv?"
    `uv` 是一个超快速的 Python 包管理器，用 Rust 编写，比 pip 快 10-100 倍。
    [官方文档](https://docs.astral.sh/uv/)

#### 安装 uv

=== "Linux / macOS"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows (PowerShell)"
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "Homebrew"
    ```bash
    brew install uv
    ```

#### 使用 uv 安装依赖

```bash
uv sync
```

这个命令会：
- 🔍 检查 Python 版本
- 📦 安装所有依赖包
- 🐍 创建虚拟环境（如果需要）
- ✅ 设置 PyTorch CUDA 索引

### 3. 验证安装

```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.\.venv\Scripts\activate   # Windows

# 验证 Python
python --version  # 应显示 3.12+

# 验证关键包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 📦 手动安装（备选）

如果要使用 pip 或 conda，可以手动安装：

### 使用 pip

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows

# 安装依赖
pip install torch torchvision transformers numpy pandas pillow scipy
pip install huggingface-hub tqdm
```

### 使用 Conda

```bash
# 创建环境
conda create -n psorad python=3.12 -y

# 激活环境
conda activate psorad

# 安装 PyTorch (CUDA 13.0)
conda install pytorch torchvision pytorch-cuda=13.0 -c pytorch -c nvidia

# 安装其他依赖
conda install transformers numpy pandas pillow scipy huggingface-hub tqdm

# 安装项目
pip install -e .
```

## 🐕 GPU 支持配置

### NVIDIA GPU (推荐)

#### 1. 检查 NVIDIA 驱动

```bash
nvidia-smi  # 应显示 GPU 信息和驱动版本
```

#### 2. 验证 CUDA

```bash
nvidia-smi --query-gpu=capability | head -1
# 应显示计算能力，例如 7.0, 8.0, 9.0 等
```

#### 3. 验证 PyTorch GPU 支持

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

!!! success "GPU 正常工作"
    如果输出显示 `CUDA Available: True` 和具体的 GPU 名称，说明 GPU 支持配置成功。

### Apple Silicon (M1/M2)

```bash
# M1/M2 芯片原生支持
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 🛠️ 开发环境配置

### IDE 推荐

!!! tip "推荐使用 VS Code"
    ```bash
    # 安装 Python 扩展
    code --install-extension ms-python.python
    code --install-extension ms-python.vscode-pylance
    ```

### 编辑器设置

=== ".vscode/settings.json"
    ```json
    {
        "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
        "python.linting.pylintEnabled": false,
        "python.linting.flake8Enabled": true,
        "python.linting.mypyEnabled": true,
        "python.formatting.provider": "black",
        "[python]": {
            "editor.formatOnSave": true,
            "editor.defaultFormatter": "ms-python.python"
        }
    }
    ```

### 代码检查工具

```bash
# 安装开发依赖
uv sync --group dev

# 运行代码检查
ruff check .    # 风格检查
mypy src        # 类型检查
pytest          # 单元测试
```

## ✅ 安装检查清单

运行以下脚本验证完整安装：

```bash
#!/bin/bash

echo "🔍 安装检查清单"
echo "=================="

# 检查 Python
echo -n "✓ Python 版本: "
python --version

# 检查 PyTorch
echo -n "✓ PyTorch: "
python -c "import torch; print(torch.__version__)"

# 检查 Transformers
echo -n "✓ Transformers: "
python -c "import transformers; print(transformers.__version__)"

# 检查 CUDA
echo -n "✓ CUDA 可用: "
python -c "import torch; print(torch.cuda.is_available())"

# 检查 GPU (如有)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo -n "✓ GPU: "
    python -c "import torch; print(torch.cuda.get_device_name(0))"
fi

echo "=================="
echo "✅ 安装检查完成！"
```

## 🔧 常见问题

### Q: ModuleNotFoundError: No module named 'torch'

**解决方案**:
```bash
# 重新安装 PyTorch
uv pip install torch torchvision -U
```

### Q: CUDA 不可用，但有 NVIDIA GPU

**解决方案**:
1. 更新 NVIDIA 驱动: `nvidia-driver-update`
2. 重新安装 PyTorch: `uv sync --upgrade`
3. 检查 CUDA 版本匹配

### Q: 内存不足 (OOM)

**解决方案**:
```bash
# 减小 batch size
uv run main.py train --batch-size 8  # 默认 16
```

### Q: 下载模型超时

**解决方案**:
```bash
# 使用 HF Mirror (中国地区推荐)
export HF_ENDPOINT=https://hf-mirror.com
uv run main.py download-models
```

## 📚 后续步骤

安装完成后，请按以下顺序操作：

1. 📖 阅读 [快速开始](quickstart.md) 了解基本用法
2. 🔧 按 [快速开始](quickstart.md) 运行 demo
3. 📚 查看 [文档](../documentation/cli-reference.md) 了解详细参数
4. 🧪 运行单元测试: `uv run pytest`

---

💬 **遇到问题?** 请 [提交 Issue](https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense/issues)
