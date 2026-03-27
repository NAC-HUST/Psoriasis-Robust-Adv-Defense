# 代码规范

项目代码规范和编程指南。

## 代码风格

### 工具

- **ruff**: 代码风格检查和格式化
- **mypy**: 类型检查
- **pytest**: 单元测试

### 运行检查

```bash
ruff check .        # 检查
ruff format .       # 自动格式化

mypy src            # 类型检查

pytest              # 测试
```

## 编码标准

### 类型注解

```python
# 推荐
def load_image(path: str) -> torch.Tensor:
    """Load image from file path."""
    ...

# 不推荐
def load_image(path):
    """Load image from file path."""
    ...
```

### 文档字符串

```python
def process_batch(
    images: torch.Tensor,
    batch_size: int = 32,
) -> list[torch.Tensor]:
    """Process image batch.
    
    Args:
        images: Input tensor of shape (N, 3, H, W)
        batch_size: Processing batch size
        
    Returns:
        List of processed tensors
        
    Raises:
        ValueError: If input tensor shape is invalid
    """
    ...
```

## 测试编写

```python
import pytest
from psorad.models import load_resnet50

class TestModelLoading:
    @pytest.fixture
    def model(self):
        return load_resnet50(pretrained=False)
    
    def test_model_forward(self, model):
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        assert y.shape == (1, 1000)
```

---

完整规范文档待补充...
