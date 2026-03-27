# 模型训练

详细的模型训练文档。

## 🎓 训练概览

项目支持两种骨干网络的训练：

| 模型 | 类型 | 优势 | 训练时间 |
|------|------|------|---------|
| **ResNet50** | CNN | 速度快，稳定 | 5-10 min (GPU) |
| **SigLIP** | Vision-Language | 0-shot，灵活 | 10-15 min (GPU) |

## 🚀 训练流程

### ResNet50 微调

```bash
uv run main.py train \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --epochs 10 \
    --batch-size 32
```

### SigLIP 冻结骨干

```bash
uv run main.py train \
    --backbone siglip \
    --datadir psoriasis_normal \
    --epochs 5 \
    --freeze-siglip-backbone
```

## 📊 输出和日志

见 [CLI 参考 - train](cli-reference.md#train)

---

完整训练文档待补充...
