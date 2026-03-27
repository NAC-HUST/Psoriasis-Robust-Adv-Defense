# 对抗攻击

详细的对抗攻击文档。

## ⚔️ 攻击概览

本项目实现了 **SAMOO**(Sparse Multi-Objective Optimization Attack) 算法。

## 🎯 攻击类型

### 无目标攻击 (Untargeted)

模型预测改变即可（→ 任意错误类别）

```bash
uv run main.py attack \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt
```

### 有目标攻击 (Targeted)

强制改变为指定类别

```bash
uv run main.py attack \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
    --targeted \
    --target-class 1
```

## 📊 输出和结果

见 [CLI 参考 - attack](cli-reference.md#attack)

---

完整攻击文档待补充...
