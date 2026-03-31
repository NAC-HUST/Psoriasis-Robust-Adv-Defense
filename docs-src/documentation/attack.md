# 对抗攻击

详细的对抗攻击文档。

## 攻击概览

本项目实现了 **SAMOO**(Sparse Multi-Objective Optimization Attack) 算法。

## 攻击类型

### 无目标攻击 (Untargeted)

模型预测改变即可（→ 任意错误类别）。推荐直接使用训练产出的 `*_train-val-split.csv`，攻击默认仅从 `val` 子集取样：

```bash
uv run main.py attack \
    --backbone resnet50 \
    --datadir psoriasis_normal \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
    --manifest-csv model/trained_classifier/resnet50/best_classifier_train-val-split.csv \
    --sample-index 0
```

说明：

- 默认 `--attack-split val`
- `--sample-index` 是子集内编号（0-based）
- 攻击日志与 summary 会同时记录 `sample_index_subset` 与 `sample_index_global`

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

## 输出和结果

见 [CLI 参考 - attack](cli-reference.md#attack)

## 本轮优化记录

本轮对 `src/psorad` 主实现的攻击优化与工程质量改进详见：

- [SAMOO 攻击优化总结](attack-optimization-summary.md)

---

完整攻击文档待补充...
