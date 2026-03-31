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

## 批量并行攻击（tools）

支持通过 `tools/batch_parallel_attack.py` 对同一子集内多个样本并行执行 SAMOO。

```bash
uv run python tools/batch_parallel_attack.py \
    --backbone resnet50 \
    --checkpoint model/trained_classifier/resnet50/best_classifier.pt \
    --manifest-csv model/trained_classifier/resnet50/best_classifier_train-val-split.csv \
    --datadir psoriasis_normal \
    --attack-split val \
    --start-index 0 \
    --max-samples 100 \
    --workers 4
```

默认输出目录：

- `output/batch_attack/<backbone>/<datadir>/`

可选 `--run-name` 追加一层运行子目录（便于保留多轮实验）。

### 终端实时进度指标

批量模式使用 `tqdm` 进度条输出，并实时显示：

- `done`：已处理样本数（成功 + 失败）
- `success` / `error`：当前成功与异常数量
- `succ_rate`：当前成功率（成功样本 / 已完成样本）
- `avg_q`：当前平均查询次数
- `avg_px`：当前平均修改像素点数量
- `elapsed_s`：累计总用时（秒）
- `avg_dur_s`：单样本平均耗时（秒）
- `spd`：当前吞吐（样本/秒）

### 批量报告文件

批量攻击结束后，会自动生成：

- `batch_report.json`：完整结构化结果（元信息 + 聚合统计 + 逐样本记录 + 错误列表）
- `batch_report.csv`：逐样本统计表（便于表格分析）
- `batch_report.md`：人类可读总结

## 输出和结果

见 [CLI 参考 - attack](cli-reference.md#attack)

## 本轮优化记录

本轮对 `src/psorad` 主实现的攻击优化与工程质量改进详见：

- [SAMOO 攻击优化总结](attack-optimization-summary.md)

---

完整攻击文档待补充...
