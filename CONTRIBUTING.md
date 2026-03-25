# Contributing Guide

感谢你参与 Psoriasis-Robust-Adv-Defense 项目。

本文档用于统一开发环境、代码规范与提交流程，帮助贡献者快速对齐协作方式。

## 1. 开发前提

- Python >= 3.12
- 建议使用 Linux / WSL
- 建议使用 `uv` 管理环境（也支持 pip）

## 2. 本地开发环境（推荐使用 uv）

```bash
uv sync

```

## 3. 基本开发流程

1. 从最新主分支创建功能分支。
2. 完成代码修改与自测。
3. 运行格式/静态检查/测试。
4. 提交代码并发起 PR。

建议分支命名：

- `feat/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `refactor/<short-description>`

## 4. 代码质量要求

提交前至少运行以下命令：

```bash
ruff check .
mypy src
pytest
```

可选：启用 pre-commit

```bash
pre-commit install
pre-commit run -a
```

## 5. 项目运行自检（建议）

若你的修改涉及训练/攻击流程，建议至少跑通以下链路：

```bash
uv run main.py preprocess
uv run main.py download-models
uv run main.py train --backbone resnet50
uv run main.py attack --backbone resnet50 --checkpoint model/trained_classifier/resnet50/best_binary_classifier.pt
```

若涉及 SigLIP，请额外验证：

```bash
uv run main.py train --backbone siglip
uv run main.py attack --backbone siglip --checkpoint model/trained_classifier/siglip/best_binary_classifier.pt
```

## 6. 提交信息建议

建议使用清晰的提交前缀：

- `feat:` 新功能
- `fix:` 缺陷修复
- `docs:` 文档修改
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建或工具调整

示例：

```text
feat: add SAMOO runner for binary psoriasis classifier
docs: expand README quickstart and CLI reference
fix: lazy-load transformers for siglip path
```

## 7. Pull Request 清单

PR 描述请尽量包含：

- 变更目的与背景
- 主要改动点
- 运行/测试结果（命令与输出摘要）
- 兼容性与风险说明

合并前请确认：

- [ ] 文档与代码行为一致
- [ ] 未提交数据集与大模型权重
- [ ] 关键路径已本地验证
- [ ] 不包含无关文件改动

## 8. 数据与模型文件注意事项

- 请勿提交 `dataset/` 下的真实数据样本。
- 请勿提交 `model/pretrained_model/` 与 `model/trained_classifier/` 的大文件权重。
- 如需复现结果，请在 PR 中说明数据来源与运行命令，不直接提交产物。

## 9. 讨论与协作

若需求或设计不确定，请先通过 Issue / PR 讨论关键接口与目录变更，再进行大规模修改。

