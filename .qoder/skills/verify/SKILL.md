---
name: verify
description: Run the repository quality gate (ruff, mypy, pytest) to verify code changes are ready. Use before committing, opening a PR, or when the user asks to check/verify their changes.
---

# verify

对本仓库的当前改动执行质量门禁，确认可以提交。

## 执行步骤

1. 先看改动范围：`git status` 与 `git diff --name-status`。
2. 依次运行以下命令（在项目根目录）：

   ```bash
   ruff check .
   mypy src
   pytest
   ```

3. 若某一步失败，区分“新增问题”与“存量问题”，优先修复本次改动引入的问题。
4. 修复后重跑对应命令，直到通过。

## 输出

给出结论：`PASS`（可提交）/ `CONDITIONAL PASS`（修复建议后可提交）/ `BLOCK`（存在阻塞问题），并逐项列出：
- ruff / mypy / pytest 各自通过或失败摘要。
- 每个问题的文件位置、原因、可执行的修复建议。

## 备注

- ruff 的 `line-length` 为 240（非默认值），格式化用 `ruff format`。
- 不要把 `dataset/` 真实数据或 `model/` 大权重纳入提交。
