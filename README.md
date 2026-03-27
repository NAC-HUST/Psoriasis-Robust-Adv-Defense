# Psoriasis Robust Adv&Defense

面向银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究。

## 项目概览

本项目当前支持以下核心流程：

1. 数据预处理（统一尺寸、生成 `class_manifest.csv`）
2. 预训练模型下载（ResNet50 / SigLIP）
3. 分类器训练（ResNet50 / SigLIP）
4. SAMOO 对抗攻击与结果导出
5. 文档站点（VitePress）手动构建并通过 `doc` 分支 `/docs` 发布

## 目录说明

- `src/`：核心代码
- `dataset/`：数据集目录
- `model/`：预训练与训练后模型
- `output/`：攻击与实验输出
- `docs-src/`：文档源码（VitePress）
- `docs/`：文档静态产物（GitHub Pages 发布目录）

## 环境准备（Python / uv）

要求：Python >= 3.12

```bash
# 安装 Python 依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

## 常用项目命令

```bash
# 预处理
uv run main.py preprocess --datadir psoriasis_normal

# 下载模型
uv run main.py download-models

# 训练
uv run main.py train --backbone resnet50 --datadir psoriasis_normal

# 攻击
uv run main.py attack --backbone resnet50 --datadir psoriasis_normal
```

---

## 文档维护与发布（重点）

当前采用 **手动构建发布**：

- 文档源码在 `docs-src/`
- 构建产物输出到 `docs/`
- GitHub Pages 从 `doc` 分支的 `/docs` 路径发布

### 1) 安装前端依赖（首次或变更后）

```bash
npm install
```

### 2) 本地预览文档

```bash
npm run docs:dev
```

默认会启动本地预览（通常是 `http://localhost:5173` 或终端提示端口）。

### 3) 手动重建发布产物

```bash
npm run docs:build
```

这个命令会：

1. 清空 `docs/` 旧内容
2. 从 `docs-src/` 构建 VitePress 静态文件到 `docs/`
3. 生成 `.nojekyll`

### 4) 提交文档更新（doc 分支）

```bash
git checkout doc

git add docs-src docs package.json package-lock.json
git commit -m "docs: update content and rebuild static site"
git push origin doc
```

---

## GitHub Pages 配置

仓库设置中确认：

- `Settings -> Pages`
- `Source: Deploy from a branch`
- `Branch: doc`
- `Folder: /docs`

保存后，GitHub 会从 `doc` 分支的 `docs/` 目录发布页面。

## 文档更新建议流程

每次文档改动建议按这个顺序执行：

1. 修改 `docs-src/` 下 Markdown 或 VitePress 配置
2. `npm run docs:dev` 本地检查
3. `npm run docs:build` 重建 `docs/`
4. `git add` + `git commit` + `git push origin doc`

## 许可证

本项目采用 GPLv3 许可证。
