# Psoriasis Robust Adv&Defense Doc

面向银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究的文档部分。

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
