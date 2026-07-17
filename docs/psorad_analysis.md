# `src/psorad/` 代码库深度分析报告

> 生成日期：2026-07-07
> 分析范围：`src/psorad/` 全部 25 个 Python 文件 + `configs/` + `pyproject.toml` + `main.py`
> 代码总量：~2150 行 Python

---

## 目录

1. [项目概况](#1-项目概况)
2. [文件清单与规模](#2-文件清单与规模)
3. [调用链与依赖图](#3-调用链与依赖图)
4. [逐模块深度分析](#4-逐模块深度分析)
   - 4.1 `cli.py` — CLI 入口
   - 4.2 `config.py` — 配置系统
   - 4.3 `preprocess/` — 预处理流水线
   - 4.4 `data/` — 数据集与加载器
   - 4.5 `models/` — 模型定义、工厂、训练
   - 4.6 `attack/` — 攻击系统
   - 4.7 `defense/` — 防御模块（空）
   - 4.8 `utils/` — 工具函数
5. [具体问题清单](#5-具体问题清单)
6. [配置系统分析](#6-配置系统分析)
7. [数据流分析](#7-数据流分析)
8. [整理建议](#8-整理建议)

---

## 1. 项目概况

**项目**：面向银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究

**入口**：`main.py` → `psorad.cli:main()`

**四个子命令**：

```
psorad preprocess      # 图像预处理流水线
psorad download-models # 下载预训练权重
psorad train           # 训练分类器
psorad attack          # SAMOO 黑盒稀疏对抗攻击
```

**研究管线三阶段**（当前仅覆盖阶段 1）：

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 稀疏对抗攻击（SAMOO 算法） | ✅ 代码基本完成 |
| 2 | 对抗训练 + 特征一致性正则化 | ❌ 未开始 |
| 3 | 攻防协同演化联合优化 | ❌ 未开始 |

**依赖关系**（简化版）：

```
main.py → cli.py (4个子命令，lazy import)
              ├─ preprocess.pipeline → utils.image
              ├─ utils.download
              ├─ config → models.train → data.dataset, models.factory
              │                           └─ models.classifier (via factory)
              └─ attack.runner → attack.losses, attack.samoo_core, models.classifier, utils.image
```

---

## 2. 文件清单与规模

| 模块 | 文件 | 行数 | 角色 |
|------|------|------|------|
| **根** | `__init__.py` | 10 | 包声明，导出子模块列表 |
| | `cli.py` | 161 | argparse CLI，4 个子命令的入口分发 |
| | `config.py` | 200 | 4 个 dataclass + 3 个 TOML 加载函数 |
| **attack** | `__init__.py` | 4 | 导出 `run_samoo_attack` |
| | `runner.py` | **775** | ⚠️ 最大文件：编排器（模型加载+攻击+导出全包） |
| | `losses.py` | 56 | `UnTargeted` 无目标攻击损失函数 |
| | `samoo_core/__init__.py` | 4 | 导出 `Attack`, `AttackParams` |
| | `samoo_core/attack.py` | 307 | NSGA-II 进化攻击主循环 |
| | `samoo_core/population.py` | 26 | 种群容器 + 评估/检索 |
| | `samoo_core/selection.py` | 89 | 非支配排序 + 拥挤距离 + 锦标赛选择 |
| | `samoo_core/solution.py` | 74 | 个体编码 + 图像生成 + 支配判定 |
| | `samoo_core/variation.py` | 72 | 交叉 + 变异算子 |
| **data** | `__init__.py` | 4 | 导出 `SkinDataset`, `build_loaders` |
| | `dataset.py` | 215 | ⚠️ 两条数据路径共存 |
| **defense** | `__init__.py` | 1 | ❌ **空文件** |
| **models** | `__init__.py` | 11 | 从 `utils.download` 和 `factory` 重新导出 |
| | `classifier.py` | 69 | `build_resnet50_classifier()`, `SiglipClassifier` |
| | `download.py` | 39 | ❌ **死代码**（与 `utils/download.py` 完全重复） |
| | `factory.py` | 43 | 注册表模式（已注册: resnet50, siglip） |
| | `train.py` | 248 | ⚠️ 两条训练路径共存 |
| **preprocess** | `__init__.py` | 4 | 导出 `run_preprocess` |
| | `pipeline.py` | 107 | 图像缩放 + class_manifest.csv 生成 |
| **utils** | `__init__.py` | 5 | 导出 `set_seed`, `center_crop_resize` |
| | `download.py` | 39 | 实际被使用的下载函数 |
| | `image.py` | 14 | `center_crop_resize()` |
| | `seed.py` | 15 | `set_seed()` |
| | **总计** | **~2150** | |

---

## 3. 调用链与依赖图

### 3.1 完整调用链

```
main.py:main()
  └─ psorad.cli:main()
      ├─ [command=preprocess]
      │   └─ preprocess.pipeline:run_preprocess()
      │       ├─ preprocess.pipeline:_discover_class_dirs()     # 发现类别目录
      │       ├─ preprocess.pipeline:_resize_and_save()         # 缩放+保存
      │       │   └─ utils.image:center_crop_resize()
      │       └─ preprocess.pipeline:preprocess_dataset_and_build_manifest()
      │
      ├─ [command=download-models]
      │   └─ utils.download:download_all_models()
      │       ├─ download_resnet50()   # torchvision→.pth
      │       └─ download_siglip()     # HF hub→本地目录
      │
      ├─ [command=train]
      │   ├─ [experiment_config 路径]
      │   │   ├─ config:load_experiment_config()     # TOML→ExperimentConfig
      │   │   └─ models.train:train_experiment()
      │   │       ├─ models.factory:build_model()    # 注册表→分类器
      │   │       │   └─ models.classifier:build_resnet50_classifier() 或 SiglipClassifier
      │   │       ├─ data.dataset:build_split_loaders()
      │   │       │   ├─ SplitDataDataset
      │   │       │   ├─ build_split_transforms()
      │   │       │   └─ _resolve_normalize()
      │   │       └─ utils.seed:set_seed()
      │   │
      │   └─ [CLI参数路径]
      │       └─ models.train:train_classifier()
      │           ├─ models.factory:build_model()
      │           ├─ data.dataset:build_loaders()         # 旧路径
      │           │   ├─ SkinDataset
      │           │   ├─ build_transforms()               # 旧路径
      │           │   └─ split_manifest()                  # 手动拆分
      │           ├─ data.dataset:split_manifest()        # 二次读取！重复
      │           └─ utils.seed:set_seed()
      │
      └─ [command=attack]
          └─ attack.runner:run_samoo_attack()
              ├─ attack.runner:_load_checkpoint()          # 加载模型
              │   ├─ models.factory:build_model()          # 通过工厂
              │   └─ models.classifier (导入，但不直接使用工厂)
              ├─ attack.runner:BinaryModelAdapter          # 模型归一化包装
              ├─ attack.runner:_load_sample_for_attack_split()
              │   └─ 自包含拆分逻辑（⚠️ 重复 data.dataset 的 split_manifest）
              ├─ attack.runner:_resolve_attack_hparams()
              │   └─ _resolution_profile()                 # 分辨率预设
              ├─ attack.runner:_apply_confidence_boost()
              ├─ attack.losses:UnTargeted                  # 损失函数
              ├─ attack.samoo_core.attack:Attack           # 进化算法
              │   ├─ population:Population
              │   ├─ selection:fast_nondominated_sort 等
              │   ├─ solution:Solution
              │   └─ variation:generate_offspring
              ├─ attack.runner:_pick_best_candidate()
              └─ attack.runner:_export_attack_artifacts()  # 导出 JSON/TXT/PNG
```

### 3.2 关键观察

- **无循环依赖**：导入图是从 CLI → 叶子模块的严格 DAG
- **`models/__init__.py` 从 `utils.download` 重新导出**：`__all__` 里有 `download_all_models` 等，但 `cli.py` 直接 `from psorad.utils.download import`，跳过了 `models/__init__`
- **`attack/runner.py` 同时通过两种方式获取模型**：`_load_checkpoint` 里用 `factory.build_model()`，但 `BinaryModelAdapter` 直接 import `models.classifier` 里的常量（归一化统计值）
- **两条训练路径之间的导入共享程度低**：`train_classifier` 走 `build_loaders` + `SkinDataset`，`train_experiment` 走 `build_split_loaders` + `SplitDataDataset`，两者除 `factory.build_model()` 外几乎没有共享代码

---

## 4. 逐模块深度分析

### 4.1 `cli.py`（161 行）

**现状**：
- 标准的 argparse 入口，定义了 `preprocess` / `download-models` / `train` / `attack` 四个子命令
- 使用延迟导入（`from psorad.xxx import` 在函数内部），启动速度快
- `_resolve_manifest_csv()` 辅助函数拼接默认路径

**问题**：
- `attack` 子命令有 16 个可选覆盖参数（`--eps`, `--iterations` 等），全部通过关键字参数透传给 `run_samoo_attack()`，函数签名多达 27 个参数
- `train` 子命令有两条路径（CLI 参数 vs `--experiment-config` TOML），两条路径的入口和结果表现不一致
- 没有 `--version` 参数

**建议**：
- `attack` 的参数传递可以考虑用一个 `AttackCLIConfig` dataclass 打包，而非逐个传递
- 两条训练路径应该合并或明确废弃一条

---

### 4.2 `config.py`（200 行）

**现状**：
- 定义了 4 个 dataclass：
  - `DatasetConfig` — 数据集配置（split_data_root, splits, label_columns, class_names 等）
  - `ModelConfig` — 模型配置（backbone, pretrained_path, freeze_backbone, num_classes）
  - `TrainConfig` — 训练配置（epochs, batch_size, lr 等）
  - `ExperimentConfig` — 实验配置（dataset + model + train 组合）
- 3 个加载函数：`load_dataset_config()`, `load_model_config()`, `load_experiment_config()`
- `_to_path()` / `_as_tuple()` / `_require_mapping()` 三个辅助函数

**问题**：

1. **`TrainConfig` 同名冲突**：`config.py:72-80` 和 `models/train.py:19-36` 各有一个 `TrainConfig`，字段完全不同
   - `config.py` 版本：`epochs`, `batch_size`, `learning_rate`, `num_workers`, `seed`, `output_dir`, `model_name`
   - `models/train.py` 版本：增加的字段有 `backbone`, `num_classes`, `manifest_csv`, `pretrained_resnet_path` 等训练特有的
   - 两者都是 `@dataclass`，但 `config.py` 用 `slots=True`，`models/train.py` 没有

2. **`_to_path()` 的 `base_dir` 参数废弃**：
   ```python
   def _to_path(value, *, base_dir=None):
       if ...:
           return path
       return path if base_dir is None else path  # ← base_dir 从未被使用
   ```
   接收了 `config_path.parent` 作为 `base_dir`，但没有做相对路径解析。如果要实现「相对路径基于配置文件所在目录」，这是个 bug。

3. **`DatasetConfig.image_size` 类型不一致**：
   - 代码中：`int` (`config.py:48`)
   - TOML 配置中：`"224x224"` (字符串，如 `configs/dataset_config/baseline_dataset.toml`)
   - `load_dataset_config()` 用 `int(data.get("image_size", 224))` 读取，传入 `"224x224"` 会导致 `int("224x224")` 抛出 `ValueError`
   - 但因为实际使用 `load_experiment_config()` 路径中也用了同样的 `int()` 读取，所以 TOML 中的 `"224x224"` 字符串必然会导致运行时错误

4. **大量 TOML 字段不消费**：详见 [第 6 章](#6-配置系统分析)

---

### 4.3 `preprocess/`（111 行）

**现状**：
- `run_preprocess()` → `preprocess_dataset_and_build_manifest()` 流程完整
- 发现 `raw_data/<datadir>/<class_name>/` 下的图片，缩放+中心裁剪后输出到 `processed_data/<datadir>/<class_name>/`，生成 `class_manifest.csv`
- `PreprocessConfig` dataclass 用于参数打包

**评价**：
- 功能层面干净、完整，是整个项目中最不需要改动的模块
- `run_preprocess()` 的返回值是 `(int, int, Path)`，第一个是 `processed_count`、第二个是 `manifest_rows`，但两者实际都是 `len(manifest)`，没有体现出差异（冗余）
- 写死了 `quality=95` 的 JPEG 输出，没有作为参数暴露

---

### 4.4 `data/`（219 行）

**现状**：**两条数据路径共存，职责重叠**

#### 路径 A（旧）：`SkinDataset` + `build_loaders()` + `split_manifest()`
```
Raw CSV → split_manifest() 随机拆分为 train/val → SkinDataset × 2 → DataLoader
```

- `SkinDataset` 接收一个 CSV 或由 `set_manifest()` 注入 DataFrame
- `build_transforms()`：基于 `for_siglip` bool 硬编码归一化参数
- `split_manifest()`：用 `torch.Generator` 做随机排列拆分

#### 路径 B（新）：`SplitDataDataset` + `build_split_loaders()`
```
split_data/ 下预先拆分好的 CSV → SplitDataDataset → DataLoader
```

- `SplitDataDataset` 读取单独 split 的 label CSV
- `build_split_transforms()`：用字符串 key（`"imagenet"` / `"siglip"`）查 `_resolve_normalize()`
- `build_split_loaders()`：接收 `DatasetConfig`，遍历 splits 创建 loaders

**问题**：

1. **归一化逻辑重复**：
   ```python
   # dataset.py:125（旧路径）
   normalize = Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) if for_siglip else Normalize([0.485,...], [0.229,...])
   
   # dataset.py:116-121（新路径）
   def _resolve_normalize(normalize):
       if normalize == "siglip": return Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
       if normalize == "imagenet": return Normalize([0.485,...], [0.229,...])
   
   # attack/runner.py:164-169（又写了一次）
   if backbone == "resnet50": mean/std = ([0.485,...], [0.229,...])
   if backbone == "siglip":   mean/std = ([0.5,...], [0.5,...])
   ```

   **同一个映射关系写了三遍。**

2. **`data/__init__.py` 只导出了旧路径**：
   ```python
   from psorad.data.dataset import SkinDataset, build_loaders
   ```
   `SplitDataDataset` 和 `build_split_loaders` 未被导出，但 `models/train.py:train_experiment()` 直接 `from psorad.data.dataset import build_split_loaders`。所以 `__init__.py` 的导出落后于实际使用。

3. **`SkinDataset` 实际上未被外部使用**（除了 `build_loaders` 自己实例化它）。新的训练路径只用 `SplitDataDataset`。

---

### 4.5 `models/`（410 行，含死代码）

#### 4.5.1 `classifier.py`（69 行）

**现状**：
- `build_resnet50_classifier(pretrained_weight_path?, num_classes=2)`：torchvision ResNet50 + 替换 fc 层
- `SiglipClassifier(pretrained_dir_or_id, num_classes=2, freeze_backbone=True)`：HF transformers AutoModel + 线性分类头
- 两者都返回 `nn.Module`

**问题**：
- `build_resnet50_classifier` 不支持 `freeze_backbone` 参数（但 ResNet50 微调时有时也需要冻结）
- `SiglipClassifier` 的 `forward` 处理了 4 种不同的输出格式（`get_image_features` 直接返回 tensor / `pooler_output` / `last_hidden_state.mean`），过于 defensive，但这是 HF 模型差异导致的

#### 4.5.2 `factory.py`（43 行）

**现状**：
- 注册表模式：`register_backbone("resnet50")` / `register_backbone("siglip")`
- `build_model(cfg: ModelConfig, num_classes: int) → nn.Module`
- `available_backbones() → list[str]`

**评价**：
- 这是 `models/` 模块中设计最好的部分。注册表模式清晰、可扩展。
- 唯一的限制：当前硬编码了 `resnet50` 和 `siglip` 两个注册函数，添加新骨干需要修改 `factory.py` 源码（而非外部注册）。对于研究项目来说足够了。

**问题**：
- `_build_siglip()` 中 `pretrained_path` 为 None 时的 fallback 是 `"model/pretrained_model/siglip"`（硬编码字符串），而 `siglip` 的目录结构可能变化。但 `_build_resnet50` 在 `pretrained_path` 为 None 时传入 None，行为是「随机初始化」。两者行为不一致。

#### 4.5.3 `train.py`（248 行）

**现状**：**两条训练路径**

| 函数 | 用途 | 数据加载 | 调用方 |
|------|------|----------|--------|
| `train_classifier(config: TrainConfig)` | 旧路 | `build_loaders()` + `SkinDataset` | `cli.py:108`（直接参数） |
| `train_experiment(exp_cfg: ExperimentConfig)` | 新路 | `build_split_loaders()` + `SplitDataDataset` | `cli.py:101`（TOML 配置） |

**问题**：

1. **`train_classifier` 重复读取 manifest**：
   ```python
   # 第 96 行：build_loaders 内部调用了 split_manifest
   train_loader, val_loader = build_loaders(manifest_csv=config.manifest_csv, ...)
   # 第 125 行：又显式调用了 split_manifest
   train_manifest, val_manifest = split_manifest(manifest_csv=config.manifest_csv, ...)
   # 第 129 行：只是为了导出 split CSV
   split_csv_path = _export_split_csv(checkpoint_path, train_manifest, val_manifest)
   ```
   **同一 CSV 被 `pd.read_csv` 读了两遍**，虽然 seed 相同所以拆分结果一致，但浪费了 IO 和内存。

2. **两个 `TrainConfig` 混淆**：
   - `train.py:19-36` 的 `TrainConfig` 包含 `backbone`, `manifest_csv`, `pretrained_resnet_path` 等训练需要的字段
   - `config.py:72-80` 的 `TrainConfig` 没有这些字段
   - 导入时如果 `from psorad.config import TrainConfig` 会拿到错误的类
   - 两个类都叫 `TrainConfig`，模糊搜索或 IDE 跳转时极易混淆

3. **两个训练循环的代码几乎相同**（epoch 循环 + 前向 + 反向 + 评估 + 保存），但并未提取公共基类或函数。后续如果修改训练逻辑（如添加 scheduler、AMP、gradient clipping），需要改两遍。

#### 4.5.4 `models/download.py`（39 行）← **死代码**

```python
# 内容完全等于 utils/download.py
def download_resnet50(): ...   # 同 utils/download.py
def download_siglip(): ...    # 同 utils/download.py
def download_all_models(): ... # 同 utils/download.py
```

- `models/__init__.py` 从 `utils.download` re-export 了这三个函数
- `cli.py` 直接从 `utils.download` import
- `models/download.py` **没有任何导入路径引用它**

**结论**：这是重构时残留的冗余文件，应删除。

---

### 4.6 `attack/`（1313 行，占整个包的 61%）

#### 4.6.1 `runner.py`（775 行）— **最大问题点**

**775 行承担了以下所有职责**：

| 功能区域 | 行数区间 | 说明 |
|----------|----------|------|
| `_safe_path_token()` | 20-23 | 路径字符串清理 |
| `_resolution_profile()` | 26-55 | 基于分辨率的预设参数 |
| `_resolve_attack_hparams()` | 58-103 | 超参合并（预设 + CLI 覆盖） |
| `_apply_confidence_boost()` | 106-156 | 高置信度时增强攻击强度 |
| `BinaryModelAdapter` | 159-222 | 模型包装（归一化 + 前向） |
| `_to_uint8_image()` | 225-227 | 图像格式转换 |
| `_build_comparison_image()` | 230-252 | 图像拼接（before + after + diff） |
| `_pick_best_candidate()` | 255-265 | 从帕累托前沿挑选最佳解 |
| `_export_attack_artifacts()` | 268-395 | 导出 JSON / TXT / PNG（**128 行**） |
| `_load_checkpoint()` | 398-419 | 模型加载（**内部用工厂，但顶部 import 非工厂**） |
| `_load_sample_from_manifest()` | 422-440 | 从 manifest 加载单样本 |
| `_load_sample_for_attack_split()` | 443-500 | **重复实现** `split_manifest()` 的拆分逻辑 |
| `AttackRunLogger` | 503-509 | 日志收集器（用 `print()` 而非 `logging`） |
| `_format_topk_probs()` | 512-516 | Top-K 置信度格式化 |
| `run_samoo_attack()` | 520-774 | **主编排函数**（255 行） |

**具体问题**：

1. **职责过于集中**：所有的「决策」（跑什么参数、是否增强）、「执行」（加载模型、加载数据、调进化算法）、「输出」（JSON、TXT、4 张 PNG）都在一个文件里。改导出格式要改这个文件，改攻击参数默认值也要改这个文件。

2. **数据拆分逻辑重复**：
   ```python
   # attack/runner.py:464-482
   # 自己实现了 train/val 拆分（用 torch.randperm + Generator）
   # 而 data/dataset.py:199-214 的 split_manifest() 做完全相同的事
   ```
   两处的代码几乎逐行对应，但一个放在 `data/dataset.py`，一个内联在 `attack/runner.py`。修补一处 bug 时另一处会漏掉。

3. **`BinaryModelAdapter` 硬编码归一化参数**（见 4.4），与 `data/dataset.py` 的同一逻辑不同步。

4. **`_load_checkpoint` 从 state_dict 推断 num_classes**：
   ```python
   for key in state_dict:
       if "fc.weight" in key or "classifier.weight" in key:
           num_classes = state_dict[key].shape[0]
   ```
   假设 weight 的 shape[0] 就是 num_classes，但如果检查点保存的是 `strict=False` 加载后的部分权重，这个推断可能出错（如只保存了分类头参数）。同时，检查点元数据中已经保存了 `backbone` 等信息，但 `_load_checkpoint` 没有利用它们。

5. **从 `.npy` 文件回传攻击结果**：`Attack.attack()` 把结果 `np.save()` 到磁盘，`run_samoo_attack()` 再 `np.load()` 读回来。这是一种「进程间通信」式的设计，但进攻和读回在同一个进程内。完全可以用内存传递代替。

6. **`print()` 作为日志**：`AttackRunLogger` 用 `print()` 输出日志，同时累积到 `self.lines`。没有日志级别、没有时间戳、不能输出到文件（除了最终被 `_export_attack_artifacts` 写入 `attack_log.txt`）。

#### 4.6.2 `losses.py`（56 行）

**现状**：
- `PredictionModel` Protocol — 定义了 `predict()` 接口
- `UnTargeted` — 无目标攻击损失函数
  - `__call__(img)` → `[is_adversarial, true_score - other_max]`
  - `get_label(img)` → 预测类别

**评价**：
- 干净、专注。Protocol 定义清晰。
- 唯一的耦合：`UnTargeted.__init__` 接受 `to_pytorch_input` 参数来决定是否在内部做 `np.ndarray → torch.Tensor` 转换。这导致了 `to_pytorch()` 和 `to_flat_numpy()` 两个辅助函数。其实可以统一在 adapter 层处理，让损失函数只操作 numpy。

#### 4.6.3 `samoo_core/`（572 行）

**总体评价**：这是整个项目中最独立、最干净的模块。字典式参数（`payload`）传递结果和回调（`progress_callback`）的设计松耦合。内部实现了一个标准的 NSGA-II 多目标进化算法。

每个文件分析：

| 文件 | 行数 | 质量 | 说明 |
|------|------|------|------|
| `attack.py` | 307 | ⭐⭐⭐ | `AttackParams` 验证严格（`__post_init__`），`Attack.attack()` 流程完整：初始化 → 迭代（排序 → 选择 → 交叉变异 → 环境选择）→ 结果保存。退火变异率、预算检查、提前停止均有实现。 |
| `population.py` | 26 | ⭐⭐⭐ | 简洁的容器 + `evaluate()` + `find_adv_solns()` |
| `solution.py` | 74 | ⭐⭐⭐ | `Solution` 编码清晰：pixels（像素索引）+ values（扰动值）→ `generate_image()` 组装对抗样本。`dominates()` 对对抗逻辑的处理正确（成功 vs 失败的支配规则不同）。 |
| `selection.py` | 89 | ⭐⭐⭐ | 标准 NSGA-II：快速非支配排序 O(MN²) + 拥挤距离 + 锦标赛选择。实现清晰。 |
| `variation.py` | 72 | ⭐⭐ | 单点交叉 + 替换式变异。交叉操作检查像素不重复（`if soln2.pixels[idx] not in soln1.pixels`），确保子代不产生重复像素。但 `crossover` 中 `delta1` 的 `np.asarray([idx for idx in range(k) if ...])` 可以用向量化加速。 |

**微小问题**：
- `variation.py:33`：`delta1` 用列表推导式构建，可以被向量化
- `attack.py:275-291`：环境选择的实现是标准的 NSGA-II 选择（按 front 依次加入，最后一个 front 按拥挤距离排序截断），但被注释为「环境选择」，可以提取出 `_environmental_selection()` 方法
- `attack.py:95-100`：`_init_population` 里 `all_pixels = np.arange(h * w)`，而 `attack()` 方法里又算了一次 `all_pixels = np.arange(h * w)`（第 159 行）。可以作为 params 传递或缓存

---

### 4.7 `defense/`（1 行，空）

`__init__.py` 内容为空（0 字节）。

- `src/psorad/__init__.py` 的 `__all__` 中已经包含了 `defense`，但由于目录下只有空文件，`import psorad.defense` 不会报错但没有任何可用内容
- 项目标题是「攻防协同」，但防御端代码完全为零
- 如果要开发阶段 2（对抗训练），预计需要：`defense/adversarial_trainer.py`、`defense/feature_regularization.py`、`defense/attacker.py`（生成对抗样本用于训练）等

---

### 4.8 `utils/`（58 行）

**现状**：

| 文件 | 行数 | 说明 |
|------|------|------|
| `image.py` | 14 | `center_crop_resize(pil_image, image_size)` — 基于 `ImageOps.fit` |
| `seed.py` | 15 | `set_seed(seed)` — 设置 random/numpy/torch/cuda 种子 |
| `download.py` | 39 | `download_resnet50()` / `download_siglip()` / `download_all_models()` |
| `__init__.py` | 5 | 从 `image` 和 `seed` re-export |

**评价**：
- 整体干净。`download.py` 是三个下载函数中**唯一被实际使用的版本**
- `image.py` 的 `center_crop_resize` 在 `preprocess/pipeline.py` 和 `attack/runner.py` 中都被使用
- `seed.py` 的 `set_seed` 在 `models/train.py` 中被使用
- 不足：没有 `__all__` 约束 `download.py` 的导出，但 `cli.py` 直接 `from psorad.utils.download import`，绕过了 `__init__.py` 的封装

---

## 5. 具体问题清单

### 🔴 P0 — 必须修

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| 1 | **`runner.py` 775 行单体文件** | `attack/runner.py` | 改任何功能都要翻这个大文件，逻辑耦合严重 |
| 2 | **两个 `TrainConfig` 同名冲突** | `config.py:72` vs `train.py:19` | 导入时可能拿错，IDE 跳转混淆 |
| 3 | **数据拆分逻辑重复 2 次** | `data/dataset.py:199` + `attack/runner.py:464` | 同一份逻辑修一个漏一个 |
| 4 | **归一化参数重复 3 次** | `data/dataset.py:125` + `data/dataset.py:116` + `attack/runner.py:164` | 改一个参数三个地方要同步 |

### 🟡 P1 — 应该修

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| 5 | **`models/download.py` 是死代码** | `models/download.py` | 冗余文件，内容完全等于 `utils/download.py` |
| 6 | **`train_classifier` 重复读取 manifest** | `models/train.py:96+125` | 同样的 CSV 读两次，浪费 IO |
| 7 | **`_to_path()` 的 `base_dir` 参数废弃** | `config.py:12-20` | 意图是解析相对路径，但 `base_dir` 传入后没使用 |
| 8 | **`DatasetConfig.image_size` 类型不一致** | `config.py:48` vs TOML | 代码用 `int`，配置写 `"224x224"`，运行时必然出错 |
| 9 | **两条训练路径并行** | `models/train.py` | 相同的训练逻辑维护两套 |
| 10 | **结果通过 `.npy` 文件在进程内传递** | `attack/runner.py:717` vs `attack.py:154` | 同一进程内写入磁盘再读回，完全可以内存传递 |

### 🟢 P2 — 建议修

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| 11 | **使用 `print()` 而非 `logging`** | `attack/runner.py:508` | 没有日志级别、时间戳、文件输出、结构化 |
| 12 | **`preprocess` 返回值冗余** | `preprocess/pipeline.py:106` | `processed_count` 和 `manifest_rows` 是同一个值 |
| 13 | **`data/__init__.py` 导出落后** | `data/__init__.py` | 只导出了旧路径 `SkinDataset`, `build_loaders` |
| 14 | **`attack()` 方法里的 `all_pixels` 重复计算** | `attack.py:97+159` | 初始化和主循环各算一次 |
| 15 | **`variation.py:33` 未向量化** | `variation.py:33` | 列表推导式可以替换为 numpy 向量化操作 |
| 16 | **环境选择没有提取为方法** | `attack.py:275-291` | 内联的 NSGA-II 环境选择可以被提取为 `_environmental_selection()` |
| 17 | **`BinaryModelAdapter` 直接 import model class** | `attack/runner.py:16` | 绕过了 `factory.py` 的注册表模式 |
| 18 | **`attack` CLI 参数 16 个逐个传递** | `cli.py:131-158` | 函数签名 27 个参数，可以用 dataclass 打包 |
| 19 | **TOML 中的 `"224x224"` 格式问题** | `baseline_dataset.toml` | 多个数据集的 `image_size` 用了 `"224x224"` 字符串，但代码用 `int()` 读取 |

---

## 6. 配置系统分析

### 6.1 TOML 中定义了但代码不消费的字段

`configs/model_config/baseline_model.toml` 包含丰富的配置节，但绝大多数未被 `config.py` 的 dataclass 或任何消费代码读取：

| TOML 节 | 定义字段 | `config.py` 对应 | 代码消费 |
|---------|---------|-----------------|---------|
| `[Train]` | `weight_decay`, `mixed_precision`, `grad_accum_steps`, `clip_grad_norm` | ❌ 无 | ❌ |
| `[Train]` | `scheduler`, `warmup_epochs`, `min_lr`, `step_size`, `gamma` | ❌ 无 | ❌ |
| `[Train]` | `loss`, `label_smoothing`, `pos_weight` | ❌ 无 | ❌ |
| `[Train]` | `metrics`, `eval_every`, `save_best_metric` | ❌ 无 | ❌ |
| `[Backbone.resnet50]` | `pretrained_source`, `weights`, `in_channels` | ❌ 无 | ❌ |
| `[Backbone.siglip]` | `pooling`, `dropout` | ❌ 无 | ❌ |
| `[Optimizer]` | `name`, `betas`, `momentum` | ❌ 无 | ❌ |
| `[Augmentation]` | `train_transforms`, `val_transforms`, `color_jitter` 等 | ❌ 无 | ❌ |
| `[Output.checkpoint]` | `save_every_epoch`, `save_top1`, `save_last` | ❌ 无 | ❌ |
| `[Output.logging]` | `save_log`, `verbose`, `visualize_metrics` | ❌ 无 | ❌ |
| `[Download]` | `hf_endpoint`, `save_path` | ❌ 无 | ❌ |
| `[Download.resnet50]` | `hf_repo_id`, `dir_name` | ❌ 无 | ❌ |
| `[Download.siglip2]` | `hf_repo_id`, `dir_name` | ❌ 无 | ❌ |

**实际被消费的仅**：
| TOML 配置 | `config.py` 字段 | 消费代码 |
|-----------|-----------------|---------|
| `backbone` | `ModelConfig.backbone` | `factory.build_model()` |
| `pretrained_path` | `ModelConfig.pretrained_path` | `factory.build_model()` |
| `freeze_backbone` | `ModelConfig.freeze_backbone` | `SiglipClassifier.__init__` |
| `num_classes` | `ModelConfig.num_classes` | `factory.build_model()` |
| `normalization` | `ModelConfig.normalization` | `build_split_transforms()` |
| `split_data_root` | `DatasetConfig.split_data_root` | `build_split_loaders()` |
| `splits` | `DatasetConfig.splits` | `build_split_loaders()` |
| `image_size` | `DatasetConfig.image_size` | `build_transforms()` |
| `label_columns` | `DatasetConfig.label_columns` | `SplitDataDataset` |
| `epochs` / `batch_size` / `learning_rate` | `TrainConfig` | `train_experiment()` |

**结论**：配置方案中约 3/4 的字段定义了但没实现。这给人「配置了就会生效」的错误预期。要么删掉未实现字段，要么逐步补齐消费代码。

### 6.2 占位符文件

| 文件 | 内容 | 状态 |
|------|------|------|
| `configs/attack_config/example_attack.toml` | 空（0 字节） | ❌ |
| `configs/defence_config/example_defence.toml` | 空（0 字节） | ❌ |

攻击和防御的示例配置从未创建。`cli.py` 的 `attack` 子命令也不支持 `--experiment-config` 方式加载攻击配置。

### 6.3 配置架构总结

当前的配置架构是分层的：
```
ExperimentConfig
  ├── DatasetConfig    ← TOML [dataset] 节
  ├── ModelConfig      ← TOML [model] 节
  └── TrainConfig      ← TOML [train] 节
```

这个方向是对的，但缺陷在于：
1. 没有 `AttackConfig`（攻击配置）
2. 没有 `DefenseConfig`（防御配置，阶段 2 需要）
3. `ExperimentConfig` 只在训练中使用，攻击路径完全靠 CLI 参数
4. 消费代码跟不上配置定义

---

## 7. 数据流分析

### 7.1 训练数据流

```
（旧路径 train_classifier）
raw_data/<class>/*.jpg
  ↓ preprocess （中心裁剪缩放 → JPEG）
processed_data/<datadir>/class_manifest.csv
  ↓ build_loaders() 内部分为 train/val
SkinDataset (train) → DataLoader  ─┐
                                    ├→ train loop → checkpoint.pt
SkinDataset (val)   → DataLoader  ─┘

（新路径 train_experiment）
split_data/<dataset>/images/{train,val,test}/*.jpg
split_data/<dataset>/labels/{train,val,test}_labels.csv
  ↓ build_split_loaders()
SplitDataDataset (train) → DataLoader  ─┐
                                        ├→ train loop → checkpoint.pt
SplitDataDataset (val)   → DataLoader  ─┘
```

**问题**：两条路径的数据格式完全不同。旧路径是「原始类别目录 → 单 CSV」，新路径是「预拆分 split_data 目录」。`split_data/` 下的数据需要外部脚本生成，项目中并没有生成 `split_data/` 的代码（`preprocess/pipeline.py` 只生成旧格式）。

### 7.2 攻击数据流

```
checkpoint.pt
  ↓ _load_checkpoint()
nn.Module
  ↓ BinaryModelAdapter（归一化包装）
adapter.predict_proba()
  ↓
原始图像（HWC numpy, [0,1]）→ SAMOO Attack
  ↓ np.save/.np.load（通过磁盘传递）
payload dict (front0_imgs, adversarial_labels, ...)
  ↓ _pick_best_candidate() + _export_attack_artifacts()
summary.json + summary.txt + attack_log.txt + before.png + after.png + diff_x4.png + before_after_diff.png
```

**关键路径**：原始图像从 `_load_sample_for_attack_split()` 返回 HWC numpy 数组，进入 `Attack` 时也是 HWC，（`AttackParams.__post_init__` 验证了 `x.ndim == 3 and x.shape[2] == 3`），最终对抗样本以同样格式输出。归一化仅在 `BinaryModelAdapter` 内部进行（`_prepare_tensor` 做 HWC→CHW +/255 + normalize）。

---

## 8. 整理建议

### 阶段 1：快速清污（消除明显的问题）

| 步骤 | 操作 | 涉及文件 | 风险 |
|------|------|----------|------|
| 1.1 | 删除 `models/download.py` | `models/download.py` | 低 — 已确认无引用 |
| 1.2 | 重命名 `models/train.py` 中的 `TrainConfig` → `TrainingRunConfig` | `models/train.py` | 中 — 需同步 `cli.py` 的 import |
| 1.3 | 修复 `_to_path()` 的 `base_dir` 未使用问题 | `config.py` | 低 — 当前行为不受影响（传了等于没传） |
| 1.4 | 修复 `image_size` 类型不一致：统一用 `int` | `baseline_dataset.toml`, `config.py` | 低 — 当前实际路径走 `load_experiment_config` 也会出错 |
| 1.5 | 更新 `data/__init__.py` 导出新路径 | `data/__init__.py` | 低 — 仅为接口整洁 |

### 阶段 2：结构拆解（消除架构债务）

| 步骤 | 操作 | 涉及文件 | 说明 |
|------|------|----------|------|
| 2.1 | **拆分 `attack/runner.py`** | 新建 3-4 个文件 | 详见下方拆解方案 |
| 2.2 | 合并归一化逻辑 | `data/dataset.py` + `attack/runner.py` | 提取到 `models/classifier.py` 或 `utils/` |
| 2.3 | 合并数据拆分逻辑 | `attack/runner.py` → `data/dataset.py` | 让 runner 复用 dataset 的拆分 API |
| 2.4 | 消除 `train_classifier` 重复读取 manifest | `models/train.py` | `build_loaders()` 返回拆分后的两个 DataFrame 或重构加载流程 |
| 2.5 | 攻击结果从「磁盘传递」改为「内存传递」 | `attack.py` + `runner.py` | `Attack.attack()` 直接返回 payload dict 而非 `np.save()` |

### 阶段 3：配置落地（让 TOML 真正生效）

| 步骤 | 操作 | 涉及文件 | 说明 |
|------|------|----------|------|
| 3.1 | 删除 TOML 中不消费的字段 | `baseline_model.toml` | 或逐步实现 |
| 3.2 | 添加 `AttackConfig` dataclass | `config.py` | 让攻击路径也支持 TOML 配置 |
| 3.3 | 为 `cli.py` 的 `attack` 命令添加 `--experiment-config` | `cli.py` | 匹配训练的 TOML 路径 |

### 阶段 4：研究推进（添加缺失功能）

| 步骤 | 操作 | 涉及文件 | 说明 |
|------|------|----------|------|
| 4.1 | 实现 `defense/` 模块 | `defense/` | 阶段 2 的对抗训练 |
| 4.2 | 添加 `logging` 替代 `print()` | 全局 | 结构化日志 |
| 4.3 | 添加测试 | `tests/` | 当前仅 1 个冒烟测试 |
| 4.4 | 废弃旧路径 | `train_classifier` + `build_loaders` + `SkinDataset` | 只保留 split_data 路径 |

### `attack/runner.py` 拆解方案（详细）

当前 775 行的 `runner.py` 可以拆为：

```
attack/
├── runner.py            (精简版: 只做编排，约 150 行)
├── adapter.py           (BinaryModelAdapter → 从 runner.py 提取)
├── hparams.py           (_resolution_profile + _resolve_attack_hparams + _apply_confidence_boost)
├── export.py            (_export_attack_artifacts + _build_comparison_image + _to_uint8_image)
└── sampler.py           (_load_sample_from_manifest + _load_sample_for_attack_split)
```

这样每个文件的职责：
- `runner.py`：串联流程（加载→攻击→导出），约 150 行
- `adapter.py`：模型归一化包装 + predict 方法
- `hparams.py`：超参数预设和解析策略，可以被测试独立验证
- `export.py`：结果导出（JSON/TXT/PNG），可以独立于攻击逻辑测试
- `sampler.py`：数据加载逻辑，与 `data/dataset.py` 对齐

当前 `_load_checkpoint()` 可以考虑保留在 `runner.py` 或移至 `models/loader.py`（后者需要新建文件）。

---

*本分析基于 `src/psorad/` 全部 25 个源文件的逐行阅读，覆盖了 `refactor/framework423` 分支上截至 2026-07-07 的全部内容。*
