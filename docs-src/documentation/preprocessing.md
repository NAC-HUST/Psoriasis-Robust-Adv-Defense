# 数据预处理

详细的数据预处理文档。

## 预处理流程

数据预处理是整个管道中的第一步。

### 输入

原始数据应按以下目录结构组织：

```
dataset/raw_data/<datadir>/
├── <class_1>/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── <class_2>/
    ├── img_003.jpg
    └── ...
```

### 处理步骤

1. **自动类别识别**: 逐级目录 → 类别标签
2. **图像加载**: 支持 JPG、PNG、BMP 等格式
3. **等比缩放**: 保持宽高比，缩放至目标尺寸
4. **中心裁剪**: 确保输出 224×224（或指定大小）
5. **格式统一**: 存储为 JPG 格式

### 输出

```
dataset/processed_data/<datadir>/
├── <class_1>/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── <class_2>/
│   └── ...
└── class_manifest.csv
```

## Manifest 文件

`class_manifest.csv` 记录数据集元信息：

| class_id | class_name | num_samples |
|----------|-----------|------------|
| 0 | normal | 150 |
| 1 | psoriasis | 120 |

## 自定义参数

见 [CLI 参考 - preprocess](cli-reference.md#preprocess)

---

完整文档待补充...
