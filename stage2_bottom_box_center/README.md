# Stage 2: BottomCenterNet

轻量级两阶段行人底部中心点预测模型，接收 YOLO 裁出的行人 crop，输出底部中心点像素坐标。

---

## 模型概述

本模型是两阶段行人定位流水线的第二阶段，上游由 YOLO 完成行人检测并裁出单人区域，本模型对裁图做精确底部中心点回归：

```
原图 ──→ [Stage 1: YOLO] ──→ 行人 bbox crop ──→ [Stage 2: BottomCenterNet] ──→ 底部中心点 (x, y)
```

**模型架构**：以完整通道宽度的 MobileNetV2 为 Backbone（最宽 320ch），在 stage5/6/7 引入 SE（Squeeze-and-Excitation）通道注意力，接更大的回归头（512→256→2），最终经 Sigmoid 输出归一化坐标。

**输入**：行人 crop（BGR 图像，letterbox 缩放至 256×256）
**输出**：归一化坐标 `(x_norm, y_norm) ∈ [0, 1]`，可一步还原为原图像素坐标

---

## 性能指标

| 指标 | 值 |
|------|----|
| 参数量 | ~2.2M |
| 输入尺寸 | 256 × 256 |
| GPU 推理延迟（batch=1） | ~3.4 ms |
| Loss 函数 | SmoothL1 |
| 评估设备 | NVIDIA RTX 3070 Ti Laptop GPU (8 GB) |

> 速度数据来自 200 次平均，CUDA 12.8，PyTorch 2.10。

---

## 目录结构

```
stage2_bottom_box_center/
├── bc_models/
│   └── bottom_center_net.py      # BottomCenterNet 模型定义（MobileNetV2 + SE）
├── bc_datasets/
│   ├── dataset.py                # Dataset 类（YOLO keypoint / COCO bbox 格式）
│   └── dataloader.py             # DataLoader 构建器
├── bc_infer.py                   # 推理脚本（含速度基准）
├── train_bc.py                   # 训练脚本
├── run_batch_infer.py            # 批量推理脚本
├── checkpoints/                  # 训练权重（gitignore）
├── logs/                         # TensorBoard 日志（gitignore）
└── README.md
```

---

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.10.18 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| OpenCV | 4.10.0 |
| NumPy | 2.2.6 |
| TensorBoard | 2.20.0 |
| GPU 显存 | 建议 ≥ 4 GB |

```bash
conda activate vru
```

---

## 快速开始

### 1. 数据准备

使用 YOLO Keypoint 格式数据集，目录结构如下：

```
pedestrian_crops_yolo/
├── images/
│   ├── train/   # 2469 张
│   ├── val/     # 732 张
│   └── test/    # 348 张
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

每个 `.txt` 标签文件格式（一行一个实例）：

```
class_id  cx  cy  bw  bh  kx  ky  kv
# 坐标均归一化到 [0, 1]；(kx, ky) 为底部中心点；kv>0 表示可见
# 示例: 0  0.5  0.5  1.0  1.0  0.476190  0.920000  2
```

### 2. 运行推理

```bash
conda activate vru
cd stage2_bottom_box_center

python bc_infer.py \
  --image /path/to/crop.jpg \
  --weights ./checkpoints/bc_net_20260329_183535/best.pt \
  --save_dir ./infer_vis
```

脚本启动时自动打印 200 次 GPU/CPU 速度基准，结果图保存至 `infer_vis/`。

**预期输出：**

```
[Speed] avg = 3.37 ms/img  (device=cuda)
crop.jpg | class=0 | predict=(128.3, 241.7)
Saved: infer_vis/crop_pred.jpg
```

### 3. 在代码中调用

**推荐方式（通过两阶段入口，支持模型复用）：**

```python
# 参见根目录 two_stage_infer.py 的 External Usage 章节
from two_stage_infer import load_models, run_image
detector, bc_model = load_models()
summary = run_image("/path/to/image.jpg", detector, bc_model)
```

**直接调用 stage2（仅 BottomCenterNet，需自行提供 crop）：**

```python
import sys
sys.path.insert(0, "/path/to/stage2_bottom_box_center")

from bc_infer import infer_images, load_model
import cv2

# 加载模型（一次）
bc_model = load_model("checkpoints/bc_net_20260329_183535/best.pt", device="cuda")

# 推理
img = cv2.imread("crop.jpg")
results = infer_images(
    images=[img],
    class_ids=[0],
    weights_path=None,            # 传入 preloaded_model 时忽略
    preloaded_model=bc_model,
    device="cuda",
)
x, y = results[0]["x_pred"], results[0]["y_pred"]
```

---

## 训练指南

### 启动训练

```bash
conda activate vru
cd stage2_bottom_box_center
python train_bc.py
```

默认路径指向 `/home/mtl/Desktop/mtl_datasets/pedestrian_crops_yolo`，无需额外配置。

**自定义参数：**

```bash
python train_bc.py \
  --img_h 256 --img_w 256 \
  --batch_size 32 \
  --epochs 150 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.2
```

**覆盖数据集路径：**

```bash
python train_bc.py \
  --images_root /path/to/dataset/images \
  --labels_root /path/to/dataset/labels \
  --data_yaml   /path/to/dataset/data.yaml
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--img_h` / `--img_w` | 256 | 输入图像尺寸 |
| `--batch_size` | 32 | 每批样本数 |
| `--epochs` | 150 | 训练总轮数 |
| `--lr` | 1e-3 | 初始学习率（CosineAnnealing 调度） |
| `--weight_decay` | 1e-4 | L2 正则化 |
| `--dropout` | 0.2 | 回归头 Dropout 率 |
| `--train_ratio` | 0.9 | 训练集比例（仅 COCO 格式有效） |
| `--dataset_format` | `yolo_keypoint` | 数据集格式（`yolo_keypoint` / `coco_bbox`） |

### 训练输出

```
checkpoints/bc_net_YYYYMMDD_HHMMSS/
├── best.pt   # 验证集 loss 最低的权重（推理使用）
└── last.pt   # 最后一个 epoch 的权重

logs/bc_net_YYYYMMDD_HHMMSS/
└── events.out.tfevents.*
```

### TensorBoard 监控

```bash
conda activate vru
tensorboard --logdir logs/
```

监控指标：`train/loss`、`val/loss`（SmoothL1）、`train/px_err`、`val/px_err`（像素误差）

---

## 推理与部署

### 坐标解码逻辑

模型输出归一化坐标，还原为原图像素坐标：

```python
x_in   = x_norm * img_w_in          # 还原到输入图尺寸
y_in   = y_norm * img_h_in
x_orig = (x_in - dw) / ratio        # 去除 letterbox padding，还原到原图坐标
y_orig = (y_in - dh) / ratio
```

### 模型架构详情

```
输入 (B, 3, 256, 256)
        │
        ▼ stem: ConvBNReLU6 s=2        →  (B, 32, 128, 128)
        │
        ▼ stage1~4（标准 InvResidual）
  s=1, e=1 →  16ch @ 128×128
  s=2, e=6 →  24ch @  64×64  ×2
  s=2, e=6 →  32ch @  32×32  ×3
  s=2, e=6 →  64ch @  16×16  ×4
        │
        ▼ stage5~7（InvResidual + SE 注意力）
  s=1, e=6 →  96ch @  16×16  ×3  ← SE
  s=2, e=6 → 160ch @   8×8   ×3  ← SE
  s=1, e=6 → 320ch @   8×8   ×1  ← SE
        │
        ▼ conv head: ConvBNReLU6 1×1  →  (B, 512, 8, 8)
        │
        ▼ AdaptiveAvgPool2d(1)         →  (B, 512)
        │
        ▼ Dropout → Linear(512→256) → ReLU → Dropout → Linear(256→2) → Sigmoid
        │
        ▼ 输出 (B, 2)  →  [x_norm, y_norm] ∈ [0, 1]
```

---

## 常见问题

**Q: CUDA out of memory 怎么办？**
A: 降低 `--batch_size`，例如改为 16 或 8。

**Q: 推理时预测点偏向图像角落（接近 0,0）？**
A: 检查输入图像是否为 BGR 格式，以及 letterbox 的 `ratio` 和 `(dw, dh)` 是否正确传入解码。

**Q: 如何在没有 GPU 的机器上推理？**
A: 脚本自动检测 CUDA 可用性并回退到 CPU，无需修改代码。`--device cpu` 也可显式指定。

**Q: 标签文件中 `kv=0` 的关键点会参与训练吗？**
A: 不会，`bc_datasets/dataset.py` 中会过滤掉不可见关键点（`kv <= 0`）的样本。

---

## 更新日志

- **[2026-03-30]** 重命名模块以避免与外部包冲突：`models/` → `bc_models/`，`datasets/` → `bc_datasets/`，`infer.py` → `bc_infer.py`，`train.py` → `train_bc.py`
- **[2026-03-30]** `bc_infer.infer_images()` 新增 `preloaded_model` 参数，支持模型复用
- **[2026-03-29]** 完成训练，发布 `bc_net_20260329_183535` 权重
- **[2026-03-29]** 初始版本：BottomCenterNet（MobileNetV2 全尺寸 + SE 注意力）
