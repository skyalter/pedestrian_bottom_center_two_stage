# Pedestrian Bottom Center Two Stage

Two-stage pipeline that first detects pedestrians with YOLO and then predicts
a sub-pixel bottom-center ground contact point with BottomCenterNet.

---

## Folder Layout

```text
pedestrian_bottom_center_two_stage/
├── two_stage_infer.py             # two-stage inference entry point
├── stage1_yolo26/                 # stage-1 YOLO detector
│   ├── train_yolo.py
│   └── yolo26_pedbbox_03-30.pt   # trained weights
├── stage2_bottom_box_center/      # stage-2 BottomCenterNet
│   ├── bc_infer.py
│   ├── train_bc.py
│   ├── run_batch_infer.py
│   ├── bc_models/                 # model definition
│   ├── bc_datasets/               # dataset & dataloader
│   └── checkpoints/               # trained weights
├── dataset_pedestrian_bbox/       # raw pedestrian bbox dataset
├── dataset_bbox_to_center/        # conversion script: bbox → crop + bottom-center
└── dataset_bottom_center/         # converted crop + bottom-center dataset
```

---

## Environment

```bash
conda activate vru
```

| Dependency | Version |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Ultralytics | — |
| OpenCV | 4.10.0 |

---

## Recommended Workflow

### 1. Prepare the bbox dataset

Store the YOLO pedestrian bbox dataset under `dataset_pedestrian_bbox/`.

### 2. Train the pedestrian detector (stage 1)

```bash
cd stage1_yolo26
python train_yolo.py
```

### 3. Convert bbox data into crop + bottom-center data

```bash
cd dataset_bbox_to_center
python yolo_bbox_to_bottom_center.py
```

Outputs go to `dataset_bottom_center/` without modifying `dataset_pedestrian_bbox/`.

### 4. Train the bottom-center model (stage 2)

```bash
cd stage2_bottom_box_center
python train_bc.py
```

### 5. Run two-stage inference

**Single image:**

```bash
python two_stage_infer.py --source /path/to/image.jpg
```

**Folder of frames (exports overlay video + CSV):**

```bash
python two_stage_infer.py --source /path/to/frames_dir --fps 10
```

`two_stage_infer.py` auto-detects image vs clip mode from whether `--source` is a file or a directory.
Default weights (`DEFAULT_DET_WEIGHTS` / `DEFAULT_BC_WEIGHTS`) are defined at the top of the file and used automatically unless overridden.

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | — | Image file or frames directory |
| `--det_weights` | `stage1_yolo26/yolo26_pedbbox_03-30.pt` | Stage-1 YOLO weights |
| `--bc_weights` | `stage2_bottom_box_center/checkpoints/bc_net_20260329_183535/best.pt` | Stage-2 BottomCenterNet weights |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--det_imgsz` | `1280` | YOLO input image size |
| `--det_conf` | `0.25` | YOLO confidence threshold |
| `--det_iou` | `0.45` | YOLO NMS IoU threshold |
| `--crop_pad` | `8` | Pixel padding around each detected bbox |
| `--bc_h` / `--bc_w` | `256` | BottomCenterNet input size |
| `--fps` | `10.0` | Output video FPS (clip mode) |
| `--save_dir` | auto | Output root directory |

---

## External Usage (Python Import)

This project is designed to be imported into an external codebase.
Models are loaded **once** at startup (including GPU warm-up), then inference
can be called repeatedly with no additional overhead.

### Setup

Add both the project root and the stage-2 subdirectory to `sys.path`:

```python
import sys
sys.path.insert(0, "/path/to/pedestrian_bottom_center_two_stage")
sys.path.insert(0, "/path/to/pedestrian_bottom_center_two_stage/stage2_bottom_box_center")
```

### Minimal usage

```python
from two_stage_infer import load_models, run_image

# Load once at startup (~1 s, includes GPU warm-up)
detector, bc_model = load_models()

# Call per frame/image — pure inference, no reload (~85–120 ms on RTX 3070 Ti)
summary = run_image("/path/to/image.jpg", detector, bc_model)
```

### `summary` dict structure

```python
{
    "num_detections": 2,
    "detections": [
        {
            "index": 0,
            "det_score": 0.87,           # YOLO confidence
            "bbox_xyxy": [x0, y0, x1, y1],
            "crop_xyxy": [cx0, cy0, cx1, cy1],
            "bc_point_full": [px, py],   # bottom-center in original image coords
            "bc_point_crop": [cx, cy],   # bottom-center in crop coords
            "bc_score": 1.0,
        },
        ...
    ],
    "overlay_path": "/path/to/overlay.jpg",
    "json_path":    "/path/to/results.json",
    "timing_ms": {
        "detector":      84.3,
        "bottom_center":  7.9,
        "total":        116.3,
    },
}
```

### Override default parameters

All parameters have defaults matching the trained models. Pass keyword arguments to override:

```python
# Custom confidence threshold and output directory
summary = run_image(
    "/path/to/image.jpg",
    detector,
    bc_model,
    det_conf=0.5,
    save_dir="/tmp/results",
)
```

### Timing breakdown (RTX 3070 Ti, CUDA 12.8)

| Phase | Time |
|---|---|
| `load_models()` (once) | ~1000 ms |
| `run_image()` — YOLO detect | ~80 ms |
| `run_image()` — BottomCenterNet | ~5–10 ms |
| `run_image()` — total | ~110–130 ms |

---

## Output Files

Each call to `run_image()` writes to `save_dir/<image_stem>_<timestamp>/`:

```
<image_stem>_overlay.jpg   # original image with bbox rectangles and bottom-center dots
<image_stem>_results.json  # full summary dict
```

---

## Changelog

- **[2026-03-30]** Refactored `two_stage_infer.py`: model loading separated from inference, GPU warm-up in `load_models()`, removed intermediate crop/overlay saving
- **[2026-03-30]** Renamed modules for uniqueness: `infer.py` → `two_stage_infer.py`, `models/` → `bc_models/`, `datasets/` → `bc_datasets/`, `infer.py` (stage2) → `bc_infer.py`, `train.py` → `train_yolo.py` / `train_bc.py`
- **[2026-03-29]** Initial release
