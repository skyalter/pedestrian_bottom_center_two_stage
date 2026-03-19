# Two-Stage Inference

This folder runs the full two-stage pipeline on one image:

1. stage1 pedestrian bbox detection
2. crop each detected pedestrian
3. stage2 bottom-center inference on each crop
4. map predicted points back to the original image

## Run

```bash
cd two_stage_inference
python infer_image.py \
  --source /path/to/image.jpg \
  --det_weights ../stage1_yolo26/runs/detect/train/weights/best.pt \
  --bc_weights ../stage2_bottom_box_center/checkpoints/<run_name>/best.pt
```

## Single-Frame Outputs

Each single-frame run writes:

- one overlay image with bbox and bottom-center point
- one json file with:
  - detector bbox results
  - bottom-center point results
  - category id
  - detector score
  - bottom-center score

## Clip Inference

```bash
cd two_stage_inference
python infer_clip.py \
  --source_dir /path/to/ordered_frames \
  --det_weights ../stage1_yolo26/runs/detect/train/weights/best.pt \
  --bc_weights ../stage2_bottom_box_center/checkpoints/<run_name>/best.pt \
  --fps 10
```

This runs single-frame inference on each image in sorted order and exports:

- per-frame overlay images
- per-frame json files
- one stitched overlay video
- one clip summary json
