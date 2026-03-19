# Pedestrian Bottom Center Two Stage

This is a standalone project for:

1. training a YOLO pedestrian bbox detector
2. keeping the original pedestrian bbox dataset untouched
3. converting that bbox dataset into a crop-based bottom-center dataset
4. training a bottom-center model on the converted dataset
5. running single-image two-stage inference

## Folder Layout

```text
pedestrian_bottom_center_two_stage/
├── stage1_yolo26/
├── dataset_pedestrian_bbox/
├── dataset_bbox_to_center/
├── dataset_bottom_center/
├── stage2_bottom_box_center/
└── two_stage_inference/
```

## Recommended Workflow

### 1. Put or sync the original bbox dataset

Store the YOLO pedestrian bbox+point dataset under `dataset_pedestrian_bbox/`.

### 2. Train the pedestrian detector

Run the stage1 training code in `stage1_yolo26/`.

### 3. Convert bbox data into crop + bottom-center data

Run the conversion script in `dataset_bbox_to_center/`.
This creates a new dataset under `dataset_bottom_center/` and does not modify `dataset_pedestrian_bbox/`.

### 4. Train bottom-center

Run the stage2 training code in `stage2_bottom_box_center/`.

### 5. Run two-stage inference on one image

Run the script in `two_stage_inference/`.
