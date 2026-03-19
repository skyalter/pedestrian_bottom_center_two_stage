# Pedestrian BBox + Point Dataset

Put the original YOLO pedestrian bbox+point dataset here.

Expected structure:

```text
dataset_pedestrian_bbox/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label is expected to contain one bbox and one point per object in YOLO pose-style format.

This folder is treated as the source dataset and should not be modified by the conversion code.
