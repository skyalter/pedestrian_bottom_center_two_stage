# Dataset Conversion

This folder converts the original pedestrian bbox+point dataset into a new crop-based bottom-center dataset.

Input:

- `../dataset_pedestrian_bbox/`

Output:

- `../dataset_bottom_center/`

Expected source label format is YOLO pose style with one bbox and one keypoint per object:

```text
class cx cy w h kpt_x kpt_y v
```

The source dataset is never modified.
