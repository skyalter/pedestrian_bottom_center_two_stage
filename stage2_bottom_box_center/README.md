# Stage 2: Bottom Box Center

This folder trains the bottom-center model on the generated crop dataset.

Default training dataset:

- `../dataset_bottom_center/images`
- `../dataset_bottom_center/labels`

## Train

```bash
cd stage2_bottom_box_center
python train.py
```

## Infer One Crop

```bash
cd stage2_bottom_box_center
python infer.py \
  --image ../dataset_bottom_center/images/val/<crop>.jpg \
  --weights ./checkpoints/<run_name>/best.pt \
  --class_id 0
```
