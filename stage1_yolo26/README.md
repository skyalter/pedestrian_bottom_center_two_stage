# Stage 1: YOLO26 Pedestrian Detection

This folder is for training the pedestrian bbox detector.

Expected dataset location:

- `../dataset_pedestrian_bbox/images/{train,val,test}`
- `../dataset_pedestrian_bbox/labels/{train,val,test}`

## Train

```bash
cd stage1_yolo26
python train.py
```
