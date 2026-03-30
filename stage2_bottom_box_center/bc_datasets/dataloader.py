# dataloader.py
# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import DataLoader, Subset

from bc_datasets.dataset import BottomCenterCocoDataset, BottomCenterYoloKeypointDataset

def build_datasets(
    dataset_format: str = "coco_bbox",
    images_root: str = "",
    ann_json: str | None = None,
    labels_root: str | None = None,
    data_yaml: str | None = None,
    img_size=(256, 256),
    stride=4,
    sigma_mode="auto",
    sigma_fixed=2.0,
    train_ratio=0.9,
    seed=42,
):
    if dataset_format == "coco_bbox":
        if ann_json is None:
            raise ValueError("ann_json is required for dataset_format='coco_bbox'")
        full_ds = BottomCenterCocoDataset(
            images_root=images_root,
            ann_json=ann_json,
            img_size=img_size,
            stride=stride,
            sigma_mode=sigma_mode,
            sigma_fixed=sigma_fixed,
            normalize=True,
            to_rgb=True,
        )

        n = len(full_ds)
        idx = np.arange(n)
        np.random.default_rng(seed).shuffle(idx)
        n_train = int(n * train_ratio)

        train_ds = Subset(full_ds, idx[:n_train].tolist())
        val_ds = Subset(full_ds, idx[n_train:].tolist())
        return train_ds, val_ds

    if dataset_format == "yolo_keypoint":
        if labels_root is None:
            raise ValueError("labels_root is required for dataset_format='yolo_keypoint'")
        train_ds = BottomCenterYoloKeypointDataset(
            images_root=f"{images_root}/train",
            labels_root=f"{labels_root}/train",
            data_yaml=data_yaml,
            img_size=img_size,
            stride=stride,
            sigma_mode=sigma_mode,
            sigma_fixed=sigma_fixed,
            normalize=True,
            to_rgb=True,
        )
        val_ds = BottomCenterYoloKeypointDataset(
            images_root=f"{images_root}/val",
            labels_root=f"{labels_root}/val",
            data_yaml=data_yaml,
            img_size=img_size,
            stride=stride,
            sigma_mode=sigma_mode,
            sigma_fixed=sigma_fixed,
            normalize=True,
            to_rgb=True,
        )
        return train_ds, val_ds

    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def build_dataloaders(
    batch_size=32,
    num_workers=4,
    **dataset_kwargs,
):
    train_ds, val_ds = build_datasets(**dataset_kwargs)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders(
        batch_size=16,
        num_workers=4,
        images_root="/home/mtl/sunky/bottom_center/data/crops_all/data",
        ann_json="/home/mtl/sunky/bottom_center/data/crops_all/bottom.json",
        img_size=(256, 256),
        stride=4,
        sigma_mode="auto",
        sigma_fixed=2.0,
        train_ratio=0.9,
        seed=42,
    )

    print("Train/Val batches:", len(train_loader), len(val_loader))
    batch = next(iter(train_loader))
    print("image:", batch["image"].shape)
    print("heatmap:", batch["heatmap"].shape)
    print("offset_map:", batch["offset_map"].shape)
    print("offset_mask:", batch["offset_mask"].shape)

    num_classes = train_loader.dataset.dataset.num_classes
    print("num_classes:", num_classes)
