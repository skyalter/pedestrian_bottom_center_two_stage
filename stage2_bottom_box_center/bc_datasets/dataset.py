# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def letterbox_resize(
    img: np.ndarray,
    new_shape: tuple[int, int] = (256, 256),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    h0, w0 = img.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])

    ratio = min(new_w / w0, new_h / h0)
    w, h = int(round(w0 * ratio)), int(round(h0 * ratio))
    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    dw, dh = (new_w - w) / 2.0, (new_h - h) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img_out = cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    return img_out, ratio, (left, top)


def bbox_to_bottom_box_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h


def gaussian2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    heatmap = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0
    return heatmap


def draw_gaussian(heatmap: np.ndarray, center_xy: tuple[float, float], sigma: float) -> None:
    h, w = heatmap.shape
    cx, cy = center_xy
    x0, y0 = int(cx), int(cy)
    if not (0 <= x0 < w and 0 <= y0 < h):
        return

    radius = max(1, int(3 * sigma))
    left, right = min(x0, radius), min(w - 1 - x0, radius)
    top, bottom = min(y0, radius), min(h - 1 - y0, radius)

    gaussian = gaussian2d((top + bottom + 1, left + right + 1), sigma)
    hm_patch = heatmap[y0 - top : y0 + bottom + 1, x0 - left : x0 + right + 1]
    np.maximum(hm_patch, gaussian, out=hm_patch)


def auto_sigma_from_box(
    bbox_resized: tuple[float, float, float, float],
    stride: int,
    k: float = 0.5,
    min_sigma: float = 1.6,
    max_sigma: float = 3.5,
) -> float:
    _, _, bw, bh = bbox_resized
    scale = max(1.0, min(bw, bh) / float(stride))
    sigma = k * scale
    return float(np.clip(sigma, min_sigma, max_sigma))


class _BaseBottomCenterDataset(Dataset):
    def __init__(
        self,
        img_size: tuple[int, int] = (256, 256),
        stride: int = 4,
        sigma_mode: str = "auto",
        sigma_fixed: float = 2.0,
        normalize: bool = True,
        to_rgb: bool = True,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.stride = int(stride)
        self.sigma_mode = sigma_mode
        self.sigma_fixed = float(sigma_fixed)
        self.normalize = normalize
        self.to_rgb = to_rgb

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.h_out = self.img_size[0] // self.stride
        self.w_out = self.img_size[1] // self.stride

    def _prep_image(self, img_path: Path) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized, ratio, (dw, dh) = letterbox_resize(img, new_shape=self.img_size)
        return img, img_resized, ratio, (dw, dh)

    def _build_sample(
        self,
        image_id: int,
        file_name: str,
        img_in: np.ndarray,
        center_xy_in: tuple[float, float],
        bbox_resized: tuple[float, float, float, float],
        cat_idx: int,
    ) -> dict[str, Any]:
        cx_in, cy_in = center_xy_in
        cx_out = cx_in / self.stride
        cy_out = cy_in / self.stride

        heatmap = np.zeros((self.num_classes, self.h_out, self.w_out), dtype=np.float32)
        sigma = (
            auto_sigma_from_box(bbox_resized, stride=self.stride)
            if self.sigma_mode == "auto"
            else float(self.sigma_fixed)
        )
        draw_gaussian(heatmap[cat_idx], (cx_out, cy_out), sigma=sigma)

        offset_map = np.zeros((2, self.h_out, self.w_out), dtype=np.float32)
        offset_mask = np.zeros((1, self.h_out, self.w_out), dtype=np.float32)
        j, i = int(cy_out), int(cx_out)
        if 0 <= i < self.w_out and 0 <= j < self.h_out:
            offset_map[0, j, i] = float(cx_out - math.floor(cx_out))
            offset_map[1, j, i] = float(cy_out - math.floor(cy_out))
            offset_mask[0, j, i] = 1.0

        img_np = img_in.astype(np.float32) / 255.0
        if self.normalize:
            img_np = (img_np - self.mean) / self.std

        return {
            "image": torch.from_numpy(np.transpose(img_np, (2, 0, 1))),
            "heatmap": torch.from_numpy(heatmap),
            "offset_map": torch.from_numpy(offset_map),
            "offset_mask": torch.from_numpy(offset_mask),
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "file_name": file_name,
            "center_xy_in": torch.tensor([cx_in, cy_in], dtype=torch.float32),
            "center_xy_out": torch.tensor([cx_out, cy_out], dtype=torch.float32),
            "bbox_resized": torch.tensor(bbox_resized, dtype=torch.float32),
            "sigma": torch.tensor(sigma, dtype=torch.float32),
            "stride": torch.tensor(self.stride, dtype=torch.int32),
            "img_size": torch.tensor(self.img_size, dtype=torch.int32),
            "cat_idx": torch.tensor(cat_idx, dtype=torch.long),
            "class_name": self.class_names[cat_idx],
        }


class BottomCenterCocoDataset(_BaseBottomCenterDataset):
    def __init__(
        self,
        images_root: str,
        ann_json: str,
        img_size: tuple[int, int] = (256, 256),
        stride: int = 4,
        sigma_mode: str = "auto",
        sigma_fixed: float = 2.0,
        normalize: bool = True,
        to_rgb: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            stride=stride,
            sigma_mode=sigma_mode,
            sigma_fixed=sigma_fixed,
            normalize=normalize,
            to_rgb=to_rgb,
        )
        self.images_root = Path(images_root)
        self.ann_json = Path(ann_json)

        coco = json.loads(self.ann_json.read_text(encoding="utf-8"))
        self.images = {img["id"]: img for img in coco.get("images", [])}

        cats = sorted(coco.get("categories", []), key=lambda x: x["id"])
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
        self.class_names = [c.get("name", str(c["id"])) for c in cats]
        self.num_classes = len(self.class_names)
        if self.num_classes == 0:
            raise RuntimeError("No categories found in COCO 'categories'.")

        self.anns_by_img: dict[int, list[dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        self.items = [
            (image_id, meta["file_name"])
            for image_id, meta in self.images.items()
            if image_id in self.anns_by_img and self.anns_by_img[image_id]
        ]
        if not self.items:
            raise RuntimeError("No images with annotations found.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id, file_name = self.items[index]
        _, img_in, ratio, (dw, dh) = self._prep_image(self.images_root / file_name)

        ann = self.anns_by_img[image_id][0]
        bbox = ann["bbox"]
        cat_id = ann["category_id"]
        cat_idx = self.cat_id_to_idx.get(cat_id, 0)

        x, y, w, h = bbox
        bbox_resized = (x * ratio + dw, y * ratio + dh, w * ratio, h * ratio)
        center_xy_in = bbox_to_bottom_box_center(bbox_resized)

        return self._build_sample(
            image_id=image_id,
            file_name=file_name,
            img_in=img_in,
            center_xy_in=center_xy_in,
            bbox_resized=bbox_resized,
            cat_idx=cat_idx,
        )


class BottomCenterYoloKeypointDataset(_BaseBottomCenterDataset):
    def __init__(
        self,
        images_root: str,
        labels_root: str,
        data_yaml: str | None = None,
        img_size: tuple[int, int] = (256, 256),
        stride: int = 4,
        sigma_mode: str = "auto",
        sigma_fixed: float = 2.0,
        normalize: bool = True,
        to_rgb: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            stride=stride,
            sigma_mode=sigma_mode,
            sigma_fixed=sigma_fixed,
            normalize=normalize,
            to_rgb=to_rgb,
        )
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)

        self.class_names = self._load_class_names(data_yaml)
        self.num_classes = len(self.class_names)
        self.items = self._build_items()
        if not self.items:
            raise RuntimeError("No YOLO keypoint samples found.")

    def _load_class_names(self, data_yaml: str | None) -> list[str]:
        if data_yaml:
            yaml_path = Path(data_yaml)
            if yaml_path.exists():
                text = yaml_path.read_text(encoding="utf-8")
                for line in text.splitlines():
                    if ":" not in line:
                        continue
                    left, right = line.split(":", 1)
                    if left.strip() == "0":
                        return [right.strip()]
                if "pedestrian" in text:
                    return ["pedestrian"]
        return ["pedestrian"]

    def _build_items(self) -> list[tuple[int, str, Path, Path]]:
        image_paths = sorted(
            p
            for p in self.images_root.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        items = []
        for idx, image_path in enumerate(image_paths):
            label_path = self.labels_root / f"{image_path.stem}.txt"
            if label_path.exists():
                items.append((idx, image_path.name, image_path, label_path))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _parse_label(self, label_path: Path, img_w: int, img_h: int) -> tuple[int, tuple[float, float], tuple[float, float, float, float]]:
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"Expected exactly 1 label row in {label_path}, got {len(lines)}")

        fields = lines[0].split()
        if len(fields) < 8:
            raise RuntimeError(f"Expected YOLO keypoint row with >=8 fields in {label_path}, got {len(fields)}")

        class_id = int(float(fields[0]))
        cx = float(fields[1]) * img_w
        cy = float(fields[2]) * img_h
        bw = float(fields[3]) * img_w
        bh = float(fields[4]) * img_h
        kx = float(fields[5]) * img_w
        ky = float(fields[6]) * img_h
        kv = float(fields[7])
        if kv <= 0:
            raise RuntimeError(f"Invisible or missing keypoint in {label_path}")

        bbox = (cx - bw / 2.0, cy - bh / 2.0, bw, bh)
        return class_id, (kx, ky), bbox

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id, file_name, image_path, label_path = self.items[index]
        img_raw, img_in, ratio, (dw, dh) = self._prep_image(image_path)
        img_h, img_w = img_raw.shape[:2]

        class_id, (kx, ky), bbox = self._parse_label(label_path, img_w=img_w, img_h=img_h)
        cat_idx = min(max(class_id, 0), self.num_classes - 1)

        center_xy_in = (kx * ratio + dw, ky * ratio + dh)
        x, y, w, h = bbox
        bbox_resized = (x * ratio + dw, y * ratio + dh, w * ratio, h * ratio)

        return self._build_sample(
            image_id=image_id,
            file_name=file_name,
            img_in=img_in,
            center_xy_in=center_xy_in,
            bbox_resized=bbox_resized,
            cat_idx=cat_idx,
        )
