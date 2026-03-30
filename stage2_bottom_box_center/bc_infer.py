"""
BottomCenterNet 推理脚本

模型直接输出 (x_norm, y_norm)，解码为原图坐标只需一次乘除。
脚本启动时自动打印 GPU/CPU 速度基准（200次平均）。

用法：
  python bc_infer.py --image /path/to/crop.jpg --weights checkpoints/bc_net_XXX/best.pt
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from bc_models.bottom_center_net import BottomCenterNet
from bc_datasets.dataset import letterbox_resize

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROJECT_ROOT  = Path(__file__).resolve().parent


def preprocess_image(
    img: np.ndarray,
    img_size: Tuple[int, int] = (256, 256),
) -> Tuple[torch.Tensor, float, Tuple[float, float]]:
    """
    Returns:
        img_t       : (3, H, W) float tensor，已归一化
        ratio       : 缩放比例
        (dw, dh)    : letterbox 水平/垂直 padding（像素）
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox_resize(img_rgb, new_shape=img_size)
    img_np  = img_in.astype(np.float32) / 255.0
    img_np  = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_t   = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float()
    return img_t, ratio, (dw, dh)


def load_model(
    weights_path: str,
    device: str = "cuda",
) -> Tuple[BottomCenterNet, Tuple[int, int]]:
    ckpt = torch.load(weights_path, map_location="cpu")

    num_classes = ckpt.get("num_classes", 1)
    img_size    = tuple(ckpt.get("img_size", (256, 256)))
    state_dict  = ckpt.get("model_state", ckpt)

    model = BottomCenterNet(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, img_size


def infer_images(
    images:         List[np.ndarray],
    class_ids:      List[int],
    weights_path:   str,
    image_paths:    Optional[List[str]] = None,
    device:         str = "cuda",
    img_size:       Optional[Tuple[int, int]] = None,
    save_dir:       Optional[str] = None,
    preloaded_model: Optional[Tuple] = None,
) -> List[Dict[str, Any]]:
    """
    对一批行人 crop 图进行底部中心点预测。

    Args:
        images          : BGR 格式的 crop 图列表
        class_ids       : 每张图对应的类别 ID（当前输出与类别无关，保留兼容性）
        weights_path    : 模型权重路径（preloaded_model 不为 None 时忽略）
        image_paths     : 图片路径（用于保存文件名），可选
        device          : "cuda" / "cpu"
        save_dir        : 若指定，保存带预测点的可视化图片
        preloaded_model : (model, img_size) 元组，传入后跳过模型加载

    Returns:
        list of {image_name, class_id, x_pred, y_pred, out_path}
    """
    assert len(images) == len(class_ids)
    device = device if torch.cuda.is_available() else "cpu"

    if preloaded_model is not None:
        model, img_size = preloaded_model
    else:
        model, img_size = load_model(weights_path, device=device)
    H_in, W_in = img_size

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for idx, img in enumerate(images):
        cls_id   = class_ids[idx]
        img_path = image_paths[idx] if image_paths else f"image_{idx}.jpg"
        img_name = os.path.basename(img_path)
        H0, W0   = img.shape[:2]

        img_t, ratio, (dw, dh) = preprocess_image(img, img_size=img_size)

        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(device))  # (1, 2)

        x_norm = pred[0, 0].item()
        y_norm = pred[0, 1].item()

        x_in   = x_norm * W_in
        y_in   = y_norm * H_in
        x_orig = float(np.clip((x_in - dw) / ratio, 0, W0 - 1))
        y_orig = float(np.clip((y_in - dh) / ratio, 0, H0 - 1))

        out_path = None
        if save_dir:
            vis = img.copy()
            cv2.circle(vis, (int(round(x_orig)), int(round(y_orig))), 5, (0, 255, 0), -1)
            out_path = str(Path(save_dir) / f"{Path(img_name).stem}_pred.jpg")
            cv2.imwrite(out_path, vis)

        results.append({
            "image_name": img_name,
            "class_id":   cls_id,
            "x_pred":     x_orig,
            "y_pred":     y_orig,
            "score":      1.0,
            "out_path":   out_path,
        })

    return results


def parse_args():
    import argparse
    yolo_root = Path("/home/mtl/Desktop/mtl_datasets/pedestrian_crops_yolo")
    parser = argparse.ArgumentParser(description="BottomCenterNet 推理")
    parser.add_argument("--image",    default=str(yolo_root / "images" / "val"))
    parser.add_argument("--weights",  default=str(PROJECT_ROOT / "checkpoints" / "bc_net_latest" / "best.pt"))
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--save_dir", default=str(PROJECT_ROOT / "infer_vis"))
    return parser.parse_args()


if __name__ == "__main__":
    import time
    import argparse

    args = parse_args()
    img  = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {args.image}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model, img_size = load_model(args.weights, device=device)
    img_t, _, _ = preprocess_image(img, img_size=img_size)
    x_in = img_t.unsqueeze(0).to(device)

    # 速度基准
    with torch.no_grad():
        for _ in range(20):
            model(x_in)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            model(x_in)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    print(f"[Speed] avg = {(t1-t0)/200*1000:.2f} ms/img  (device={device})")

    results = infer_images(
        images=[img],
        class_ids=[args.class_id],
        image_paths=[args.image],
        weights_path=args.weights,
        device=args.device,
        save_dir=args.save_dir,
    )

    for r in results:
        print(f"{r['image_name']} | class={r['class_id']} "
              f"| predict=({r['x_pred']:.1f}, {r['y_pred']:.1f})")
        if r["out_path"]:
            print(f"Saved: {r['out_path']}")
