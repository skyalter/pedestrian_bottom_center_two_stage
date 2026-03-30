"""
批量推理脚本：从 test 目录随机取 N 张图，统计耗时并保存 overlay 结果图。
用法：
  python run_batch_infer.py [--n 50] [--weights checkpoints/bc_net_xxx/best.pt]
"""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bc_models.bottom_center_net import BottomCenterNet
from bc_datasets.dataset import letterbox_resize

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROJECT_ROOT  = Path(__file__).resolve().parent
TEST_DIR      = Path("/home/mtl/Desktop/mtl_datasets/pedestrian_crops_yolo/images/test")


def load_model(weights_path: str, device: str):
    ckpt       = torch.load(weights_path, map_location="cpu")
    num_classes = ckpt.get("num_classes", 1)
    img_size    = tuple(ckpt.get("img_size", (256, 256)))
    state_dict  = ckpt.get("model_state", ckpt)
    model = BottomCenterNet(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, img_size


def preprocess(img: np.ndarray, img_size):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox_resize(img_rgb, new_shape=img_size)
    img_np  = img_in.astype(np.float32) / 255.0
    img_np  = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_t   = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float()
    return img_t, ratio, (dw, dh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=50, help="推理图片数量")
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "checkpoints" / "bc_net_20260329_183535" / "best.pt"))
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--save_dir", default=str(PROJECT_ROOT / "infer_vis"))
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Weights] {args.weights}")

    # 收集图片路径
    all_imgs = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.png"))
    random.seed(args.seed)
    selected = random.sample(all_imgs, min(args.n, len(all_imgs)))
    print(f"[Data] test set size={len(all_imgs)}, selected={len(selected)}")

    # 加载模型
    model, img_size = load_model(args.weights, device)
    H_in, W_in = img_size
    print(f"[Model] img_size={img_size}")

    # 预热（GPU）
    if device == "cuda":
        dummy = torch.zeros(1, 3, H_in, W_in).to(device)
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
        torch.cuda.synchronize()

    # 速度基准（单张）
    dummy_img = cv2.imread(str(selected[0]))
    img_t, _, _ = preprocess(dummy_img, img_size)
    x_in = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(x_in)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            model(x_in)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    bench_ms = (t1 - t0) / 200 * 1000
    print(f"[Speed benchmark] avg={bench_ms:.2f} ms/img  (200次平均, device={device})")

    # 批量推理 + overlay 保存
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t_total_start = time.perf_counter()

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 无法读取: {img_path}")
            continue

        H0, W0 = img.shape[:2]
        img_t, ratio, (dw, dh) = preprocess(img, img_size)

        t_s = time.perf_counter()
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(device))
        if device == "cuda":
            torch.cuda.synchronize()
        t_e = time.perf_counter()
        infer_ms = (t_e - t_s) * 1000

        x_norm = pred[0, 0].item()
        y_norm = pred[0, 1].item()
        x_in_  = x_norm * W_in
        y_in_  = y_norm * H_in
        x_orig = float(np.clip((x_in_ - dw) / ratio, 0, W0 - 1))
        y_orig = float(np.clip((y_in_ - dh) / ratio, 0, H0 - 1))

        # 绘制：1px 绿色点
        vis = img.copy()
        cx, cy = int(round(x_orig)), int(round(y_orig))
        vis[cy:cy+2, cx:cx+2] = (0, 255, 0)

        out_path = save_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), vis)

        results.append({
            "name":     img_path.name,
            "x":        x_orig,
            "y":        y_orig,
            "infer_ms": infer_ms,
        })

    t_total_end = time.perf_counter()
    total_ms = (t_total_end - t_total_start) * 1000
    n = len(results)

    # 统计
    times = [r["infer_ms"] for r in results]
    print("\n" + "=" * 60)
    print(f"[Results] 推理图片数: {n}")
    print(f"  总耗时 (含IO/预处理): {total_ms:.1f} ms  ({total_ms/1000:.2f} s)")
    print(f"  纯推理 avg: {np.mean(times):.2f} ms/img")
    print(f"  纯推理 min: {np.min(times):.2f} ms")
    print(f"  纯推理 max: {np.max(times):.2f} ms")
    print(f"  纯推理 p50: {np.percentile(times, 50):.2f} ms")
    print(f"  纯推理 p95: {np.percentile(times, 95):.2f} ms")
    print(f"[Speed benchmark (200次)] avg: {bench_ms:.2f} ms/img")
    print(f"[Overlay 图片保存至] {save_dir}")
    print("=" * 60)

    print("\n[逐图结果]")
    for r in results:
        print(f"  {r['name']:60s}  predict=({r['x']:6.1f}, {r['y']:6.1f})  {r['infer_ms']:.2f}ms")


if __name__ == "__main__":
    main()
