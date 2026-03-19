import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json

from models.bottom_center import BottomCenterModel
from datasets.dataset import letterbox_resize
from utils.viz import heatmap_to_color

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROJECT_ROOT = Path(__file__).resolve().parent


def preprocess_image(img: np.ndarray,
                     img_size: Tuple[int, int] = (256, 256),
                     to_rgb: bool = True,
                     normalize: bool = True):

    img_rgb0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if to_rgb else img.copy()
    img_in, r, (dw, dh) = letterbox_resize(img_rgb0, new_shape=img_size, color=(114, 114, 114))
    img_np = img_in.astype(np.float32) / 255.0
    if normalize:
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_chw = np.transpose(img_np, (2, 0, 1))
    img_t = torch.from_numpy(img_chw).float()
    return img_t, img_in, r, (dw, dh), img_rgb0


def load_model(weights_path: str, num_classes: Optional[int] = None, device: str = "cuda"):
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)
    ckpt_num_classes = ckpt.get("num_classes", None)
    model_num_classes = ckpt_num_classes if ckpt_num_classes is not None else num_classes
    if model_num_classes is None:
        raise ValueError("Num_classes not defined, error.")

    model = BottomCenterModel(num_classes=model_num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    stride = ckpt.get("stride", 4)
    return model, int(stride)


def infer_images(
    images: List[np.ndarray],
    class_ids: List[int],
    weights_path: str,
    ann_json: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    device: str = "cuda",
    img_size: Tuple[int, int] = (256, 256),
    save_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:

    assert len(images) == len(class_ids), "images and class_ids must have the same length"
    if ann_json and not image_paths:
        print("[Warning] ann_json provided but image_paths is None, cannot match GT annotations.")

    device = device if torch.cuda.is_available() else "cpu"
    model, stride = load_model(weights_path, num_classes=num_classes, device=device)

    anns_by_name = {}
    if ann_json and os.path.exists(ann_json):
        with open(ann_json, "r", encoding="utf-8") as f:
            coco = json.load(f)
        imgs = {img["id"]: img["file_name"] for img in coco.get("images", [])}
        anns_by_img = {}
        for ann in coco.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)
        anns_by_name = {
            os.path.basename(imgs[iid]): anns[0]["bbox"]
            for iid, anns in anns_by_img.items() if iid in imgs and len(anns) > 0
        }

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for idx, img in enumerate(images):
        cls_id = class_ids[idx]
        img_path = image_paths[idx] if image_paths else f"image_{idx}.jpg"
        img_name = os.path.basename(img_path)

        img_t, img_in_rgb, r, (dw, dh), img_rgb0 = preprocess_image(
            img, img_size=img_size, to_rgb=True, normalize=True
        )
        H0, W0 = img_rgb0.shape[:2]

        with torch.no_grad():
            x = img_t.unsqueeze(0).to(device)
            heatmap_pred, offset_pred = model(x)

        # Decode per-class channel
        hm_1 = heatmap_pred[:, cls_id:cls_id + 1, :, :]
        hm_sig = torch.sigmoid(hm_1)
        b, _, h, w = hm_sig.shape
        flat_idx = torch.argmax(hm_sig.view(b, -1), dim=1)
        jj = (flat_idx // w).float()
        ii = (flat_idx %  w).float()
        offx = offset_pred[:, 0, jj.long(), ii.long()]
        offy = offset_pred[:, 1, jj.long(), ii.long()]
        x_in = (ii + offx) * stride
        y_in = (jj + offy) * stride

        x_orig = (x_in.item() - dw) / r
        y_orig = (y_in.item() - dh) / r
        x_orig = float(np.clip(x_orig, 0, W0 - 1))
        y_orig = float(np.clip(y_orig, 0, H0 - 1))
        score = float(hm_sig.max().item())

        gt_bbox = anns_by_name.get(img_name, None)

        hm_up = F.interpolate(hm_sig, size=img_size, mode="bilinear", align_corners=False)[0, 0]
        hm_color = heatmap_to_color(hm_up.cpu().numpy())
        hm_color = cv2.resize(hm_color, (W0, H0), interpolation=cv2.INTER_LINEAR)
        img_bgr0 = cv2.cvtColor(img_rgb0, cv2.COLOR_RGB2BGR)
        blend = cv2.addWeighted(img_bgr0, 0.6, hm_color, 0.4, 0)

        # GREEN predict
        cv2.circle(blend, (int(round(x_orig)), int(round(y_orig))), 4, (0, 255, 0), -1)

        # BLUE GT
        if gt_bbox is not None:
            x, y, w, h = gt_bbox
            x2, y2 = x + w, y + h
            cx, cy = x + w / 2.0, y + h / 2.0
            cv2.rectangle(blend, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(blend, (int(cx), int(cy)), 4, (255, 0, 0), -1)

        out_path = None
        if save_dir:
            out_path = str(Path(save_dir) / f"{Path(img_name).stem}_cls{cls_id}_overlay.jpg")
            cv2.imwrite(out_path, blend)

        results.append({
            "image_name": img_name,
            "class_id": cls_id,
            "x_pred": x_orig,
            "y_pred": y_orig,
            "score": score,
            "gt_bbox": gt_bbox,
            "out_path": out_path,
        })

    return results



def parse_args():
    import argparse

    yolo_root = PROJECT_ROOT.parent / "dataset_bottom_center"
    parser = argparse.ArgumentParser(description="Infer bottom-box-center on one crop image")
    parser.add_argument(
        "--image",
        default=str(yolo_root / "images" / "val" / "example.jpg"),
    )
    parser.add_argument("--ann_json", default=None)
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "checkpoints" / "bottom_center_latest" / "best.pt"))
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default=str(PROJECT_ROOT / "infer_vis"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {args.image}")

    images = [img]
    class_ids = [args.class_id]
    image_paths = [args.image]

    results = infer_images(
        images=images,
        class_ids=class_ids,
        image_paths=image_paths,
        weights_path=args.weights,
        ann_json=args.ann_json,
        device=args.device,
        img_size=(256, 256),
        save_dir=args.save_dir,
    )

    for r in results:
        print(f"{r['image_name']} | class={r['class_id']} | predict=({r['x_pred']:.1f},{r['y_pred']:.1f}) | GT={r['gt_bbox']} | score={r['score']:.3f}")
        if r["out_path"]:
            print(f"Saved: {r['out_path']}")
