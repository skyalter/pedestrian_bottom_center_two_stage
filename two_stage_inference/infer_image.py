import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE2_ROOT = PROJECT_ROOT / "stage2_bottom_box_center"
if str(STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2_ROOT))

from infer import infer_images  # noqa: E402


def clip_box(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    x0_i = max(0, min(width, int(math.floor(x0))))
    y0_i = max(0, min(height, int(math.floor(y0))))
    x1_i = max(0, min(width, int(math.ceil(x1))))
    y1_i = max(0, min(height, int(math.ceil(y1))))
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return x0_i, y0_i, x1_i, y1_i


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage inference on a single image")
    parser.add_argument("--source", required=True, help="Image path")
    parser.add_argument(
        "--det_weights",
        default=str(PROJECT_ROOT / "stage1_yolo26" / "runs" / "detect" / "train" / "weights" / "best.pt"),
        help="Stage1 detector weights",
    )
    parser.add_argument(
        "--bc_weights",
        default=str(PROJECT_ROOT / "stage2_bottom_box_center" / "checkpoints" / "bottom_center_latest" / "best.pt"),
        help="Stage2 bottom-center weights",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--det_imgsz", type=int, default=1280)
    parser.add_argument("--det_conf", type=float, default=0.25)
    parser.add_argument("--det_iou", type=float, default=0.45)
    parser.add_argument("--crop_pad", type=int, default=8)
    parser.add_argument("--bc_h", type=int, default=256)
    parser.add_argument("--bc_w", type=int, default=256)
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--save_dir", default=str(PROJECT_ROOT / "two_stage_inference" / "runs"))
    return parser.parse_args()


def run_two_stage_image(args: argparse.Namespace) -> dict:
    source_path = Path(args.source).resolve()
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source_path}")

    height, width = image.shape[:2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path(args.save_dir) / f"{source_path.stem}_{stamp}"
    crop_dir = work_dir / "crops"
    crop_overlay_dir = work_dir / "crop_overlays"
    work_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(args.det_weights)
    det_results = detector.predict(
        source=str(source_path),
        imgsz=args.det_imgsz,
        conf=args.det_conf,
        iou=args.det_iou,
        device=args.device,
        verbose=False,
    )
    if not det_results:
        raise RuntimeError("Detector returned no result")

    boxes = det_results[0].boxes
    crops: list[np.ndarray] = []
    crop_paths: list[str] = []
    det_metas: list[dict] = []

    if boxes is not None:
        for idx in range(len(boxes)):
            x0, y0, x1, y1 = boxes.xyxy[idx].tolist()
            det_score = float(boxes.conf[idx].item())
            det_class = int(boxes.cls[idx].item())
            crop_xyxy = clip_box(
                x0 - args.crop_pad,
                y0 - args.crop_pad,
                x1 + args.crop_pad,
                y1 + args.crop_pad,
                width=width,
                height=height,
            )
            if crop_xyxy is None:
                continue

            crop_x0, crop_y0, crop_x1, crop_y1 = crop_xyxy
            crop = image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
            if crop.size == 0:
                continue

            crop_path = crop_dir / f"{source_path.stem}_det{idx:03d}.jpg"
            cv2.imwrite(str(crop_path), crop)

            crops.append(crop)
            crop_paths.append(str(crop_path))
            det_metas.append(
                {
                    "index": idx,
                    "det_class_id": det_class,
                    "det_score": det_score,
                    "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                    "crop_xyxy": [crop_x0, crop_y0, crop_x1, crop_y1],
                    "crop_path": str(crop_path),
                }
            )

    bc_results = []
    if crops:
        bc_results = infer_images(
            images=crops,
            class_ids=[args.class_id] * len(crops),
            image_paths=crop_paths,
            weights_path=args.bc_weights,
            device=args.device,
            img_size=(args.bc_h, args.bc_w),
            save_dir=str(crop_overlay_dir),
        )

    overlay = image.copy()
    detections = []
    for det_meta, bc_result in zip(det_metas, bc_results):
        x0, y0, x1, y1 = det_meta["bbox_xyxy"]
        crop_x0, crop_y0, _, _ = det_meta["crop_xyxy"]
        point_x = float(np.clip(crop_x0 + bc_result["x_pred"], 0, width - 1))
        point_y = float(np.clip(crop_y0 + bc_result["y_pred"], 0, height - 1))

        cv2.rectangle(
            overlay,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (0, 200, 255),
            2,
        )
        cv2.circle(overlay, (int(round(point_x)), int(round(point_y))), 4, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            f"det={det_meta['det_score']:.2f} bc={bc_result['score']:.2f}",
            (int(round(x0)), max(18, int(round(y0)) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "index": det_meta["index"],
                "det_class_id": det_meta["det_class_id"],
                "det_score": det_meta["det_score"],
                "bbox_xyxy": det_meta["bbox_xyxy"],
                "crop_xyxy": det_meta["crop_xyxy"],
                "crop_path": det_meta["crop_path"],
                "bc_point_crop": [float(bc_result["x_pred"]), float(bc_result["y_pred"])],
                "bc_point_full": [point_x, point_y],
                "bc_score": float(bc_result["score"]),
                "crop_overlay": bc_result["out_path"],
            }
        )

    overlay_path = work_dir / f"{source_path.stem}_overlay.jpg"
    json_path = work_dir / f"{source_path.stem}_results.json"
    cv2.imwrite(str(overlay_path), overlay)

    summary = {
        "source": str(source_path),
        "image_size": [height, width],
        "det_weights": args.det_weights,
        "bc_weights": args.bc_weights,
        "num_detections": len(detections),
        "params": {
            "det_imgsz": args.det_imgsz,
            "det_conf": args.det_conf,
            "det_iou": args.det_iou,
            "crop_pad": args.crop_pad,
            "bc_img_size": [args.bc_h, args.bc_w],
            "class_id": args.class_id,
        },
        "detections": detections,
        "overlay_path": str(overlay_path),
        "json_path": str(json_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_two_stage_image(args)

    print(f"[OK] detections: {summary['num_detections']}")
    print(f"[OK] overlay: {summary['overlay_path']}")
    print(f"[OK] json: {summary['json_path']}")


if __name__ == "__main__":
    main()
