"""
Two-stage pedestrian bottom-center inference.

Stage 1: YOLO pedestrian bbox detection
Stage 2: BottomCenterNet bottom-center point prediction on each crop

Usage — single image:
    python two_stage_infer.py --source /path/to/image.jpg

Usage — image folder (clip):
    python two_stage_infer.py --source /path/to/frames_dir --fps 10

Usage — import:
    from two_stage_infer import load_models, run_image
    detector, bc_model = load_models()
    summary = run_image("/path/to/image.jpg", detector, bc_model)
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
STAGE2_ROOT  = PROJECT_ROOT / "stage2_bottom_box_center"
if str(STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2_ROOT))

# ── default model weights (override via CLI args or function kwargs) ───────────
DEFAULT_DET_WEIGHTS = str(PROJECT_ROOT / "stage1_yolo26" / "yolo26_pedbbox_03-30.pt")
DEFAULT_BC_WEIGHTS  = str(PROJECT_ROOT / "stage2_bottom_box_center" / "checkpoints" / "bc_net_20260329_183535" / "best.pt")
DEFAULT_DEVICE      = "cuda"
DEFAULT_DET_IMGSZ   = 1280
DEFAULT_DET_CONF    = 0.25
DEFAULT_DET_IOU     = 0.45
DEFAULT_CROP_PAD    = 8
DEFAULT_BC_H        = 256
DEFAULT_BC_W        = 256
DEFAULT_CLASS_ID    = 0

from bc_infer import infer_images, load_model as load_bc_model  # noqa: E402  (stage2)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _clip_box(
    x0: float, y0: float, x1: float, y1: float,
    width: int, height: int,
) -> tuple[int, int, int, int] | None:
    x0_i = max(0, min(width,  int(math.floor(x0))))
    y0_i = max(0, min(height, int(math.floor(y0))))
    x1_i = max(0, min(width,  int(math.ceil(x1))))
    y1_i = max(0, min(height, int(math.ceil(y1))))
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return x0_i, y0_i, x1_i, y1_i


def _list_frames(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(
    det_weights: str = DEFAULT_DET_WEIGHTS,
    bc_weights:  str = DEFAULT_BC_WEIGHTS,
    device:      str = DEFAULT_DEVICE,
    det_imgsz:   int = DEFAULT_DET_IMGSZ,
    bc_h:        int = DEFAULT_BC_H,
    bc_w:        int = DEFAULT_BC_W,
) -> tuple:
    """Load both models once and warm up GPU. Returns (detector, bc_model_tuple)."""
    device = device if torch.cuda.is_available() else "cpu"

    detector = YOLO(det_weights)
    bc_model = load_bc_model(bc_weights, device=device)

    # warm up: one dummy forward so subsequent calls have no CUDA kernel init cost
    dummy_img = np.zeros((bc_h, bc_w, 3), dtype=np.uint8)
    detector.predict(source=dummy_img, imgsz=det_imgsz, device=device, verbose=False)
    model, img_size = bc_model
    dummy_t = torch.zeros(1, 3, *img_size, device=device)
    with torch.no_grad():
        model(dummy_t)

    return detector, bc_model


# ── single-image inference ────────────────────────────────────────────────────

def run_image(
    source:      str,
    detector,
    bc_model:    tuple,
    save_dir:    str  = str(PROJECT_ROOT / "runs" / "two_stage"),
    device:      str  = DEFAULT_DEVICE,
    det_weights: str  = DEFAULT_DET_WEIGHTS,
    bc_weights:  str  = DEFAULT_BC_WEIGHTS,
    det_imgsz:   int  = DEFAULT_DET_IMGSZ,
    det_conf:    float = DEFAULT_DET_CONF,
    det_iou:     float = DEFAULT_DET_IOU,
    crop_pad:    int  = DEFAULT_CROP_PAD,
    bc_h:        int  = DEFAULT_BC_H,
    bc_w:        int  = DEFAULT_BC_W,
    class_id:    int  = DEFAULT_CLASS_ID,
) -> dict:
    """Run two-stage inference on a single image. Returns summary dict."""
    t0_total    = time.perf_counter()
    source_path = Path(source).resolve()
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source_path}")

    height, width = image.shape[:2]
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path(save_dir) / f"{source_path.stem}_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── stage 1: detect ──
    t0_det = time.perf_counter()
    det_results = detector.predict(
        source=str(source_path),
        imgsz=det_imgsz,
        conf=det_conf,
        iou=det_iou,
        device=device,
        verbose=False,
    )
    if not det_results:
        raise RuntimeError("Detector returned no result")
    t_det_ms = (time.perf_counter() - t0_det) * 1000.0

    boxes      = det_results[0].boxes
    crops:      list[np.ndarray] = []
    crop_names: list[str]        = []
    det_metas:  list[dict]       = []

    if boxes is not None:
        for idx in range(len(boxes)):
            x0, y0, x1, y1 = boxes.xyxy[idx].tolist()
            det_score = float(boxes.conf[idx].item())
            det_class = int(boxes.cls[idx].item())
            crop_xyxy = _clip_box(
                x0 - crop_pad, y0 - crop_pad,
                x1 + crop_pad, y1 + crop_pad,
                width=width, height=height,
            )
            if crop_xyxy is None:
                continue
            cx0, cy0, cx1, cy1 = crop_xyxy
            crop = image[cy0:cy1, cx0:cx1].copy()
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_names.append(f"{source_path.stem}_det{idx:03d}.jpg")
            det_metas.append({
                "index":        idx,
                "det_class_id": det_class,
                "det_score":    det_score,
                "bbox_xyxy":    [float(x0), float(y0), float(x1), float(y1)],
                "crop_xyxy":    [cx0, cy0, cx1, cy1],
            })

    # ── stage 2: bottom-center ──
    bc_results = []
    t_bc_ms    = 0.0
    if crops:
        t0_bc = time.perf_counter()
        bc_results = infer_images(
            images=crops,
            class_ids=[class_id] * len(crops),
            image_paths=crop_names,
            weights_path=bc_weights,
            device=device,
            img_size=(bc_h, bc_w),
            save_dir=None,
            preloaded_model=bc_model,
        )
        t_bc_ms = (time.perf_counter() - t0_bc) * 1000.0

    # ── overlay + results ──
    overlay    = image.copy()
    detections = []
    for det_meta, bc_result in zip(det_metas, bc_results):
        x0, y0, x1, y1 = det_meta["bbox_xyxy"]
        cx0, cy0        = det_meta["crop_xyxy"][:2]
        point_x = float(np.clip(cx0 + bc_result["x_pred"], 0, width  - 1))
        point_y = float(np.clip(cy0 + bc_result["y_pred"], 0, height - 1))

        cv2.rectangle(overlay,
                      (int(round(x0)), int(round(y0))),
                      (int(round(x1)), int(round(y1))),
                      (0, 200, 255), 2)
        cv2.circle(overlay, (int(round(point_x)), int(round(point_y))), 4, (0, 255, 0), -1)
        cv2.putText(overlay,
                    f"det={det_meta['det_score']:.2f} bc={bc_result['score']:.2f}",
                    (int(round(x0)), max(18, int(round(y0)) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        detections.append({
            "index":         det_meta["index"],
            "det_class_id":  det_meta["det_class_id"],
            "det_score":     det_meta["det_score"],
            "bbox_xyxy":     det_meta["bbox_xyxy"],
            "crop_xyxy":     det_meta["crop_xyxy"],
            "bc_point_crop": [float(bc_result["x_pred"]), float(bc_result["y_pred"])],
            "bc_point_full": [point_x, point_y],
            "bc_score":      float(bc_result["score"]),
        })

    overlay_path = work_dir / f"{source_path.stem}_overlay.jpg"
    json_path    = work_dir / f"{source_path.stem}_results.json"
    cv2.imwrite(str(overlay_path), overlay)

    summary = {
        "source":         str(source_path),
        "image_size":     [height, width],
        "det_weights":    det_weights,
        "bc_weights":     bc_weights,
        "num_detections": len(detections),
        "params": {
            "det_imgsz":   det_imgsz,
            "det_conf":    det_conf,
            "det_iou":     det_iou,
            "crop_pad":    crop_pad,
            "bc_img_size": [bc_h, bc_w],
            "class_id":    class_id,
        },
        "detections":   detections,
        "overlay_path": str(overlay_path),
        "json_path":    str(json_path),
        "timing_ms": {
            "detector":      t_det_ms,
            "bottom_center": t_bc_ms,
            "total":         (time.perf_counter() - t0_total) * 1000.0,
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ── clip (folder) inference ───────────────────────────────────────────────────

def run_clip(args: argparse.Namespace) -> None:
    """Run two-stage inference on every frame in a folder and export a video."""
    t0_clip    = time.perf_counter()
    source_dir = Path(args.source).resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {source_dir}")

    frames = _list_frames(source_dir)
    if not frames:
        raise RuntimeError(f"No image frames found in {source_dir}")

    save_root      = Path(args.save_dir).resolve()
    clip_dir       = save_root / source_dir.name
    frame_runs_dir = clip_dir / "frame_runs"
    clip_dir.mkdir(parents=True, exist_ok=True)
    frame_runs_dir.mkdir(parents=True, exist_ok=True)

    print("[Info] Loading models...")
    t0_load = time.perf_counter()
    detector, bc_model = load_models(
        det_weights=args.det_weights,
        bc_weights=args.bc_weights,
        device=args.device,
        det_imgsz=args.det_imgsz,
        bc_h=args.bc_h,
        bc_w=args.bc_w,
    )
    print(f"[Info] Models loaded in {(time.perf_counter()-t0_load)*1000:.0f} ms")

    frame_summaries = []
    video_writer    = None
    video_path      = clip_dir / f"{source_dir.name}_overlay.mp4"

    for frame_idx, frame_path in enumerate(frames):
        t0_frame = time.perf_counter()
        summary  = run_image(
            source=str(frame_path),
            detector=detector,
            bc_model=bc_model,
            save_dir=str(frame_runs_dir),
            device=args.device,
            det_weights=args.det_weights,
            bc_weights=args.bc_weights,
            det_imgsz=args.det_imgsz,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            crop_pad=args.crop_pad,
            bc_h=args.bc_h,
            bc_w=args.bc_w,
            class_id=args.class_id,
        )
        frame_total_ms = (time.perf_counter() - t0_frame) * 1000.0

        overlay = cv2.imread(summary["overlay_path"], cv2.IMREAD_COLOR)
        if overlay is None:
            raise RuntimeError(f"Cannot read frame overlay: {summary['overlay_path']}")

        if video_writer is None:
            h, w = overlay.shape[:2]
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (w, h),
                True,
            )
        video_writer.write(overlay)

        frame_summaries.append({
            "frame_index":    frame_idx,
            "frame_name":     frame_path.name,
            "source":         str(frame_path),
            "overlay_path":   summary["overlay_path"],
            "json_path":      summary["json_path"],
            "num_detections": summary["num_detections"],
            "timing_ms":      summary["timing_ms"],
            "frame_total_ms": frame_total_ms,
        })

    if video_writer is not None:
        video_writer.release()

    clip_summary = {
        "source_dir": str(source_dir),
        "num_frames": len(frames),
        "fps":        args.fps,
        "video_path": str(video_path),
        "timing_ms": {
            "total": (time.perf_counter() - t0_clip) * 1000.0,
            "avg_per_frame": (
                sum(f["frame_total_ms"] for f in frame_summaries) / len(frame_summaries)
                if frame_summaries else 0.0
            ),
            "sum_frame_total": sum(f["frame_total_ms"] for f in frame_summaries),
        },
        "frames": frame_summaries,
        "params": {
            "det_weights": args.det_weights,
            "bc_weights":  args.bc_weights,
            "device":      args.device,
            "det_imgsz":   args.det_imgsz,
            "det_conf":    args.det_conf,
            "det_iou":     args.det_iou,
            "crop_pad":    args.crop_pad,
            "bc_img_size": [args.bc_h, args.bc_w],
            "class_id":    args.class_id,
        },
    }
    summary_path = clip_dir / f"{source_dir.name}_summary.json"
    csv_path     = clip_dir / f"{source_dir.name}_timing.csv"
    summary_path.write_text(json.dumps(clip_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_index", "frame_name", "num_detections",
                         "detector_ms", "bottom_center_ms", "frame_total_ms",
                         "overlay_path", "json_path"])
        for f in frame_summaries:
            writer.writerow([
                f["frame_index"],
                f["frame_name"],
                f["num_detections"],
                f"{f['timing_ms']['detector']:.4f}",
                f"{f['timing_ms']['bottom_center']:.4f}",
                f"{f['frame_total_ms']:.4f}",
                f["overlay_path"],
                f["json_path"],
            ])

    print(f"[OK] frames: {len(frames)}")
    print(f"[OK] timing_ms: total={clip_summary['timing_ms']['total']:.2f}, "
          f"avg_per_frame={clip_summary['timing_ms']['avg_per_frame']:.2f}")
    print(f"[OK] video: {video_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] csv: {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage inference: YOLO pedestrian detection → BottomCenterNet.\n"
            "Pass an image file for single-image mode or a directory for clip mode."
        )
    )
    parser.add_argument("--source",      required=True,
                        help="Image file (single-image mode) or folder of frames (clip mode)")
    parser.add_argument("--det_weights", default=DEFAULT_DET_WEIGHTS)
    parser.add_argument("--bc_weights",  default=DEFAULT_BC_WEIGHTS)
    parser.add_argument("--device",      default=DEFAULT_DEVICE)
    parser.add_argument("--det_imgsz",   type=int,   default=DEFAULT_DET_IMGSZ)
    parser.add_argument("--det_conf",    type=float, default=DEFAULT_DET_CONF)
    parser.add_argument("--det_iou",     type=float, default=DEFAULT_DET_IOU)
    parser.add_argument("--crop_pad",    type=int,   default=DEFAULT_CROP_PAD)
    parser.add_argument("--bc_h",        type=int,   default=DEFAULT_BC_H)
    parser.add_argument("--bc_w",        type=int,   default=DEFAULT_BC_W)
    parser.add_argument("--class_id",    type=int,   default=DEFAULT_CLASS_ID)
    parser.add_argument("--fps",         type=float, default=10.0,
                        help="Output video FPS (clip mode only)")
    parser.add_argument("--save_dir",    default=None,
                        help="Output root directory (defaults next to source)")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    source = Path(args.source).resolve()

    if args.save_dir is None:
        if source.is_dir():
            args.save_dir = str(source.parent / f"{source.name}_two_stage_runs")
        else:
            args.save_dir = str(PROJECT_ROOT / "runs" / "two_stage")

    print("[Info] Loading models...")
    t0_load = time.perf_counter()
    detector, bc_model = load_models(
        det_weights=args.det_weights,
        bc_weights=args.bc_weights,
        device=args.device,
        det_imgsz=args.det_imgsz,
        bc_h=args.bc_h,
        bc_w=args.bc_w,
    )
    print(f"[Info] Models loaded in {(time.perf_counter()-t0_load)*1000:.0f} ms")

    if source.is_dir():
        run_clip(args)
    else:
        summary = run_image(
            source=str(source),
            detector=detector,
            bc_model=bc_model,
            save_dir=args.save_dir,
            device=args.device,
            det_weights=args.det_weights,
            bc_weights=args.bc_weights,
            det_imgsz=args.det_imgsz,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            crop_pad=args.crop_pad,
            bc_h=args.bc_h,
            bc_w=args.bc_w,
            class_id=args.class_id,
        )
        print(f"[OK] detections: {summary['num_detections']}")
        print(f"[OK] timing_ms: total={summary['timing_ms']['total']:.2f}, "
              f"detector={summary['timing_ms']['detector']:.2f}, "
              f"bottom_center={summary['timing_ms']['bottom_center']:.2f}")
        print(f"[OK] overlay: {summary['overlay_path']}")
        print(f"[OK] json: {summary['json_path']}")


if __name__ == "__main__":
    main()
