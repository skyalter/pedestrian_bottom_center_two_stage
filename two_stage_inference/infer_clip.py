import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from infer_image import run_two_stage_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage inference on an ordered image folder and export a video")
    parser.add_argument("--source_dir", required=True, help="Folder containing ordered image frames")
    parser.add_argument(
        "--det_weights",
        default=None,
        help="Stage1 detector weights. If omitted, infer_image.py defaults are used.",
    )
    parser.add_argument(
        "--bc_weights",
        default=None,
        help="Stage2 bottom-center weights. If omitted, infer_image.py defaults are used.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--det_imgsz", type=int, default=1280)
    parser.add_argument("--det_conf", type=float, default=0.25)
    parser.add_argument("--det_iou", type=float, default=0.45)
    parser.add_argument("--crop_pad", type=int, default=8)
    parser.add_argument("--bc_h", type=int, default=256)
    parser.add_argument("--bc_w", type=int, default=256)
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0, help="Output video fps")
    parser.add_argument("--save_dir", default=None, help="Output root for clip results")
    return parser.parse_args()


def list_frames(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    t0_clip = time.perf_counter()
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {source_dir}")

    frames = list_frames(source_dir)
    if not frames:
        raise RuntimeError(f"No image frames found in {source_dir}")

    save_root = Path(args.save_dir).resolve() if args.save_dir else source_dir.parent / f"{source_dir.name}_two_stage_runs"
    clip_dir = save_root / source_dir.name
    frame_runs_dir = clip_dir / "frame_runs"
    clip_dir.mkdir(parents=True, exist_ok=True)
    frame_runs_dir.mkdir(parents=True, exist_ok=True)

    frame_summaries = []
    video_writer = None
    video_path = clip_dir / f"{source_dir.name}_overlay.mp4"

    for frame_idx, frame_path in enumerate(frames):
        t0_frame = time.perf_counter()
        frame_args = argparse.Namespace(
            source=str(frame_path),
            det_weights=args.det_weights or str(
                PROJECT_ROOT / "stage1_yolo26" / "runs" / "detect" / "train" / "weights" / "best.pt"
            ),
            bc_weights=args.bc_weights or str(
                PROJECT_ROOT / "stage2_bottom_box_center" / "checkpoints" / "bottom_center_latest" / "best.pt"
            ),
            device=args.device,
            det_imgsz=args.det_imgsz,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            crop_pad=args.crop_pad,
            bc_h=args.bc_h,
            bc_w=args.bc_w,
            class_id=args.class_id,
            save_dir=str(frame_runs_dir),
        )

        summary = run_two_stage_image(frame_args)
        frame_total_ms = (time.perf_counter() - t0_frame) * 1000.0
        overlay = cv2.imread(summary["overlay_path"], cv2.IMREAD_COLOR)
        if overlay is None:
            raise RuntimeError(f"Cannot read frame overlay: {summary['overlay_path']}")

        if video_writer is None:
            height, width = overlay.shape[:2]
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (width, height),
                True,
            )

        video_writer.write(overlay)
        frame_summaries.append(
            {
                "frame_index": frame_idx,
                "frame_name": frame_path.name,
                "source": str(frame_path),
                "overlay_path": summary["overlay_path"],
                "json_path": summary["json_path"],
                "num_detections": summary["num_detections"],
                "timing_ms": summary["timing_ms"],
                "frame_total_ms": frame_total_ms,
            }
        )

    if video_writer is not None:
        video_writer.release()

    clip_summary = {
        "source_dir": str(source_dir),
        "num_frames": len(frames),
        "fps": args.fps,
        "video_path": str(video_path),
        "timing_ms": {
            "total": (time.perf_counter() - t0_clip) * 1000.0,
            "avg_per_frame": (
                sum(frame["frame_total_ms"] for frame in frame_summaries) / len(frame_summaries)
                if frame_summaries else 0.0
            ),
            "sum_frame_total": sum(frame["frame_total_ms"] for frame in frame_summaries),
        },
        "frames": frame_summaries,
        "params": {
            "det_weights": args.det_weights,
            "bc_weights": args.bc_weights,
            "device": args.device,
            "det_imgsz": args.det_imgsz,
            "det_conf": args.det_conf,
            "det_iou": args.det_iou,
            "crop_pad": args.crop_pad,
            "bc_img_size": [args.bc_h, args.bc_w],
            "class_id": args.class_id,
        },
    }
    summary_path = clip_dir / f"{source_dir.name}_summary.json"
    csv_path = clip_dir / f"{source_dir.name}_timing.csv"
    summary_path.write_text(json.dumps(clip_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "frame_name",
                "num_detections",
                "detector_ms",
                "bottom_center_ms",
                "frame_total_ms",
                "overlay_path",
                "json_path",
            ]
        )
        for frame in frame_summaries:
            writer.writerow(
                [
                    frame["frame_index"],
                    frame["frame_name"],
                    frame["num_detections"],
                    f"{frame['timing_ms']['detector']:.4f}",
                    f"{frame['timing_ms']['bottom_center']:.4f}",
                    f"{frame['frame_total_ms']:.4f}",
                    frame["overlay_path"],
                    frame["json_path"],
                ]
            )

    print(f"[OK] frames: {len(frames)}")
    print(
        "[OK] timing_ms: "
        f"total={clip_summary['timing_ms']['total']:.2f}, "
        f"avg_per_frame={clip_summary['timing_ms']['avg_per_frame']:.2f}"
    )
    print(f"[OK] video: {video_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] csv: {csv_path}")


if __name__ == "__main__":
    main()
