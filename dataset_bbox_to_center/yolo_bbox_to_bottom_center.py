import argparse
import math
import shutil
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Convert YOLO bbox+point dataset to crop-based bottom-center dataset")
    parser.add_argument("--src_root", default=str(project_root / "dataset_pedestrian_bbox"))
    parser.add_argument("--dst_root", default=str(project_root / "dataset_bottom_center"))
    parser.add_argument("--pad", type=int, default=8, help="Extra padding around each bbox crop")
    parser.add_argument("--overwrite", action="store_true", help="Clear destination image/label folders before writing")
    return parser.parse_args()


def clip_box(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    x0_i = max(0, min(width, int(math.floor(x0))))
    y0_i = max(0, min(height, int(math.floor(y0))))
    x1_i = max(0, min(width, int(math.ceil(x1))))
    y1_i = max(0, min(height, int(math.ceil(y1))))
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return x0_i, y0_i, x1_i, y1_i


def clear_split_dir(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_split(src_root: Path, dst_root: Path, split: str, pad: int) -> int:
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split
    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for label_path in sorted(src_lbl_dir.glob("*.txt")):
        image_path = find_image(src_img_dir, label_path.stem)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]

        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for obj_idx, line in enumerate(lines):
            fields = line.split()
            if len(fields) < 8:
                continue

            class_id = int(float(fields[0]))
            cx = float(fields[1]) * width
            cy = float(fields[2]) * height
            bw = float(fields[3]) * width
            bh = float(fields[4]) * height
            kpt_x_abs = float(fields[5]) * width
            kpt_y_abs = float(fields[6]) * height
            kpt_vis = float(fields[7])
            if kpt_vis <= 0:
                continue

            x0 = cx - bw / 2.0
            y0 = cy - bh / 2.0
            x1 = cx + bw / 2.0
            y1 = cy + bh / 2.0

            crop_xyxy = clip_box(x0 - pad, y0 - pad, x1 + pad, y1 + pad, width=width, height=height)
            if crop_xyxy is None:
                continue

            crop_x0, crop_y0, crop_x1, crop_y1 = crop_xyxy
            crop = image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            bbox_x0 = x0 - crop_x0
            bbox_y0 = y0 - crop_y0
            bbox_x1 = x1 - crop_x0
            bbox_y1 = y1 - crop_y0
            bbox_cx = ((bbox_x0 + bbox_x1) / 2.0) / crop_w
            bbox_cy = ((bbox_y0 + bbox_y1) / 2.0) / crop_h
            bbox_w = (bbox_x1 - bbox_x0) / crop_w
            bbox_h = (bbox_y1 - bbox_y0) / crop_h

            kpt_x = (kpt_x_abs - crop_x0) / crop_w
            kpt_y = (kpt_y_abs - crop_y0) / crop_h

            if not (0.0 <= kpt_x <= 1.0 and 0.0 <= kpt_y <= 1.0):
                continue

            crop_stem = f"{image_path.stem}__obj{obj_idx:03d}"
            out_img_path = dst_img_dir / f"{crop_stem}.jpg"
            out_lbl_path = dst_lbl_dir / f"{crop_stem}.txt"

            cv2.imwrite(str(out_img_path), crop)
            out_lbl_path.write_text(
                f"{class_id} {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f} {kpt_x:.6f} {kpt_y:.6f} {int(kpt_vis)}\n",
                encoding="utf-8",
            )
            written += 1

    return written


def write_data_yaml(dst_root: Path) -> None:
    yaml_path = dst_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dst_root}",
                "",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "kpt_shape: [1, 3]",
                "flip_idx: [0]",
                "names:",
                "  0: pedestrian",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    if args.overwrite:
        for split in ("train", "val", "test"):
            clear_split_dir(dst_root / "images" / split)
            clear_split_dir(dst_root / "labels" / split)

    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = convert_split(src_root, dst_root, split=split, pad=args.pad)

    write_data_yaml(dst_root)

    for split in ("train", "val", "test"):
        print(f"[OK] {split}: wrote {counts[split]} crops")
    print(f"[OK] output: {dst_root}")


if __name__ == "__main__":
    main()
