import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train YOLO pedestrian bbox detector")
    parser.add_argument("--model", default="yolo26x.pt", help="YOLO pretrained checkpoint or model yaml")
    parser.add_argument("--data", default=str(project_root / "pedestrian.yaml"), help="YOLO dataset yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--close_mosaic", type=int, default=20)
    parser.add_argument("--project", default=str(project_root / "runs" / "detect"))
    parser.add_argument("--name", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
