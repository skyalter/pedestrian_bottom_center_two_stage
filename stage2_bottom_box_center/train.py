import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloader import build_dataloaders
from models.bottom_center import BottomCenterModel
from losses.loss import CenterLoss
from utils.viz import visualize_batch, add_val_visuals


def step_loop(model, dataloader, criterion, device, optimizer=None):

    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss_sum = 0.0
    total_hm_sum  = 0.0
    total_off_sum = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        heatmap_gt = batch["heatmap"].to(device, non_blocking=True)
        offset_gt = batch["offset_map"].to(device, non_blocking=True)
        offset_mask = batch["offset_mask"].to(device, non_blocking=True)
        cat_idx = batch["cat_idx"].to(device, non_blocking=True).long()
        B = images.size(0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        heatmap_pred, offset_pred = model(images)
        losses = criterion(heatmap_pred, offset_pred, heatmap_gt, offset_gt, offset_mask, cat_idx)
        loss = losses["total"]

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss_sum += loss.item() * B
        total_hm_sum   += losses["heatmap"].item() * B
        total_off_sum  += losses["offset"].item() * B
        total_samples  += B

    denom = max(total_samples, 1)

    return {
        "total":        total_loss_sum / denom,
        "heatmap_loss": total_hm_sum / denom,
        "offset_loss":  total_off_sum / denom,
    }


def save_checkpoint(state, is_best, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "last.pt")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(out_dir, "best.pt")
        torch.save(state, best_path)


def parse_args():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train bottom-box-center model")
    yolo_root = project_root.parent / "dataset_bottom_center"
    parser.add_argument("--dataset_format", choices=["coco_bbox", "yolo_keypoint"], default="yolo_keypoint")
    parser.add_argument("--images_root", default=str(yolo_root / "images"))
    parser.add_argument("--labels_root", default=str(yolo_root / "labels"))
    parser.add_argument("--data_yaml", default=str(yolo_root / "data.yaml"))
    parser.add_argument("--ann_json", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def unwrap_dataset(ds):
    return ds.dataset if hasattr(ds, "dataset") else ds


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # paths & TensorBoard
    time_str = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", f"bottom_center_{time_str}")
    ckpt_dir = os.path.join("checkpoints", f"bottom_center_{time_str}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # dataloader
    train_loader, val_loader = build_dataloaders(
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        images_root=args.images_root,
        ann_json=args.ann_json,
        labels_root=args.labels_root,
        data_yaml=args.data_yaml,
        img_size=(args.img_h, args.img_w),
        stride=4,
        sigma_mode="auto",
        sigma_fixed=2.0,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    num_classes = unwrap_dataset(train_loader.dataset).num_classes

    # define model & loss
    model = BottomCenterModel(num_classes=num_classes).to(device)

    criterion = CenterLoss(alpha=2.0, beta=4.0, lambda_off=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epochs = args.epochs

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = step_loop(model, train_loader, criterion, device, optimizer)

        # Val
        model.eval()
        with torch.no_grad():
            val_metrics = step_loop(model, val_loader, criterion, device, optimizer=None)

            val_first_batch = next(iter(val_loader))
            images_val = val_first_batch["image"].to(device, non_blocking=True)
            heatmap_pred_val, offset_pred_val = model(images_val)
            add_val_visuals(writer, epoch, val_first_batch, heatmap_pred_val, offset_pred_val)

        # Scheduler
        scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]

        # Console Logs
        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"LR={cur_lr:.2e}  ||  "
            f"Train total={train_metrics['total']:.4f} | hm={train_metrics['heatmap_loss']:.4f} | off={train_metrics['offset_loss']:.4f}  ||  "
            f"Val total={val_metrics['total']:.4f} | hm={val_metrics['heatmap_loss']:.4f} | off={val_metrics['offset_loss']:.4f}"
        )

        # ---- TensorBoard Scalars ----
        writer.add_scalar("lr", cur_lr, epoch)
        writer.add_scalar("train/total", train_metrics["total"], epoch)
        writer.add_scalar("train/heatmap", train_metrics["heatmap_loss"], epoch)
        writer.add_scalar("train/offset", train_metrics["offset_loss"], epoch)
        writer.add_scalar("val/total", val_metrics["total"], epoch)
        writer.add_scalar("val/heatmap", val_metrics["heatmap_loss"], epoch)
        writer.add_scalar("val/offset", val_metrics["offset_loss"], epoch)

        # ---- Save ----
        is_best = val_metrics["total"] < best_val
        if is_best:
            best_val = val_metrics["total"]

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_total_loss": val_metrics["total"],
                "train_total_loss": train_metrics["total"],
                "num_classes": num_classes,
                "stride": 4,    
            },
            is_best=is_best,
            out_dir=ckpt_dir,
        )


    print(f"Best Val total loss: {best_val:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
