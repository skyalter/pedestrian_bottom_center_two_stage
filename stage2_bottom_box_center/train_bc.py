"""
BottomCenterNet 训练脚本（直接坐标回归）

损失函数：SmoothL1 on (x_norm, y_norm)
默认输入尺寸：256×256
默认 batch_size：32
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from bc_datasets.dataloader import build_dataloaders
from bc_models.bottom_center_net import BottomCenterNet


def pixel_error(pred_norm: torch.Tensor, target_norm: torch.Tensor,
                img_w: float, img_h: float) -> torch.Tensor:
    """每个样本的像素欧氏距离（输入分辨率下）。"""
    pred_px = pred_norm * torch.tensor([img_w, img_h], device=pred_norm.device)
    tgt_px  = target_norm * torch.tensor([img_w, img_h], device=target_norm.device)
    return torch.norm(pred_px - tgt_px, dim=1)  # (B,)


def step_loop(model, dataloader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    criterion = torch.nn.SmoothL1Loss()

    total_loss    = 0.0
    total_px_err  = 0.0
    total_samples = 0

    for batch in dataloader:
        images    = batch["image"].to(device, non_blocking=True)         # (B,3,H,W)
        center_xy = batch["center_xy_in"].to(device, non_blocking=True)  # (B,2) pixels
        img_size  = batch["img_size"]                                     # (B,2) [H, W]
        B = images.size(0)

        H = float(img_size[0, 0])
        W = float(img_size[0, 1])
        target = torch.stack([center_xy[:, 0] / W,
                               center_xy[:, 1] / H], dim=1)  # (B,2) ∈ [0,1]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(images)                              # (B,2) ∈ [0,1]
        loss = criterion(pred, target)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        with torch.no_grad():
            px_err = pixel_error(pred.detach(), target, W, H).mean().item()

        total_loss    += loss.item() * B
        total_px_err  += px_err * B
        total_samples += B

    denom = max(total_samples, 1)
    return {
        "loss":   total_loss   / denom,
        "px_err": total_px_err / denom,
    }


def save_checkpoint(state: dict, is_best: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, "last.pt"))
    if is_best:
        torch.save(state, os.path.join(out_dir, "best.pt"))
        print(f"  ★ 保存 best.pt  (val_loss={state['val_loss']:.5f})")


def parse_args():
    yolo_root = Path("/home/mtl/Desktop/mtl_datasets/pedestrian_crops_yolo")

    parser = argparse.ArgumentParser(description="Train BottomCenterNet")
    parser.add_argument("--dataset_format", choices=["coco_bbox", "yolo_keypoint"],
                        default="yolo_keypoint")
    parser.add_argument("--images_root", default=str(yolo_root / "images"))
    parser.add_argument("--labels_root", default=str(yolo_root / "labels"))
    parser.add_argument("--data_yaml",   default=str(yolo_root / "data.yaml"))
    parser.add_argument("--ann_json",    default=None)

    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--img_h",        type=int,   default=256)
    parser.add_argument("--img_w",        type=int,   default=256)
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--train_ratio",  type=float, default=0.9)
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


def unwrap_dataset(ds):
    return ds.dataset if hasattr(ds, "dataset") else ds


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    time_str = time.strftime("%Y%m%d_%H%M%S")
    run_dir  = os.path.join("logs",        f"bc_net_{time_str}")
    ckpt_dir = os.path.join("checkpoints", f"bc_net_{time_str}")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # ── 数据 ──────────────────────────────────────────────────────────
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

    # ── 模型 ──────────────────────────────────────────────────────────
    model = BottomCenterNet(num_classes=num_classes,
                            dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] 参数量: {n_params:,} ({n_params/1e6:.3f}M)")

    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = step_loop(model, train_loader, device, optimizer)

        with torch.no_grad():
            val_metrics = step_loop(model, val_loader, device)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:03d}/{args.epochs}]  lr={cur_lr:.2e}  "
            f"train loss={train_metrics['loss']:.5f} px_err={train_metrics['px_err']:.2f}px  "
            f"val loss={val_metrics['loss']:.5f} px_err={val_metrics['px_err']:.2f}px"
        )

        writer.add_scalar("lr",           cur_lr,                  epoch)
        writer.add_scalar("train/loss",   train_metrics["loss"],   epoch)
        writer.add_scalar("train/px_err", train_metrics["px_err"], epoch)
        writer.add_scalar("val/loss",     val_metrics["loss"],     epoch)
        writer.add_scalar("val/px_err",   val_metrics["px_err"],   epoch)

        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]

        save_checkpoint(
            state={
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss":        val_metrics["loss"],
                "val_px_err":      val_metrics["px_err"],
                "num_classes":     num_classes,
                "img_size":        (args.img_h, args.img_w),
                "model_type":      "bc_net",
            },
            is_best=is_best,
            out_dir=ckpt_dir,
        )

    print(f"\n训练完成。最优 val loss: {best_val:.5f}")
    writer.close()


if __name__ == "__main__":
    main()
