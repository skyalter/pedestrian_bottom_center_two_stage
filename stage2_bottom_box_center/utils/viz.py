import numpy as np
import cv2
import torch
import torchvision.utils as vutils
from typing import List, Tuple

# Denormalization params for ImageNet pre-trained models
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    """3xHxW tensor -> HxWx3 uint8 RGB"""
    x = img_t.detach().cpu().numpy()
    x = (np.transpose(x, (1, 2, 0)) * STD + MEAN)
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def heatmap_to_color(hm: np.ndarray) -> np.ndarray:
    """hm: (H,W) in [0,1] or arbitrary"""
    h = hm.astype(np.float32)
    if h.max() > 0:
        h = h / (h.max() + 1e-6)
    h = (h * 255).astype(np.uint8)
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)  # BGR
    return h


def decode_pred_point(heatmap_pred: torch.Tensor, offset_pred: torch.Tensor, stride: int = 4):
    with torch.no_grad():
        B, C, H, W = heatmap_pred.shape
        hm = torch.sigmoid(heatmap_pred)
        hm_red = hm.max(dim=1).values
        idx = hm_red.view(B, -1).argmax(dim=1)
        jj = (idx // W).long()
        ii = (idx % W).long()
        b_idx = torch.arange(B, device=heatmap_pred.device)
        offx = offset_pred[b_idx, 0, jj, ii]
        offy = offset_pred[b_idx, 1, jj, ii]
        x_in = (ii.float() + offx) * stride
        y_in = (jj.float() + offy) * stride
        pts = [(float(x_in[b].item()), float(y_in[b].item())) for b in range(B)]
    return pts



def visualize_batch(batch, images, heatmap_pred, offset_pred, stride: int = 4, max_vis: int = 8) -> np.ndarray:

    B = images.shape[0]
    n = min(B, max_vis)

    hm_1ch = heatmap_pred.sigmoid().max(dim=1, keepdim=True).values
    hm_up = torch.nn.functional.interpolate(
        hm_1ch, size=images.shape[-2:], mode="bilinear", align_corners=False
    )  


    pred_xy_in = decode_pred_point(heatmap_pred, offset_pred, stride=stride)

    vis_list = []
    for b in range(n):
        img_rgb = denormalize(images[b])
        if "center_xy_in" in batch:
            cx_gt, cy_gt = batch["center_xy_in"][b].detach().cpu().numpy().tolist()
        else:
            m = batch["offset_mask"][b, 0]
            jj, ii = torch.nonzero(m, as_tuple=True)
            if len(jj) == 0:
                cx_gt, cy_gt = -1, -1
            else:
                cx_gt = (ii[0].item() + batch["offset_map"][b, 0, jj[0], ii[0]].item()) * stride
                cy_gt = (jj[0].item() + batch["offset_map"][b, 1, jj[0], ii[0]].item()) * stride

        hm = hm_up[b, 0].detach().cpu().numpy()
        hm_color_bgr = heatmap_to_color(hm)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, hm_color_bgr, 0.4, 0)

        cv2.circle(overlay, (int(round(cx_gt)), int(round(cy_gt))), 3, (255, 0, 0), -1)
        px, py = pred_xy_in[b]
        cv2.circle(overlay, (int(round(px)), int(round(py))), 3, (0, 255, 0), -1)

        vis_list.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    canvas = np.concatenate(vis_list, axis=1) if len(vis_list) > 1 else vis_list[0]
    return canvas


def add_val_visuals(writer, epoch, batch, heatmap_pred, offset_pred):

    hm_gt = batch["heatmap"].detach().cpu().float() 
    print(f"[Debug] batch['heatmap'].shape = {hm_gt.shape}, dtype = {hm_gt.dtype}")
    hm_pred = torch.sigmoid(heatmap_pred.detach().cpu())

    hm_gt_vis   = hm_gt.max(dim=1, keepdim=True).values
    hm_pred_vis = hm_pred.max(dim=1, keepdim=True).values 

    n = min(8, hm_gt.shape[0])
    grid_gt = vutils.make_grid(hm_gt_vis[:n], nrow=n, normalize=True)
    grid_pred = vutils.make_grid(hm_pred_vis[:n], nrow=n, normalize=True)

    writer.add_image("val/heatmap_gt", grid_gt, global_step=epoch)
    writer.add_image("val/heatmap_pred", grid_pred, global_step=epoch)

    off_gt = batch["offset_map"].detach().cpu()
    off_pred = offset_pred.detach().cpu()

    grid_offx_gt = vutils.make_grid(off_gt[:, 0:1, :, :][:n], nrow=n, normalize=True)
    grid_offy_gt = vutils.make_grid(off_gt[:, 1:2, :, :][:n], nrow=n, normalize=True)
    grid_offx_pred = vutils.make_grid(off_pred[:, 0:1, :, :][:n], nrow=n, normalize=True)
    grid_offy_pred = vutils.make_grid(off_pred[:, 1:2, :, :][:n], nrow=n, normalize=True)

    writer.add_image("val/offset_x_gt", grid_offx_gt, epoch)
    writer.add_image("val/offset_y_gt", grid_offy_gt, epoch)
    writer.add_image("val/offset_x_pred", grid_offx_pred, epoch)
    writer.add_image("val/offset_y_pred", grid_offy_pred, epoch)

    images = batch["image"]
    canvas = visualize_batch(batch, images, heatmap_pred, offset_pred, stride=4, max_vis=8)
    writer.add_image("val/overlay", torch.from_numpy(canvas).permute(2, 0, 1), epoch)