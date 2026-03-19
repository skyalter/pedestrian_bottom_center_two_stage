# loss.py
"""
Final data structure:
{
    "total": total_loss,          # total = heatmap_loss + λ * offset_loss
    "heatmap": heatmap_loss,      # scalar tensor
    "offset": offset_loss         # scalar tensor
}
Each input:
    heatmap_pred: [B, C, H, W]  - predicted logits
    offset_pred : [B, 2, H, W]
    heatmap_gt  : [B, C, H, W]  - Gaussian heatmap (values in [0,1])
    offset_gt   : [B, 2, H, W]
    offset_mask : [B, 1, H, W]
    cat_idx     : [B]           - category index for each sample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLossHeatmap(nn.Module):
    """
    CenterNet-style focal loss for Gaussian heatmaps (single channel).
    Expected input shapes: pred/gt ∈ [B, 1, H, W]
      - pred: logits (not passed through sigmoid)
      - gt  : Gaussian heatmap (0~1, peak usually = 1)
    Formulation:
      pos: log(p) * (1 - p)^α
      neg: log(1 - p) * p^α * (1 - gt)^β
    Normalized by the number of positive samples.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        assert pred.dim() == 4 and gt.dim() == 4, "FocalLossHeatmap expects 4D tensors"
        assert pred.size(1) == 1 and gt.size(1) == 1, "FocalLossHeatmap expects [B,1,H,W]"

        pred = torch.clamp(pred.sigmoid(), 1e-6, 1.0 - 1e-6)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_weight = torch.pow(1 - gt, self.beta)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weight * neg_inds

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos > 0:
            loss = -(pos_loss + neg_loss) / num_pos
        else:
            loss = -neg_loss
        return loss


class MaskedL1Loss(nn.Module):
    """
    L1 loss computed only where mask == 1 (for offset branch).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert pred.shape == gt.shape, "pred and gt must have the same shape"
        assert mask.dim() == 4 and mask.size(1) == 1, "mask must be [B,1,H,W]"

        mask = mask.expand_as(gt).float()
        loss = F.l1_loss(pred * mask, gt * mask, reduction="sum")
        denom = mask.sum() + 1e-4
        return loss / denom


class CenterLoss(nn.Module):
    """
    Combined loss for bottom-center point prediction:
      - Heatmap: multi-channel input [B,C,H,W], select one by cat_idx → [B,1,H,W],
        then apply FocalLossHeatmap
      - Offset : Masked L1 loss, computed only on mask==1

    Example:
        loss_dict = criterion(
            heatmap_pred=pred_hm, offset_pred=pred_off,
            heatmap_gt=gt_hm, offset_gt=gt_off,
            offset_mask=gt_mask, cat_idx=cat_idx
        )
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, lambda_off: float = 1.0):
        super().__init__()
        self.heatmap_loss = FocalLossHeatmap(alpha, beta)
        self.offset_loss = MaskedL1Loss()
        self.lambda_off = lambda_off

    @staticmethod
    def _gather_by_cat_idx(tensor: torch.Tensor, cat_idx: torch.Tensor) -> torch.Tensor:
        """
        Gather one channel per sample from [B,C,H,W] using cat_idx → [B,1,H,W].
        """
        assert tensor.dim() == 4, "expect [B,C,H,W]"
        assert cat_idx.dim() == 1 and cat_idx.size(0) == tensor.size(0), "cat_idx must be [B]"
        B = tensor.size(0)
        b_idx = torch.arange(B, device=tensor.device)
        return tensor[b_idx, cat_idx.long(), :, :].unsqueeze(1)

    def forward(
        self,
        heatmap_pred: torch.Tensor,
        offset_pred: torch.Tensor,
        heatmap_gt: torch.Tensor,
        offset_gt: torch.Tensor,
        offset_mask: torch.Tensor,
        cat_idx: torch.Tensor
    ):
        heatmap_pred_1 = self._gather_by_cat_idx(heatmap_pred, cat_idx)
        heatmap_gt_1   = self._gather_by_cat_idx(heatmap_gt, cat_idx)

        hm_loss = self.heatmap_loss(heatmap_pred_1, heatmap_gt_1)
        off_loss = self.offset_loss(offset_pred, offset_gt, offset_mask)

        total = hm_loss + self.lambda_off * off_loss
        return {"total": total, "heatmap": hm_loss, "offset": off_loss}
