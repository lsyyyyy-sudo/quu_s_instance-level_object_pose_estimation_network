"""角点热图损失 = focal loss (+ 可选的亚像素偏移回归)。

对应 BoxDreamer 的 ``lossesV3.py``：BoxDreamer 同时监督
``bbox_feat``（角点表征）与射线/位姿；本项目只监督角点热图这一个头。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.utils.focal_loss import focal_loss


class CornerHeatmapLoss(nn.Module):
    """Args 与 ``configs/model/loss/default.yaml`` 对应。"""

    def __init__(
        self,
        focal_weight: float = 1.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        offset_weight: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.focal_weight = float(focal_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_beta = float(focal_beta)
        self.offset_weight = float(offset_weight)
        self.eps = float(eps)

    def forward(
        self,
        pred_heatmap: torch.Tensor,
        gt_heatmap: torch.Tensor,
        pred_offset: Optional[torch.Tensor] = None,
        corner_2d: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
        heatmap_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_heatmap: ``[B, K, h, w]`` logits
            gt_heatmap:   ``[B, K, h, w]`` 高斯峰标签
            pred_offset:  ``[B, 2K, h, w]`` 亚像素偏移预测（可选）
            corner_2d:    ``[B, K, 2]`` GT 角点（图像像素），用于算偏移标签
            image_size / heatmap_size: 二者的换算比例

        Returns:
            ``{"loss": 标量, "loss_focal": ..., "loss_offset": ...}``
        """
        loss_focal = focal_loss(
            pred_heatmap, gt_heatmap, alpha=self.focal_alpha, beta=self.focal_beta, eps=self.eps
        )

        loss_offset = pred_heatmap.new_zeros(())
        if pred_offset is not None and self.offset_weight > 0.0 and corner_2d is not None:
            loss_offset = self._offset_loss(
                pred_offset, corner_2d, heatmap_size, image_size
            )

        total = self.focal_weight * loss_focal + self.offset_weight * loss_offset
        return {
            "loss": total,
            "loss_focal": loss_focal.detach(),
            "loss_offset": loss_offset.detach(),
        }

    @staticmethod
    def _offset_loss(
        pred_offset: torch.Tensor,
        corner_2d: torch.Tensor,
        heatmap_size: int,
        image_size: int,
    ) -> torch.Tensor:
        """在 GT 角点所在的热图格子上监督「真实位置 - 格子左上角」的偏移。"""
        B, K, _ = corner_2d.shape
        scale = float(heatmap_size) / float(image_size)
        centers = corner_2d.detach() * scale                      # [B, K, 2] 热图坐标
        floor = torch.floor(centers)
        residual = centers - floor                                # [B, K, 2]，值域 (-1, 1) 附近

        ix = floor[..., 0].clamp(0, heatmap_size - 1).long()
        iy = floor[..., 1].clamp(0, heatmap_size - 1).long()

        pred_offset = pred_offset.reshape(B, K, 2, heatmap_size, heatmap_size)
        gather_x = pred_offset[torch.arange(B, device=pred_offset.device).view(B, 1), torch.arange(K, device=pred_offset.device).view(1, K), 0, iy, ix]
        gather_y = pred_offset[torch.arange(B, device=pred_offset.device).view(B, 1), torch.arange(K, device=pred_offset.device).view(1, K), 1, iy, ix]

        pred_res = torch.stack([gather_x, gather_y], dim=-1)      # [B, K, 2]
        return F.l1_loss(pred_res, residual)
