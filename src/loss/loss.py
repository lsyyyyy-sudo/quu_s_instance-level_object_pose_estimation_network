"""角点热图损失：粗损失（整张热图）+ 细损失（8 个角点坐标）。

对应 BoxDreamer 论文 Sec 3.3 的::

    L = L_coarse + λ · L_fine,      λ = 2.0

- **coarse**：整张热图的逐元素损失。BoxDreamer 用 ``nn.SmoothL1Loss``，
  我们在配置里也用该默认值，同时保留 ``focal``（CenterNet 风格）作为备选。
- **fine**：8 个角点坐标的损失，把角点位置直接拉准，系数 λ 默认 2.0。

与 BoxDreamer 代码的差异
------------------------
BoxDreamer 的 fine 项监督的是一个**单独的坐标回归分支** ``regression_boxes``
（见它 ``configs/model/loss/default.yaml`` 里那行被注释掉的 ``weight: [2.0, 0.0]``，
在官方配置中**默认是关闭的**）。

我们为了控制参数量（目标是 8GB 显存能训），没有加额外的回归头，而是对**预测热图
做可微的 soft-argmax** 得到角点坐标，再与 GT 角点做 SmoothL1。这样 fine 项直接
优化我们最终真正使用的那个量。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.utils.focal_loss import focal_loss
from src.models.utils.prediction_utils import soft_argmax_2d, to_zero_one


def _smooth_l1(pred: torch.Tensor, gt: torch.Tensor, beta: float) -> torch.Tensor:
    """Huber / Smooth L1。``beta=1.0`` 时等价于 ``nn.SmoothL1Loss``。"""
    return F.smooth_l1_loss(pred, gt, beta=beta)


class CornerHeatmapLoss(nn.Module):
    """Args 与 ``configs/model/loss/default.yaml`` 一一对应。

    约定：**网络输出已经做过激活**，本模块不再动它。
    - ``heatmap_loss='smooth_l1'`` 时，预测与 GT 都在 ``[-1, 1]``（BoxDreamer 风格）
    - ``heatmap_loss='focal'`` 时，预测是 logits（CenterNet 风格）
    """

    def __init__(
        self,
        heatmap_loss: str = "smooth_l1",
        heatmap_weight: float = 1.0,
        smooth_l1_beta: float = 1.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        fine_weight: float = 2.0,
        fine_beta: float = 25.0,
        fine_loss: str = "smooth_l1",
        offset_weight: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if heatmap_loss not in ("smooth_l1", "focal"):
            raise ValueError(
                f"Unknown heatmap_loss: {heatmap_loss!r} (expected 'smooth_l1' or 'focal')"
            )
        if fine_loss not in ("smooth_l1", "l1"):
            raise ValueError(f"Unknown fine_loss: {fine_loss!r} (expected 'smooth_l1' or 'l1')")

        self.heatmap_loss = heatmap_loss
        self.heatmap_weight = float(heatmap_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.focal_alpha = float(focal_alpha)
        self.focal_beta = float(focal_beta)
        self.fine_weight = float(fine_weight)
        self.fine_beta = float(fine_beta)
        self.fine_loss = fine_loss
        self.offset_weight = float(offset_weight)
        self.eps = float(eps)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred_heatmap: torch.Tensor,
        gt_heatmap: torch.Tensor,
        corner_2d: Optional[torch.Tensor] = None,
        image_size: Optional[int] = None,
        heatmap_size: Optional[int] = None,
        pred_offset: Optional[torch.Tensor] = None,
        heatmap_range: str = "minus_one_to_one",
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_heatmap:  ``[B, K, h, w]`` 网络输出（已激活）
            gt_heatmap:    ``[B, K, h, w]`` 热图标签
            corner_2d:     ``[B, K, 2]`` GT 角点，图像像素坐标（fine 项用）
            image_size / heatmap_size: 二者的换算比例
            pred_offset:   ``[B, 2K, h, w]`` 亚像素偏移预测（可选）
            heatmap_range: 预测热图的取值区间，fine 项提取前会转到 ``[0, 1]``

        Returns:
            ``{"loss", "loss_coarse", "loss_fine", "loss_offset"}``
        """
        # ---- 粗损失：整张热图 ----
        if self.heatmap_loss == "smooth_l1":
            loss_coarse = _smooth_l1(pred_heatmap, gt_heatmap, self.smooth_l1_beta)
        else:
            loss_coarse = focal_loss(
                pred_heatmap, gt_heatmap,
                alpha=self.focal_alpha, beta=self.focal_beta, eps=self.eps,
            )

        # ---- 细损失：8 个角点坐标 ----
        loss_fine = pred_heatmap.new_zeros(())
        if self.fine_weight > 0.0 and corner_2d is not None:
            loss_fine = self._fine_loss(
                pred_heatmap, corner_2d, image_size, heatmap_size, heatmap_range
            )

        # ---- 可选的亚像素偏移 ----
        loss_offset = pred_heatmap.new_zeros(())
        if pred_offset is not None and self.offset_weight > 0.0 and corner_2d is not None:
            loss_offset = self._offset_loss(pred_offset, corner_2d, heatmap_size, image_size)

        total = (
            self.heatmap_weight * loss_coarse
            + self.fine_weight * loss_fine
            + self.offset_weight * loss_offset
        )
        return {
            "loss": total,
            "loss_coarse": loss_coarse.detach(),
            "loss_fine": loss_fine.detach(),
            "loss_offset": loss_offset.detach(),
        }

    # ------------------------------------------------------------------ #
    def _fine_loss(
        self,
        pred_heatmap: torch.Tensor,
        corner_2d: torch.Tensor,
        image_size: Optional[int],
        heatmap_size: Optional[int],
        heatmap_range: str,
    ) -> torch.Tensor:
        """对预测热图做 soft-argmax 得到角点，再与 GT 角点算 SmoothL1。

        坐标统一换算到**图像像素**空间，这样 fine 项的数值量级与
        BoxDreamer 用裁剪图像素坐标的设定接近，λ=2.0 才有可比性。
        """
        if image_size is None or heatmap_size is None:
            raise ValueError("fine loss requires image_size and heatmap_size")

        # soft-argmax 前统一到 [0, 1]，避免 [-1, 1] 与 [0, 1] 两种量程下
        # 同一个 beta 给出完全不同的锐度
        heat = to_zero_one(pred_heatmap, heatmap_range)
        corner_hm = soft_argmax_2d(heat, beta=self.fine_beta)
        corner_px = corner_hm * (float(image_size) / float(heatmap_size))
        if self.fine_loss == "smooth_l1":
            return _smooth_l1(corner_px, corner_2d, self.smooth_l1_beta)
        return F.l1_loss(corner_px, corner_2d)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _offset_loss(
        pred_offset: torch.Tensor,
        corner_2d: torch.Tensor,
        heatmap_size: Optional[int],
        image_size: Optional[int],
    ) -> torch.Tensor:
        """在 GT 角点所在的热图格子上监督「真实位置 − 格子左上角」的偏移。"""
        B, K, _ = corner_2d.shape
        if heatmap_size is None or image_size is None:
            raise ValueError("offset loss requires image_size and heatmap_size")

        scale = float(heatmap_size) / float(image_size)
        centers = corner_2d.detach() * scale
        floor = torch.floor(centers)
        residual = centers - floor

        ix = floor[..., 0].clamp(0, heatmap_size - 1).long()
        iy = floor[..., 1].clamp(0, heatmap_size - 1).long()
        bidx = torch.arange(B, device=pred_offset.device).view(B, 1)
        kidx = torch.arange(K, device=pred_offset.device).view(1, K)

        pred_offset = pred_offset.reshape(B, K, 2, heatmap_size, heatmap_size)
        gx = pred_offset[bidx, kidx, 0, iy, ix]
        gy = pred_offset[bidx, kidx, 1, iy, ix]
        pred_res = torch.stack([gx, gy], dim=-1)

        return F.l1_loss(pred_res, residual)
