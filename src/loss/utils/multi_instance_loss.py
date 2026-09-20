"""多实例损失：中心热图 focal + 角点回归 + 几何一致性。

坐标系约定（**必须统一，否则角点会错位**）
------------------------------------------
热图格子 ``(hx, hy)`` 对应裁剪图的 ``(hx*s, hy*s)``，``s = image_size / heatmap_size``。

``pred_offset[b, :, hy, hx]`` 存的是 **8 个角点在【裁剪像素】坐标系里的位置，
相对于 ``(hx*s, hy*s)`` 的偏移**。

于是某个峰 ``(hx, hy)`` 处的实例角点 = ``pred_offset[:, hy, hx] + (hx*s, hy*s)``。

损失三项
--------
1. ``center``: CenterNet 式 focal loss（正样本是 GT 中心的高斯峰）
2. ``corner``: 在 GT 中心所在的格子上，回归 8 个角点（smooth L1）
3. ``geo``:    几何一致性 —— 预测的 8 点必须构成合法长方体投影（零内参）
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.loss.utils.geo_consistency import geo_consistency_loss


def center_focal_loss(pred_logits: torch.Tensor, gt_heatmap: torch.Tensor,
                      alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet 式 focal loss（对中心热图）。

    Args:
        pred_logits: ``[B, 1, h, w]`` **未过 sigmoid** 的 logits
        gt_heatmap:  ``[B, 1, h, w]`` 取值 [0,1] 的高斯峰
        alpha/beta:  CenterNet 原文的超参（2, 4）

    Returns:
        标量损失（按正样本数归一化；无正样本时返回 0，不会 NaN）
    """
    if pred_logits.shape != gt_heatmap.shape:
        raise ValueError(f"形状不一致: {tuple(pred_logits.shape)} vs {tuple(gt_heatmap.shape)}")
    p = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    # ⚠️ 正样本判据放宽到 0.9（而不是 >= 1-1e-4）。
    #    原来那个判据要求 GT 峰【精确】等于 1.0；只要数据集把峰放在浮点位置，
    #    就会一个正样本都匹配不到 -> 中心头没有监督 -> 塌缩（ALGO-08 第二次踩）。
    #    现在数据集已对齐格点，这里是第二道保险。
    pos = gt_heatmap.ge(0.9).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(p) * torch.pow(1 - p, alpha) * pos
    neg_loss = -torch.log(1 - p) * torch.pow(p, alpha) * torch.pow(1 - gt_heatmap, beta) * neg
    n_pos = pos.sum()
    if float(n_pos) == 0.0:
        # 不再静默：没有正样本 = 这一项根本没起作用，必须大声说出来
        import warnings
        warnings.warn(
            "center_focal_loss: 正样本数为 0！GT 热图里没有任何值 >= 0.9。"
            "检查数据集是否把高斯峰放在整数格点上（见 src/datasets/bop_pbr.py）。",
            RuntimeWarning, stacklevel=2,
        )
    loss = (pos_loss + neg_loss).sum()
    return loss / n_pos.clamp_min(1.0)


def _sample_offsets(offset: torch.Tensor, hx: torch.Tensor, hy: torch.Tensor) -> torch.Tensor:
    """在给定热图格子处取出 16 个值。

    Args:
        offset: ``[B, 16, h, w]``
        hx, hy: ``[B, N]`` long

    Returns:
        ``[B, N, 8, 2]``
    """
    B, C, h, w = offset.shape
    N = hx.shape[1]
    flat = offset.reshape(B, C, h * w)
    idx = (hy * w + hx).unsqueeze(1).expand(B, C, N)
    vals = flat.gather(2, idx).permute(0, 2, 1).reshape(B, N, 8, 2)
    return vals


def corner_regression_loss(pred_offset: torch.Tensor,
                           gt_corners: torch.Tensor,
                           gt_centers: torch.Tensor,
                           valid: torch.Tensor,
                           image_size: int,
                           heatmap_size: int,
                           beta: float = 1.0) -> torch.Tensor:
    """在 GT 中心所在格子上回归 8 个角点。

    Args:
        pred_offset: ``[B, 16, h, w]``（相对格子对应裁剪像素的偏移）
        gt_corners:  ``[B, N, 8, 2]`` **裁剪像素**坐标
        gt_centers:  ``[B, N, 2]``    **裁剪像素**坐标
        valid:       ``[B, N]`` 1/0
        image_size / heatmap_size: 用于求 stride

    Returns:
        标量损失（按有效实例数归一化）
    """
    B, N = valid.shape
    s = float(image_size) / float(heatmap_size)
    dev = pred_offset.device

    hx = (gt_centers[..., 0] / s).round().long().clamp(0, heatmap_size - 1)
    hy = (gt_centers[..., 1] / s).round().long().clamp(0, heatmap_size - 1)

    pred = _sample_offsets(pred_offset, hx, hy)                  # [B,N,8,2]
    base = torch.stack([hx.float() * s, hy.float() * s], dim=-1).unsqueeze(2)  # [B,N,1,2]
    pred_abs = pred + base

    m = valid.unsqueeze(-1).unsqueeze(-1)                        # [B,N,1,1]
    if m.sum() <= 0:
        return (pred_abs * 0.0).sum()
    loss = F.smooth_l1_loss(pred_abs, gt_corners, beta=beta, reduction="none")
    return (loss * m).sum() / (m.sum() * 8 * 2)


def multi_instance_loss(pred_heatmap: torch.Tensor,
                        pred_offset: torch.Tensor,
                        batch: Dict[str, torch.Tensor],
                        image_size: int,
                        heatmap_size: int,
                        center_weight: float = 1.0,
                        corner_weight: float = 1.0,
                        geo_weight: float = 1.0,
                        geo_beta: float = 1.0) -> Dict[str, torch.Tensor]:
    """多实例总损失。

    Args:
        pred_heatmap: ``[B,1,h,w]`` logits
        pred_offset:  ``[B,16,h,w]``
        batch:        数据集返回的 dict（需含 center_heatmap / inst_corners /
                      inst_centers / inst_valid）
        geo_weight:   **几何一致性权重**。0 表示关闭。

    Returns:
        dict: ``{"loss": 总, "center": ..., "corner": ..., "geo": ...}``
    """
    gt_hm = batch["center_heatmap"]
    l_center = center_focal_loss(pred_heatmap, gt_hm)
    l_corner = corner_regression_loss(
        pred_offset, batch["inst_corners"], batch["inst_centers"],
        batch["inst_valid"], image_size, heatmap_size,
    )

    if geo_weight > 0:
        s = float(image_size) / float(heatmap_size)
        B, N = batch["inst_valid"].shape
        dev = pred_offset.device
        hx = (batch["inst_centers"][..., 0] / s).round().long().clamp(0, heatmap_size - 1)
        hy = (batch["inst_centers"][..., 1] / s).round().long().clamp(0, heatmap_size - 1)
        pred = _sample_offsets(pred_offset, hx, hy)
        base = torch.stack([hx.float() * s, hy.float() * s], dim=-1).unsqueeze(2)
        pred_abs = (pred + base).reshape(B * N, 8, 2)
        m = batch["inst_valid"].reshape(B * N)
        if m.sum() > 0:
            # 只对有效实例算几何项
            pg = pred_abs[m > 0]
            l_geo = geo_consistency_loss(pg)
        else:
            l_geo = pred_abs.sum() * 0.0
    else:
        l_geo = pred_heatmap.sum() * 0.0

    total = center_weight * l_center + corner_weight * l_corner + geo_weight * l_geo
    return {"loss": total, "center": l_center.detach(),
            "corner": l_corner.detach(), "geo": l_geo.detach()}


__all__ = ["center_focal_loss", "corner_regression_loss", "multi_instance_loss"]
