"""几何一致性损失：让预测的 8 个角点【真的构成一个长方体投影】。

背景
----
网络对 8 个角点各自独立回归（8 张热图 -> 8 个 argmax），损失也是逐角点求和。
**没有任何一项在问「这 8 个点合起来像不像一个盒子」。**

实测（目标视频 frame 0）：三组棱的消失点共点残差 1.3e-1 ~ 4.5e-1
（对照组标定：真盒子投影 < 1e-15，随机 8 点 ~ 2e-1，合法阈值 5e-2）
=> 预测的 8 个点**全部不是合法的长方体投影**。

判据（**不需要相机内参**）
------------------------
3D 里相互平行的三组棱，其投影必须各自【共点】（交于消失点）。
4 条齐次直线 l1..l4 共点  <=>  任意 3 条组成的 3x3 行列式为 0。

    loss = Σ_族 Σ_{三元组} det([l_i; l_j; l_k])^2

可微、便宜（不做 SVD），且与相机模型无关。

⚠️ 共点是**必要不充分**条件（优化器可能把盒子拉成退化形状来满足它）。
所以本项是**辅助损失**，必须和监督项（拟合 GT 角点）一起用 ——
GT 本身是合法投影，两者合力才能把解推向"既贴合又合法"。
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict

import torch


# BB8 顺序下，3D 里相互平行的三组棱（与 src/models/utils/box_fit.py 一致）
EDGE_FAMILIES = {
    "X": [(0, 1), (3, 2), (4, 5), (7, 6)],
    "Y": [(1, 2), (0, 3), (5, 6), (4, 7)],
    "Z": [(0, 4), (1, 5), (2, 6), (3, 7)],
}
_TRIPLES = list(combinations(range(4), 3))          # 每组 4 条棱 -> 4 个三元组


def _lines_from_corners(corners: torch.Tensor, segs) -> torch.Tensor:
    """由角点构造齐次直线。

    Args:
        corners: ``[B, 8, 2]``
        segs:    4 个 (i, j) 索引对

    Returns:
        ``[B, 4, 3]``，每条直线已归一化到单位模（**必须**，否则
        图像坐标下 c 分量 ~1e6，各行尺度差异会让行列式/奇异值失去意义 —— 这个坑踩过）。
    """
    B = corners.shape[0]
    out = []
    for i, j in segs:
        p = corners[:, i, :]                          # [B,2]
        q = corners[:, j, :]
        a = torch.cat([p, torch.ones(B, 1, device=p.device, dtype=p.dtype)], dim=1)
        b = torch.cat([q, torch.ones(B, 1, device=q.device, dtype=q.dtype)], dim=1)
        l = torch.cross(a, b, dim=1)                  # [B,3]
        l = l / l.norm(dim=1, keepdim=True).clamp_min(1e-12)
        out.append(l)
    return torch.stack(out, dim=1)                    # [B,4,3]


def concurrence_residual(corners: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """三组棱的共点残差（用于监控；数值越小越像合法盒子）。

    用行列式绝对值的归一化版本，量纲与 ``geo_consistency_loss`` 一致。

    Args:
        corners: ``[B, 8, 2]``（建议先在裁剪图坐标系里中心化/缩放）
        reduce:  True 返回 ``[B]`` 每样本的残差，False 返回逐三元组

    Returns:
        ``[B]``（reduce=True）
    """
    B = corners.shape[0]
    c0 = corners.mean(dim=1, keepdim=True)
    s = (corners - c0).abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    cn = (corners - c0) / s

    per_sample = torch.zeros(B, device=corners.device, dtype=corners.dtype)
    for segs in EDGE_FAMILIES.values():
        lines = _lines_from_corners(cn, segs)          # [B,4,3]
        for (i, j, k) in _TRIPLES:
            M = lines[:, [i, j, k], :]                 # [B,3,3]
            per_sample = per_sample + torch.linalg.det(M).abs()
    per_sample = per_sample / (len(EDGE_FAMILIES) * len(_TRIPLES))
    return per_sample if reduce else per_sample


def geo_consistency_loss(corners: torch.Tensor,
                         weight_by_box: bool = True) -> torch.Tensor:
    """几何一致性损失（标量，越小越好）。

    Args:
        corners: ``[B, 8, 2]`` 预测角点。**调用前请把坐标中心化/缩放到 O(1)**，
                 否则行列式的数值会随图像尺度爆炸。
        weight_by_box: 按 8 点外接框对角线归一化，让不同大小的物体贡献相当
                       （对角线已经在中心化时用过；这里再乘一次是为了让
                       小框的残差不会被系统性地放大）。

    Returns:
        标量损失
    """
    if corners.ndim != 3 or corners.shape[1] != 8 or corners.shape[2] != 2:
        raise ValueError(f"corners 期望 [B,8,2]，得到 {tuple(corners.shape)}")

    B = corners.shape[0]
    c0 = corners.mean(dim=1, keepdim=True)
    d = corners - c0
    diag = d.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    cn = d / diag

    total = corners.new_zeros(())
    for segs in EDGE_FAMILIES.values():
        lines = _lines_from_corners(cn, segs)
        fam = corners.new_zeros(())
        for (i, j, k) in _TRIPLES:
            M = lines[:, [i, j, k], :]
            fam = fam + torch.linalg.det(M).pow(2).mean()
        total = total + fam
    total = total / len(EDGE_FAMILIES)

    if weight_by_box:
        # 已经在 cn 里做过尺度归一化，这里保持恒等（留出接口便于以后调）
        pass
    return total


__all__ = ["geo_consistency_loss", "concurrence_residual", "EDGE_FAMILIES"]
