"""多实例解码：中心热图 -> N 个实例 -> 每个实例的 8 个角点。

与单实例的区别
--------------
单实例：8 张【角点】热图 -> 每张取 argmax -> 8 个点（只能表示一个物体）。
多实例（CenterNet 式）：
    1 张【中心】热图     -> 局部极大值（NMS）= 实例中心，个数可变
    16 通道【角点偏移】  -> 在每个峰的位置读出 8 个角点（相对该像素的偏移）

这样「预测几个」由热图上的峰数自然决定，不需要外部分类/检测器。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def nms_heatmap(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """3x3（或 kernel x kernel）最大池化 NMS：只保留局部极大值。

    Args:
        heatmap: ``[B, 1, h, w]``
        kernel:  奇数，邻域大小

    Returns:
        同形状，非极大值处被置 0
    """
    if kernel % 2 == 0:
        raise ValueError("kernel 必须是奇数")
    pad = kernel // 2
    # max_pool 的 padding 要小心：用 -inf 填充，避免边界被 0 误判为极大值
    x = F.pad(heatmap, (pad, pad, pad, pad), value=float("-inf"))
    pooled = F.max_pool2d(x, kernel_size=kernel, stride=1)
    return torch.where(heatmap >= pooled, heatmap, torch.zeros_like(heatmap))


def topk_peaks(heatmap: torch.Tensor, k: int = 100,
               thr: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """取每张图里得分最高的 k 个峰。

    Args:
        heatmap: ``[B, 1, h, w]``（已做过 NMS）
        k:       每张图最多取几个
        thr:     低于此分数丢弃

    Returns:
        ``(scores [B,k], ys [B,k], xs [B,k])``，不足 k 个的位置分数为 0
    """
    B, C, h, w = heatmap.shape
    assert C == 1, f"期望单通道中心热图，得到 {C} 通道"
    flat = heatmap.reshape(B, -1)
    kk = min(k, flat.shape[1])
    scores, idx = flat.topk(kk, dim=1)                      # [B,kk]
    ys = torch.div(idx, w, rounding_mode="floor")
    xs = idx % w
    if kk < k:                                              # 补齐到 k
        pad = k - kk
        scores = F.pad(scores, (0, pad), value=0.0)
        ys = F.pad(ys, (0, pad), value=0)
        xs = F.pad(xs, (0, pad), value=0)
    valid = scores >= thr
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    return scores, ys, xs


def gather_corners(offset: torch.Tensor, ys: torch.Tensor,
                   xs: torch.Tensor, stride: float = 1.0) -> torch.Tensor:
    """在给定像素位置读出 8 个角点。

    坐标系约定（与 ``src/loss/utils/multi_instance_loss.py`` 严格一致）
    ----------------------------------------------------------------
    热图格子 ``(hx, hy)`` 对应裁剪图的 ``(hx*stride, hy*stride)``。
    ``offset[:, :, hy, hx]`` = 8 个角点在**裁剪像素**下相对该点的偏移。
    所以角点 = ``offset + (hx*stride, hy*stride)``。

    Args:
        offset: ``[B, 16, h, w]``（8 角点 x 2 坐标，相对格子对应位置）
        ys, xs: ``[B, k]`` 峰位置（热图坐标系）
        stride: ``image_size / heatmap_size``，把热图格子换算成裁剪像素

    Returns:
        ``[B, k, 8, 2]``，**裁剪像素坐标系**下每个实例的 8 个角点
    """
    B, C, h, w = offset.shape
    if C != 16:
        raise ValueError(f"offset 期望 16 通道（8 角点 x 2），得到 {C}")
    k = ys.shape[1]
    flat = offset.reshape(B, C, h * w)
    idx = (ys * w + xs).unsqueeze(1).expand(B, C, k)        # [B,C,k]
    vals = flat.gather(2, idx)                              # [B,C,k]
    vals = vals.permute(0, 2, 1).reshape(B, k, 8, 2)        # [B,k,8,2]
    base = (torch.stack([xs, ys], dim=-1).unsqueeze(2).to(vals.dtype) * float(stride))
    return vals + base


def greedy_nms(scores: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor,
               min_dist: float = 3.0) -> torch.Tensor:
    """按距离的贪心 NMS：同一个实例附近只保留得分最高的峰。

    为什么需要它
    ------------
    max-pool NMS 用 ``>=`` 比较，遇到【平台/宽脊】会把整片都留下
    （实测 4x4 的平台被数成 16 个实例）。真实热图虽然不会完全平坦，
    但可能有很宽的脊。这里再按距离做一次贪心抑制，彻底解决。

    Args:
        scores, ys, xs: ``[B, k]``（已按分数降序更好，本函数自己排）
        min_dist: 两个峰距离小于它就算同一个实例（单位 = 热图格）

    Returns:
        ``[B, k]`` 的 bool 掩码，True = 保留
    """
    B, k = scores.shape
    keep = torch.zeros(B, k, dtype=torch.bool, device=scores.device)
    for b in range(B):
        order = torch.argsort(scores[b], descending=True)
        kept_yx = []
        for idx in order.tolist():
            if float(scores[b, idx]) <= 0:
                continue
            y, x = float(ys[b, idx]), float(xs[b, idx])
            ok = True
            for (ky, kx) in kept_yx:
                if (y - ky) ** 2 + (x - kx) ** 2 < min_dist ** 2:
                    ok = False
                    break
            if ok:
                keep[b, idx] = True
                kept_yx.append((y, x))
    return keep


def decode_instances(heatmap: torch.Tensor, offset: torch.Tensor,
                     thr: float = 0.05, topk: int = 100,
                     nms_kernel: int = 3, min_dist: float = 3.0,
                     stride: float = 1.0) -> List[List[dict]]:
    """端到端解码：中心热图 + 角点偏移 -> 每张图的实例列表。

    Args:
        heatmap: ``[B, 1, h, w]`` 中心热图（logits 或已 sigmoid，本函数内做 sigmoid）
        offset:  ``[B, 16, h, w]`` 角点偏移（热图坐标系单位）
        thr:     中心得分阈值
        topk:    每张图最多输出几个实例
        nms_kernel: max-pool NMS 邻域（奇数）
        min_dist:   贪心 NMS 的最小实例间距（热图格）
        stride:     image_size / heatmap_size，输出角点用裁剪像素

    Returns:
        长度为 B 的列表；每个元素是实例 dict 列表，每个 dict::

            {"score": float, "center": (x, y), "corners": np.ndarray [8,2]}
    """
    if heatmap.shape[1] != 1:
        raise ValueError(f"heatmap 期望 1 通道，得到 {heatmap.shape[1]}")
    hm = torch.sigmoid(heatmap.detach())
    hm = nms_heatmap(hm, nms_kernel)
    scores, ys, xs = topk_peaks(hm, k=topk, thr=thr)
    keep = greedy_nms(scores, ys, xs, min_dist=min_dist)
    scores = torch.where(keep, scores, torch.zeros_like(scores))
    corners = gather_corners(offset.detach(), ys, xs, stride=stride)  # [B,k,8,2]

    B = heatmap.shape[0]
    out: List[List[dict]] = []
    for b in range(B):
        inst = []
        for i in range(scores.shape[1]):
            s = float(scores[b, i])
            if s <= 0:
                continue
            inst.append({
                "score": s,
                "center": (float(xs[b, i]), float(ys[b, i])),
                "corners": corners[b, i].cpu().numpy(),
            })
        out.append(inst)
    return out


__all__ = ["nms_heatmap", "topk_peaks", "gather_corners", "greedy_nms",
           "decode_instances"]
