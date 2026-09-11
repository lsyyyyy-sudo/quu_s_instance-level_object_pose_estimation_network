"""张量打包与监督目标构造。

对应 BoxDreamer/src/models/utils/data_processing.py 与
``src/datasets/utils/base/bbox_utils.py`` 的 ``type == "heatmap"`` 分支。

两种热图风格
------------
``boxdreamer``（默认，临摹 BoxDreamer）
    尺度自适应的衰减函数，每个角点用自己的尺度::

        d          = 网格点到角点的欧氏像素距离（**不是**距离平方）
        s_i        = 角点 i 到物体 2D 中心（8 个角点的均值）的像素距离
        scale_i    = (s_i / 10) ** 2          # 即论文里的 2σ²
        H_i        = exp(-d / scale_i)
        H_i       /= max(H_i)                 # 峰值归一化到 1
        H_i        = H_i * 2 - 1              # 映射到 [-1, 1]

    注意它不是标准高斯：指数里是 ``-d`` 而不是 ``-d²/(2σ²)``。
    论文明确说 CornerNet 的原始超参直接搬过来不好用，所以重新定义了这个函数。

``centernet``
    固定 σ 的标准高斯，取值 ``[0, 1]``（我们最初的实现，保留作为对照）。
"""

from typing import Dict, List, Optional, Sequence

import torch


def _grid(H: int, W: int, device, dtype=torch.float32):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return xs, ys


# --------------------------------------------------------------------------- #
# 单通道高斯（centernet 风格）
# --------------------------------------------------------------------------- #
def gaussian_2d(
    size: int,
    center_x: float,
    center_y: float,
    sigma: float = 2.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """在 ``size x size`` 的热图上画一个标准高斯峰。"""
    xs, ys = _grid(size, size, device, dtype)
    dist2 = (xs - center_x) ** 2 + (ys - center_y) ** 2
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


# --------------------------------------------------------------------------- #
# BoxDreamer 的尺度自适应热图
# --------------------------------------------------------------------------- #
def boxdreamer_heatmap_2d(
    center: torch.Tensor,
    scale_factor: torch.Tensor,
    H: int,
    W: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """单个角点的尺度自适应热图，返回 ``[H, W]``，取值 ``[-1, 1]``。"""
    xs, ys = _grid(H, W, center.device, center.dtype)
    d = torch.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    scale = scale_factor.clamp(min=eps)
    heat = torch.exp(-d / scale)
    peak = heat.max().clamp(min=eps)
    heat = heat / peak
    return heat * 2.0 - 1.0


def make_heatmap_target(
    corner_2d: torch.Tensor,
    image_size: int,
    heatmap_size: int,
    sigma: float = 2.0,
    style: str = "boxdreamer",
    min_scale: float = 1.0,
) -> torch.Tensor:
    """由图像像素坐标的 2D 角点生成热图监督。

    Args:
        corner_2d:    ``[B, K, 2]``，坐标以 ``image_size`` 为基准
        image_size:   网络输入边长
        heatmap_size: 输出热图边长
        sigma:        仅 ``style='centernet'`` 使用，高斯标准差（热图像素）
        style:        ``'boxdreamer'`` 或 ``'centernet'``
        min_scale:    仅 ``style='boxdreamer'`` 使用，``(s_i/10)²`` 的下限。

            为什么需要它：当角点很靠近物体 2D 中心时 ``s_i → 0``，
            BoxDreamer 原式会让 ``exp(-d / scale)`` 在 d≥1 处全部下溢成 0，
            热图退化成 δ 函数。这不只是数值问题——**背景全为 0 会让 top-k
            提取出现大量并列，取出的位置基本是随机的**（实测平均偏 14~18 px）。
            真实投影包围盒的角点离中心不会太近，但合成数据/异常姿态下会踩到。
            下限取 1 个热图像素，既保留"近中心角点更锐"的语义，又避免退化。

    Returns:
        ``[B, K, heatmap_size, heatmap_size]``
        - ``boxdreamer`` 风格取值 ``[-1, 1]``
        - ``centernet`` 风格取值 ``[0, 1]``

    被遮挡的角点同样要监督（amodal），所以这里不做可见性过滤。
    """
    corner_2d = torch.as_tensor(corner_2d, dtype=torch.float32)
    B, K, _ = corner_2d.shape
    ratio = float(heatmap_size) / float(image_size)
    centers = corner_2d * ratio                       # [B, K, 2] 热图坐标

    if style == "boxdreamer":
        # 物体 2D 中心 = 8 个角点投影的均值（对齐 BoxDreamer 的 bbox.mean(dim=1)）
        obj_center = centers.mean(dim=1, keepdim=True)          # [B, 1, 2]
        # 每个角点自己的尺度：它到物体中心的像素距离
        dis = torch.linalg.norm(centers - obj_center, dim=-1)   # [B, K]
        scale_factor = ((dis / 10.0) ** 2).clamp(min=float(min_scale))   # [B, K]

        heatmap = torch.empty(B, K, heatmap_size, heatmap_size,
                              dtype=torch.float32, device=corner_2d.device)
        for b in range(B):
            for k in range(K):
                heatmap[b, k] = boxdreamer_heatmap_2d(
                    centers[b, k], scale_factor[b, k], heatmap_size, heatmap_size
                )
        return heatmap

    if style == "centernet":
        heatmap = torch.zeros(B, K, heatmap_size, heatmap_size,
                              dtype=torch.float32, device=corner_2d.device)
        for b in range(B):
            for k in range(K):
                cx = float(centers[b, k, 0].clamp(0, heatmap_size - 1))
                cy = float(centers[b, k, 1].clamp(0, heatmap_size - 1))
                heatmap[b, k] = gaussian_2d(
                    heatmap_size, cx, cy, sigma=sigma, device=corner_2d.device
                )
        return heatmap

    raise ValueError(
        f"Unknown heatmap style: {style!r} (expected 'boxdreamer' or 'centernet')"
    )


# --------------------------------------------------------------------------- #
# batch 工具
# --------------------------------------------------------------------------- #
def batch_index_select(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """按 batch 维做 gather，``index`` 形状 ``[B]``。"""
    return tensor[index]


def to_device(batch: Dict, device) -> Dict:
    """把 batch 里所有 tensor 搬到 device（非 tensor 原样保留）。"""
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
            out[key] = [v.to(device, non_blocking=True) for v in value]
        else:
            out[key] = value
    return out


def stack_optional(tensors: Sequence, fill_value: float = 0.0) -> torch.Tensor:
    """把可能为 None 的序列堆叠起来，None 用 ``fill_value`` 填充。"""
    ref = next((t for t in tensors if t is not None), None)
    if ref is None:
        raise ValueError("All entries are None; cannot infer shape.")
    filled: List[torch.Tensor] = [
        t if t is not None else torch.full_like(ref, fill_value) for t in tensors
    ]
    return torch.stack(filled, dim=0)


def heatmap_value_range(style: str) -> str:
    """返回该风格热图的取值区间标签，供预测侧统一处理。"""
    return "minus_one_to_one" if style == "boxdreamer" else "zero_to_one"
