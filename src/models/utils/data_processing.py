"""张量打包与监督目标构造。

对应 BoxDreamer/src/models/utils/data_processing.py。
"""

from typing import Dict, List, Sequence

import torch


def gaussian_2d(
    size: int,
    center_x: float,
    center_y: float,
    sigma: float = 2.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """在 ``size x size`` 的热图上画一个高斯峰（CenterNet 风格）。"""
    ys, xs = torch.meshgrid(
        torch.arange(size, device=device, dtype=dtype),
        torch.arange(size, device=device, dtype=dtype),
        indexing="ij",
    )
    dist2 = (xs - center_x) ** 2 + (ys - center_y) ** 2
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


def make_heatmap_target(
    corner_2d: torch.Tensor,
    image_size: int,
    heatmap_size: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """由图像像素坐标的 2D 角点生成高斯热图监督。

    Args:
        corner_2d:    ``[B, K, 2]``，坐标以 ``image_size`` 为基准
        image_size:   网络输入边长
        heatmap_size: 输出热图边长
        sigma:        高斯标准差（热图像素单位）

    Returns:
        ``[B, K, heatmap_size, heatmap_size]``，取值 ``[0, 1]``

    被遮挡的角点同样要监督（amodal），所以这里不做可见性过滤。
    超出图像范围的角点会被 clamp 到边界，仍给出一个峰。
    """
    corner_2d = torch.as_tensor(corner_2d, dtype=torch.float32)
    B, K, _ = corner_2d.shape
    scale = float(heatmap_size) / float(image_size)
    centers = corner_2d * scale

    heatmap = torch.zeros(B, K, heatmap_size, heatmap_size, dtype=torch.float32, device=corner_2d.device)
    for b in range(B):
        for k in range(K):
            cx = float(centers[b, k, 0].clamp(0, heatmap_size - 1))
            cy = float(centers[b, k, 1].clamp(0, heatmap_size - 1))
            heatmap[b, k] = gaussian_2d(heatmap_size, cx, cy, sigma=sigma, device=corner_2d.device)
    return heatmap


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
