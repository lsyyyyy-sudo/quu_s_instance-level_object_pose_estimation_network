"""从热图提取角点，以及「热图 -> 2D 角点 -> 位姿」的推理后处理。

对应 BoxDreamer/src/models/utils/prediction_utils.py 中的角点提取部分。

BoxDreamer 支持三种 bbox 表征（``heatmap`` / ``voting`` / ``cornernet``），
本项目当前只实现 ``heatmap``（config 里的 ``bbox_representation: heatmap``）。
"""

from typing import Optional

import torch

from src.models.utils.box_utils import recover_pose_from_bb8


def _pixel_grid(H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
    """返回 ``[H*W, 2]`` 的 (x, y) 坐标网格。"""
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2).to(dtype)


def soft_argmax_2d(heatmap: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    """对每个通道做空间 softmax 后求期望坐标（亚像素，可微）。

    Args:
        heatmap: ``[B, K, H, W]`` logits
        beta:    温度系数，越大越接近 argmax

    Returns:
        ``[B, K, 2]``，单位是热图像素（x, y）
    """
    B, K, H, W = heatmap.shape
    flat = heatmap.reshape(B, K, -1)
    weights = torch.softmax(flat * beta, dim=-1)              # [B, K, H*W]
    grid = _pixel_grid(H, W, heatmap.device, heatmap.dtype)   # [H*W, 2]
    return (weights.unsqueeze(-1) * grid.view(1, 1, -1, 2)).sum(dim=2)


def argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """取每个通道最大值位置（整数像素）。

    Returns:
        ``[B, K, 2]``，单位是热图像素（x, y）
    """
    B, K, H, W = heatmap.shape
    idx = heatmap.reshape(B, K, -1).argmax(dim=-1)   # [B, K]
    x = (idx % W).to(heatmap.dtype)
    y = torch.div(idx, W, rounding_mode="floor").to(heatmap.dtype)
    return torch.stack([x, y], dim=-1)


def heatmap_to_corners(
    heatmap: torch.Tensor,
    method: str = "soft_argmax",
    beta: float = 100.0,
) -> torch.Tensor:
    """从热图提角点，结果以**热图像素**为单位。"""
    if method == "soft_argmax":
        return soft_argmax_2d(heatmap, beta=beta)
    if method == "argmax":
        return argmax_2d(heatmap)
    raise ValueError(f"Unknown extraction method: {method}")


def scale_corners(corners: torch.Tensor, src_size: int, dst_size: int) -> torch.Tensor:
    """把角点从 ``src_size`` 边长的坐标系线性缩放到 ``dst_size``。

    约定图像是正方形（本项目统一 resize 成 ``image_size x image_size``）。
    """
    return corners * (float(dst_size) / float(src_size))


def normalize_corners(corners: torch.Tensor, size: int) -> torch.Tensor:
    """像素 -> ``[0, 1]`` 归一化坐标（BoxDreamer 的 ``normalized_keypoints_2d``）。"""
    return corners / float(size)


def predict_corners_and_pose(
    pred_heatmap: torch.Tensor,
    bbox_3d: torch.Tensor,
    K: torch.Tensor,
    image_size: int,
    heatmap_size: int,
    method: str = "soft_argmax",
    beta: float = 100.0,
    solve_pose: bool = True,
):
    """推理后处理：热图 -> 图像像素坐标的 2D 角点 -> PnP 位姿。

    Args:
        pred_heatmap: ``[B, K, h, w]`` 网络输出
        bbox_3d:      ``[B, 8, 3]`` 物体坐标系下的 3D 角点
        K:            ``[B, 3, 3]`` / ``[3, 3]`` 相机内参（与 ``image_size`` 对应）
        image_size:   网络输入边长
        heatmap_size: 热图边长

    Returns:
        dict:
            ``corner_2d``      ``[B, 8, 2]`` 图像像素坐标
            ``corner_2d_norm`` ``[B, 8, 2]`` 归一化坐标
            ``poses``          ``[B, 4, 4]`` 位姿（``solve_pose=False`` 时为 None）
    """
    corner_hm = heatmap_to_corners(pred_heatmap, method=method, beta=beta)
    corner_2d = scale_corners(corner_hm, heatmap_size, image_size)
    corner_2d_norm = normalize_corners(corner_2d, image_size)

    poses: Optional[torch.Tensor] = None
    if solve_pose:
        poses, _ = recover_pose_from_bb8(corner_2d, bbox_3d, K)

    return {
        "corner_2d": corner_2d,
        "corner_2d_norm": corner_2d_norm,
        "poses": poses,
    }


def corner_error_px(corner_2d_pred: torch.Tensor, corner_2d_gt: torch.Tensor) -> torch.Tensor:
    """逐样本的角点平均 L2 像素误差，``[B]``。"""
    return torch.linalg.norm(corner_2d_pred - corner_2d_gt, dim=-1).mean(dim=-1)


def corner_pck(
    corner_2d_pred: torch.Tensor,
    corner_2d_gt: torch.Tensor,
    thresholds=(0.05, 0.10, 0.15),
) -> dict:
    """PCK：误差小于「GT 2D 包围框对角线 × threshold」的角点比例。

    Returns:
        ``{"pck@0.05": float, ...}``（对 batch 和角点取平均）
    """
    diff = torch.linalg.norm(corner_2d_pred - corner_2d_gt, dim=-1)   # [B, 8]
    xy_min = corner_2d_gt.amin(dim=1)                                  # [B, 2]
    xy_max = corner_2d_gt.amax(dim=1)
    diag = torch.linalg.norm(xy_max - xy_min, dim=-1, keepdim=True)     # [B, 1]
    diag = diag.clamp(min=1e-6)

    out = {}
    for t in thresholds:
        out[f"pck@{t:g}"] = (diff < t * diag).float().mean().item()
    return out
