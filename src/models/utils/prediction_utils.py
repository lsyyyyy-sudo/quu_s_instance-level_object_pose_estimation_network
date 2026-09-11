"""从热图提取角点，以及「热图 -> 2D 角点 -> 位姿」的推理后处理。

对应 BoxDreamer/src/models/utils/box_utils.py 的 ``recover_bb8_corners``
与 src/models/utils/prediction_utils.py。

三种提取方式
------------
``argmax``       取每通道最大值位置（整数像素）
``soft_argmax``  空间 softmax 求期望（亚像素、可微）—— 我们训练 fine loss 时用它
``topk``         取 top-k 个位置求平均（**BoxDreamer 官方实现**，k=20）

BoxDreamer 的热图取值是 ``[-1, 1]``，取 top-k 之前会先转成 ``[0, 1]``；
这里用 ``heatmap_range`` 参数显式声明，避免靠猜。
"""

from typing import Optional

import torch

from src.models.utils.box_utils import recover_pose_from_bb8

# BoxDreamer 里 top-k 的 k
DEFAULT_TOPK = 20


def _grid(H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
    """返回 ``[H*W, 2]`` 的 (x, y) 坐标网格。"""
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2).to(dtype)


def to_zero_one(heatmap: torch.Tensor, heatmap_range: str = "auto") -> torch.Tensor:
    """把热图统一到 ``[0, 1]``。

    Args:
        heatmap_range: ``'minus_one_to_one'`` / ``'zero_to_one'`` / ``'auto'``
    """
    if heatmap_range == "zero_to_one":
        return heatmap
    if heatmap_range == "minus_one_to_one":
        return (heatmap + 1.0) / 2.0
    if heatmap_range == "auto":
        # 只在明确探测到负值时才当作 [-1, 1]，避免误判
        return (heatmap + 1.0) / 2.0 if float(heatmap.min()) < -0.5 else heatmap
    raise ValueError(f"Unknown heatmap_range: {heatmap_range!r}")


# --------------------------------------------------------------------------- #
# 三种提取方式
# --------------------------------------------------------------------------- #
def soft_argmax_2d(heatmap: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    """空间 softmax 后求期望坐标（亚像素，可微）。

    Args:
        heatmap: ``[B, K, H, W]``（任意取值区间，softmax 对平移不敏感）
        beta:    温度系数，越大越接近 argmax

    Returns:
        ``[B, K, 2]``，单位是热图像素（x, y）
    """
    B, K, H, W = heatmap.shape
    flat = heatmap.reshape(B, K, -1)
    weights = torch.softmax(flat * beta, dim=-1)              # [B, K, H*W]
    grid = _grid(H, W, heatmap.device, heatmap.dtype)         # [H*W, 2]
    return (weights.unsqueeze(-1) * grid.view(1, 1, -1, 2)).sum(dim=2)


def argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """取每通道最大值位置（整数像素）。"""
    B, K, H, W = heatmap.shape
    idx = heatmap.reshape(B, K, -1).argmax(dim=-1)   # [B, K]
    x = (idx % W).to(heatmap.dtype)
    y = torch.div(idx, W, rounding_mode="floor").to(heatmap.dtype)
    return torch.stack([x, y], dim=-1)


def topk_2d(
    heatmap: torch.Tensor,
    k: int = DEFAULT_TOPK,
    heatmap_range: str = "auto",
) -> torch.Tensor:
    """取每通道 top-k 个位置求平均（BoxDreamer 官方做法）。

    BoxDreamer 用这个代替 argmax：峰如果被噪声打歪，argmax 会一次跳很远，
    而 top-k 平均能稳住。代价是对双峰不敏感。

    Args:
        heatmap: ``[B, K, H, W]``
        k:       取多少个位置，BoxDreamer 用 20
        heatmap_range: 热图取值区间，取 top-k 前会先转到 ``[0, 1]``

    Returns:
        ``[B, K, 2]``，单位是热图像素（x, y）
    """
    heatmap = to_zero_one(heatmap, heatmap_range)
    B, K, H, W = heatmap.shape
    k = max(1, min(int(k), H * W))

    flat = heatmap.reshape(B, K, -1)
    _, idxs = torch.topk(flat, k=k, dim=-1)          # [B, K, k]
    xs = (idxs % W).to(heatmap.dtype)
    ys = torch.div(idxs, W, rounding_mode="floor").to(heatmap.dtype)
    return torch.stack([xs.mean(dim=-1), ys.mean(dim=-1)], dim=-1)


def heatmap_to_corners(
    heatmap: torch.Tensor,
    method: str = "soft_argmax",
    beta: float = 100.0,
    k: int = DEFAULT_TOPK,
    heatmap_range: str = "auto",
) -> torch.Tensor:
    """从热图提角点，结果以**热图像素**为单位。"""
    if method == "soft_argmax":
        return soft_argmax_2d(heatmap, beta=beta)
    if method == "argmax":
        return argmax_2d(heatmap)
    if method == "topk":
        return topk_2d(heatmap, k=k, heatmap_range=heatmap_range)
    raise ValueError(
        f"Unknown extraction method: {method!r} "
        "(expected 'soft_argmax', 'argmax' or 'topk')"
    )


# --------------------------------------------------------------------------- #
# 坐标换算
# --------------------------------------------------------------------------- #
def scale_corners(corners: torch.Tensor, src_size: int, dst_size: int) -> torch.Tensor:
    """把角点从 ``src_size`` 边长的坐标系线性缩放到 ``dst_size``。"""
    return corners * (float(dst_size) / float(src_size))


def normalize_corners(corners: torch.Tensor, size: int) -> torch.Tensor:
    """像素 -> ``[0, 1]`` 归一化坐标（BoxDreamer 的 ``normalized_keypoints_2d``）。"""
    return corners / float(size)


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def predict_corners_and_pose(
    pred_heatmap: torch.Tensor,
    bbox_3d: torch.Tensor,
    K: torch.Tensor,
    image_size: int,
    heatmap_size: int,
    method: str = "soft_argmax",
    beta: float = 100.0,
    k: int = DEFAULT_TOPK,
    heatmap_range: str = "auto",
    solve_pose: bool = True,
):
    """推理后处理：热图 -> 图像像素坐标的 2D 角点 -> PnP 位姿。

    Args:
        pred_heatmap:  ``[B, K, h, w]`` 网络输出
        bbox_3d:       ``[B, 8, 3]`` 物体坐标系下的 3D 角点
        K:             ``[B, 3, 3]`` / ``[3, 3]`` 相机内参（与 ``image_size`` 对应）
        image_size:    网络输入边长
        heatmap_size:  热图边长

    Returns:
        dict:
            ``corner_2d``      ``[B, 8, 2]`` 图像像素坐标
            ``corner_2d_norm`` ``[B, 8, 2]`` 归一化坐标
            ``poses``          ``[B, 4, 4]`` 位姿（``solve_pose=False`` 时为 None）
    """
    corner_hm = heatmap_to_corners(
        pred_heatmap, method=method, beta=beta, k=k, heatmap_range=heatmap_range
    )
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


# --------------------------------------------------------------------------- #
# 指标辅助
# --------------------------------------------------------------------------- #
def corner_error_px(corner_2d_pred: torch.Tensor, corner_2d_gt: torch.Tensor) -> torch.Tensor:
    """逐样本的角点平均 L2 像素误差，``[B]``。"""
    return torch.linalg.norm(corner_2d_pred - corner_2d_gt, dim=-1).mean(dim=-1)


def corner_pck(
    corner_2d_pred: torch.Tensor,
    corner_2d_gt: torch.Tensor,
    thresholds=(0.05, 0.10, 0.15),
) -> dict:
    """PCK：误差小于「GT 2D 包围框对角线 × threshold」的角点比例。"""
    diff = torch.linalg.norm(corner_2d_pred - corner_2d_gt, dim=-1)   # [B, 8]
    xy_min = corner_2d_gt.amin(dim=1)
    xy_max = corner_2d_gt.amax(dim=1)
    diag = torch.linalg.norm(xy_max - xy_min, dim=-1, keepdim=True).clamp(min=1e-6)

    out = {}
    for t in thresholds:
        out[f"pck@{t:g}"] = (diff < t * diag).float().mean().item()
    return out
