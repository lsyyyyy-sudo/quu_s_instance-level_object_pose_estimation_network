"""可视化工具：把预测的 8 个角点和 3D 包围盒画到图像上。

对应 BoxDreamer/src/lightning/utils/vis/。
"""

from typing import Optional, Sequence

import cv2
import numpy as np
import torch

# bb8 顺序下，把 8 个角点连成 12 条棱（索引按 BB8_BITS 的顺序）
BB8_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),   # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7),   # 侧面
]

_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
]


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """``[3, H, W]`` 的 ``[0,1]`` 张量 -> ``[H, W, 3]`` uint8 RGB。"""
    image = image.detach().cpu().float().clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def draw_corners(
    image: np.ndarray,
    corner_2d: torch.Tensor,
    corner_2d_gt: Optional[torch.Tensor] = None,
    radius: int = 3,
) -> np.ndarray:
    """在图上画预测角点（实心）与 GT 角点（空心）。"""
    canvas = image.copy()
    pts = corner_2d.detach().cpu().numpy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, _COLORS[i % len(_COLORS)], -1)

    if corner_2d_gt is not None:
        gt = corner_2d_gt.detach().cpu().numpy()
        for i, (x, y) in enumerate(gt):
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius + 3, _COLORS[i % len(_COLORS)], 1)
    return canvas


def draw_bbox3d(
    image: np.ndarray,
    corner_2d: torch.Tensor,
    color=(0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """按 bb8 拓扑把角点连成 3D 包围盒的线框。"""
    canvas = image.copy()
    pts = corner_2d.detach().cpu().numpy()
    for i, j in BB8_EDGES:
        p1 = (int(round(pts[i, 0])), int(round(pts[i, 1])))
        p2 = (int(round(pts[j, 0])), int(round(pts[j, 1])))
        cv2.line(canvas, p1, p2, color, thickness)
    return canvas


def make_vis_panel(
    image: torch.Tensor,
    corner_2d: Optional[torch.Tensor] = None,
    corner_2d_gt: Optional[torch.Tensor] = None,
    pose_corner_2d: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """拼一张可视化图：原图 + GT 角点 + 预测角点 + PnP 位姿投影。"""
    canvas = denormalize_image(image)
    if corner_2d_gt is not None:
        canvas = draw_corners(canvas, corner_2d_gt, None, radius=2)
        canvas = draw_bbox3d(canvas, corner_2d_gt, color=(0, 200, 0), thickness=1)
    if pose_corner_2d is not None:
        canvas = draw_bbox3d(canvas, pose_corner_2d, color=(255, 0, 0), thickness=2)
    if corner_2d is not None:
        canvas = draw_corners(canvas, corner_2d, None, radius=3)
    return canvas


def to_chw_float(panel: np.ndarray) -> torch.Tensor:
    """``[H, W, 3]`` uint8 RGB -> ``[3, H, W]`` float，供 TensorBoard 记录。"""
    return torch.from_numpy(panel.astype(np.float32) / 255.0).permute(2, 0, 1)
