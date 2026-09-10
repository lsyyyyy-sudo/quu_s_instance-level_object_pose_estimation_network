"""位姿相关的数学工具与评估指标。

对应 BoxDreamer/src/models/utils/pose_utils.py。

坐标系约定
----------
- 位姿 ``T_co``：把**物体坐标系**下的点变换到**相机坐标系**，形状 ``[4, 4]``。
- 相机坐标系：x 向右，y 向下，z 向前（OpenCV / BOP 约定）。
"""

import math
from typing import Optional

import torch

from src.models.utils.box_utils import project_points, transform_points


# --------------------------------------------------------------------------- #
# 位姿基本运算
# --------------------------------------------------------------------------- #
def rt_to_pose(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``R [3,3]`` + ``t [3]`` -> ``[4, 4]``。"""
    R = torch.as_tensor(R, dtype=torch.float32)
    t = torch.as_tensor(t, dtype=torch.float32).reshape(3)
    pose = torch.eye(4, dtype=torch.float32, device=R.device)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def pose_to_rt(pose: torch.Tensor):
    """``[4, 4]`` -> ``(R, t)``。"""
    pose = torch.as_tensor(pose, dtype=torch.float32)
    return pose[..., :3, :3], pose[..., :3, 3]


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    """位姿求逆（利用 R 是正交矩阵，比通用求逆快且稳）。"""
    pose = torch.as_tensor(pose, dtype=torch.float32)
    R, t = pose[..., :3, :3], pose[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(pose)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def compose_poses(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """位姿复合 ``a @ b``。"""
    return torch.as_tensor(a, dtype=torch.float32) @ torch.as_tensor(b, dtype=torch.float32)


def is_valid_pose(pose: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """判断 PnP 是否成功（失败时位姿是单位阵以外的全 0，最后一行为 0）。"""
    pose = torch.as_tensor(pose, dtype=torch.float32)
    return pose[..., 3, 3].abs() > eps


# --------------------------------------------------------------------------- #
# 旋转误差
# --------------------------------------------------------------------------- #
def rotation_angle_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """两个旋转矩阵之间的测地线夹角（度），``[...]`` -> ``[...]``。"""
    R_pred = torch.as_tensor(R_pred, dtype=torch.float32)
    R_gt = torch.as_tensor(R_gt, dtype=torch.float32)
    rel = R_pred.transpose(-1, -2) @ R_gt
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


# --------------------------------------------------------------------------- #
# 位姿精度指标
# --------------------------------------------------------------------------- #
def add_error(pose_pred: torch.Tensor, pose_gt: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """ADD：模型点集在预测/真值位姿下的平均对应点距离。

    注意：本项目没有 mesh，只有 3D 包围盒的 8 个角点，因此 ``points`` 传
    ``bbox_3d`` 即可（``[8, 3]`` 或 ``[B, 8, 3]``）。这与 BOP 官方用稠密
    模型点算出的 ADD 数值不完全可比，评估口径见 ``docs/``。

    Returns:
        ``[B]`` 平均距离
    """
    pose_pred = torch.as_tensor(pose_pred, dtype=torch.float32)
    pose_gt = torch.as_tensor(pose_gt, dtype=torch.float32)
    points = torch.as_tensor(points, dtype=torch.float32)

    pred = transform_points(points, pose_pred)
    gt = transform_points(points, pose_gt)
    return torch.linalg.norm(pred - gt, dim=-1).mean(dim=-1)


def adds_error(pose_pred: torch.Tensor, pose_gt: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """ADD-S：对每个点取最近邻，适用于旋转对称物体。"""
    pose_pred = torch.as_tensor(pose_pred, dtype=torch.float32)
    pose_gt = torch.as_tensor(pose_gt, dtype=torch.float32)
    points = torch.as_tensor(points, dtype=torch.float32)

    pred = transform_points(points, pose_pred)   # [B, N, 3]
    gt = transform_points(points, pose_gt)       # [B, N, 3]
    dist = torch.cdist(pred, gt)                 # [B, N, N]
    return dist.min(dim=-1).values.mean(dim=-1)


def pose_error_5cm5deg(
    pose_pred: torch.Tensor,
    pose_gt: torch.Tensor,
    points: torch.Tensor,
    translation_threshold_mm: float = 50.0,
    rotation_threshold_deg: float = 5.0,
    symmetric: bool = False,
) -> torch.Tensor:
    """BOP 常用的 5cm/5° 命中率。返回 ``[B]`` bool。

    ``translation_threshold_mm`` 的单位跟随 3D 模型（通常渲染数据是 mm）。
    """
    if symmetric:
        trans_err = adds_error(pose_pred, pose_gt, points)
    else:
        trans_err = add_error(pose_pred, pose_gt, points)

    R_pred = torch.as_tensor(pose_pred, dtype=torch.float32)[..., :3, :3]
    R_gt = torch.as_tensor(pose_gt, dtype=torch.float32)[..., :3, :3]
    rot_err = rotation_angle_deg(R_pred, R_gt)

    return (trans_err < translation_threshold_mm) & (rot_err < rotation_threshold_deg)


def reprojection_error(
    pose_pred: torch.Tensor,
    corner_2d_gt: torch.Tensor,
    bbox_3d: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """把 3D 角点用预测位姿投影回去，与 GT 2D 角点的平均像素距离。"""
    corner_2d_pred = project_points(bbox_3d, K, pose_pred)
    corner_2d_gt = torch.as_tensor(corner_2d_gt, dtype=torch.float32)
    return torch.linalg.norm(corner_2d_pred - corner_2d_gt, dim=-1).mean(dim=-1)


def geodesic_distance_deg(R1: torch.Tensor, R2: torch.Tensor) -> float:
    """标量版本的旋转误差，方便 numpy 侧调用。"""
    return float(rotation_angle_deg(R1, R2))


__all__ = [
    "rt_to_pose",
    "pose_to_rt",
    "invert_pose",
    "compose_poses",
    "is_valid_pose",
    "rotation_angle_deg",
    "add_error",
    "adds_error",
    "pose_error_5cm5deg",
    "reprojection_error",
    "geodesic_distance_deg",
]
