"""3D 包围盒角点（bb8）的定义、投影与 PnP 求解。

对应 BoxDreamer/src/models/utils/box_utils.py。

与 BoxDreamer 的差异
--------------------
BoxDreamer 是**多视角**的，角点张量形状是 ``[B, T, ...]``（T = 参考图 + 查询图）。
本项目是**单图**输入，所有张量都去掉 T 这一维，形状为 ``[B, ...]``。

角点顺序（bb8 约定，与 BoxDreamer 的 ``consist_bbox3d`` 完全一致）：
先底面逆时针 4 个点，再顶面逆时针 4 个点::

    bits = [[0,0,0], [0,1,0], [1,1,0], [1,0,0],
            [0,0,1], [0,1,1], [1,1,1], [1,0,1]]

    corner_i = min_xyz + bits_i * (max_xyz - min_xyz)
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# 8 个角点在包围盒 min/max 上的 0/1 编码（bb8 顺序）
BB8_BITS = (
    (0, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (1, 0, 1),
)


def _bits_tensor(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(BB8_BITS, device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# 角点构造
# --------------------------------------------------------------------------- #
def bbox3d_from_minmax(min_xyz, max_xyz) -> torch.Tensor:
    """由包围盒 min/max 构造 8 个角点。

    Args:
        min_xyz: ``[3]`` 或 ``[B, 3]``
        max_xyz: ``[3]`` 或 ``[B, 3]``

    Returns:
        ``[8, 3]`` 或 ``[B, 8, 3]``
    """
    min_xyz = torch.as_tensor(min_xyz, dtype=torch.float32)
    max_xyz = torch.as_tensor(max_xyz, dtype=torch.float32)
    bits = _bits_tensor(min_xyz.device)  # [8, 3]

    if min_xyz.ndim == 1:
        extent = (max_xyz - min_xyz).unsqueeze(0)          # [1, 3]
        return min_xyz.unsqueeze(0) + bits * extent        # [8, 3]
    extent = (max_xyz - min_xyz).unsqueeze(1)              # [B, 1, 3]
    return min_xyz.unsqueeze(1) + bits.unsqueeze(0) * extent  # [B, 8, 3]


def bbox3d_from_models_info(models_info: dict, obj_id: int) -> torch.Tensor:
    """从 BOP 的 ``models_info.json`` 里取某个物体的 3D 包围盒角点。

    文件由 HCCEPose 的 ``s1_p3_obj_infos.py`` 生成，结构::

        {"1": {"min_x": .., "min_y": .., "min_z": ..,
               "max_x": .., "max_y": .., "max_z": ..,
               "diameter": ..}, ...}
    """
    key = str(int(obj_id))
    if key not in models_info:
        raise KeyError(f"obj_id={obj_id} not found in models_info.json")
    info = models_info[key]
    min_xyz = [info["min_x"], info["min_y"], info["min_z"]]
    max_xyz = [info["max_x"], info["max_y"], info["max_z"]]
    return bbox3d_from_minmax(min_xyz, max_xyz)


def bbox3d_diameter(bbox_3d: torch.Tensor) -> torch.Tensor:
    """包围盒角点两两之间的最大距离（物体尺度，用于归一化误差）。"""
    bbox_3d = torch.as_tensor(bbox_3d, dtype=torch.float32)
    diff = bbox_3d.unsqueeze(-2) - bbox_3d.unsqueeze(-3)   # [..., 8, 8, 3]
    return torch.linalg.norm(diff, dim=-1).amax(dim=(-1, -2))


# --------------------------------------------------------------------------- #
# 投影 / 变换
# --------------------------------------------------------------------------- #
def transform_points(pts_3d: torch.Tensor, Rt: torch.Tensor) -> torch.Tensor:
    """用 ``[3,4]`` / ``[4,4]`` 位姿变换点。支持带 batch 的广播。

    Args:
        pts_3d: ``[..., N, 3]``
        Rt:     ``[..., 3, 4]`` 或 ``[..., 4, 4]``

    Returns:
        ``[..., N, 3]``
    """
    pts_3d = torch.as_tensor(pts_3d, dtype=torch.float32)
    Rt = torch.as_tensor(Rt, dtype=torch.float32)
    R = Rt[..., :3, :3]
    t = Rt[..., :3, 3]
    return pts_3d @ R.transpose(-1, -2) + t.unsqueeze(-2)


# BoxDreamer 里叫 bbox_3d 的变换，这里单独给个别名保持可读性
transform_bbox3d = transform_points


def project_points(
    pts_3d: torch.Tensor,
    K: torch.Tensor,
    Rt: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """针孔投影。``pts_3d`` 为 ``[..., N, 3]``（相机坐标系，或给 ``Rt`` 先变换）。

    Args:
        pts_3d: ``[..., N, 3]``
        K:      ``[3, 3]`` 或 ``[..., 3, 3]``
        Rt:     可选位姿，给了就先变换到相机坐标系

    Returns:
        ``[..., N, 2]`` 像素坐标
    """
    pts_3d = torch.as_tensor(pts_3d, dtype=torch.float32)
    K = torch.as_tensor(K, dtype=torch.float32)
    if Rt is not None:
        pts_3d = transform_points(pts_3d, Rt)

    z = pts_3d[..., 2].clamp(min=eps)
    x = pts_3d[..., 0] / z
    y = pts_3d[..., 1] / z

    fx, fy = K[..., 0, 0], K[..., 1, 1]
    cx, cy = K[..., 0, 2], K[..., 1, 2]
    skew = K[..., 0, 1]

    u = fx.unsqueeze(-1) * x + skew.unsqueeze(-1) * y + cx.unsqueeze(-1)
    v = fy.unsqueeze(-1) * y + cy.unsqueeze(-1)
    return torch.stack([u, v], dim=-1)


def project_bbox3d(bbox_3d: torch.Tensor, K: torch.Tensor, Rt: torch.Tensor) -> torch.Tensor:
    """把 3D 包围盒角点投影到图像上，返回 ``[..., 8, 2]``。"""
    return project_points(bbox_3d, K, Rt)


# --------------------------------------------------------------------------- #
# 从 2D 角点解位姿
# --------------------------------------------------------------------------- #
def recover_pose_from_bb8(
    corner_2d: torch.Tensor,
    bbox_3d: torch.Tensor,
    K: torch.Tensor,
    use_ransac: bool = False,
    ransac_reproj_threshold: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """用 ``cv2.solvePnP`` 从 2D-3D 角点对应解出物体位姿。

    对应 BoxDreamer 的 ``recover_pose_from_bb8``。差异：
    BoxDreamer 的输入是**原始网络输出**（还要先跑 recover_bb8_corners 提取角点），
    这里直接吃已经提取好的 2D 角点 ``[B, 8, 2]``（单图，无 T 维）。

    Args:
        corner_2d: ``[B, 8, 2]`` 预测（或 GT）的角点像素坐标
        bbox_3d:   ``[B, 8, 3]`` 物体坐标系下的角点
        K:         ``[B, 3, 3]`` 或 ``[3, 3]`` 相机内参
        use_ransac: 是否先用 RANSAC 剔除外点（BoxDreamer 实测关掉了）

    Returns:
        ``(poses, corner_2d)``，其中 ``poses`` 为 ``[B, 4, 4]``。
        PnP 失败的样本位姿保持全 0，可由调用方过滤。
    """
    corner_2d = torch.as_tensor(corner_2d, dtype=torch.float32)
    bbox_3d = torch.as_tensor(bbox_3d, dtype=torch.float32)
    K = torch.as_tensor(K, dtype=torch.float32)

    if corner_2d.ndim == 2:  # 允许传入 [8, 2]
        corner_2d = corner_2d.unsqueeze(0)
        bbox_3d = bbox_3d.unsqueeze(0) if bbox_3d.ndim == 2 else bbox_3d

    B = corner_2d.shape[0]
    poses = torch.zeros(B, 4, 4, dtype=torch.float32, device=corner_2d.device)
    poses[:, 3, 3] = 1.0

    for b in range(B):
        pts_2d_np = corner_2d[b].detach().cpu().numpy().astype(np.float64)
        pts_3d_np = bbox_3d[b].detach().cpu().numpy().astype(np.float64)
        K_np = (K if K.ndim == 2 else K[b]).detach().cpu().numpy().astype(np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        rvec = tvec = None
        ok = False
        if use_ransac:
            try:
                ok, rvec, tvec, _ = cv2.solvePnPRansac(
                    pts_3d_np, pts_2d_np, K_np, dist,
                    reprojectionError=ransac_reproj_threshold,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                ok = False

        if not ok:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    pts_3d_np, pts_2d_np, K_np, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
            except cv2.error:
                ok = False

        if not ok or rvec is None:
            continue

        R_np, _ = cv2.Rodrigues(rvec)
        pose = np.hstack((R_np, tvec.reshape(3, 1)))  # [3, 4]
        poses[b, :3, :] = torch.from_numpy(pose).to(poses.dtype)

    return poses, corner_2d
