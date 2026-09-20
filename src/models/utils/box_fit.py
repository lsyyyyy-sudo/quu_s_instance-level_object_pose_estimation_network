"""把预测的 8 个 2D 角点投影回「合法长方体」。

问题
----
网络对 8 个角点**各自独立**回归（8 张热图 -> 8 个 argmax），没有任何几何约束。
实测预测出的 8 点深度棱夹角 72~90 度（合法投影应 < 10 度），面还会自交。

解法
----
给定 8 个 2D 点 + **已知的 3D 盒尺寸**，反解出一个「相机 + 位姿」，
再用它重新投影 —— 输出的 8 个点**必然是某个长方体的透视投影**。

关键：**焦距也是未知量**，所以不需要事先知道相机内参。

    未知数: f(1) + rvec(3) + tvec(3) = 7
    方程数: 8 点 x 2 = 16              -> 超定

主点固定为图像中心（裁剪以目标为中心，目标中心本就落在画面中心附近）。

四步
----
(1) DLT 线性初始化 3x4 相机矩阵 P
(2) RQ 分解 P -> (K, R, t)，归一化 K[2,2]=1
(3) torch 梯度精修 (log f, rvec, tvec)
(4) 重投影 3D 盒角点
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# (1) DLT
# --------------------------------------------------------------------------- #
def dlt_camera(points_2d: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """从 N>=6 组对应解 3x4 相机矩阵 P（最小二乘）。

    Args:
        points_2d: ``[N, 2]``
        points_3d: ``[N, 3]``

    Returns:
        ``[3, 4]`` 相机矩阵（未归一化尺度）
    """
    n = points_2d.shape[0]
    A = np.zeros((2 * n, 12), dtype=np.float64)
    for i in range(n):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]
    # 最小奇异值对应的右奇异向量
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    if abs(P[2, 3]) > 1e-12:
        P = P / P[2, 3]
    return P


def rq_decompose(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 3x4 的 P 分解成 K(3x3), R(3x3), t(3)。"""
    M = P[:, :3]
    # numpy 没有 rq，用 qr 的转置技巧
    Q, R_ = np.linalg.qr(np.flipud(M).T)
    R_ = np.flipud(R_.T)
    Q = np.flipud(Q.T)
    # 让 R_ 的对角为正
    sign = np.sign(np.diag(R_))
    sign[sign == 0] = 1.0
    R_ = R_ @ np.diag(sign)
    Q = np.diag(sign) @ Q
    if np.linalg.det(Q) < 0:
        R_ = -R_
        Q = -Q
    K = R_
    if abs(K[2, 2]) > 1e-12:
        K = K / K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    return K, Q, t


# --------------------------------------------------------------------------- #
# (3)(4) 精修 + 重投影
# --------------------------------------------------------------------------- #
def _rodrigues(rvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues 公式，rvec ``[3]`` -> R ``[3,3]``。"""
    theta = torch.linalg.norm(rvec) + 1e-12
    k = rvec / theta
    Kx = torch.zeros(3, 3, dtype=rvec.dtype, device=rvec.device)
    Kx[0, 1], Kx[0, 2] = -k[2], k[1]
    Kx[1, 0], Kx[1, 2] = k[2], -k[0]
    Kx[2, 0], Kx[2, 1] = -k[1], k[0]
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    return I + torch.sin(theta) * Kx + (1 - torch.cos(theta)) * (Kx @ Kx)


def project(f: torch.Tensor, rvec: torch.Tensor, tvec: torch.Tensor,
            pts3d: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
    """用 (f, rvec, tvec) 把 3D 点投影到 2D。pts3d ``[N,3]`` -> ``[N,2]``。"""
    R = _rodrigues(rvec)
    cam = (R @ pts3d.T).T + tvec            # [N,3]
    z = cam[:, 2].clamp(min=1e-6)
    u = f * cam[:, 0] / z + cx
    v = f * cam[:, 1] / z + cy
    return torch.stack([u, v], dim=-1)


def fit_box_to_corners(
    corners_2d: np.ndarray | torch.Tensor,
    bbox_3d: np.ndarray | torch.Tensor,
    image_size: int = 256,
    iters: int = 400,
    lr: float = 0.02,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """把 8 个 2D 角点约束成「合法长方体投影」—— **不需要任何相机内参**。

    原理
    ----
    8 个点必须是某个长方体在**某个 3x4 投影矩阵 P** 下的像：

        minimize over P (11 自由度)   sum_i ||proj(P, X_i) - p_i||^2

    方程 16 个、未知 11 个 -> **5 个独立约束**，这 5 个就是几何约束。
    ``P`` 可以是任意投影相机 —— 不假设焦距、主点、像素宽高比。
    所以本函数**完全不涉及内参**。

    实现
    ----
    (1) DLT 线性解出 P（11 自由度，16 方程，最小二乘）
    (2) 用 torch 在 P 上做几步高斯-牛顿/Adam 精修（Huber 损失抗外点）
    (3) 用 P 重投影 3D 盒角点 -> 8 个必然合法的点

    Args:
        corners_2d: ``[8, 2]`` 预测的角点
        bbox_3d:    ``[8, 3]`` 物体坐标系下的 3D 角点（用已知的盒尺寸）
        image_size: 只用于把坐标归一化到稳定数值范围（不是相机模型！）
        iters/lr:   精修迭代数与学习率

    Returns:
        ``(corners_refined [8,2], info)``；info 含 ``fit_err_px``（拟合残差）
        与 ``n_valid``。**残差大就说明原预测离"任何合法盒子"都很远，
        此时约束后的结果不可信 —— 调用方应当据此标记失败，而不是照单全收。**
    """
    c2 = np.asarray(corners_2d, dtype=np.float64)
    p3 = np.asarray(bbox_3d, dtype=np.float64)
    info = {"ok": False, "fit_err_px": float("nan")}

    # 数值稳定：把 2D/3D 都平移到各自质心、按尺度归一化
    c_mean, p_mean = c2.mean(0), p3.mean(0)
    c_scale = max(float(np.ptp(c2, axis=0).max()), 1e-6)
    p_scale = max(float(np.ptp(p3, axis=0).max()), 1e-6)
    cn = (c2 - c_mean) / c_scale
    pn = (p3 - p_mean) / p_scale

    # --- (1) DLT ---
    try:
        P = dlt_camera(cn, pn)                     # [3,4]，11 自由度
    except Exception as e:
        info["error"] = f"DLT 失败: {e}"
        return c2.astype(np.float32), info

    # --- (2) 精修 P（Huber）---
    Pt = torch.tensor(P, dtype=torch.float64, requires_grad=True)
    tgt = torch.tensor(cn, dtype=torch.float64)
    src = torch.tensor(pn, dtype=torch.float64)

    def _proj(Pm):
        ones = torch.ones(src.shape[0], 1, dtype=torch.float64)
        X = torch.cat([src, ones], dim=1)          # [8,4]
        q = (Pm @ X.T).T                           # [8,3]
        z = q[:, 2:3]
        # 避免 z=0（点在相机平面上）：加一个和符号一致的小量
        z = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
        return q[:, :2] / z

    opt = torch.optim.Adam([Pt], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        r = torch.linalg.norm(_proj(Pt) - tgt, dim=-1)
        delta = 0.05
        loss = torch.where(r < delta, 0.5 * r * r / delta, r - 0.5 * delta).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_n = _proj(Pt).numpy()
    pred = pred_n * c_scale + c_mean
    err = float(np.linalg.norm(pred - c2, axis=-1).mean())
    info["fit_err_px"] = err
    info["ok"] = bool(np.all(np.isfinite(pred)))
    # 投影矩阵是否退化（前两行秩 < 2 说明拟合不可信）
    info["P_rank"] = int(np.linalg.matrix_rank(Pt.detach().numpy()[:, :3]))
    return pred.astype(np.float32), info


# --------------------------------------------------------------------------- #
# 合法性度量（用于验证「约束前 vs 约束后」）
#
# ⚠️ 曾经的错误判据：要求四条深度棱在图像里【近似平行】。
#    这是错的！长方体在透视投影下，3D 里平行的棱在图像里【交于消失点】，
#    除非消失点在无穷远，否则它们在图像里并不平行。
#    「要求平行」= 要求仿射/正交投影，会把结果压成扁平的假盒子。
#
# 正确且【零内参】的判据：三组棱各自必须【共点】（交于消失点）。
# --------------------------------------------------------------------------- #
# BB8 顺序下，3D 里相互平行的三组棱
EDGE_FAMILIES = {
    "X": [(0, 1), (3, 2), (4, 5), (7, 6)],
    "Y": [(1, 2), (0, 3), (5, 6), (4, 7)],
    "Z": [(0, 4), (1, 5), (2, 6), (3, 7)],
}
FACES = [(0, 1, 2, 3), (4, 5, 6, 7)]


def _line_from_seg(p, q, c0, s):
    """线段 -> 归一化的齐次直线 l = p x q（每条线都除以自己的模）。

    ⚠️ 必须归一化：直线的 ``c`` 分量在图像坐标下是 ~1e6 量级，
    各行尺度差异巨大时 SVD 的最小奇异值会给出虚假的"共点"。
    同时把坐标中心化到原点、按尺度缩放，改善条件数。
    """
    a = np.array([(p[0] - c0[0]) / s, (p[1] - c0[1]) / s, 1.0])
    b = np.array([(q[0] - c0[0]) / s, (q[1] - c0[1]) / s, 1.0])
    l = np.cross(a, b)
    n = np.linalg.norm(l)
    return l / n if n > 1e-12 else l


def _concurrence_residual(lines: np.ndarray) -> float:
    """4 条【已归一化】的齐次直线是否共点。

    共点 <=> 4x3 的直线矩阵秩为 2 <=> 最小奇异值为 0。
    返回 最小/最大 奇异值比（0 最好）。
    """
    if lines.shape[0] < 3:
        return float("nan")
    s = np.linalg.svd(lines, compute_uv=False)
    return float(s[-1] / max(s[0], 1e-12))


def cuboid_metrics(c: np.ndarray) -> dict:
    """返回「像不像一个长方体投影」的指标（**不需要内参**）。

    正确判据：3D 里相互平行的三组棱，其投影必须各自【共点】（交于消失点）。
    返回 ``vp_residual_max``（三组中最差的一个，0 最好）。
    """
    c = np.asarray(c, dtype=np.float64)
    c0 = c.mean(axis=0)
    s = max(float(np.ptp(c, axis=0).max()), 1e-6)
    out = {}
    for fam, segs in EDGE_FAMILIES.items():
        lines = np.array([_line_from_seg(c[i], c[j], c0, s) for i, j in segs])
        out[f"vp_residual_{fam}"] = _concurrence_residual(lines)
    out["vp_residual_max"] = max(out[f"vp_residual_{f}"] for f in EDGE_FAMILIES)

    L = [float(np.linalg.norm(c[j] - c[i])) for i, j in EDGE_FAMILIES["Y"]]
    out["depth_len_ratio"] = float(max(L) / max(min(L), 1e-6))
    out["faces_simple"] = [bool(is_simple_quad(c[list(f)])) for f in FACES]
    # 阈值 5e-2 由对照组标定：真盒子投影 + 1px 噪声时残差约 2.7e-2
    out["is_cuboidish"] = bool(out["vp_residual_max"] < 5e-2) and all(out["faces_simple"])
    return out


def is_simple_quad(p) -> bool:
    """四边形是否简单（不自交）。"""
    p = np.asarray(p, dtype=np.float64)
    a, b, cc, d = p

    def seg_int(p1, p2, p3, p4):
        def cr(o, x, y):
            return (x[0] - o[0]) * (y[1] - o[1]) - (x[1] - o[1]) * (y[0] - o[0])
        d1, d2 = cr(p3, p4, p1), cr(p3, p4, p2)
        d3, d4 = cr(p1, p2, p3), cr(p1, p2, p4)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
    return not (seg_int(a, b, cc, d) or seg_int(b, cc, d, a))
