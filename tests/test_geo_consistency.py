"""几何一致性损失的单元测试 —— **带对照组**。

今天已经在判据上错过两次，所以任何"像不像盒子"的度量都必须先过这一关：
  ✅ 真盒子投影          -> 残差 ~0
  ❌ 随机 8 点           -> 残差明显大
  ✅ 真投影 + 小噪声     -> 残差随噪声单调增大
  ✅ 梯度可回传且有限
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.loss.utils.geo_consistency import (  # noqa: E402
    concurrence_residual,
    geo_consistency_loss,
)

BI = np.array([-34.97, -22.435, -16.545])
BA = np.array([34.97, 22.435, 16.545])
BITS = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0),
        (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]
X3 = np.array([[(BA if b[k] else BI)[k] for k in range(3)] for b in BITS])


def _rand_R(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _project(R, t, f=800.0, c=(1600.0, 1200.0)):
    cam = (R @ X3.T).T + t
    return np.hstack([f * cam[:, :2] / cam[:, 2:3] + np.array(c)])


def test_true_box_projection_has_near_zero_residual():
    """真盒子投影的共点残差应当接近 0。"""
    rng = np.random.default_rng(0)
    res = []
    for _ in range(20):
        R = _rand_R(rng)
        t = np.array([rng.uniform(-100, 100), rng.uniform(-100, 100),
                      rng.uniform(500, 1200)])
        c = torch.tensor(_project(R, t), dtype=torch.float64)[None]
        res.append(float(concurrence_residual(c)[0]))
    res = np.array(res)
    assert res.max() < 1e-9, f"真投影残差应当 ~0，实际 max={res.max():.2e}"


def test_random_points_have_large_residual():
    """随机 8 点的残差必须明显大于真投影。"""
    rng = np.random.default_rng(1)
    res = []
    for _ in range(20):
        c = torch.tensor(rng.uniform(0, 3200, size=(8, 2)), dtype=torch.float64)[None]
        res.append(float(concurrence_residual(c)[0]))
    res = np.array(res)
    assert res.min() > 1e-3, f"随机点残差应当很大，实际 min={res.min():.2e}"


def test_separation_between_positive_and_negative():
    """正负样本之间必须有数量级上的分离（否则这个指标没有判别力）。"""
    rng = np.random.default_rng(2)
    pos, neg = [], []
    for _ in range(30):
        R = _rand_R(rng)
        t = np.array([rng.uniform(-100, 100), rng.uniform(-100, 100),
                      rng.uniform(500, 1200)])
        pos.append(float(concurrence_residual(
            torch.tensor(_project(R, t), dtype=torch.float64)[None])[0]))
        neg.append(float(concurrence_residual(
            torch.tensor(rng.uniform(0, 3200, size=(8, 2)), dtype=torch.float64)[None])[0]))
    assert np.median(neg) > 1e4 * max(np.median(pos), 1e-12)


def test_residual_increases_with_noise():
    """加噪应当让残差单调增大（说明指标确实在度量几何合法性）。"""
    rng = np.random.default_rng(3)
    R = _rand_R(rng)
    base = _project(R, np.array([30.0, -20.0, 700.0]))
    meds = []
    for sig in [0.0, 1.0, 3.0, 10.0, 30.0]:
        rs = [float(concurrence_residual(
            torch.tensor(base + rng.standard_normal((8, 2)) * sig,
                         dtype=torch.float64)[None])[0]) for _ in range(20)]
        meds.append(float(np.median(rs)))
    for a, b in zip(meds, meds[1:]):
        assert b > a, f"残差未随噪声增大: {meds}"


def test_loss_is_differentiable_and_finite():
    """损失必须可回传、梯度有限 —— 否则加进训练会炸。"""
    rng = np.random.default_rng(4)
    R = _rand_R(rng)
    c = torch.tensor(_project(R, np.array([10.0, 5.0, 800.0])),
                     dtype=torch.float32)[None].clone().requires_grad_(True)
    # 加一点噪声，避免在精确解处梯度为 0
    with torch.no_grad():
        c += torch.randn_like(c) * 2.0
    loss = geo_consistency_loss(c)
    assert torch.isfinite(loss), "损失不是有限值"
    loss.backward()
    assert c.grad is not None and torch.isfinite(c.grad).all(), "梯度不是有限值"
    assert c.grad.abs().sum() > 0, "梯度全 0，损失没有作用"


def test_loss_prefers_valid_box_over_random():
    """同一批里，合法盒子的损失应当【远小于】随机点。

    ⚠️ 不用绝对值阈值：轻微噪声下合法盒子的损失也是 O(1e-3)，不是 0。
    有判别力的性质是【相对分离度】，损失的大小只影响 geo_weight 怎么调。
    """
    rng = np.random.default_rng(5)
    R = _rand_R(rng)
    good = torch.tensor(_project(R, np.array([0.0, 0.0, 700.0])),
                        dtype=torch.float64)[None]
    with torch.no_grad():
        good = good + torch.randn_like(good) * 0.5     # 轻微扰动
    bad = torch.tensor(rng.uniform(0, 3200, size=(1, 8, 2)), dtype=torch.float64)
    lg, lb = float(geo_consistency_loss(good)), float(geo_consistency_loss(bad))
    print(f"\n    合法盒子(0.5px 噪声) loss = {lg:.3e}")
    print(f"    随机 8 点          loss = {lb:.3e}")
    print(f"    分离度 = {lb / lg:.1f}x")
    assert lg < 0.05, f"合法盒子损失过大 {lg:.2e}（说明损失标定有问题）"
    assert lb > 1e-4, f"随机点损失过小 {lb:.2e}"
    # 阈值 50x：实测在 90~1700x 之间波动（取决于随机种子与扰动幅度），
    # 只要有两个数量级的分离就足够说明指标有判别力。
    assert lb / lg > 50, f"分离度不足: {lb/lg:.1f}x"


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        geo_consistency_loss(torch.zeros(2, 7, 2))
    with pytest.raises(ValueError):
        geo_consistency_loss(torch.zeros(2, 8, 3))
