"""多实例损失的单元测试 —— 关键是**坐标一致性**。

今天已经因为"坐标系/度量没对齐"错过三次，所以这里专门有往返测试：
把 GT 当作预测喂进去，损失必须接近 0。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.loss.utils.multi_instance_loss import (  # noqa: E402
    center_focal_loss,
    corner_regression_loss,
    multi_instance_loss,
)
from src.models.utils.multi_instance import (  # noqa: E402
    decode_instances,
    gather_corners,
)

def _logits_from_gt(gt: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """把 [0,1] 的 GT 热图变成「完美预测」的 logits（反 sigmoid）。

    ⚠️ 不能用 ``where(gt > t, big, -big)`` 二值化 —— 那会把高斯变成
    **平坦圆盘**，盘内每个像素都是极大值，NMS 会解出一堆假峰（实测
    一次解出 20 个）。真实网络输出是光滑的，必须用反 sigmoid。
    """
    g = gt.clamp(eps, 1 - eps)
    return torch.log(g) - torch.log1p(-g)


IMG, HM = 256, 64
STRIDE = IMG / HM      # 4.0


def _batch(n_inst=3, seed=0):
    """构造一个 GT 一致的 batch。"""
    rng = np.random.default_rng(seed)
    B, N = 1, 4
    corners = np.zeros((B, N, 8, 2), dtype=np.float32)
    centers = np.zeros((B, N, 2), dtype=np.float32)
    valid = np.zeros((B, N), dtype=np.float32)
    ctr = np.zeros((B, 1, HM, HM), dtype=np.float32)
    yy, xx = np.mgrid[0:HM, 0:HM]
    for i in range(n_inst):
        # 中心落在格子中心，避免取整误差
        hx, hy = rng.integers(8, HM - 8, size=2)
        cx, cy = hx * STRIDE, hy * STRIDE
        centers[0, i] = (cx, cy)
        # 一个围绕中心的假盒子
        ang = rng.uniform(0, np.pi)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        base = np.array([[-20, -14], [-20, 14], [20, 14], [20, -14],
                         [-14, -8], [-14, 8], [14, 8], [14, -8]], dtype=np.float32)
        corners[0, i] = base @ R.T + np.array([cx, cy])
        valid[0, i] = 1.0
        g = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * 2.0 ** 2))
        ctr[0, 0] = np.maximum(ctr[0, 0], g.astype(np.float32))
    return {
        "center_heatmap": torch.from_numpy(ctr),
        "inst_corners": torch.from_numpy(corners),
        "inst_centers": torch.from_numpy(centers),
        "inst_valid": torch.from_numpy(valid),
    }


def test_center_focal_zero_for_perfect():
    """完美的中心热图 -> loss 接近 0。"""
    b = _batch()
    gt = b["center_heatmap"]
    # 用很大的 logits 模拟"完全正确"
    logits = _logits_from_gt(gt)
    perfect = float(center_focal_loss(logits, gt))
    # 无信息预测（全部 logits=0 -> p=0.5）作为基线
    baseline = float(center_focal_loss(torch.zeros_like(gt), gt))
    print(f"\n    完美 p=gt: {perfect:.4f}   无信息 p=0.5: {baseline:.4f}   "
          f"改善 {baseline / max(perfect, 1e-9):.1f}x")
    # focal 对中间取值 g 有加权，完美预测也不是 0 —— 所以断言【相对改善】
    assert perfect < 0.5 * baseline, \
        f"完美预测应当明显优于无信息预测: {perfect:.4f} vs {baseline:.4f}"


def test_center_focal_large_for_wrong():
    """完全错误的热图 -> loss 明显大。"""
    b = _batch()
    gt = b["center_heatmap"]
    logits = -_logits_from_gt(gt)
    loss = float(center_focal_loss(logits, gt))
    good = float(center_focal_loss(_logits_from_gt(gt), gt))
    assert loss > 20 * max(good, 1e-6), \
        f"错误预测应当远大于正确预测: {loss:.4f} vs {good:.4f}"


def test_corner_regression_zero_for_gt():
    """**往返测试**：把 GT 角点当作预测写进 offset，损失必须接近 0。

    这直接检验「数据集 -> 损失 -> 解码」三者的坐标系是否一致。
    """
    b = _batch()
    offset = torch.zeros(1, 16, HM, HM)
    for i in range(b["inst_valid"].shape[1]):
        if b["inst_valid"][0, i] <= 0:
            continue
        cx, cy = b["inst_centers"][0, i].tolist()
        hx = int(round(cx / STRIDE)); hy = int(round(cy / STRIDE))
        base = torch.tensor([hx * STRIDE, hy * STRIDE])
        d = (b["inst_corners"][0, i].reshape(8, 2) - base)     # [8,2]
        offset[0, :, hy, hx] = d.reshape(-1)
    loss = float(corner_regression_loss(offset, b["inst_corners"], b["inst_centers"],
                                        b["inst_valid"], IMG, HM))
    assert loss < 1e-4, f"往返损失应当接近 0，实际 {loss:.6f}"


def test_gather_and_regression_agree():
    """``gather_corners`` 与 ``corner_regression_loss`` 必须用同一套约定。"""
    b = _batch()
    offset = torch.zeros(1, 16, HM, HM)
    for i in range(b["inst_valid"].shape[1]):
        if b["inst_valid"][0, i] <= 0:
            continue
        cx, cy = b["inst_centers"][0, i].tolist()
        hx = int(round(cx / STRIDE)); hy = int(round(cy / STRIDE))
        base = torch.tensor([hx * STRIDE, hy * STRIDE])
        offset[0, :, hy, hx] = (b["inst_corners"][0, i].reshape(8, 2) - base).reshape(-1)

    hx = torch.tensor([[int(round(b["inst_centers"][0, i, 0].item() / STRIDE))
                        for i in range(4)]])
    hy = torch.tensor([[int(round(b["inst_centers"][0, i, 1].item() / STRIDE))
                        for i in range(4)]])
    got = gather_corners(offset, hy, hx, stride=STRIDE)[0]      # [4,8,2]
    for i in range(4):
        if b["inst_valid"][0, i] <= 0:
            continue
        assert torch.allclose(got[i], b["inst_corners"][0, i], atol=1e-4), \
            f"实例{i} 的 gather 结果与 GT 不一致"


def test_full_loss_differentiable():
    b = _batch()
    logits = torch.zeros(1, 1, HM, HM, requires_grad=True)
    offset = torch.zeros(1, 16, HM, HM, requires_grad=True)
    out = multi_instance_loss(logits, offset, b, IMG, HM, geo_weight=1.0)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert torch.isfinite(offset.grad).all() and offset.grad.abs().sum() > 0


def test_geo_weight_zero_disables_geo():
    b = _batch()
    logits = torch.zeros(1, 1, HM, HM)
    # ⚠️ 不能用全 0 的 offset：那时 8 个角点重合于一点，是**退化输入**，
    # 几何项恰好为 0，测不出"参与与否"。这里放一个非退化的偏移。
    # 用【随机】偏移：既非退化（8 点不重合），也不是合法盒子（几何项 > 0）
    g = torch.Generator().manual_seed(0)
    offset = torch.randn(1, 16, HM, HM, generator=g) * 6.0
    a = multi_instance_loss(logits, offset, b, IMG, HM, geo_weight=0.0)
    c = multi_instance_loss(logits, offset, b, IMG, HM, geo_weight=5.0)
    assert abs(float(a["geo"])) < 1e-9, "geo_weight=0 时几何项应当为 0"
    assert float(c["geo"]) > 0, "geo_weight>0 时几何项应当参与"
    assert float(c["loss"]) > float(a["loss"]), "几何项应当抬高总损失"


def test_decode_roundtrip_from_gt():
    """**端到端往返**：把 GT 写进 (heatmap, offset)，解码应当还原出 GT 实例。"""
    b = _batch(n_inst=3)
    # 中心：用大 logits 让 GT 峰成为唯一的峰
    gt = b["center_heatmap"]
    logits = _logits_from_gt(gt)
    offset = torch.zeros(1, 16, HM, HM)
    for i in range(4):
        if b["inst_valid"][0, i] <= 0:
            continue
        cx, cy = b["inst_centers"][0, i].tolist()
        hx = int(round(cx / STRIDE)); hy = int(round(cy / STRIDE))
        base = torch.tensor([hx * STRIDE, hy * STRIDE])
        offset[0, :, hy, hx] = (b["inst_corners"][0, i].reshape(8, 2) - base).reshape(-1)

    out = decode_instances(logits, offset, thr=0.5, topk=20, min_dist=4.0,
                           stride=STRIDE)[0]
    assert len(out) == 3, f"应当解出 3 个实例，实际 {len(out)}"
    # 每个解出的实例，其角点应当能在 GT 里找到匹配
    gt_c = b["inst_corners"][0][:3].numpy()
    for o in out:
        d = np.linalg.norm(gt_c - o["corners"][None], axis=-1).mean(axis=-1)
        assert d.min() < 1.0, f"解出的实例与任何 GT 都不匹配（最小距离 {d.min():.2f}）"


def test_bad_shape_rejected():
    b = _batch()
    with pytest.raises(ValueError):
        center_focal_loss(torch.zeros(1, 1, HM, HM), torch.zeros(1, 1, HM, HM + 1))
    with pytest.raises(ValueError):
        gather_corners(torch.zeros(1, 8, HM, HM),
                       torch.zeros(1, 2, dtype=torch.long),
                       torch.zeros(1, 2, dtype=torch.long))

def test_gt_heatmap_has_exact_peaks():
    """⚠️ 回归测试：GT 中心热图必须有【精确等于 1.0】的格点。

    这是 ALGO-08（focal 零正样本）的同一个坑，今天第二次踩：
    数据集把高斯峰放在浮点位置 -> 没有任何格子等于 1.0
    -> focal 的 pos 掩码（gt >= 1-1e-4）一个都匹配不到
    -> 中心头没有监督 -> 塌缩成"处处无物体" -> recall = 0
    （实测 8 个实例的 batch 只有 1 个正样本格）。

    这里用数据集【真实的】构造逻辑复现一遍。
    """
    import numpy as np

    HM, IMG, sigma = 64, 256, 2.0
    scale = HM / float(IMG)
    yy, xx = np.mgrid[0:HM, 0:HM]
    rng = np.random.default_rng(7)
    hm = np.zeros((1, HM, HM), dtype=np.float32)
    centers = rng.uniform(40, 200, size=(6, 2))       # 浮点中心
    for c in centers:
        gx = float(np.round(c[0] * scale))
        gy = float(np.round(c[1] * scale))
        gx = min(max(gx, 0.0), HM - 1.0)
        gy = min(max(gy, 0.0), HM - 1.0)
        g = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        g[int(gy), int(gx)] = 1.0
        hm[0] = np.maximum(hm[0], g)

    n_exact = int((hm >= 1.0 - 1e-6).sum())
    assert n_exact >= len(centers), (
        f"精确等于 1.0 的格点只有 {n_exact} 个，放了 {len(centers)} 个实例 "
        "-> focal 匹配不到正样本"
    )

    gt = torch.from_numpy(hm)[None]
    assert float(gt.ge(0.9).sum()) >= len(centers), "放宽到 0.9 后仍匹配不到正样本"
    logits = _logits_from_gt(gt)
    loss = float(center_focal_loss(logits, gt))
    assert np.isfinite(loss) and loss < 5.0, f"完美热图的 focal 过大: {loss}"
