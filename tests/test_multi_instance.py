"""多实例解码的单元测试 —— 用【合成热图】验证端到端正确性。

重点验证：
  · 峰数 = 实例数（数量可变，不写死）
  · 读出角点后能还原到正确位置
  · 相邻实例不会被合并（NMS 只压制同一个局部极大值的邻域）
  · 低分峰被阈值滤掉
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.utils.multi_instance import (  # noqa: E402
    decode_instances,
    gather_corners,
    nms_heatmap,
    topk_peaks,
)


def _make_heatmap(h, w, centers, sigma=1.5):
    """在给定中心放高斯峰（logits：峰值 +8，背景 -8）。

    ⚠️ 这里踩过一个坑：不能写 ``maximum(hm, 8*exp(...))``。
    高斯在远处下溢到 0，``maximum(-8, 0) = 0`` 会把背景从 -8 抬到 0，
    而 ``sigmoid(0) = 0.5`` 会越过任何合理阈值 -> 全是假峰。
    正确做法是让高斯项【相对背景】偏移：``8*g - 8``，远处仍是 -8。
    """
    hm = torch.full((1, 1, h, w), -8.0)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    for (cx, cy) in centers:
        d2 = (xx - cx).float() ** 2 + (yy - cy).float() ** 2
        g = torch.exp(-d2 / (2 * sigma ** 2))
        hm[0, 0] = torch.maximum(hm[0, 0], 16.0 * g - 8.0)
    return hm


def test_peak_count_matches_instance_count():
    """峰数必须等于实例数（这是多实例的核心要求）。"""
    for n in [1, 2, 3, 5]:
        centers = [(10 + 12 * i, 20 + 3 * i) for i in range(n)]
        hm = _make_heatmap(64, 64, centers)
        off = torch.zeros(1, 16, 64, 64)
        out = decode_instances(hm, off, thr=0.3, topk=20)
        assert len(out) == 1
        assert len(out[0]) == n, f"{n} 个实例只解出 {len(out[0])} 个"


def test_centers_are_accurate():
    """解出的中心应当落在真值附近。"""
    centers = [(10, 20), (40, 15), (30, 50)]
    hm = _make_heatmap(64, 64, centers)
    off = torch.zeros(1, 16, 64, 64)
    out = decode_instances(hm, off, thr=0.3, topk=20)[0]
    assert len(out) == 3
    got = sorted([tuple(np.round(o["center"]).astype(int)) for o in out])
    for (gx, gy), (tx, ty) in zip(got, sorted(centers)):
        assert abs(gx - tx) <= 1 and abs(gy - ty) <= 1, f"中心偏差过大 {gx,gy} vs {tx,ty}"


def test_corners_gathered_at_correct_location():
    """每个实例应当读到【它自己】那个位置的偏移。"""
    h = w = 64
    centers = [(10, 10), (50, 50)]
    hm = _make_heatmap(h, w, centers)
    off = torch.zeros(1, 16, h, w)
    # 在 (10,10) 处放 +100，在 (50,50) 处放 -100，方便区分
    off[0, :, 10, 10] = 100.0
    off[0, :, 50, 50] = -100.0
    out = decode_instances(hm, off, thr=0.3, topk=20)[0]
    by = {tuple(np.round(o["center"]).astype(int)): o["corners"] for o in out}
    assert (10, 10) in by and (50, 50) in by
    assert np.allclose(by[(10, 10)], 100.0 + np.array([[10, 10]] * 8))
    assert np.allclose(by[(50, 50)], -100.0 + np.array([[50, 50]] * 8))


def test_nms_suppresses_plateau():
    """一个宽峰（平台/宽脊）只能算一个实例，不能被数成多个。

    ⚠️ 要测【真正的入口】``decode_instances``。只测 max-pool NMS 是测不出来的 ——
    max-pool 用 ``>=`` 比较，平台会被整片保留（实测 4x4 -> 16 个"峰"）。
    真正兜住它的是后面那层【按距离的贪心 NMS】。
    """
    hm = torch.full((1, 1, 32, 32), -8.0)
    hm[0, 0, 10:14, 10:14] = 8.0          # 4x4 的平台
    out = decode_instances(hm, torch.zeros(1, 16, 32, 32),
                           thr=0.5, topk=20, min_dist=4.0)[0]
    assert len(out) == 1, f"平台被数成了 {len(out)} 个实例"
    cx, cy = out[0]["center"]
    assert 10 <= cx <= 13 and 10 <= cy <= 13, f"中心落在平台外: {(cx, cy)}"


def test_low_score_peaks_filtered():
    """低于阈值的峰要被滤掉。"""
    hm = _make_heatmap(64, 64, [(10, 10)])
    hm[0, 0, 50, 50] = -5.0               # 很弱的峰（sigmoid 后 ~0.007）
    out = decode_instances(hm, torch.zeros(1, 16, 64, 64), thr=0.3, topk=20)[0]
    assert len(out) == 1, f"弱峰没被滤掉，解出 {len(out)} 个"


def test_close_instances_survive():
    """相距几个像素的两个实例应当都能保留（NMS 邻域只有 3x3）。"""
    hm = _make_heatmap(64, 64, [(20, 20), (26, 20)], sigma=1.0)
    out = decode_instances(hm, torch.zeros(1, 16, 64, 64), thr=0.3, topk=20)[0]
    assert len(out) == 2, f"相邻实例被合并了，只解出 {len(out)} 个"


def test_batch_independence():
    """batch 内不同图的结果不能互相串。"""
    hm = torch.cat([_make_heatmap(64, 64, [(10, 10)]),
                    _make_heatmap(64, 64, [(30, 30), (40, 40)])], dim=0)
    out = decode_instances(hm, torch.zeros(2, 16, 64, 64), thr=0.3, topk=20)
    assert len(out) == 2
    assert len(out[0]) == 1
    assert len(out[1]) == 2


def test_bad_channels_rejected():
    import pytest
    with pytest.raises(ValueError):
        decode_instances(torch.zeros(1, 8, 64, 64), torch.zeros(1, 16, 64, 64))
    with pytest.raises(ValueError):
        gather_corners(torch.zeros(1, 8, 64, 64), torch.zeros(1, 2, dtype=torch.long),
                       torch.zeros(1, 2, dtype=torch.long))

def test_center_is_in_crop_pixels_like_corners():
    """⚠️ 回归测试：decode 返回的 center 必须和 corners 同一坐标系（裁剪像素）。

    这个 bug 的表现非常隐蔽：解码出的【数量】和 GT 完全一致，
    只是位置全都差了 stride 倍（4x），于是 recall 只有 3%。
    如果不是逐级追踪（GT 数 / NMS 后格点数 / topk / 匹配数），
    很容易误判成"检测器没学会"。

    这里直接验证：offset 里 8 个角点相对中心的偏移恒为 (dx, dy) 时，
    解出的 center 和 corners 之间必须正好差 (dx, dy)。
    """
    h = w = 32
    stride = 4.0
    hm = _make_heatmap(h, w, [(8, 8)])
    off = torch.zeros(1, 16, h, w)
    dx, dy = 5.0, -3.0
    off[0, 0::2, 8, 8] = dx       # x 偏移
    off[0, 1::2, 8, 8] = dy       # y 偏移

    out = decode_instances(hm, off, thr=0.3, topk=10, min_dist=3.0,
                           stride=stride)[0]
    assert len(out) == 1
    inst = out[0]

    # center 必须是裁剪像素（= 格坐标 x stride），不是格坐标
    assert inst["center"] == (8 * stride, 8 * stride), (
        f"center 应为裁剪像素 {(8*stride, 8*stride)}，实际 {inst['center']}"
    )
    # corners = offset + 格位置(裁剪像素)
    expect = np.array([[8 * stride + dx, 8 * stride + dy]] * 8)
    assert np.allclose(inst["corners"], expect), (
        f"corners 与 center 坐标系不一致:\n{inst['corners']}\nvs\n{expect}"
    )
