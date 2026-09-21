"""端到端自洽测试：用【真实数据集样本】验证整条链路。

为什么需要它
------------
静态读代码抓不到轴/转置类错误。这个测试构造"完美预测"：
  center_heatmap 用数据集的 GT（峰值 1.0）
  offset 用 GT 角点减去所在格子的位置
然后验证：
  ① 中心 focal loss 应该很小（说明热图头的位置对）
  ② 角点回归 loss 应该 ~0（说明 offset 的【坐标系和轴顺序】对）
  ③ decode 出来的实例数 = GT 实例数（说明解码链路对）
  ④ decode 出来的角点 = GT 角点（说明 stride 和中心坐标对）

**任何一处转置/轴错位都会让 ② 或 ④ 失败。**
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loss.utils.multi_instance_loss import (  # noqa: E402
    center_focal_loss,
    corner_regression_loss,
)
from src.models.utils.multi_instance import decode_instances  # noqa: E402

DSET = os.environ.get("MI_TEST_DATASET", r"D:\AAA_Projects\psd_zju3dv_coding_exam\data\dji_action4_hybrid")
IMG, HM = 256, 64
STRIDE = IMG / float(HM)


pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DSET, "train_pbr")),
    reason="本地没有数据集（远端才有），跳过端到端测试",
)


def _real_batch(n=4):
    from src.datasets.bop_pbr import BOPPBRDataset
    ds = BOPPBRDataset(dataset_root=DSET, split="train_pbr", obj_ids=[1],
                       image_size=IMG, heatmap_size=HM, augment=False,
                       multi_instance=True, multi_crop_scale=2.5, max_instances=8)
    items = [ds[i] for i in range(min(n, len(ds)))]
    batch = {}
    for k in items[0]:
        v = items[0][k]
        if torch.is_tensor(v):
            batch[k] = torch.stack([it[k] for it in items])
    return batch


def _perfect_offset(batch):
    """把 GT 角点写进 offset，构造"完美预测"。"""
    B, N = batch["inst_valid"].shape
    off = torch.zeros(B, 16, HM, HM)
    for b in range(B):
        for i in range(N):
            if float(batch["inst_valid"][b, i]) <= 0:
                continue
            cx, cy = batch["inst_centers"][b, i].tolist()
            hx = int(round(cx / STRIDE)); hy = int(round(cy / STRIDE))
            hx = min(max(hx, 0), HM - 1); hy = min(max(hy, 0), HM - 1)
            base = torch.tensor([hx * STRIDE, hy * STRIDE])
            d = batch["inst_corners"][b, i].reshape(8, 2) - base
            off[b, :, hy, hx] = d.reshape(-1)
    return off


def test_gt_heatmap_has_one_peak_per_instance():
    """数据集的热图必须每个实例一个精确的峰。"""
    b = _real_batch(4)
    gt = b["center_heatmap"]
    n_inst = int(b["inst_valid"].sum())
    n_peak = int((gt >= 1.0 - 1e-6).sum())
    assert n_peak == n_inst, (
        f"GT 热图有 {n_peak} 个精确峰，但有 {n_inst} 个实例 —— "
        "focal 会匹配不到正样本（ALGO-08）"
    )


def test_loss_is_near_zero_for_gt():
    """用 GT 构造完美预测 -> 两项损失都应该很小。"""
    b = _real_batch(4)
    gt_hm = b["center_heatmap"]
    logits = torch.log(gt_hm.clamp(1e-4, 1 - 1e-4)) - torch.log1p(-gt_hm.clamp(1e-4, 1 - 1e-4))
    lc = float(center_focal_loss(logits, gt_hm))
    off = _perfect_offset(b)
    lk = float(corner_regression_loss(off, b["inst_corners"], b["inst_centers"],
                                      b["inst_valid"], IMG, HM))
    print(f"\n    center focal = {lc:.4f}   corner = {lk:.6f}")
    assert lk < 1e-3, f"完美 offset 的角点损失应当 ~0，实际 {lk:.6f}（坐标系/轴顺序有错）"
    assert lc < 5.0, f"完美热图的 focal 过大 {lc:.4f}"


def test_decode_recovers_gt_on_real_samples():
    """端到端：完美预测 -> decode 必须还原出所有 GT 实例。"""
    b = _real_batch(4)
    gt_hm = b["center_heatmap"]
    logits = torch.log(gt_hm.clamp(1e-4, 1 - 1e-4)) - torch.log1p(-gt_hm.clamp(1e-4, 1 - 1e-4))
    off = _perfect_offset(b)

    out = decode_instances(logits, off, thr=0.5, topk=20, min_dist=3.0, stride=STRIDE)
    total_gt = int(b["inst_valid"].sum())
    total_pred = sum(len(o) for o in out)
    print(f"\n    GT 实例 {total_gt}   decode 出 {total_pred}")
    assert total_pred == total_gt, (
        f"完美输入下 decode 应当还原 {total_gt} 个实例，实际 {total_pred} 个"
    )

    # 逐个验证角点
    worst = 0.0
    for bidx, insts in enumerate(out):
        for inst in insts:
            pc = torch.tensor(inst["corners"])
            # 找最近的 GT
            best = 1e9
            for i in range(b["inst_valid"].shape[1]):
                if float(b["inst_valid"][bidx, i]) <= 0:
                    continue
                d = float(torch.linalg.norm(pc - b["inst_corners"][bidx, i], dim=-1).mean())
                best = min(best, d)
            worst = max(worst, best)
    print(f"    与最近 GT 的最大平均角点距离 {worst:.4f} px")
    assert worst < 0.5, (
        f"decode 的角点与 GT 差 {worst:.3f} px —— stride / 中心坐标 / 轴顺序有错"
    )
