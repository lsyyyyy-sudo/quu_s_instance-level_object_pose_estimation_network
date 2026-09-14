"""PnP 往返验证：用真实数据的 GT 角点解位姿，必须还原出 GT 位姿。

未训练模型预测的角点是垃圾、PnP 全失败是正常的；但如果**喂 GT 角点**都解不出来，
那就是 K / bbox3d / 角点顺序 / 位姿约定之间有 bug —— 这条不查出来，
训练跑再久也不知道位姿分支是坏的。

    python scripts/verify_pnp.py --dataset-root data/dji_action4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.bop_pbr import BOPPBRDataset  # noqa: E402
from src.models.utils.box_utils import recover_pose_from_bb8  # noqa: E402
from src.models.utils.pose_utils import (  # noqa: E402
    add_error,
    rotation_angle_deg,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/dji_action4")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    ds = BOPPBRDataset(
        dataset_root=args.dataset_root, split="train_pbr", obj_ids=[1],
        image_size=args.image_size, heatmap_size=64, heatmap_style="boxdreamer",
        augment=False, use_gt_crop=True,
    )
    n = min(args.n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).astype(int).tolist()

    c2d = torch.stack([ds[i]["corner_2d"] for i in idxs])       # [n,8,2]
    b3d = torch.stack([ds[i]["bbox_3d"] for i in idxs])         # [n,8,3]
    K = torch.stack([ds[i]["cam_K"] for i in idxs])             # [n,3,3]
    pg = torch.stack([ds[i]["pose_gt"] for i in idxs])          # [n,4,4]

    poses, _ = recover_pose_from_bb8(c2d, b3d, K)
    valid = poses[:, 3, 3].abs() > 1e-8
    print(f"[pnp] 用 GT 角点解位姿：{int(valid.sum())} / {n} 成功")

    if valid.sum() == 0:
        print("✗ 一个都没解出来 —— 说明 K / 角点 / 位姿约定之间有 bug")
        return 1

    pp, pgt, bb = poses[valid], pg[valid], b3d[valid]
    rot = rotation_angle_deg(pp[:, :3, :3], pgt[:, :3, :3])
    trans = torch.linalg.norm(pp[:, :3, 3] - pgt[:, :3, 3], dim=-1)
    adde = add_error(pp, pgt, bb)

    print(f"[pnp] 旋转误差 (deg): median {rot.median():.4f}  max {rot.max():.4f}")
    print(f"[pnp] 平移误差 (mm) : median {trans.median():.4f}  max {trans.max():.4f}")
    print(f"[pnp] ADD    (mm)   : median {adde.median():.4f}  max {adde.max():.4f}")

    ok = rot.max() < 0.5 and trans.max() < 1.0 and adde.max() < 1.0
    print()
    if ok:
        print("✅ PnP 往返通过：GT 角点能精确还原 GT 位姿。"
              "位姿分支本身没问题，之前指标里缺 val/add 是因为模型还没训练。")
        return 0
    print("✗ PnP 往返误差过大，检查 K / bbox3d / 角点顺序 / 位姿约定")
    return 1


if __name__ == "__main__":
    sys.exit(main())
