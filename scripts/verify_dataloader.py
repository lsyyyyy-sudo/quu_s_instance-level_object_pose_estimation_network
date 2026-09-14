"""验证 BOP PBR dataloader：这是"数据"和"网络"之间的那道检查。

不只是看形状对不对，重点是**验证标签自洽**：

  · ``corner_2d`` 是不是 8 个 3D 角点用 GT 位姿投影到裁剪图上的结果
  · ``heatmap`` 的峰值位置 × (image_size/heatmap_size) 是不是落在 ``corner_2d`` 上
  · 裁剪后的 ``cam_K`` 是否仍然和裁剪图匹配（再投影一次应该重合）

用法::

    python scripts/verify_dataloader.py --dataset-root data/dji_action4 \
        --out data/render_ws/out/dataloader_check.png --n 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.bop_pbr import BOPPBRDataset  # noqa: E402
from src.models.utils.box_utils import project_points  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/dji_action4")
    ap.add_argument("--out", default="data/render_ws/out/dataloader_check.png")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--heatmap-size", type=int, default=64)
    ap.add_argument("--heatmap-style", default="boxdreamer")
    args = ap.parse_args()

    ds = BOPPBRDataset(
        dataset_root=args.dataset_root,
        split="train_pbr",
        obj_ids=[1],
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_style=args.heatmap_style,
        augment=False,
        use_gt_crop=True,
    )
    print(f"[ds] 样本数 = {len(ds)}   (500 帧 x 10 实例 = 5000)")

    step = max(1, len(ds) // args.n)
    idxs = list(range(0, len(ds), step))[: args.n]

    stride = args.image_size / float(args.heatmap_size)
    print(f"[ds] image_size={args.image_size} heatmap_size={args.heatmap_size} "
          f"stride={stride} style={args.heatmap_style}")

    problems = []
    panels = []
    for k, i in enumerate(idxs):
        s = ds[i]
        img = s["image"]              # [3,H,W]
        K = s["cam_K"]                # [3,3]
        b3d = s["bbox_3d"]            # [8,3]
        c2d = s["corner_2d"]          # [8,2]
        hm = s["heatmap"]             # [8,h,w]
        pose = s["pose_gt"]           # [4,4]

        H = W = args.image_size
        hh, hw = hm.shape[-2:]

        # ---- 形状/范围 ----
        assert img.shape == (3, H, W), img.shape
        assert K.shape == (3, 3), K.shape
        assert b3d.shape == (8, 3), b3d.shape
        assert c2d.shape == (8, 2), c2d.shape
        assert hm.shape == (8, hh, hw), hm.shape
        assert pose.shape == (4, 4), pose.shape
        if not (0.0 <= img.min() and img.max() <= 1.0):
            problems.append(f"#{i} image 范围 [{img.min():.3f},{img.max():.3f}] 超出 [0,1]")

        # ---- 核心：自己按裁剪后的 K 重新投影一次，必须和 corner_2d 完全一致 ----
        reproj = project_points(b3d.unsqueeze(0), K.unsqueeze(0), pose.unsqueeze(0))[0]
        err = (reproj - c2d).abs().max().item()
        if err > 1e-3:
            problems.append(f"#{i} 重新投影与 corner_2d 不一致，max err = {err:.5f}")

        # ---- 热图峰值位置 vs 角点位置 ----
        # GT 的构造是 centers = corner_2d * ratio，热图在整数网格上取值，
        # 所以峰值格 = round(centers)。直接比**格子索引**，避免自己引入重建偏移。
        ratio = hw / float(W)
        expected_cells = np.rint(c2d.numpy() * ratio).astype(np.int64)
        peak_cells = np.array(
            [[int(torch.argmax(hm[c]).item()) % hw,
              int(torch.argmax(hm[c]).item()) // hw] for c in range(8)],
            dtype=np.int64,
        )
        cell_err = np.abs(peak_cells - expected_cells).max()
        peak_err_px = np.linalg.norm((peak_cells - expected_cells) * stride, axis=1)
        if cell_err > 1:
            problems.append(f"#{i} 热图峰值格子偏离 {cell_err} 格（>1），"
                            f"corner_2d={c2d.numpy().round(1).tolist()} "
                            f"peaks={peak_cells.tolist()}")

        inb = ((c2d[:, 0] >= 0) & (c2d[:, 0] < W) & (c2d[:, 1] >= 0) & (c2d[:, 1] < H))
        print(f"  #{i:5d} frame={int(s['frame_id']):3d}  "
              f"corner_2d 在界内 {int(inb.sum())}/8  "
              f"峰值格子误差 max {cell_err} 格 ({peak_err_px.max():.2f}px)  "
              f"热图范围 [{hm.min():.3f},{hm.max():.3f}]")

        peaks = peak_cells * stride          # 只用于可视化打点

        # ---- 拼一张可视：裁剪图 + 8 角点 + 热图最大投影 ----
        rgb = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        pim = Image.fromarray(rgb)
        d = ImageDraw.Draw(pim)
        for c in range(8):
            u, v = c2d[c].tolist()
            d.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(255, 255, 0), width=2)
            d.text((u + 6, v - 6), str(c + 1), fill=(255, 255, 0))
        # 峰位置画红点，应该落在黄圈里
        for c in range(8):
            u, v = peaks[c]
            d.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 60, 60))

        hm_np = hm.numpy()
        hm_vis = hm_np.max(axis=0)
        hm_vis = (hm_vis - hm_vis.min()) / max(hm_vis.max() - hm_vis.min(), 1e-6)
        hm_vis = (hm_vis * 255).astype(np.uint8)
        hm_vis = np.stack([hm_vis] * 3, axis=-1)
        hm_vis = np.asarray(Image.fromarray(hm_vis).resize((H, W), Image.NEAREST))

        panel = Image.new("RGB", (W * 2 + 12, H + 22), (20, 20, 24))
        dd = ImageDraw.Draw(panel)
        dd.text((2, 3), f"#{i}  crop + 8 corners (yellow) / heatmap peaks (red)", fill=(235, 235, 235))
        dd.text((W + 14, 3), f"heatmap max-proj  [{hm_np.min():.2f}, {hm_np.max():.2f}]", fill=(235, 235, 235))
        panel.paste(pim, (0, 20))
        panel.paste(Image.fromarray(hm_vis), (W + 12, 20))
        panels.append(panel)

    cols = 2
    rows = (len(panels) + cols - 1) // cols
    pw, ph = panels[0].size
    sheet = Image.new("RGB", (cols * pw + (cols + 1) * 8, rows * ph + (rows + 1) * 8), (16, 16, 18))
    for k, p in enumerate(panels):
        r, c = divmod(k, cols)
        sheet.paste(p, (8 + c * (pw + 8), 8 + r * (ph + 8)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\nwrote {out}  {sheet.size}")

    print("\n" + "=" * 60)
    if problems:
        print("发现问题：")
        for p in problems:
            print("  ✗", p)
        return 1
    print("✅ 全部检查通过：dataloader 的标签自洽（重投影一致、热图峰对准角点）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
