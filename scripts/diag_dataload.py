"""定位 dataloader 卡死：逐根、逐个样本计时。

背景：v1 + GEO7 合并训练时 8 个 worker 全部 100% CPU 空转、GPU 空闲、
日志零增长，但主进程只是 futex 等待 —— 说明卡在 __getitem__ 内部。
容器禁 ptrace，py-spy 用不了，所以用 faulthandler 在进程内取栈。

用法::

    python scripts/diag_dataload.py \
        --root /root/autodl-tmp/bop/v1/dji_action4_hybrid \
        --root /root/autodl-tmp/bop/v3/GEO7/dji_action4_hybrid \
        --n 20 --hang 60
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.bop_pbr import BOPPBRDataset  # noqa: E402


def build(roots, split="train_pbr", augment=False, obj_mask_ratio=None,
          obj_paste_prob=0.0, rgb_augmethods=None):
    return BOPPBRDataset(
        dataset_root=list(roots),
        split=split,
        image_size=256,
        heatmap_size=64,
        heatmap_style="boxdreamer",
        sigma=2.0,
        crop_scale=1.4,
        use_gt_crop=True,
        augment=augment,
        min_px_visib=64,
        min_visib_fract=0.10,
        multi_instance=False,
        obj_mask_ratio=obj_mask_ratio,
        obj_paste_prob=obj_paste_prob,
        rgb_augmethods=rgb_augmethods,
    )


def probe(label, roots, n, hang, **build_kw):
    print(f"\n{'=' * 68}\n[{label}] roots={roots}\n  {build_kw}\n{'=' * 68}", flush=True)
    t0 = time.time()
    ds = build(roots, **build_kw)
    print(f"  建索引耗时 {time.time() - t0:.1f}s，样本数 {len(ds)}", flush=True)

    times = []
    for i in range(min(n, len(ds))):
        # 每个样本前武装 faulthandler：卡住超过 hang 秒就把栈打出来并退出
        faulthandler.dump_traceback_later(hang, exit=True)
        t = time.time()
        try:
            item = ds[i]
        finally:
            faulthandler.cancel_dump_traceback_later()
        dt = time.time() - t
        times.append(dt)
        shape = tuple(item["image"].shape) if hasattr(item, "get") and "image" in item else "?"
        print(f"  [{i:>4}] {dt:7.3f}s  image={shape}  {ds.samples[i][0]}"
              f"  frame={ds.samples[i][1]} gt={ds.samples[i][2]}", flush=True)

    if times:
        times_sorted = sorted(times)
        print(f"  --- 中位 {times_sorted[len(times) // 2]:.3f}s  "
              f"最慢 {max(times):.3f}s  总 {sum(times):.1f}s", flush=True)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--hang", type=float, default=60.0,
                    help="单样本超过这么多秒就打印栈并退出")
    ap.add_argument("--split", default="train_pbr")
    ap.add_argument("--augment", action="store_true",
                    help="开启增强（训练时的设置）")
    ap.add_argument("--obj-mask-ratio", type=float, nargs=2, default=None)
    ap.add_argument("--obj-paste-prob", type=float, default=0.0)
    ap.add_argument("--rgb-augmethods", nargs="*", default=None)
    args = ap.parse_args()

    kw = dict(
        augment=args.augment,
        obj_mask_ratio=tuple(args.obj_mask_ratio) if args.obj_mask_ratio else None,
        obj_paste_prob=args.obj_paste_prob,
        rgb_augmethods=args.rgb_augmethods,
    )

    probe("多根合并", args.root, args.n, args.hang, **kw)


if __name__ == "__main__":
    main()
