"""量 DataLoader 的真实多进程吞吐，并 A/B 测试两个怀疑点。

背景：v1+GEO7 合并训练（8927 样本）时 8 个 worker 全程 100% CPU，
200 秒内一个 batch 都完不成；栈反复落在
``bop_pbr.py::_load_scene_json`` → ``json.decoder.raw_decode``，
faulthandler 还打出 ``Garbage-collecting``。

⚠️ 两次方法论错误的教训（都记在 docs/TROUBLESHOOTING.md）：
  1. 用 **num_workers=0 顺序读 12 个样本** 测出 0.05 s/样本，据此错误排除了
     数据加载 —— 既没触发多进程、也没触发累积效应。
  2. 改成多进程但**只跑 45 秒** —— 而故障现象是"短跑正常、全量必卡"，
     45 秒量不出累积。**跑得久比跑得多重要。**
所以这个脚本必须：真实 worker 数 + 长时间 + 逐段报告
吞吐 / 子进程 RSS / 子进程累计 CPU，用来抓"随时间崩掉"这类故障。

用法::

    python scripts/diag_throughput.py \
        --root /path/v1 --root /path/GEO7 \
        --workers 8 --seconds 300 [--gc-disable] [--cache-json] [--augment]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.bop_pbr import BOPPBRDataset  # noqa: E402


def patch_json_cache():
    """把 _load_scene_json 换成带缓存的版本（模拟"本该有的"缓存）。"""
    raw = BOPPBRDataset._load_scene_json
    cache: dict = {}

    def cached(self, scene_dir, name):
        key = (scene_dir, name)
        hit = cache.get(key)
        if hit is None:
            hit = raw(self, scene_dir, name)
            cache[key] = hit
        return hit

    BOPPBRDataset._load_scene_json = cached
    BOPPBRDataset._json_cache = cache
    return cache


def _read_proc(pid: int):
    """返回 (rss_MB, cpu_ticks)；读不到返回 None。"""
    try:
        rss = 0
        with open(f"/proc/{pid}/statm") as f:
            rss = int(f.read().split()[1]) * 4096 / 1e6  # pages -> MB
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        cpu = int(parts[13]) + int(parts[14])
        return rss, cpu
    except (OSError, IndexError, ValueError):
        return None


def _children_snapshot():
    """(子进程数, RSS 合计 MB, CPU ticks 合计)。"""
    me = os.getpid()
    n = 0
    rss = 0.0
    cpu = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                parts = f.read().split()
            if int(parts[3]) != me:
                continue
        except (OSError, IndexError, ValueError):
            continue
        got = _read_proc(int(d))
        if got:
            n += 1
            rss += got[0]
            cpu += got[1]
    return n, rss, cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--gc-disable", action="store_true")
    ap.add_argument("--cache-json", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--report-every", type=float, default=15.0)
    args = ap.parse_args()

    if args.cache_json:
        patch_json_cache()
    if args.gc_disable:
        gc.disable()

    ds = BOPPBRDataset(
        dataset_root=list(args.root),
        split="train_pbr",
        image_size=256,
        heatmap_size=64,
        heatmap_style="boxdreamer",
        sigma=2.0,
        crop_scale=1.4,
        use_gt_crop=True,
        augment=args.augment,
        min_px_visib=64,
        min_visib_fract=0.10,
        multi_instance=False,
        obj_mask_ratio=(0.0, 0.15) if args.augment else None,
        obj_paste_prob=0.2 if args.augment else 0.0,
        rgb_augmethods=["mobile"] if args.augment else None,
    )

    from torch.utils.data import DataLoader

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    label = args.label or (
        f"workers={args.workers} aug={args.augment} "
        f"gc_disable={args.gc_disable} cache_json={args.cache_json}"
    )
    print(f"\n=== {label} ===  样本数={len(ds)}  时长上限={args.seconds:.0f}s", flush=True)

    t0 = time.time()
    n_batch = 0
    n_sample = 0
    first_batch_t = None
    last_t = t0
    last_n = 0
    last_kids = _children_snapshot()

    try:
        for batch in dl:
            now = time.time()
            if first_batch_t is None:
                first_batch_t = now - t0
                print(f"  ⏱ 首个 batch: {first_batch_t:.2f}s", flush=True)
            n_batch += 1
            n_sample += int(batch["image"].shape[0])
            el = now - t0
            if el >= args.seconds:
                break
            if now - last_t >= args.report_every:
                dt = now - last_t
                inst = (n_sample - last_n) / dt
                kids = _children_snapshot()
                d_rss = kids[1] - last_kids[1]
                d_cpu = (kids[2] - last_kids[2]) / 100.0  # ticks -> 秒
                print(
                    f"    t={el:6.1f}s  累计 {n_sample:6d} 样本  "
                    f"瞬时 {inst:6.1f} 样本/s  "
                    f"子进程 {kids[0]:2d} 个  RSS {kids[1]:7.0f}MB "
                    f"({d_rss:+6.0f})  子CPU {d_cpu:5.1f}s/{dt:.0f}s",
                    flush=True,
                )
                last_t = now
                last_n = n_sample
                last_kids = kids
    except KeyboardInterrupt:
        pass

    el = time.time() - t0
    print(f"  --- 汇总 ---")
    print(f"  {el:.1f}s 内 {n_batch} batch / {n_sample} 样本")
    print(f"  总吞吐 = {n_sample / el:.1f} 样本/s   batch/s = {n_batch / el:.2f}")
    if first_batch_t is not None:
        print(f"  首个 batch 延迟 = {first_batch_t:.2f}s")
    if args.cache_json:
        print(f"  JSON 缓存条目 = {len(BOPPBRDataset._json_cache)}")
    print(f"  判据：≥30 样本/s 够喂 GPU；<5 样本/s 就是数据加载拖死训练")


if __name__ == "__main__":
    main()
