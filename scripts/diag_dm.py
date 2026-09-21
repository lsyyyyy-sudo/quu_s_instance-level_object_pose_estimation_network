"""用**真实 datamodule** 复现训练时的数据加载，逐个加变量定位卡死点。

已排除（都有实测）：
  · 数据本身：8 worker 跑满一整个 epoch，113.9 样本/s，RSS 平稳无泄漏
  · CUDA：独立进程矩阵乘正常
  · 磁盘：写 1.1 GB/s；图片 1024x768 解码 13 ms
  · 内存：cgroup 上限 128 GB，oom_kill=0
  · 训练进程的主线程栈：卡在 DataLoader._try_get_data -> queue.get -> wait
    （即**主进程在等 worker 出数据**，两次采样 45 秒完全一样）

但独立吞吐测试和真实训练只差两点，本脚本就是把这两点分别打开：

  --subset   用 Subset(train_full, 打乱后切 95%)，和 datamodule 一致
  --cuda     先初始化 CUDA 上下文并把模型放到卡上，再 fork worker
             （fork-after-CUDA-init 是经典的多进程坑）

用法::

    python scripts/diag_dm.py --root A --root B --workers 8 --seconds 60 \\
        [--subset] [--cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--subset", action="store_true",
                    help="用 Subset 打乱后切 95%（和 datamodule 一致）")
    ap.add_argument("--cuda", action="store_true",
                    help="fork worker 之前先初始化 CUDA")
    ap.add_argument("--report-every", type=float, default=10.0)
    args = ap.parse_args()

    # ---- 关键顺序：CUDA 先初始化（模拟真实训练：模型已上卡才 fork）----
    if args.cuda:
        import torch

        print("  初始化 CUDA ...", flush=True)
        a = torch.randn(2000, 2000, device="cuda")
        b = (a @ a).sum().item()
        print(f"  CUDA 就绪（sum={b:.0f}）", flush=True)

    from src.datamodules.corner_pose_datamodule import CornerPoseDataModule

    dm = CornerPoseDataModule(
        dataset_root=list(args.root),
        train_split="train_pbr",
        val_split="train_pbr",
        val_ratio=0.05,
        obj_ids=(1,),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        image_size=256,
        heatmap_size=64,
        heatmap_style="boxdreamer",
        sigma=2.0,
        crop_scale=1.4,
        use_gt_crop=True,
        augment=args.augment,
        max_train_samples=None,
        max_val_samples=512,
        shuffle_train=True,
        seed=42,
    )
    dm.setup()

    inner = dm.data_train
    print(f"\n=== subset={args.subset} cuda={args.cuda} workers={args.workers} ===",
          flush=True)
    print(f"  data_train 类型={type(inner).__name__}  len={len(inner)}", flush=True)

    if not args.subset and hasattr(inner, "dataset"):
        # 退回直接迭代原始数据集（去掉 Subset 这一层）
        print("  -> 用原始数据集替换 Subset", flush=True)
        dm.data_train = inner.dataset

    dl = dm.train_dataloader()
    t0 = time.time()
    n_batch = n_sample = 0
    first_t = None
    last_t, last_n = t0, 0
    try:
        for batch in dl:
            now = time.time()
            if first_t is None:
                first_t = now - t0
                print(f"  ⏱ 首个 batch: {first_t:.2f}s", flush=True)
            n_batch += 1
            n_sample += int(batch["image"].shape[0])
            el = now - t0
            if el >= args.seconds:
                break
            if now - last_t >= args.report_every:
                dt = now - last_t
                print(f"    t={el:6.1f}s 累计 {n_sample:6d}  "
                      f"瞬时 {(n_sample - last_n) / dt:6.1f} 样本/s", flush=True)
                last_t, last_n = now, n_sample
    except KeyboardInterrupt:
        pass

    el = time.time() - t0
    print(f"  --- 汇总 ---")
    print(f"  {el:.1f}s  {n_batch} batch / {n_sample} 样本  "
          f"= {n_sample / el:.1f} 样本/s")
    if first_t is not None:
        print(f"  首个 batch 延迟 = {first_t:.2f}s")
    print(f"  判定：{'✅ 数据侧正常' if n_sample / el > 30 else '❌ 数据侧有问题'}")


if __name__ == "__main__":
    main()
