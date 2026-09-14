"""把多个分片渲染出的 BOP 数据集合并成一份。

**为什么需要它**：Windows 上 BlenderProc 的 pyrender 进程池用不了
（multiprocessing 只有 spawn，子进程重新导入 `blenderproc` 时会 `import bpy` 失败），
所以只能用 `BP_NUM_WORKER=0` 在进程内**串行**算掩码 —— 一个 Blender 进程只吃一个核。

绕法是把场景切成 N 份，同时跑 N 个 Blender 进程（每个写自己的数据集目录），
再用本脚本把 N 份合成一份标准 BOP 数据集。见 docs/RENDER_SETUP.md。

用法::

    python scripts/merge_bop_shards.py --out <merged_root> <shard_root_1> <shard_root_2> ...

每个 ``shard_root`` 的结构需为::

    <shard_root>/train_pbr/000000/{rgb,depth,mask,mask_visib}/ + scene_*.json

合并时**逐帧重编号**（帧号连续、实例文件名跟着改），并把三个 json 按帧号拼起来。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

IMAGE_DIRS = ("rgb", "depth", "mask", "mask_visib")
JSON_NAMES = ("scene_gt.json", "scene_camera.json", "scene_gt_info.json")


def _find_chunk(shard_root: str) -> str:
    """在分片里找 ``train_pbr/<chunk>``，返回该 chunk 目录。"""
    split = os.path.join(shard_root, "train_pbr")
    if not os.path.isdir(split):
        raise FileNotFoundError(f"找不到 train_pbr: {split}")
    chunks = sorted(d for d in os.listdir(split) if os.path.isdir(os.path.join(split, d)))
    if not chunks:
        raise FileNotFoundError(f"train_pbr 下没有 chunk: {split}")
    if len(chunks) > 1:
        print(f"[merge] ⚠️ {shard_root} 有多个 chunk {chunks}，只取第一个")
    return os.path.join(split, chunks[0])


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="合并后的数据集根目录")
    ap.add_argument("--models-from", default=None,
                    help="从哪个分片拷 models/ 与 camera.json（默认取第一个分片）")
    ap.add_argument("shards", nargs="+", help="各分片的数据集根目录")
    ap.add_argument("--chunk-name", default="000000")
    ap.add_argument("--clean", action="store_true", help="合并前先删掉输出目录")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    out_chunk = os.path.join(out_root, "train_pbr", args.chunk_name)
    if args.clean and os.path.isdir(out_root):
        shutil.rmtree(out_root)
    for d in IMAGE_DIRS:
        os.makedirs(os.path.join(out_chunk, d), exist_ok=True)

    merged: Dict[str, Dict[str, list]] = {n: {} for n in JSON_NAMES}
    next_frame = 0
    total_inst = 0

    for si, shard in enumerate(args.shards):
        chunk = _find_chunk(shard)
        gt = _load(os.path.join(chunk, "scene_gt.json"))
        gti = _load(os.path.join(chunk, "scene_gt_info.json")) if os.path.isfile(
            os.path.join(chunk, "scene_gt_info.json")) else {}
        cam = _load(os.path.join(chunk, "scene_camera.json")) if os.path.isfile(
            os.path.join(chunk, "scene_camera.json")) else {}

        frames = sorted((int(k) for k in gt.keys()))
        n_inst_this = 0
        for fid in frames:
            new_fid = next_frame
            next_frame += 1
            old = f"{fid:06d}"
            new = f"{new_fid:06d}"

            # 图像：逐帧改名
            for d in ("rgb", "depth"):
                src = os.path.join(chunk, d, f"{old}.png")
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(out_chunk, d, f"{new}.png"))

            # 掩码：<frame>_<inst>.png
            anns = gt[str(fid)]
            n_inst_this += len(anns)
            for i in range(len(anns)):
                for d in ("mask", "mask_visib"):
                    src = os.path.join(chunk, d, f"{old}_{i:06d}.png")
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(out_chunk, d, f"{new}_{i:06d}.png"))

            merged["scene_gt.json"][str(new_fid)] = gt[str(fid)]
            if str(fid) in gti:
                merged["scene_gt_info.json"][str(new_fid)] = gti[str(fid)]
            if str(fid) in cam:
                merged["scene_camera.json"][str(new_fid)] = cam[str(fid)]

        print(f"[merge] 分片 {si} {shard}: {len(frames)} 帧 / {n_inst_this} 实例")
        total_inst += n_inst_this

    for name, data in merged.items():
        if data:
            with open(os.path.join(out_chunk, name), "w", encoding="utf-8") as f:
                json.dump(data, f)

    # models / camera.json
    src_root = args.models_from or args.shards[0]
    for item in ("models", "camera.json"):
        s = os.path.join(src_root, item)
        if os.path.exists(s):
            d = os.path.join(out_root, item)
            if os.path.isdir(s):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    print()
    print("=" * 62)
    print(f"合并完成 -> {out_root}")
    print(f"  帧 = {next_frame}   实例 = {total_inst}")
    print(f"  rgb = {len(os.listdir(os.path.join(out_chunk, 'rgb')))}"
          f"   depth = {len(os.listdir(os.path.join(out_chunk, 'depth')))}"
          f"   mask_visib = {len(os.listdir(os.path.join(out_chunk, 'mask_visib')))}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
