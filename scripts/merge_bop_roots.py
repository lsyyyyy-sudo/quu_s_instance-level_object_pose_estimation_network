"""把多个 BOP 数据集根合并成一个（默认用符号链接，不复制数据）。

动机：多路径（``dataset_root=[v1, GEO7]``）虽然能用，但每条样本的
``scene_dir`` 指向不同根，路径拼装、``models/`` 查找、缓存都要特判。
合并成一个根之后训练侧就是**普通单根数据集**，没有任何特例。

布局（输出根 ``--out``）::

    <out>/
    ├── models -> <第一个含 models/models_info.json 的根>/models
    └── train_pbr/
        ├── v1_000000 -> <root1>/train_pbr/000000
        ├── v1_000001 -> ...
        ├── g7_000000 -> <root2>/train_pbr/000000
        └── ...

场景名会加 ``前缀_`` 前缀 —— 各根的 BOP 场景名都是从 ``000000`` 开始的，
**不加前缀必然重名覆盖**。

BOPPBRDataset 只做 ``os.listdir(split_dir)`` + ``os.path.join``，
``os.path.isdir`` / ``isfile`` 都会跟随符号链接，所以链接布局透明可用。

用法::

    python scripts/merge_bop_roots.py \\
        --root /data/v1/dji_action4_hybrid --prefix v1 \\
        --root /data/GEO7/dji_action4_hybrid --prefix g7 \\
        --out /data/merged

    # 需要真实文件（例如要打包上传）时：
    python scripts/merge_bop_roots.py ... --out /data/merged --copy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def find_models_root(roots: list[str]) -> str | None:
    for r in roots:
        if os.path.isfile(os.path.join(r, "models", "models_info.json")):
            return r
    return None


def link_or_copy(src: str, dst: str, copy: bool) -> None:
    if os.path.lexists(dst):
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    if copy:
        shutil.copytree(src, dst, symlinks=False)
    else:
        os.symlink(os.path.abspath(src), dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True,
                    help="输入 BOP 根，可重复")
    ap.add_argument("--prefix", action="append", required=True,
                    help="与 --root 一一对应的场景名前缀")
    ap.add_argument("--out", required=True, help="输出 BOP 根")
    ap.add_argument("--split", default="train_pbr")
    ap.add_argument("--copy", action="store_true",
                    help="真复制而不是建符号链接（占磁盘）")
    args = ap.parse_args()

    if len(args.root) != len(args.prefix):
        print("错误：--root 与 --prefix 数量必须一致", file=sys.stderr)
        return 2

    print(f"合并 {len(args.root)} 个根 -> {args.out}   模式={'复制' if args.copy else '符号链接'}")
    models_root = find_models_root(args.root)
    if models_root is None:
        print("错误：没有任何根含 models/models_info.json", file=sys.stderr)
        return 2
    print(f"  models/ 取自: {models_root}")

    out_split = os.path.join(args.out, args.split)
    os.makedirs(out_split, exist_ok=True)
    link_or_copy(os.path.join(models_root, "models"),
                 os.path.join(args.out, "models"), args.copy)

    total_scenes = 0
    total_instances = 0
    for root, prefix in zip(args.root, args.prefix):
        split_dir = os.path.join(root, args.split)
        if not os.path.isdir(split_dir):
            print(f"  跳过 {root}：没有 {args.split}/", file=sys.stderr)
            continue
        scenes = sorted(os.listdir(split_dir))
        n_inst = 0
        for scene in scenes:
            src = os.path.join(split_dir, scene)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(out_split, f"{prefix}_{scene}")
            link_or_copy(src, dst, args.copy)
            gt = os.path.join(dst, "scene_gt.json")
            if os.path.isfile(gt):
                with open(gt, "r", encoding="utf-8") as f:
                    n_inst += sum(len(v) for v in json.load(f).values())
        total_scenes += len(scenes)
        total_instances += n_inst
        print(f"  {prefix}: {len(scenes)} 场景, {n_inst} 个实例  ({root})")

    # ---- 校验：合并后必须能读通 ----
    have = sorted(os.listdir(out_split))
    print(f"\n结果：{args.out}")
    print(f"  {args.split}/ 下 {len(have)} 个场景（预期 {total_scenes}）")
    print(f"  实例总数（scene_gt.json 统计）{total_instances}")
    bad = []
    for s in have:
        if not os.path.isfile(os.path.join(out_split, s, "scene_gt.json")):
            bad.append(s)
    if bad:
        print(f"  ⚠️ {len(bad)} 个场景读不到 scene_gt.json（前 5 个：{bad[:5]}）")
        return 1
    print("  ✅ 每个场景的 scene_gt.json 都可读")
    mi = os.path.join(args.out, "models", "models_info.json")
    print(f"  {'✅' if os.path.isfile(mi) else '❌'} models/models_info.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
