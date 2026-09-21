"""量化「单实例裁剪任务」到底有多病态。

背景
----
单实例架构的隐含假设是：**裁剪图里那个目标是可以被认出来的**。
但我们的渲染脚本是 `idx_l = np.random.choice(models_ids, size=num_objs, replace=True)`，
而 `models_info.json` 里只有 `obj 1` —— **每一帧的每一个物体都是同一款 DJI**。
v1 每帧 10 个、GEO7 每帧 6 个，**一个异类物体都没有**。

于是"哪台是目标"在训练里**只能靠位置回答**（框居中的那个）。
这个脚本就是要量出这个捷径有多不可靠。

对每条样本（= 一个目标实例）算三件事：

1. ``n_others_in_crop``  1.4× GT 框内还有几个**其他**实例（裁剪污染）
2. ``is_nearest``        目标是不是**离裁剪中心最近**的那个实例
                         —— 即"选中间那个"这个捷径**答对了吗**
3. ``rank``              目标按"到裁剪中心距离"排第几（1 = 最近）

⚠️ **不 import torch**：无卡模式 cgroup 内存上限 2 GB，import torch 就要 ~0.5 GB。
本脚本只用 numpy + json，可以在 1 核 / 2 GB 下跑。

用法::

    python3 scripts/analyze_instance_ambiguity.py --root /path/v1 --root /path/GEO7
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CROP_SCALE = 1.4          # 与 configs/datamodule/bop.yaml 一致
MIN_VISIB = 0.10          # 与 min_visib_fract 一致（训练用的过滤门槛）


def bbox3d_from_models_info(mi: dict) -> np.ndarray:
    """[8, 3] 的物体系角点（顺序无关，本脚本只用它们的投影包络）。"""
    mn = np.array([mi["min_x"], mi["min_y"], mi["min_z"]], dtype=np.float64)
    mx = np.array([mi["max_x"], mi["max_y"], mi["max_z"]], dtype=np.float64)
    return np.array(
        [[mn[0] if bx == 0 else mx[0],
          mn[1] if by == 0 else mx[1],
          mn[2] if bz == 0 else mx[2]]
         for bx in (0, 1) for by in (0, 1) for bz in (0, 1)],
        dtype=np.float64,
    )


def project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """[N,3] 相机坐标 -> [N,2] 像素。"""
    z = np.clip(pts_cam[:, 2], 1e-6, None)
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def crop_window(bbox_xywh, scale=CROP_SCALE):
    """复刻 src/datasets/bop_pbr.py::crop_and_resize 的裁剪窗口。"""
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(max(w, h) * scale, 1.0)
    return cx, cy, side, cx - side / 2.0, cy - side / 2.0


def analyze_root(root: str, split: str = "train_pbr",
                 jitter: float = 0.0, trials: int = 12, seed: int = 0):
    """jitter: 裁剪中心相对 GT 框中心的随机偏移，以「裁剪边长」为单位。
    用来模拟**真实检测器给出的框不居中** —— 那才是部署时的情形。
    """
    rng = np.random.default_rng(seed)
    mi_path = os.path.join(root, "models", "models_info.json")
    if not os.path.isfile(mi_path):
        return None
    models_info = json.load(open(mi_path, encoding="utf-8"))
    bbox3d = {int(k): bbox3d_from_models_info(v) for k, v in models_info.items()}
    target_ids = set(bbox3d.keys())

    split_dir = os.path.join(root, split)
    n_frames = n_targets = 0
    inst_per_frame = []
    n_others = []
    is_nearest = []
    ranks = []
    visibs = []
    n_inst_in_crop = []
    jit_ok = []

    for scene in sorted(os.listdir(split_dir)):
        scene_dir = os.path.join(split_dir, scene)
        p_gt = os.path.join(scene_dir, "scene_gt.json")
        p_cam = os.path.join(scene_dir, "scene_camera.json")
        p_info = os.path.join(scene_dir, "scene_gt_info.json")
        if not (os.path.isfile(p_gt) and os.path.isfile(p_cam)):
            continue
        gt = json.load(open(p_gt, encoding="utf-8"))
        cam = json.load(open(p_cam, encoding="utf-8"))
        info = json.load(open(p_info, encoding="utf-8")) if os.path.isfile(p_info) else {}

        for fid, anns in gt.items():
            if fid not in cam:
                continue
            K = np.array(cam[fid]["cam_K"], dtype=np.float64).reshape(3, 3)
            # 只保留目标类别的实例（与数据集的 obj_ids 过滤一致）
            keep = [i for i, a in enumerate(anns) if int(a["obj_id"]) in target_ids]
            if len(keep) < 2:
                continue
            n_frames += 1
            inst_per_frame.append(len(keep))

            # 投影所有实例 -> 2D 包络框 + 中心
            boxes, centers = [], []
            for i in keep:
                a = anns[i]
                R = np.array(a["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                t = np.array(a["cam_t_m2c"], dtype=np.float64).reshape(3)
                pc = (R @ bbox3d[int(a["obj_id"])].T).T + t
                uv = project(pc, K)
                boxes.append([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
                centers.append(uv.mean(axis=0))
            boxes = np.array(boxes)
            centers = np.array(centers)

            frame_info = info.get(fid, [])
            for m, i in enumerate(keep):
                # 目标框：与训练一致，优先 bbox_visib
                bb = None
                if m < len(frame_info):
                    fi = frame_info[m]
                    if "bbox_visib" in fi:
                        bb = fi["bbox_visib"]
                    vf = float(fi.get("visib_fract", 1.0))
                else:
                    vf = 1.0
                if bb is None or bb[2] <= 1 or bb[3] <= 1:
                    bx0, by0, bx1, by1 = boxes[m]
                    bb = [bx0, by0, bx1 - bx0, by1 - by0]
                if vf < MIN_VISIB:
                    continue
                n_targets += 1
                visibs.append(vf)

                cx, cy, side, x0, y0 = crop_window(bb)
                # 其他实例：框与裁剪窗口相交就算"被裁进来"
                ox0, oy0 = x0, y0
                ox1, oy1 = x0 + side, y0 + side
                inter = (boxes[:, 0] < ox1) & (boxes[:, 2] > ox0) & \
                        (boxes[:, 1] < oy1) & (boxes[:, 3] > oy0)
                # 目标中心在裁剪窗口内、且尺寸合理的实例才算"混进来了"
                cin = (centers[:, 0] > ox0) & (centers[:, 0] < ox1) & \
                      (centers[:, 1] > oy0) & (centers[:, 1] < oy1) & \
                      ((boxes[:, 2] - boxes[:, 0]) > 4) & ((boxes[:, 3] - boxes[:, 1]) > 4)
                others = np.ones(len(keep), dtype=bool)
                others[m] = False
                n_others.append(int((inter & others).sum()))
                n_inst_in_crop.append(int((cin & others).sum()))

                # 到裁剪中心（= 框中心）的距离；只看"混进来"的其他实例
                d = np.linalg.norm(centers - np.array([cx, cy]), axis=1)
                cand = others & inter
                if cand.sum() == 0:
                    is_nearest.append(True)
                    ranks.append(1)
                else:
                    is_nearest.append(bool(d[m] < d[cand].min()))
                    ranks.append(int((d[cand] < d[m]).sum()) + 1)

                # ---- 框不居中时捷径还成立吗（模拟检测器框的偏移）----
                if jitter > 0.0:
                    off = rng.uniform(-jitter / 2.0, jitter / 2.0, size=(trials, 2)) * side
                    cc = np.array([cx, cy]) + off                 # [T,2]
                    dd = np.linalg.norm(centers[None, :, :] - cc[:, None, :], axis=2)  # [T,N]
                    if cand.sum() == 0:
                        ok = np.ones(trials, dtype=bool)
                    else:
                        ok = dd[:, m] < dd[:, cand].min(axis=1)
                    jit_ok.extend(ok.tolist())

    if not n_targets:
        return None

    def pct(a, q):
        return float(np.percentile(np.asarray(a, dtype=float), q))

    return {
        "root": root,
        "frames": n_frames,
        "targets": n_targets,
        "inst_per_frame_median": float(np.median(inst_per_frame)),
        "n_others_mean": float(np.mean(n_others)),
        "n_others_median": pct(n_others, 50),
        "n_others_max": int(np.max(n_others)),
        "frac_with_others": float(np.mean(np.asarray(n_others) > 0)),
        "frac_with_2plus": float(np.mean(np.asarray(n_others) >= 2)),
        "frac_with_3plus": float(np.mean(np.asarray(n_others) >= 3)),
        "centers_in_crop_mean": float(np.mean(n_inst_in_crop)),
        "frac_nearest": float(np.mean(is_nearest)),
        "frac_NOT_nearest": float(1.0 - np.mean(is_nearest)),
        "rank_median": pct(ranks, 50),
        "rank_p90": pct(ranks, 90),
        "rank_max": int(np.max(ranks)),
        "visib_median": pct(visibs, 50),
        "jitter": jitter,
        "jitter_ok": float(np.mean(jit_ok)) if jit_ok else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="把结果 JSON 写到这里")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="裁剪中心随机偏移（以裁剪边长为单位），模拟检测器框不居中")
    ap.add_argument("--trials", type=int, default=12)
    args = ap.parse_args()

    names = args.names or [os.path.basename(os.path.dirname(r.rstrip("/"))) for r in args.root]
    results = []
    for nm, r in zip(names, args.root):
        print(f"\n{'=' * 70}\n  {nm}   {r}\n{'=' * 70}", flush=True)
        res = analyze_root(r, jitter=args.jitter, trials=args.trials)
        if res is None:
            print("  跳过（没有 models_info.json 或样本不足）", flush=True)
            continue
        res["name"] = nm
        results.append(res)
        print(f"  帧 {res['frames']}   目标实例 {res['targets']}"
              f"   每帧实例中位 {res['inst_per_frame_median']:.0f}"
              f"   visib 中位 {res['visib_median']:.3f}")
        print(f"  ── 裁剪污染（1.4× GT 框内，还有其他同类实例）")
        print(f"     平均 {res['n_others_mean']:.1f} 个   中位 {res['n_others_median']:.0f}"
              f"   最多 {res['n_others_max']} 个")
        print(f"     含 ≥1 个的裁剪占比  {res['frac_with_others'] * 100:6.2f}%")
        print(f"     含 ≥2 个的裁剪占比  {res['frac_with_2plus'] * 100:6.2f}%")
        print(f"     含 ≥3 个的裁剪占比  {res['frac_with_3plus'] * 100:6.2f}%")
        print(f"     （中心落在裁剪内的其他实例平均 {res['centers_in_crop_mean']:.2f} 个）")
        print(f"  ── ⭐ 位置捷径「选离裁剪中心最近的那台」")
        print(f"     答对 {res['frac_nearest'] * 100:6.2f}%"
              f"     ❌ 答错 {res['frac_NOT_nearest'] * 100:6.2f}%")
        print(f"     目标按中心距离的排名：中位 {res['rank_median']:.0f}"
              f"  p90 {res['rank_p90']:.0f}  最差 {res['rank_max']}")
        if res.get("jitter_ok") is not None:
            print(f"  ── 框不居中时（偏移 ±{res['jitter'] * 100:.0f}% 边长，模拟检测器框）")
            print(f"     捷径仍然答对 {res['jitter_ok'] * 100:6.2f}%"
                  f"     ❌ 答错 {(1 - res['jitter_ok']) * 100:6.2f}%")

    if args.out and results:
        json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2)
        print(f"\n结果已写入 {args.out}")

    if len(results) >= 2:
        print(f"\n{'=' * 70}\n  汇总\n{'=' * 70}")
        print(f"  {'数据集':<10} {'目标数':>7} {'含其他实例':>10} {'捷径答错':>9} {'框偏移后答错':>12}")
        for r in results:
            j = f"{(1 - r['jitter_ok']) * 100:>11.2f}%" if r.get("jitter_ok") is not None else " " * 12
            print(f"  {r['name']:<10} {r['targets']:>7} "
                  f"{r['frac_with_others'] * 100:>9.2f}% {r['frac_NOT_nearest'] * 100:>8.2f}% {j}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
