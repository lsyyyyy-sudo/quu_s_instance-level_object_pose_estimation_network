"""往渲染场景里加 Objaverse 物体当干扰物。

为什么需要（docs/RESULTS.md §3.6 的延伸）
---------------------------------------
我们原来的场景是 **12 个同款相机摆在纯色平面上** —— 太单调。真实视频里是
**手、牛仔布、网格垫、白桌**，还有别的杂物。

Objaverse 的作用有三层：
  1. **场景真实感** —— 有别的物体，不像"商品摆拍"
  2. **自然的遮挡** —— 别的物体会挡住我们的 DJI。**这比料箱堆一堆同款相机真实得多**
     （料箱那个方案失败了：它造出大量"裁剪图里根本没有目标"的无效样本，见 DATA-19）
  3. **形状多样性** —— 逼模型不要把"周围总是同款相机"当成线索

物体从哪来
---------
Objaverse-LVIS 子集，按 **"桌面级"类别**筛（camera/bottle/cup/book/tool/…，192 个类别），
随机抽 300 个下载、保留 287 个 ≤25MB 的（合计 1.08 GB）。
清单存在 ``BP_OBJAVERSE_KEEP`` 指向的 json 里（uid -> glb 路径）。

⚠️ 与 screen_planes.py 相同的两个约束
------------------------------------
1. **不能用 `src.*` 包导入** —— BlenderProc 跑在 Blender 内嵌 Python 里，没有 torch。
   本文件因此必须按【文件路径】被加载。
2. Objaverse 的 GLB **尺度任意**（有的 1 cm、有的 10 m），必须**归一化**到
   跟目标物体可比的尺寸，否则会渲出巨型物体糊满画面。
"""

from __future__ import annotations

import json
import os

import bpy
import numpy as np

__all__ = ["add_objaverse_props", "load_keep_list"]


def load_keep_list(path: str) -> list[str]:
    """读 uid->glb 的清单，返回 glb 路径列表。"""
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        return [v for v in d.values() if isinstance(v, str) and os.path.isfile(v)]
    return [p for p in d if isinstance(p, str) and os.path.isfile(p)]


def _join(objs):
    """把一个 GLB 里的多个 mesh 合成一个。"""
    if len(objs) == 1:
        return objs[0]
    try:
        import blenderproc as bproc
        return bproc.object.join_objects_many_list(objs)
    except Exception:
        # 退路：直接用 bpy 把除第一个以外都删掉（保底能跑）
        keep = objs[0]
        for o in objs[1:]:
            try:
                o.blender_obj.select_set(True)
            except Exception:
                pass
        return keep


def _world_extent(obj) -> np.ndarray:
    """算物体的世界坐标包围盒尺寸（米）。

    ⚠️ 为什么不用 ``obj.get_bound_box()``：实测刚 import 完的 GLTF 物体，
    它返回**全 0**（depsgraph 还没评估）。这会让"尺寸归一化"静默失败 ——
    6 个物体全被判成"尺寸非法"而跳过，而且日志里只有一行汇总，很难查。

    所以这里直接从 mesh 顶点算（大网格做子采样，避免慢）。
    """
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    bnds = []
    try:
        parts = obj if isinstance(obj, (list, tuple)) else [obj]
    except Exception:
        parts = [obj]
    for o in parts:
        try:
            bo = getattr(o, "blender_obj", o)
            me = bo.data
            n = len(me.vertices)
            if n == 0:
                continue
            step = max(1, n // 4000)          # 子采样上限 ~4000 点
            mw = bo.matrix_world
            idx = range(0, n, step)
            vs = np.array([mw @ me.vertices[i].co for i in idx], dtype=np.float64)
            bnds.append((vs.min(axis=0), vs.max(axis=0)))
        except Exception:
            continue
    if not bnds:
        return np.zeros(3)
    lo = np.min([b[0] for b in bnds], axis=0)
    hi = np.max([b[1] for b in bnds], axis=0)
    return hi - lo


def _normalize_size(obj, target_min: float, target_max: float, rng) -> float:
    """把物体的最长边缩放到 [target_min, target_max]（米）。返回缩放后的最长边。

    失败返回 -1.0（调用方会记一次 fail 并打印）。
    """
    ext = _world_extent(obj)
    cur = float(np.max(ext)) if ext.size else 0.0
    if not np.isfinite(cur) or cur <= 1e-9:
        return -1.0
    # 在目标区间里随机挑一个尺寸再等比缩放 —— 让大小也有多样性
    want = float(rng.uniform(target_min, target_max))
    s = want / cur
    obj.set_scale([s, s, s])
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    return want


def add_objaverse_props(keep_path: str, rng: np.random.Generator,
                        n_props: int = 5,
                        size_min: float = 0.025, size_max: float = 0.10,
                        bin_area: float = 0.16, spawn_z0: float = 0.05,
                        spawn_z_step: float = 0.03,
                        log_prefix: str = "[props]"):
    """往场景里加 n_props 个 Objaverse 物体（随机大小/朝向，悬在箱子上方交给物理塌落）。

    Args:
        keep_path: uid->glb 清单 json 路径
        rng: 随机数发生器
        n_props: 加几个
        size_min/max: 归一化后的最长边范围（米）。默认 2.5~10 cm，
                      和目标物体（69.9 x 44.9 x 33.1 mm）可比
        bin_area: 箱子内边长（米），用来决定水平撒点范围
        spawn_z0/step: 起始高度与层间距（米）

    Returns:
        (加载成功的 MeshObject 列表, 统计 dict)
    """
    import blenderproc as bproc

    paths = load_keep_list(keep_path)
    if not paths:
        raise SystemExit(f"{log_prefix} no GLB found in {keep_path}")
    picks = [paths[i] for i in rng.choice(len(paths), size=min(n_props, len(paths)),
                                          replace=False)]
    out, ok, fail, sizes = [], 0, 0, []
    half = bin_area / 2.0 * 0.75        # 别贴到箱壁

    for k, p in enumerate(picks):
        try:
            objs = bproc.loader.load_obj(p)
        except Exception as e:
            fail += 1
            print(f"{log_prefix} 跳过 {os.path.basename(p)}: {type(e).__name__}: {e}")
            continue
        if not objs:
            fail += 1
            print(f"{log_prefix} 跳过 {os.path.basename(p)}: load_obj 返回空")
            continue
        try:
            o = _join(objs)
            o.set_name(f"prop_{k:02d}_{os.path.basename(p)[:8]}")
            longest = _normalize_size(o, size_min, size_max, rng)
            if longest < 0:
                fail += 1
                print(f"{log_prefix} 跳过 {os.path.basename(p)}: 包围盒退化"
                      f"（顶点数 {len(getattr(o.blender_obj, 'data').vertices) if hasattr(o, 'blender_obj') else '?'}）")
                continue
            sizes.append(longest)
            o.set_rotation_euler(bproc.sampler.uniformSO3())
            o.set_location([
                float(rng.uniform(-half, half)),
                float(rng.uniform(-half, half)),
                float(spawn_z0 + k * spawn_z_step),
            ])
            o.enable_rigidbody(True, mass=1.0, friction=100.0,
                               linear_damping=0.99, angular_damping=0.99)
            out.append(o)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"{log_prefix} 处理失败 {os.path.basename(p)}: "
                  f"{type(e).__name__}: {e}")

    stats = {"ok": ok, "fail": fail, "n_pick": len(picks),
             "size_median": float(np.median(sizes)) if sizes else -1.0}
    print(f"{log_prefix} 加了 {ok}/{len(picks)} 个 Objaverse 物体"
          f"（失败 {fail}，最长边中位 {stats['size_median']*1000:.1f} mm）")
    return out, stats
