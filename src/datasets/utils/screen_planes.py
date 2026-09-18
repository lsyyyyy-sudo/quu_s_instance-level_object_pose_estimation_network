"""给 BOP 物体挂"屏幕平面"并把屏幕渲染成随机内容。

调用点在 gen_pbr_data_demo.py 里，物理模拟之后、渲染之前 —— 因为平面要跟着
物体最终的位姿走。

屏幕参数（由正交视图 + 毫米标尺量出，见 data/render_ws/measure_screen.py）
------------------------------------------------------------------------
    DJI Action 4 是【双屏】：
        背面大屏:  中心 (0,     0,   -16.55)  尺寸 60.2 x 36.9 mm  法线 (0,0,-1)
        镜头面小屏: 中心 (-15.35, -0.3, +16.55) 尺寸 28.2 x 36.4 mm  法线 (0,0,+1)
    （机身坐标：+X 长轴 69.94 / +Y 高 44.87 / +Z 镜头方向 33.09，单位 mm）

⚠️ 两个已经踩过的坑
-----------------
1. 物体的 BOP PLY 用非标准 UV 属性名 ``texture_u/v``，Blender 读不到 —— 但那是
   导入机身时的坑，这里只挂平面，不受影响。（见 ply_to_glb.py）
2. **`load_bop_objs(mm2m=True)` 会把物体缩放到米** —— 所以屏幕坐标（mm）要乘以
   0.001，而这里是通过"父级矩阵"自动继承缩放的，不能再手动乘，否则会小 1000 倍。

设计意图见 docs/RESULTS.md §3.6：**让模型学会"那是屏幕"** ——
屏幕内容每次都变 -> 内容不再是可靠线索 -> 模型只能依赖那个区域稳定的东西
（即屏幕边界 = 机身正面的四条边）。
"""

from __future__ import annotations

import importlib.util
import os

import bpy
import numpy as np

# ⚠️ 必须【按文件路径】加载 screen_content，**不能**用
#     `from src.datasets.utils.screen_content import ...`
# 因为那会先执行 `src/datasets/__init__.py`，而它 import 了 `bop_pbr.py`，后者需要 torch。
# **BlenderProc 跑在 Blender 内嵌的 Python 里，没有 torch** —— 用包导入会直接 ImportError。
_HERE = os.path.dirname(os.path.abspath(__file__))     # src/datasets/utils
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))   # -> 仓库根
_spec = importlib.util.spec_from_file_location(
    "_screen_content_standalone",
    os.path.join(_HERE, "screen_content.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_screen_content = _mod.make_screen_content
pick_kind = _mod.pick_kind

# (名字, 局部中心 mm, 宽 mm, 高 mm, 是否翻转朝 -Z)
SCREENS = [
    ("SCREEN_BACK",  (0.0, 0.0, -16.55), 60.2, 36.9, True),
    ("SCREEN_FRONT", (-15.35, -0.3, 16.55), 28.2, 36.4, False),
]
EPS = 0.06      # mm，往外挪一点避免 z-fighting


def _mk_material(name: str, img_path: str, rgb, emission: bool = True):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    if emission:
        node = nt.nodes.new("ShaderNodeEmission")
        sock = node.inputs["Color"]
        nt.links.new(node.outputs["Emission"], out_n.inputs["Surface"])
    else:
        node = nt.nodes.new("ShaderNodeBsdfPrincipled")
        sock = node.inputs["Base Color"]
        nt.links.new(node.outputs["BSDF"], out_n.inputs["Surface"])
    if img_path:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(img_path)
        tex.interpolation = "Linear"
        nt.links.new(tex.outputs["Color"], sock)
    else:
        sock.default_value = rgb
    return mat


def attach_screen_planes(bop_obj, rng: np.random.Generator, content_dir: str,
                         log_prefix: str = "[screen]", force_kind: dict | None = None):
    """给一个 BOP 物体挂上两块屏幕平面，内容按 30/40/30 混合随机。

    Args:
        bop_obj: BlenderProc 的 BOP 物体（Entity），已设定最终位姿
        rng: 随机数发生器（建议每个场景一个，保证可复现）
        content_dir: 生成的内容图放哪
        force_kind: 仅用于调试，如 {"SCREEN_BACK": "tag"} 强制某块屏用某种内容

    Returns:
        dict: {屏幕名: 内容类型}，'native' 表示没挂平面（保留原始纹理）
    """
    parent = bop_obj.blender_obj
    parent_mw = parent.matrix_world.copy()
    chosen = {}

    for name, ctr, w, h, flip in SCREENS:
        kind = (force_kind or {}).get(name) or pick_kind(rng)
        chosen[name] = kind
        if kind == "native":
            continue                      # 保留原始熄屏纹理，不挂平面

        # 生成内容图（长边给足像素，避免糊）
        px_w = int(round(w * 16))
        px_h = int(round(h * 16))
        img_path = os.path.join(content_dir, f"{name}_{kind}_{abs(hash((name, kind, rng.integers(1<<30)))) % (1<<24):06x}.png")
        make_screen_content(kind, px_w, px_h, img_path, rng)

        # 建平面（size=1 -> 1x1，从 -0.5 到 0.5）
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        pl = bpy.context.active_object
        pl.name = f"{bop_obj.get_name()}__{name}"
        pl.data.materials.append(_mk_material(pl.name + "_mat", img_path, (1, 1, 1)))
        pl.scale = (w, h, 1.0)
        if flip:
            pl.rotation_euler = (np.pi, 0.0, 0.0)
        # 放到物体的局部坐标系里（物体的 mm2m 缩放会通过父级自动继承）
        loc = np.array([ctr[0], ctr[1], ctr[2] + (-EPS if flip else EPS)], dtype=float)
        pl.location = loc
        # ⚠️ 只设 parent，**不要**碰 matrix_parent_inverse。
        # 设成 parent_mw.inverted() 会把父级变换整个抵消 -> 平面的 60.2 单位
        # 不再被 mm2m(0.001) 缩放 -> 世界尺寸变成 60200 mm（大 1000 倍）。
        # 实测踩过：world size 报 60200 x 36900 mm，期望 60.2 x 36.9。
        # 保持单位阵时，child.matrix_world = parent.matrix_world @ child.matrix_basis，
        # 位置(mm)和尺寸(mm)都会被父级正确地带入米制。
        pl.parent = parent
        pl.hide_render = False

    if log_prefix:
        print(f"{log_prefix} {bop_obj.get_name()}: " +
              ", ".join(f"{k}={v}" for k, v in chosen.items()))
    return chosen
