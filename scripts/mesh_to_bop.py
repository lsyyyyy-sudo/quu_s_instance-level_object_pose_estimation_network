"""把 3D 生成模型的输出转成 BlenderProc BOP loader 能吃的 PLY。

为什么需要这个脚本
------------------
Hunyuan3D / TRELLIS 这类模型默认输出 ``.glb``（纹理内嵌）或 ``.obj``（配 .mtl + 贴图），
但 HCCEPose 的 BlenderProc 渲染脚本走的是 ``bproc.loader.load_bop_objs()``，
对模型文件有硬性要求（以下每一条都从 BlenderProc 源码核实过）：

===============================================  ==================================================
要求                                              源码依据
===============================================  ==================================================
文件名必须是 ``obj_000001.ply``（6 位补零）        ``bop_toolkit_lib/dataset_params.py`` L152
必须放在 ``models/`` 下                           同 L142
**必须是文本(ASCII) PLY，不能是二进制**           ``loader/ObjectLoader.py`` L56 以文本模式读取并做字符串替换
需要顶点坐标 + 顶点法线                           ``s1_p1_obj_rename_center.py`` 的保存参数
纹理：PLY 头写 ``comment TextureFile <名>``       ``ObjectLoader.py`` L52/L60-68
　　　（纹理图必须与 PLY 同目录）
或者：不写 TextureFile，改用**顶点色**            ``ObjectLoader.py`` L83-88 (``map_vertex_color()``)
单位 mm（脚本用 ``mm2m=True``）                    ``s2_p1_gen_pbr_data.py`` L185
居中到原点                                       包围盒/8 角点的对称性依赖它
===============================================  ==================================================

用法
----
::

    python scripts/mesh_to_bop.py \\
        --input data/mesh/generated/dji.glb \\
        --out-dir data/dji_action4/models \\
        --obj-id 1

默认会把它**等比例缩放到 DJI Action 4 的官方尺寸**（70.5 x 44.2 x 32.8 mm），
因为生成的 mesh 尺度是任意的，而尺度直接决定 PnP 解出的平移量。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# DJI Osmo Action 4 官方尺寸（长 x 宽 x 高，mm），来源：DJI 官方技术参数页
DJI_ACTION4_DIMS_MM = (70.5, 44.2, 32.8)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _require_pymeshlab():
    try:
        import pymeshlab  # noqa: F401
    except ImportError:
        sys.exit(
            "缺少 pymeshlab。装一下：\n"
            "  .venv\\Scripts\\python.exe -m pip install pymeshlab trimesh pygltflib"
        )
    import pymeshlab

    return pymeshlab


def _bbox_of(ms):
    """返回 (min[3], max[3], extents[3], diagonal)。"""
    import numpy as np

    v = ms.current_mesh().vertex_matrix()
    mn, mx = v.min(axis=0), v.max(axis=0)
    ext = mx - mn
    return mn, mx, ext, float(np.linalg.norm(ext))


def _diagonal(dims) -> float:
    import numpy as np

    return float(np.linalg.norm(np.asarray(dims, dtype=float)))


def load_mesh(ms, path: Path) -> None:
    """加载 mesh。pymeshlab 读不了的格式（如 .glb）先用 trimesh 转一道。"""
    try:
        ms.load_new_mesh(str(path))
        return
    except Exception as e:  # noqa: BLE001
        print(f"  pymeshlab 直接加载失败（{type(e).__name__}: {e}）")

    if path.suffix.lower() not in (".glb", ".gltf"):
        raise RuntimeError(f"无法加载 {path}")

    print("  尝试用 trimesh 把 glb/gltf 转成中间格式 ...")
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    tmp = path.with_name(path.stem + "__converted.ply")
    scene.export(str(tmp))
    print(f"  已转换 -> {tmp.name}")
    ms.load_new_mesh(str(tmp))


def find_texture(ms, mesh_path: Path):
    """找出纹理文件路径；没有则返回 None。

    - .glb/.gltf：纹理**内嵌**在容器里，用 trimesh 抽出来存成 PNG
    - .obj：查同目录同名 .png/.jpg，或解析 .mtl 的 map_Kd
    - .ply：查 PLY 头部的 ``comment TextureFile``
    """
    # GLB / GLTF：内嵌纹理，需要抽出来
    if mesh_path.suffix.lower() in (".glb", ".gltf"):
        try:
            import trimesh

            obj = trimesh.load(str(mesh_path), force="mesh")
            material = getattr(getattr(obj, "visual", None), "material", None)
            if material is None:
                print("      ⚠️ glb 里没有 material")
                return None

            # trimesh 不同版本/不同导入器下，纹理可能挂在 image 或 baseColorTexture
            img = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
            if img is None:
                print(f"      ⚠️ material({type(material).__name__}) 里没有纹理图，"
                      f"可用字段: {[a for a in dir(material) if 'image' in a.lower() or 'texture' in a.lower()]}")
                return None

            out = mesh_path.with_name(mesh_path.stem + "__texture.png")
            img.convert("RGB").save(out)
            print(f"      从 {mesh_path.suffix} 抽出内嵌纹理 -> {out.name}  {img.size}")
            return out
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠️ 从 {mesh_path.suffix} 提取纹理失败：{type(e).__name__}: {e}")
        return None

    # PLY 头部
    if mesh_path.suffix.lower() == ".ply":
        head = mesh_path.read_text(encoding="latin-1", errors="ignore")[:4000]
        for line in head.splitlines():
            if line.strip().startswith("comment TextureFile"):
                name = line.strip().split(maxsplit=2)[-1].strip()
                p = mesh_path.parent / name
                if p.is_file():
                    return p

    # OBJ -> MTL -> map_Kd
    if mesh_path.suffix.lower() == ".obj":
        mtl = mesh_path.with_suffix(".mtl")
        if mtl.is_file():
            for line in mtl.read_text(encoding="latin-1", errors="ignore").splitlines():
                if line.strip().startswith("map_Kd"):
                    name = line.strip().split(maxsplit=1)[-1].strip()
                    p = mesh_path.parent / name
                    if p.is_file():
                        return p

    # 同目录同名图片
    for ext in (".png", ".PNG", ".jpg", ".JPG", ".jpeg"):
        p = mesh_path.with_suffix(ext)
        if p.is_file():
            return p

    # 同目录下唯一的图片
    imgs = [p for p in mesh_path.parent.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if len(imgs) == 1:
        return imgs[0]

    return None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def convert(
    input_path: Path,
    out_dir: Path,
    obj_id: int,
    target_dims,
    scale_mode: str,
    texture_relative_to: Path | None,
) -> Path:
    ml = _require_pymeshlab()
    out_dir.mkdir(parents=True, exist_ok=True)

    ms = ml.MeshSet()
    print(f"[1/6] 加载 {input_path}")
    load_mesh(ms, input_path)

    m = ms.current_mesh()
    print(f"      顶点 {m.vertex_number()}  面 {m.face_number()}")

    texture = find_texture(ms, input_path)
    print(f"[2/6] 纹理：{texture.name if texture else '无（将使用顶点色）'}")

    # ---- 居中 ----
    mn, mx, ext, diag = _bbox_of(ms)
    print(f"[3/6] 居中前包围盒（模型单位）：{ext[0]:.4f} x {ext[1]:.4f} x {ext[2]:.4f}"
          f"  对角线 {diag:.4f}")
    center = (mn + mx) / 2.0
    ms.compute_matrix_from_translation_rotation_scale(
        translationx=float(-center[0]),
        translationy=float(-center[1]),
        translationz=float(-center[2]),
    )

    # ---- 缩放到官方尺寸 ----
    target_diag = _diagonal(target_dims)
    if scale_mode == "diagonal":
        s = target_diag / diag
    elif scale_mode == "none":
        s = 1.0
    else:
        raise ValueError(f"未知 scale_mode: {scale_mode}")

    if s != 1.0:
        ms.compute_matrix_from_translation_rotation_scale(scalex=s, scaley=s, scalez=s)
        print(f"      等比例缩放 x{s:.4f}（按对角线对齐 {target_diag:.2f} mm）")

    # ---- 法线 ----
    print("[4/6] 计算顶点法线")
    ms.compute_normal_per_vertex()

    # ---- 纹理：wedge UV -> vertex UV ----
    has_texture = texture is not None
    if has_texture:
        try:
            if ms.current_mesh().has_wedge_tex_coord():
                ms.compute_texcoord_transfer_wedge_to_vertex()
                print("      wedge UV -> vertex UV 转换完成")
            else:
                print("      已是 vertex UV，跳过转换")
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠️ UV 转换失败（{e}），回退到顶点色")
            has_texture = False

    # ---- 保存 ----
    out_ply = out_dir / f"obj_{obj_id:06d}.ply"
    print(f"[5/6] 保存 {out_ply}")
    save_kwargs = dict(
        binary=False,              # ★ 必须文本格式
        save_vertex_normal=True,
        save_vertex_coord=True,
        save_wedge_texcoord=False,
        save_vertex_color=not has_texture,
        save_textures=False,       # ★ 纹理我们自己存（pymeshlab 会因为名字没扩展名而报错）
    )
    try:
        ms.save_current_mesh(str(out_ply), **save_kwargs)
    except TypeError:
        # 老版本 pymeshlab 没有 save_textures 参数
        save_kwargs.pop("save_textures", None)
        ms.save_current_mesh(str(out_ply), **save_kwargs)

    # ---- 纹理文件放到 PLY 旁边 + 修正 TextureFile 头 ----
    if has_texture and texture is not None:
        dst_tex = out_ply.with_suffix(".png")   # BlenderProc 只要同目录 + 头部指对名字
        if texture.resolve() != dst_tex.resolve():
            try:
                from PIL import Image

                Image.open(texture).convert("RGB").save(dst_tex)
            except Exception:  # noqa: BLE001
                shutil.copy2(texture, dst_tex)
        _ensure_texture_file_comment(out_ply, dst_tex.name)
        print(f"      纹理已放到 {dst_tex.name}")

    # ---- 校验 ----
    print("[6/6] 校验输出")
    ms2 = ml.MeshSet()
    ms2.load_new_mesh(str(out_ply))
    mn2, mx2, ext2, diag2 = _bbox_of(ms2)

    head = out_ply.read_text(encoding="latin-1", errors="ignore")[:200]
    is_ascii = "ply" in head and "binary" not in head
    has_normal = "nx" in head

    print(f"      输出包围盒(mm)：{ext2[0]:.2f} x {ext2[1]:.2f} x {ext2[2]:.2f}"
          f"  对角线 {diag2:.2f}")
    print(f"      官方尺寸(mm)  ：{target_dims[0]} x {target_dims[1]} x {target_dims[2]}"
          f"  对角线 {target_diag:.2f}")
    print(f"      文本(ASCII) PLY : {is_ascii}")
    print(f"      含顶点法线      : {has_normal}")
    print(f"      纹理            : {'TextureFile 头' if has_texture else '顶点色'}")

    # 各轴对比（两边都按降序排，忽略朝向）
    got = sorted([float(v) for v in ext2], reverse=True)
    want = sorted([float(v) for v in target_dims], reverse=True)
    print("      各轴对比（降序）:", end=" ")
    for g, w in zip(got, want):
        print(f"{g:.1f}/{w:.1f}({100*g/w:.0f}%)", end="  ")
    print()

    # 写一份 models_info.json 的输入摘要（真正生成用 HCCEPose 的 s1_p3_obj_infos.py）
    summary = {
        "obj_id": obj_id,
        "ply": out_ply.name,
        "extents_mm": [float(v) for v in ext2],
        "diagonal_mm": diag2,
        "target_dims_mm": [float(v) for v in target_dims],
        "has_texture": bool(has_texture),
        "ascii_ply": bool(is_ascii),
    }
    (out_dir / f"obj_{obj_id:06d}_convert_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      摘要 -> obj_{obj_id:06d}_convert_summary.json")

    # ---- models_info.json（BOP 必需；也是我们 8 个 3D 角点的来源）----
    make_models_info(ml, ms2, out_dir, obj_id, ext2)
    return out_ply


def make_models_info(ml, ms2, out_dir: Path, obj_id: int, extents) -> None:
    """生成 BOP 的 ``models_info.json``。

    字段与 HCCEPose 的 ``s1_p3_obj_infos.py`` 一致：
    ``diameter`` / ``min_*`` / ``max_*`` / ``size_*``。

    ``diameter`` 按 BOP 约定是**模型点之间的最大距离**。
    对 60 多万个顶点做 O(N²) 不现实，所以在**凸包顶点**上算 ——
    最大距离一定出现在凸包上，结果等价而快得多。
    """
    import numpy as np

    mn = float(np.min(extents))  # 占位，下面用真实 min/max
    v = ms2.current_mesh().vertex_matrix()
    vmin, vmax = v.min(axis=0), v.max(axis=0)

    diameter = float(np.linalg.norm(vmax - vmin))
    try:
        hull = ml.MeshSet()
        hull.load_new_mesh(str(out_dir / f"obj_{obj_id:06d}.ply"))
        hull.generate_convex_hull()
        hv = hull.current_mesh().vertex_matrix()
        if len(hv) >= 2:
            from scipy.spatial.distance import pdist

            diameter = float(pdist(hv).max())
            print(f"      直径（凸包 {len(hv)} 点）: {diameter:.2f} mm")
    except Exception as e:  # noqa: BLE001
        print(f"      ⚠️ 凸包算直径失败（回退用包围盒对角线）: {type(e).__name__}: {e}")

    info = {
        f"{obj_id}": {
            "diameter": diameter,
            "min_x": float(vmin[0]), "min_y": float(vmin[1]), "min_z": float(vmin[2]),
            "max_x": float(vmax[0]), "max_y": float(vmax[1]), "max_z": float(vmax[2]),
            "size_x": float(vmax[0] - vmin[0]),
            "size_y": float(vmax[1] - vmin[1]),
            "size_z": float(vmax[2] - vmin[2]),
        }
    }
    p = out_dir / "models_info.json"
    p.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      models_info.json 已写 -> {p.name}")


def _ensure_texture_file_comment(ply_path: Path, texture_name: str) -> None:
    """确保 PLY 头部有 ``comment TextureFile <texture_name>``，且名字正确。

    BlenderProc 正是靠这一行找纹理（``ObjectLoader.py`` L52/L60）。

    ⚠️ 注意：**不能只在缺失时插入**。MeshLab 保存时会写一个占位名字
    （实测是 ``texture_0``，因为它拿不到真实文件名），
    如果不替换掉，BlenderProc 就会去找一个不存在的 ``texture_0`` → 渲染失败。
    """
    text = ply_path.read_text(encoding="latin-1")
    line = f"comment TextureFile {texture_name}\n"

    if "comment TextureFile" in text:
        # 替换已有那一行（保留其它 comment 不动）
        new_lines = []
        replaced = False
        for raw in text.splitlines(keepends=True):
            if raw.strip().startswith("comment TextureFile"):
                new_lines.append(line)
                replaced = True
            else:
                new_lines.append(raw)
        if replaced:
            ply_path.write_text("".join(new_lines), encoding="latin-1")
            return

    idx = text.find("end_header")
    if idx < 0:
        raise RuntimeError(f"{ply_path} 不是合法的 PLY（找不到 end_header）")
    ply_path.write_text(text[:idx] + line + text[idx:], encoding="latin-1")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="mesh -> BlenderProc 可用的 BOP 格式 PLY")
    ap.add_argument("--input", required=True, type=Path, help="生成模型输出（.glb/.obj/.ply）")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="输出目录，通常是 data/dji_action4/models")
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--target-dims", type=float, nargs=3, default=list(DJI_ACTION4_DIMS_MM),
                    metavar=("L", "W", "H"), help="目标尺寸 (mm)，默认 DJI Action 4 官方值")
    ap.add_argument("--scale-mode", choices=["diagonal", "none"], default="diagonal",
                    help="diagonal=等比例缩放到目标对角线；none=不动尺度")
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"输入不存在：{args.input}")

    print("=" * 66)
    convert(
        input_path=args.input,
        out_dir=args.out_dir,
        obj_id=args.obj_id,
        target_dims=tuple(args.target_dims),
        scale_mode=args.scale_mode,
        texture_relative_to=args.input.parent,
    )
    print("=" * 66)
    print("下一步：把 models/ 交给 HCCEPose 的 s1_p3_obj_infos.py 生成 models_info.json")


if __name__ == "__main__":
    main()
