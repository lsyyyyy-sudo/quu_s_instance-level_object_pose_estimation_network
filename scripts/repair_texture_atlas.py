"""修复生成模型纹理图集里的「空白噪声」，消除渲染时的黑条纹。

问题
----
Hunyuan3D-2 输出的纹理图集里，UV 岛之间的未使用区域被纹理网络填成了**黑白噪点**。
渲染时纹理是被**缩小采样**的（2048² 的图集在画面上只有两三百像素），mip 层会把
UV 岛的颜色和旁边那片近乎全黑的空白平均掉，于是物体表面沿 UV 岛边界出现
**黑色锯齿条纹**。

这不是 UV 错了——实测简化后顶点 UV 距原始 UV 最大只有 2.9 px（见
``docs/TROUBLESHOOTING.md`` 的 `DATA-08`）。纯粹是图集空白区没有 padding。

做法
----
1. 把网格的 UV 三角形**光栅化**成一张「有效区域」掩码；
2. 对掩码外的每个像素，取**最近的有效像素**颜色（精确欧氏距离变换，一步到位）；
3. 只对掩码外区域做一次轻度模糊，让 mip 过渡自然，岛内像素保持原样。

这是渲染器/游戏引擎的标准 texture padding 流程。

用法
----
::

    python scripts/repair_texture_atlas.py \\
        --mesh data/dji_action4/models/obj_000001.ply \\
        --texture data/dji_action4/models/obj_000001.png \\
        --out data/dji_action4/models/obj_000001.png \\
        --report data/dji_action4/models/obj_000001_atlas_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _load_uvs(mesh_path: Path):
    """返回 (uv[N,2], faces[F,3])，UV 已归一化到 [0,1]。"""
    import pymeshlab as ml

    ms = ml.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    m = ms.current_mesh()

    faces = m.face_matrix().astype(np.int64)
    if m.has_vertex_tex_coord():
        uv = m.vertex_tex_coord_matrix().astype(np.float64)
    elif m.has_wedge_tex_coord():
        # wedge UV 是每三角形角一份；本项目里网格在缝处已完全未焊接，
        # 所以每个顶点只有一个 UV，直接散射不会丢信息。
        wedge = m.wedge_tex_coord_matrix().astype(np.float64)
        uv = np.zeros((m.vertex_number(), 2), dtype=np.float64)
        uv[faces.reshape(-1)] = wedge
    else:
        raise SystemExit(f"{mesh_path} 没有纹理坐标")

    return uv, faces, m.vertex_number(), m.face_number()


def main() -> int:
    ap = argparse.ArgumentParser(description="纹理图集 padding：把 UV 岛颜色外扩填满空白区")
    ap.add_argument("--mesh", required=True, type=Path, help="PLY/GLB，用来取 UV")
    ap.add_argument("--texture", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--grow", type=int, default=4,
                    help="光栅化时给每个三角形额外加粗的像素数（补偿采样误差）")
    ap.add_argument("--blur", type=float, default=1.2,
                    help="只作用于空白区的模糊半径，让 mip 过渡自然；0 = 关掉")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    for p in (args.mesh, args.texture):
        if not p.is_file():
            sys.exit(f"找不到 {p}")

    uv, faces, n_verts, n_faces = _load_uvs(args.mesh)
    print(f"[atlas] mesh {n_verts} 顶点 / {n_faces} 面，UV {uv.shape}")

    img = Image.open(args.texture).convert("RGB")
    W, H = img.size
    print(f"[atlas] texture {W}x{H}")

    # ---- 1. 光栅化有效区域 ----
    # UV 的 v=0 在图像底部（Blender/OpenGL 约定），PIL 的 y=0 在顶部。
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    px = np.clip(np.rint(uv[:, 0] * (W - 1)), 0, W - 1)
    py = np.clip(np.rint((1.0 - uv[:, 1]) * (H - 1)), 0, H - 1)
    tri = np.stack([px[faces], py[faces]], axis=-1)          # (F, 3, 2)
    width = max(1, int(args.grow))
    for t in tri:
        draw.polygon([tuple(t[0]), tuple(t[1]), tuple(t[2])], fill=255, outline=255, width=width)
    mask = np.asarray(mask_img) > 0
    coverage = float(mask.mean())
    print(f"[atlas] UV 覆盖了图集的 {100*coverage:.2f}%，"
          f"空白 {100*(1-coverage):.2f}%（这片就是噪声区）")

    # ---- 2. 最近有效像素填充 ----
    from scipy.ndimage import distance_transform_edt

    dist, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    arr = np.asarray(img, dtype=np.uint8)
    filled = arr[iy, ix]
    print(f"[atlas] 空白区最近有效像素距离：中位 {np.median(dist[~mask]):.1f} px，"
          f"最大 {dist[~mask].max():.1f} px")

    # ---- 3. 只对空白区做一次轻模糊（岛内保持原样） ----
    if args.blur and args.blur > 0:
        soft = np.asarray(
            Image.fromarray(filled).filter(ImageFilter.GaussianBlur(args.blur)),
            dtype=np.uint8,
        )
        m3 = mask[..., None]
        out = np.where(m3, arr, soft)
    else:
        out = np.where(mask[..., None], arr, filled)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.resolve() == args.texture.resolve():
        backup = args.out.with_name(args.out.stem + "_original" + args.out.suffix)
        if not backup.exists():
            img.save(backup)
            print(f"[atlas] 原图已备份 -> {backup.name}")
    Image.fromarray(out.astype(np.uint8)).save(args.out)
    print(f"[atlas] 修复后的图集 -> {args.out}")

    if args.report:
        args.report.write_text(json.dumps({
            "mesh": str(args.mesh),
            "texture": str(args.texture),
            "size": [W, H],
            "vertices": int(n_verts),
            "faces": int(n_faces),
            "uv_coverage_fraction": coverage,
            "gutter_fraction": 1.0 - coverage,
            "max_fill_distance_px": float(dist[~mask].max()) if (~mask).any() else 0.0,
            "grow_px": width,
            "blur_px": args.blur,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[atlas] 报告 -> {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
