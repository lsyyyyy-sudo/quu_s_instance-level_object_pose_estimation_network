"""沿 ±X / ±Y / ±Z 六个方向正交投影，用来判定模型当前的朝向。

零成本（纯 numpy+PIL，不走 Cycles）。每个面板标了"相机沿哪个轴看"，
看哪一面是镜头就能定出物体的坐标系。

    python axis_views.py <mesh.ply> <texture.png> <out.png>
"""

from __future__ import annotations

import sys

import numpy as np
import pymeshlab as ml
from PIL import Image, ImageDraw


def load(mesh_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    m = ms.current_mesh()
    faces = m.face_matrix().astype(np.int64)
    if m.has_vertex_tex_coord():
        uv = m.vertex_tex_coord_matrix().astype(np.float64)
    else:
        wedge = m.wedge_tex_coord_matrix().astype(np.float64)
        uv = np.zeros((m.vertex_number(), 2))
        uv[faces.reshape(-1)] = wedge
    return m.vertex_matrix(), uv


def sample(tex, uv):
    h, w = tex.shape[:2]
    u = np.clip(uv[:, 0], 0, 1)
    v = np.clip(uv[:, 1], 0, 1)
    xs = np.clip(np.rint(u * (w - 1)).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint((1.0 - v) * (h - 1)).astype(np.int64), 0, h - 1)
    return tex[ys, xs]


def splat(pts, cols, right, up, view, size=360, bg=(26, 26, 30)):
    """orthographic: project onto (right, up); depth along `view` (points toward camera)."""
    x = pts @ right
    y = -(pts @ up)          # image rows grow downward, so negate to keep `up` = up
    z = pts @ view
    xy = np.stack([x, y], axis=1)
    lo, hi = xy.min(0), xy.max(0)
    span = max((hi - lo).max(), 1e-9)
    scale = (size - 24) / span
    px = ((xy - (lo + hi) / 2) * scale + size / 2).astype(int)
    img = np.full((size, size, 3), bg, np.uint8)
    order = np.argsort(z)          # far -> near
    xs, ys, cs = px[order, 0], px[order, 1], cols[order]
    ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    xs, ys, cs = xs[ok], ys[ok], cs[ok]
    for dx in (0, 1):
        for dy in (0, 1):
            img[np.clip(ys + dy, 0, size - 1), np.clip(xs + dx, 0, size - 1)] = cs
    return img


if __name__ == "__main__":
    mesh_path, tex_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    pts, uv = load(mesh_path)
    tex = np.asarray(Image.open(tex_path).convert("RGB"), dtype=np.uint8)
    cols = sample(tex, uv)

    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    print(f"verts {len(pts)}")
    print(f"bbox min {np.round(lo, 2)}")
    print(f"bbox max {np.round(hi, 2)}")
    print(f"extent  {np.round(ext, 2)}  (X, Y, Z)")
    order = np.argsort(ext)[::-1]
    print(f"axes by length, longest first: {[('XYZ'[i], round(float(ext[i]), 2)) for i in order]}")

    # ---- 主轴分析：物体自身的惯性主轴有多贴合坐标轴？----
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    print("\nprincipal axes (inertia) vs coordinate axes:")
    for i in range(3):
        a = V[:, i]
        j = int(np.argmax(np.abs(a)))
        ang = np.degrees(np.arccos(min(1.0, abs(a[j]))))
        print(f"  pc{i}: dir={np.round(a, 4)}  -> closest to {'XYZ'[j]}  "
              f"off-axis {ang:.2f} deg  var={w[i]:.1f}")
    print("  (off-axis 接近 0 度 = 模型自身坐标系已经和坐标轴对齐)")

    # 每个面板：(标签, 相机所在方向 look_dir, right, up)
    # 相机在 +A 处朝 -A 看
    panels = [
        ("cam at +X  (looking -X)", np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), np.array([0.0, 0, 1])),
        ("cam at -X  (looking +X)", np.array([-1.0, 0, 0]), np.array([0.0, -1, 0]), np.array([0.0, 0, 1])),
        ("cam at +Y  (looking -Y)", np.array([0.0, 1, 0]), np.array([1.0, 0, 0]), np.array([0.0, 0, 1])),
        ("cam at -Y  (looking +Y)", np.array([0.0, -1, 0]), np.array([-1.0, 0, 0]), np.array([0.0, 0, 1])),
        ("cam at +Z  (looking -Z)", np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])),
        ("cam at -Z  (looking +Z)", np.array([0.0, 0, -1]), np.array([-1.0, 0, 0]), np.array([0.0, 1, 0])),
    ]
    imgs = [splat(pts, cols, r, u, v) for _, v, r, u in panels]

    S = imgs[0].shape[0]
    pad, lab = 8, 20
    cols_n, rows_n = 3, 2
    sheet = Image.new("RGB", (cols_n * S + (cols_n + 1) * pad,
                              rows_n * (S + lab) + (rows_n + 1) * pad), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for k, (name, img) in enumerate(zip([p[0] for p in panels], imgs)):
        r, c = divmod(k, cols_n)
        x = pad + c * (S + pad)
        y = pad + r * (S + lab + pad)
        d.text((x + 2, y), name, fill=(235, 235, 235))
        sheet.paste(Image.fromarray(img), (x, y + lab))
    sheet.save(out_path)
    print("wrote", out_path, sheet.size)
