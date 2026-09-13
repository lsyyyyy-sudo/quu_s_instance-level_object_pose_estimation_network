"""验证 BOP 训练集的标注是否正确：把 3D 包围盒按 GT 位姿投影到 RGB 上。

这是"数据造出来了"和"数据是对的"之间的那道检查。投影用的是**独立的 numpy 实现**
（不是训练管线里的代码），所以能同时验证 `scene_gt.json` 的位姿、
`scene_camera.json` 的内参、`models_info.json` 的尺寸三者是否自洽。

用法::

    python verify_bop.py <verify_dir> <models_info.json> <out.png>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# 12 条棱，按 8 个角点的下标
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0),
         (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def corners_from_models_info(info: dict):
    mn = np.array([info["min_x"], info["min_y"], info["min_z"]], dtype=np.float64)
    mx = np.array([info["max_x"], info["max_y"], info["max_z"]], dtype=np.float64)
    return np.array([[x, y, z] for x in (mn[0], mx[0])
                     for y in (mn[1], mx[1]) for z in (mn[2], mx[2])], dtype=np.float64)


def project(pts_obj, R, t, K):
    cam = (R @ pts_obj.T).T + t.reshape(1, 3)
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3], cam[:, 2]


def main() -> int:
    vdir = Path(sys.argv[1])
    models_info = json.loads(Path(sys.argv[2]).read_text())
    out_path = Path(sys.argv[3])

    info = models_info["1"]
    corners = corners_from_models_info(info)
    print(f"models_info['1']: size = {info['size_x']:.2f} x {info['size_y']:.2f} x {info['size_z']:.2f} mm")
    print(f"                   diameter = {info['diameter']:.2f} mm")
    print(f"8 corners (mm):\n{np.round(corners, 1)}")

    gt = json.loads((vdir / "scene_gt.json").read_text())
    cam = json.loads((vdir / "scene_camera.json").read_text())
    gti = json.loads((vdir / "scene_gt_info.json").read_text())

    panels = []
    for frame in ("0", "5", "10"):
        img_path = vdir / f"rgb_{int(frame):06d}.png"
        if not img_path.is_file():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        K = np.array(cam[frame]["cam_K"], dtype=np.float64).reshape(3, 3)
        print(f"\nframe {frame}: K = fx {K[0,0]:.1f} fy {K[1,1]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f}  ({W}x{H})")

        for inst_i, inst in enumerate(gt[frame]):
            R = np.array(inst["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t = np.array(inst["cam_t_m2c"], dtype=np.float64).reshape(3)
            uv, z = project(corners, R, t, K)

            # 独立算一遍投影后的包围框，和 scene_gt_info 的 bbox_visib 对比
            u0, v0 = uv.min(axis=0)
            u1, v1 = uv.max(axis=0)
            bx, by, bw, bh = gti[frame][inst_i]["bbox_visib"]
            print(f"  inst {inst_i} obj_id={inst['obj_id']} t={np.round(t,1)} |t|={np.linalg.norm(t):.0f}mm")
            print(f"    投影包围框   [{u0:7.1f},{v0:7.1f},{u1-u0:6.1f},{v1-v0:6.1f}]")
            print(f"    bbox_visib   [{bx:7.1f},{by:7.1f},{bw:6.1f},{bh:6.1f}]  "
                  f"px_count_visib={gti[frame][inst_i]['px_count_visib']}")

            col = (255, 60, 60) if inst_i == 0 else (60, 220, 255)
            for a, b in EDGES:
                d.line([tuple(uv[a]), tuple(uv[b])], fill=col, width=3)
            for k, (u, v) in enumerate(uv):
                d.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(255, 255, 0))
                d.text((u + 7, v - 7), str(k + 1), fill=(255, 255, 0))

        panels.append(img.resize((W // 2, H // 2)))

    if not panels:
        print("没有可用的帧")
        return 1
    pad = 8
    sheet = Image.new("RGB", (sum(p.width for p in panels) + pad * (len(panels) + 1),
                              max(p.height for p in panels) + 2 * pad), (20, 20, 24))
    x = pad
    for p in panels:
        sheet.paste(p, (x, pad))
        x += p.width + pad
    sheet.save(out_path)
    print(f"\nwrote {out_path}  {sheet.size}")
    print("看什么：黄点应落在物体包围盒的 8 个角上，红线框应紧贴物体。")
    print("若红框明显偏离物体 -> 位姿/内参/尺寸三者至少有一个不对。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
