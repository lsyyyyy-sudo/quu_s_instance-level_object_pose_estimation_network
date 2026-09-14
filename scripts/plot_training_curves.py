"""把训练过程的 metrics.csv 画成曲线图（纯 PIL，不依赖 matplotlib）。

用法::

    python scripts/plot_training_curves.py <metrics.csv> <out.png>
"""

from __future__ import annotations

import csv
import os
import sys

from PIL import Image, ImageDraw

W, H = 1100, 760
PAD_L, PAD_R, PAD_T, PAD_B = 78, 24, 56, 52
BG = (16, 16, 20)
FG = (232, 232, 236)
GRID = (48, 48, 56)


def _load(path):
    rows = list(csv.DictReader(open(path)))
    series = {}
    for r in rows:
        for k, v in r.items():
            if not k or v in ("", None):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            series.setdefault(k, []).append((int(float(r["step"])), fv))
    return series


def _panel(d, box, title, keys, colors, ylabel, logy=False):
    x0, y0, x1, y1 = box
    d.rectangle([x0, y0, x1, y1], fill=(22, 22, 27), outline=GRID)
    d.text((x0 + 6, y0 - 20), title, fill=FG)

    pts = []
    for k in keys:
        pts += series.get(k, [])
    if not pts:
        d.text((x0 + 8, y0 + 8), "(no data)", fill=(160, 160, 170))
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if logy:
        import math
        ys2 = [math.log10(max(v, 1e-6)) for v in ys]
        ymin, ymax = min(ys2), max(ys2)
        def ty(v):
            import math
            return math.log10(max(v, 1e-6))
    else:
        def ty(v):
            return v
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    if xmax - xmin < 1:
        xmax = xmin + 1

    def sx(v):
        return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

    def sy(v):
        return y1 - (ty(v) - ymin) / (ymax - ymin) * (y1 - y0)

    # 网格 + y 轴刻度
    for i in range(5):
        yy = y0 + (y1 - y0) * i / 4
        d.line([x0, yy, x1, yy], fill=GRID)
        val = ymax - (ymax - ymin) * i / 4
        lab = f"{10**val:.3g}" if logy else f"{val:.4g}"
        # 标签必须贴在**本面板**左边，不能写死 x=8：多个面板共用一张图时
        # 后面的面板会把前面的标签覆盖掉（踩过）
        d.text((x0 - 62, yy - 6), lab, fill=(150, 150, 162))
    for i in range(5):
        xx = x0 + (x1 - x0) * i / 4
        d.line([xx, y0, xx, y1], fill=GRID)
        d.text((xx - 16, y1 + 6), f"{int(xmin + (xmax-xmin)*i/4)}", fill=(150, 150, 162))
    d.text((x0 - 60, y1 + 24), "step", fill=(170, 170, 180))
    d.text((x0 - 62, y0 - 34), ylabel, fill=(210, 210, 220))

    # 曲线
    for k, c in zip(keys, colors):
        s = sorted(series.get(k, []))
        if not s:
            continue
        d.line([(sx(a), sy(b)) for a, b in s], fill=c, width=2)
        lx, ly = sx(s[-1][0]), sy(s[-1][1])
        d.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=c)
        d.text((min(lx + 6, x1 - 150), ly - 6), f"{k.split('/')[-1]} {s[-1][1]:.3g}", fill=c)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    global series
    series = _load(sys.argv[1])
    out = sys.argv[2]

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD_L - 60, 14), "v1 training curves  (val every 5 epochs)", fill=(250, 250, 250))
    d.text((PAD_L - 60, 30), os.path.basename(sys.argv[1]), fill=(150, 150, 160))

    mid = (PAD_T + (H - PAD_B)) // 2
    _panel(d, (PAD_L, PAD_T, W - PAD_R, mid - 34),
           "train loss (epoch mean)", ["train/loss_epoch", "train/loss_coarse", "train/loss_fine"],
           [(90, 200, 255), (140, 220, 140), (255, 180, 90)], "loss", logy=True)
    _panel(d, (PAD_L, mid + 26, (W - PAD_R) // 2 + 90, H - PAD_B),
           "val corner error (px, 256 crop)", ["val/corner_err_px"],
           [(255, 120, 120)], "px")
    _panel(d, ((W - PAD_R) // 2 + 110, mid + 26, W - PAD_R, H - PAD_B),
           "val PCK", ["val/pck@0.05", "val/pck@0.1", "val/pck@0.15"],
           [(120, 200, 255), (150, 230, 150), (255, 200, 110)], "ratio")

    img.save(out)
    print(f"wrote {out}  {img.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
