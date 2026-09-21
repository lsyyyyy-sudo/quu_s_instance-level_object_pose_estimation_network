"""把目标视频上的角点预测结果拼成「可读的大图」，供 PPT / 汇报复用。

输入是 `scripts/demo_video_gdino.py` 一类脚本产出的结果表，布局固定为
**3 行 x 2 列**：

    左列 = 整帧（青框 = 检测框，橙线框 = 预测的 8 角点，红点 = 角点）
    右列 = 目标裁剪放大

前两行是预测较好的帧，第三行是失败帧。

产出两张图：

1. ``<模型>/good_cases.png`` —— 单模型：每行「整帧 | 放大」，好/坏样本分别标注
2. ``model_compare_zoom.png``  —— 多模型并排（同一批帧），只留放大面板，
   用来一眼看出谁准

⚠️ 两个坑（本脚本已处理）：

* **PIL 默认字体没有中文字形**，标签会渲染成一排方框。必须显式
  `ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", ...)`。
* **面板边界要自动探测**，不能写死像素。分隔带是近黑色，
  用「行列均值 < 20 的连续段」找出来；探测失败再退回按比例切。

用法::

    python scripts/make_video_result_sheets.py \\
        --dir data/results/video_eval \\
        --models train_v1 GEO1 GEO7
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
PANEL_LABEL = {0: "整帧  (青=检测框, 橙=预测的 8 角点)", 1: "目标裁剪放大"}
CASES = [("好样本 #1", 0, True), ("好样本 #2", 1, True), ("失败样本", 2, False)]


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    print("  ⚠️ 没找到中文字体，标签会显示成方框（把字体放到 FONT_CANDIDATES 里）")
    return ImageFont.load_default()


def _bands(profile, thresh=20, min_len=3):
    dark = profile < thresh
    out, s = [], None
    for i, d in enumerate(dark):
        if d and s is None:
            s = i
        elif not d and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    if s is not None and len(dark) - s >= min_len:
        out.append((s, len(dark)))
    return out


def _segments(bands, total, min_gap=20):
    segs, prev = [], 0
    for s, e in bands:
        if s - prev > min_gap:
            segs.append((prev, s))
        prev = e
    if total - prev > min_gap:
        segs.append((prev, total))
    return segs


def panel_grid(path: str):
    """返回 (image, rows, cols)。自动探测失败时按 3x2 比例切。"""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    a = np.asarray(im.convert("L"), dtype=np.float32)
    rows = _segments(_bands(a.mean(axis=1)), H)
    cols = _segments(_bands(a.mean(axis=0)), W)
    if len(rows) < 3 or len(cols) < 2:
        print(f"    ⚠️ {path} 自动切分失败（rows={len(rows)} cols={len(cols)}），按比例切")
        rows = [(int(H * i / 3), int(H * (i + 1) / 3)) for i in range(3)]
        cols = [(0, int(W * 0.60)), (int(W * 0.60), W)]
    return im, rows, cols


def _put(d, font, x, y, s, col):
    """带黑描边的文字 —— 亮背景上也读得清。"""
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                d.text((x + dx, y + dy), s, fill=(0, 0, 0), font=font)
    d.text((x, y), s, fill=col, font=font)


def sheet_single(src: str, out: str) -> None:
    im, rows, cols = panel_grid(src)
    cell_h, pad, label_h = 420, 10, 26
    tiles = []
    for name, ri, _good in CASES:
        for ci in (0, 1):
            x0, x1 = cols[ci]
            y0, y1 = rows[ri]
            crop = im.crop((x0, y0, x1, y1))
            sc = cell_h / crop.height
            tiles.append((name, ri, ci,
                          crop.resize((max(int(crop.width * sc), 1), cell_h), Image.LANCZOS)))
    cw = max(t.width for *_, t in tiles)
    total_w = pad + (pad + cw) * 2
    row_h = [max(t.height for _, r, _, t in tiles if r == ri) + label_h + pad for ri in range(3)]
    sh = Image.new("RGB", (total_w, pad + sum(row_h)), (14, 14, 18))
    d = ImageDraw.Draw(sh)
    f = load_font(17)
    y = pad
    for ri, (name, _, good) in enumerate(CASES):
        x = pad
        for _n, r, ci, t in tiles:
            if r != ri:
                continue
            col = (120, 240, 140) if good else (250, 130, 120)
            d.rectangle([x, y, x + t.width, y + label_h - 2], fill=(26, 26, 32))
            _put(d, f, x + 6, y + 4, f"{os.path.basename(os.path.dirname(src))}  {name}  {PANEL_LABEL[ci]}", col)
            sh.paste(t, (x, y + label_h))
            x += t.width + pad
        y += row_h[ri]
    sh.save(out)
    print(f"    -> {out}  {sh.size}")


def sheet_compare(base: str, models, out: str) -> None:
    data = {}
    for key, _label in models:
        p = os.path.join(base, key, "stage5_corners.jpg")
        if not os.path.isfile(p):
            print(f"  ❌ 缺少 {p}，跳过对比图")
            return
        im, rows, cols = panel_grid(p)
        data[key] = (im, rows, cols)
        print(f"  {key}: {im.size} rows={len(rows)} cols={len(cols)}")

    cell_h, pad, label_h = 360, 12, 54
    tiles = {}
    for key, _ in models:
        im, rows, cols = data[key]
        for name, ri, _g in CASES:
            x0, x1 = cols[1]
            y0, y1 = rows[ri]
            crop = im.crop((x0, y0, x1, y1))
            sc = cell_h / crop.height
            tiles[(key, name)] = crop.resize((max(int(crop.width * sc), 1), cell_h), Image.LANCZOS)

    cw = max(t.width for t in tiles.values())
    total_w = pad + (pad + cw) * len(models)
    total_h = pad + sum(label_h + cell_h + pad for _ in CASES)
    sh = Image.new("RGB", (total_w, total_h), (14, 14, 18))
    d = ImageDraw.Draw(sh)
    f = load_font(19)
    y = pad
    for name, _ri, _g in CASES:
        x = pad
        for key, label in models:
            d.rectangle([x, y, x + cw, y + label_h - 4], fill=(26, 26, 32))
            col = (120, 240, 140) if key == "GEO7" else (200, 200, 210)
            _put(d, f, x + 8, y + 4, f"{name}   {label}", col)
            _put(d, f, x + 8, y + 28, f"（{PANEL_LABEL[1]}）", (150, 150, 160))
            sh.paste(tiles[(key, name)], (x, y + label_h))
            x += cw + pad
        y += label_h + cell_h + pad
    sh.save(out)
    print(f"  -> {out}  {sh.size}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"data\results\video_eval",
                    help="含 <模型>/stage5_corners.jpg 的目录")
    ap.add_argument("--models", nargs="*", default=["train_v1", "GEO1", "GEO7"])
    ap.add_argument("--labels", nargs="*", default=None,
                    help="与 --models 对应的显示名；缺省用目录名")
    args = ap.parse_args()

    labels = args.labels or args.models
    models = list(zip(args.models, labels))

    print("=== 单模型图 ===")
    for key, _ in models:
        p = os.path.join(args.dir, key, "stage5_corners.jpg")
        if os.path.isfile(p):
            sheet_single(p, os.path.join(args.dir, key, "good_cases.png"))
        else:
            print(f"  ❌ 缺少 {p}")

    print("\n=== 多模型横向对比（放大面板）===")
    sheet_compare(args.dir, models, os.path.join(args.dir, "model_compare_zoom.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
