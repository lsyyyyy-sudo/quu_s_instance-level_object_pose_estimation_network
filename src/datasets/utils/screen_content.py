"""屏幕内容生成器 —— 让模型学会"那是屏幕"。

设计意图（见 docs/RESULTS.md §3.6）
---------------------------------
真实视频里目标机器的**屏幕是亮着的、还贴着标签**，而我们的训练数据里屏幕是
**熄屏深蓝**这个固定外观。屏幕占正面约 40% 面积、是最大的高对比度面，
它的图案一变，网络赖以定位的纹理线索就全废了。

**核心逻辑**：如果屏幕内容每次都变，那"内容"就不再是可靠线索，
模型只能转而依赖那个区域**稳定的东西** —— 也就是**屏幕的边界**。
而屏幕的边界恰好就是机身正面的四条边。

> **⇒ 这等于在教模型"看屏幕的轮廓，别看它的内容"。**
> 真实视频里屏幕内容虽然变了，**边界没变** —— 所以这个能力正好迁移得过去。

⚠️ 刻意**不全随机**（三类混合）
----------------------------
全随机有风险：模型可能学到"屏幕就是噪声"，然后干脆忽略整个正面，
把有用的边界信息也丢掉。所以按比例混合，保留"正常外观"的锚点：

    原生  30%  ——  不用平面（保留原始熄屏纹理）
    真实感 40%  ——  取景画面 / 菜单 UI / **标签图案**（贴近测试域）
    随机  30%  ——  纯色 / 棋盘 / 噪声 / 渐变 / 文字块（逼模型忽略内容）

    from src.datasets.utils.screen_content import make_screen_content
    make_screen_content("tag", 512, 320, "out.png")
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# 三类内容
NATIVE = "native"
REALISTIC = ("tag", "viewfinder", "menu")
RANDOM = ("solid", "checker", "noise", "gradient", "text")

# 混合比例（刻意不全随机，理由见模块文档）
MIX = [
    (NATIVE, 0.30),
    ("tag", 0.15),
    ("viewfinder", 0.15),
    ("menu", 0.10),
    ("solid", 0.08),
    ("checker", 0.08),
    ("noise", 0.06),
    ("gradient", 0.05),
    ("text", 0.03),
]


def pick_kind(rng: np.random.Generator) -> str:
    kinds = [k for k, _ in MIX]
    probs = np.array([p for _, p in MIX], dtype=np.float64)
    return str(rng.choice(kinds, p=probs / probs.sum()))


# --------------------------------------------------------------------------- #
# 各类内容
# --------------------------------------------------------------------------- #
def _solid(w, h, rng):
    c = rng.integers(0, 256, 3)
    return np.tile(c.astype(np.uint8), (h, w, 1))


def _checker(w, h, rng):
    cell = int(rng.integers(max(4, min(w, h) // 32), max(8, min(w, h) // 6)))
    ys, xs = np.mgrid[0:h, 0:w]
    m = ((xs // cell + ys // cell) % 2).astype(np.uint8)
    c1 = rng.integers(0, 256, 3).astype(np.uint8)
    c2 = rng.integers(0, 256, 3).astype(np.uint8)
    return np.where(m[..., None] > 0, c1, c2).astype(np.uint8)


def _noise(w, h, rng):
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    k = int(rng.integers(1, 4)) | 1
    return cv2.GaussianBlur(base, (k, k), 0)


def _gradient(w, h, rng):
    a = rng.integers(0, 256, 3).astype(np.float32)
    b = rng.integers(0, 256, 3).astype(np.float32)
    ang = rng.uniform(0, np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.cos(ang) * xs / max(w - 1, 1) + np.sin(ang) * ys / max(h - 1, 1)
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    return (a[None, None, :] * (1 - t[..., None]) + b[None, None, :] * t[..., None]).astype(np.uint8)


def _text(w, h, rng):
    bg = np.full((h, w, 3), int(rng.integers(0, 90)), np.uint8)
    fg = tuple(int(v) for v in rng.integers(120, 256, 3))
    n = int(rng.integers(3, 9))
    lh = h / (n + 1)
    for i in range(n):
        # 不用 % 格式化，直接拼随机串（避免 %% 和 %d 混在一起出问题）
        style = int(rng.integers(0, 6))
        if style == 0:
            txt = "IMG_%04d" % int(rng.integers(0, 10000))
        elif style == 1:
            txt = "REC %02d:%02d" % (int(rng.integers(0, 60)), int(rng.integers(0, 60)))
        elif style == 2:
            txt = "%dK %dFPS" % (int(rng.choice([2, 4])), int(rng.choice([24, 30, 60])))
        elif style == 3:
            txt = "ISO %d00" % int(rng.integers(1, 7))
        elif style == 4:
            txt = "BAT %d" % int(rng.integers(10, 100))
        else:
            txt = "ACTION 4"
        y = int((i + 1) * lh)
        cv2.putText(bg, txt, (int(w * 0.05), y), cv2.FONT_HERSHEY_SIMPLEX,
                    lh / 45.0, fg, max(1, int(h / 160)), cv2.LINE_AA)
    return bg


def _tag(w, h, rng):
    """AprilTag 风格：黑边框 + 内部二值格。"""
    s = int(min(w, h) * rng.uniform(0.55, 0.85))
    canvas = np.full((h, w, 3), int(rng.integers(200, 256)), np.uint8)
    x0, y0 = (w - s) // 2, (h - s) // 2
    canvas[y0:y0 + s, x0:x0 + s] = 0
    border = max(2, s // 12)
    inner = s - 2 * border
    g = int(rng.choice([4, 5, 6]))
    cell = inner // g
    bits = rng.integers(0, 2, (g, g))
    for i in range(g):
        for j in range(g):
            if bits[i, j]:
                yy = y0 + border + i * cell
                xx = x0 + border + j * cell
                canvas[yy:yy + cell, xx:xx + cell] = 255
    return canvas


def _viewfinder(w, h, rng):
    """取景画面：模糊的彩色场景 + 取景框/网格线 + 状态条。"""
    small = rng.integers(0, 256, (max(4, h // 24), max(4, w // 24), 3), dtype=np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (0, 0), max(1.0, min(w, h) / 40.0))
    if rng.random() < 0.5:
        for i in range(1, 3):
            cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (255, 255, 255), 1)
            cv2.line(img, (0, h * i // 3), (w, h * i // 3), (255, 255, 255), 1)
    m = int(min(w, h) * 0.12)
    for (a, b, c, d) in ((m, m, m * 2, m), (m, m, m, m * 2),
                         (w - m, m, w - m * 2, m), (w - m, m, w - m, m * 2),
                         (m, h - m, m * 2, h - m), (m, h - m, m, h - m * 2),
                         (w - m, h - m, w - m * 2, h - m), (w - m, h - m, w - m, h - m * 2)):
        cv2.line(img, (a, b), (c, d), (255, 255, 255), max(1, min(w, h) // 80))
    bar = max(2, h // 14)
    img[:bar, :] = (int(rng.integers(0, 60)),) * 3
    return img.astype(np.uint8)


def _menu(w, h, rng):
    """深色菜单 UI：标题 + 若干条白色条目 + 高亮行。"""
    bg = np.full((h, w, 3), int(rng.integers(10, 40)), np.uint8)
    fg = (235, 235, 235)
    rows = int(rng.integers(4, 8))
    lh = h / (rows + 1.5)
    hl = int(rng.integers(0, rows))
    for i in range(rows):
        y = int((i + 1.2) * lh)
        if i == hl:
            cv2.rectangle(bg, (0, int(y - lh * 0.7)), (w, int(y + lh * 0.25)),
                          (int(rng.integers(40, 120)),) * 3, -1)
        cv2.putText(bg, "SETTING %02d" % (i + 1), (int(w * 0.06), y),
                    cv2.FONT_HERSHEY_SIMPLEX, lh / 55.0,
                    (255, 255, 255) if i == hl else fg, max(1, int(h / 180)), cv2.LINE_AA)
    return bg


_MAKERS = {
    "solid": _solid, "checker": _checker, "noise": _noise,
    "gradient": _gradient, "text": _text, "tag": _tag,
    "viewfinder": _viewfinder, "menu": _menu,
}


def make_screen_content(kind: str, w: int, h: int, out_path: str,
                        rng: np.random.Generator | None = None) -> str:
    """生成一张 w×h 的屏幕内容图并写到 out_path。kind='native' 时返回空串。"""
    if kind == NATIVE:
        return ""
    if kind not in _MAKERS:
        raise ValueError(f"unknown screen content kind: {kind!r}; "
                         f"available: {sorted(_MAKERS) + [NATIVE]}")
    rng = rng or np.random.default_rng()
    img = _MAKERS[kind](int(w), int(h), rng)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path


if __name__ == "__main__":
    import sys
    outd = sys.argv[1] if len(sys.argv) > 1 else "."
    rng = np.random.default_rng(0)
    tiles = []
    for k in ["tag", "viewfinder", "menu", "solid", "checker", "noise",
              "gradient", "text"]:
        p = os.path.join(outd, f"preview_{k}.png")
        make_screen_content(k, 512, 320, p, rng)
        tiles.append((k, cv2.imread(p)))
        print(f"  {k:12s} -> {p}")
    # 拼一张预览
    rows = [np.hstack([t[1] for t in tiles[i:i + 3]]) for i in range(0, len(tiles), 3)]
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.vstack(rows)
    cv2.imwrite(os.path.join(outd, "screen_content_preview.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  -> {os.path.join(outd, 'screen_content_preview.jpg')}")
