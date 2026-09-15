"""BoxDreamer 式的训练时增强 —— **训练数据的遮挡与域差异在这里补**。

为什么需要它（见 docs/RESULTS.md）
--------------------------------
v1 的模型在两个维度上崩：

* **遮挡**：完全可见样本的角点失败率 **0.79%**，严重遮挡样本 **45.0%**（差 57 倍）
* **跨域**：合成数据中位误差 1.44 px，真实视频上角点明显跑偏

而训练集里**几乎没有任何遮挡**（`DATA-16`：95% 实例完全可见），
也**没有真实相机的光度特性**。BoxDreamer 正是用训练时增强补这两块的
（`refs/BoxDreamer/src/datasets/utils/aug.py`）：

    rgb_augmethods: ['dark', 'mobile']      # 模拟暗光 / 手机相机
    obj_truncation_ratio: [0.0, 0.2]
    obj_mask_ratio: [0.0, 0.4]              # 抠掉物体 bbox 内一块
    obj_paste_prob: 0.4                     # 把别的物体贴上来

本模块实现其中三项（截断项未实现，我们的裁剪本来就含边界外扩）：

* :func:`apply_dark_aug` / :func:`apply_mobile_aug`
  —— 按 BoxDreamer 的参数范围用 cv2/numpy 重写。
  **刻意不依赖 albumentations**：它在 py3.11 上要编译 `stringzilla`，装不上。
* :func:`mask_bbox_region`
  —— 对应 BoxDreamer 的 ``random_mask_image_with_bbox``：在物体 bbox 内随机取一个
  ``bbox边长 × ratio`` 的矩形，用**别处的像素**替换掉（物体被"挖掉一块"）。
* :func:`paste_foreign_object`
  —— 对应 ``random_paste_obj_with_bbox``：把一个**别的物体**的像素贴到目标上。
  我们直接用**同一帧里的兄弟实例**（`mask_visib` 给 alpha），零额外 I/O 且更真实。

关键约定：**角点标签不因遮挡而改变**（amodal 监督）—— 遮住的部分仍要预测出来，
这正是要教给网络的能力。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 光度增强（BoxDreamer: apply_dark_aug / apply_mobile_aug）
# --------------------------------------------------------------------------- #
def _sample(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _motion_blur(img: np.ndarray, ksize: int, rng: np.random.Generator) -> np.ndarray:
    """线性核做运动模糊（对应 A.MotionBlur）。"""
    ksize = int(max(3, ksize | 1))
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    ang = rng.uniform(0, np.pi)
    cx = cy = ksize // 2
    for t in np.linspace(-cx, cx, ksize * 3):
        x = int(round(cx + t * np.cos(ang)))
        y = int(round(cy + t * np.sin(ang)))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s <= 0:
        return img
    kernel /= s
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = ((np.arange(256) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _hsv_shift(img: np.ndarray, dh: int, ds: int, dv: int) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _iso_noise(img: np.ndarray, rng: np.random.Generator, strength: float = 0.06) -> np.ndarray:
    """近似 A.ISONoise：亮度相关的乘性+加性噪声。

    强度刻意调到**真实传感器的量级**（约 1~3%），而不是照抄 albumentations 的默认
    强度 —— 实测按默认参数会把图糊成雪花，那不是"手机相机"，那是"坏掉的相机"。
    """
    x = img.astype(np.float32) / 255.0
    lum = x.mean(axis=-1, keepdims=True)
    sigma = strength * (0.25 + lum)          # 暗处噪声相对更强
    noise = rng.normal(0.0, 1.0, x.shape) * sigma
    return (np.clip(x + noise, 0, 1) * 255).astype(np.uint8)


def _rain(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """近似 A.RandomRain。

    BoxDreamer 开这一项是为了手机拍户外/雨天的场景；**我们的目标数据是室内**
    （桌面上折衣服），雨丝只会变成不真实的白色斜线。所以这里保留接口但把强度
    压到很低（细、暗、少），当作一般的"杂线/划痕"扰动。
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    for _ in range(int(rng.integers(3, 10))):
        x0 = int(rng.integers(-w // 4, w))
        y0 = int(rng.integers(0, h))
        L = int(rng.integers(h // 10, h // 3))
        overlay = out.copy()
        cv2.line(overlay, (x0, y0), (x0 + L // 5, y0 + L), (200, 205, 215), 1, cv2.LINE_AA)
        out = out * 0.85 + overlay * 0.15
    return out.clip(0, 255).astype(np.uint8)


def apply_dark_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """对应 BoxDreamer 的 ``apply_dark_aug``（整体 p=0.75，各子项独立概率）。

    参数范围照抄：
        RandomBrightnessContrast(p=0.75, brightness_limit=(-0.6, 0.0),
                                 contrast_limit=(-0.5, 0.3))
        Blur(p=0.1, blur_limit=(3, 9))
        MotionBlur(p=0.2, blur_limit=(3, 25))
        RandomGamma(p=0.1, gamma_limit=(15, 65))       # -> gamma 0.15~0.65
        HueSaturationValue(p=0.1, val_shift_limit=(-100, -40))
    """
    if rng.random() >= 0.75:
        return img
    out = img
    if rng.random() < 0.75:
        b = _sample(rng, -0.6, 0.0)          # brightness 相对偏移
        c = _sample(rng, -0.5, 0.3)          # contrast 相对偏移
        x = out.astype(np.float32)
        mean = x.mean()
        x = (x - mean) * (1.0 + c) + mean * (1.0 + b) + 255.0 * b * 0.5
        out = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.1:
        k = int(rng.integers(3, 10))
        out = cv2.GaussianBlur(out, (k | 1, k | 1), 0)
    if rng.random() < 0.2:
        out = _motion_blur(out, int(rng.integers(3, 26)), rng)
    if rng.random() < 0.1:
        out = _gamma(out, _sample(rng, 0.15, 0.65))
    if rng.random() < 0.1:
        out = _hsv_shift(out, 0, 0, int(_sample(rng, -100, -40)))
    return out


def apply_mobile_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """对应 BoxDreamer 的 ``apply_mobile_aug``（整体 p=1.0）。

        MotionBlur(p=0.25), ColorJitter(p=0.5), RandomRain(p=0.1), ISONoise(p=0.25)
    """
    out = img
    if rng.random() < 0.25:
        out = _motion_blur(out, int(rng.integers(3, 26)), rng)
    if rng.random() < 0.5:
        # A.ColorJitter 默认：brightness/contrast/saturation/hue 各 ±0.2，**全局**抖动。
        # ⚠️ 不能写成"逐通道增益" —— 那是色偏（color cast），会渲出整张绿/紫的图，
        #    既不是手机相机的特性，也会把模型带偏（踩过）。
        x = out.astype(np.float32)
        b = _sample(rng, -0.2, 0.2) * 255.0          # brightness
        c = 1.0 + _sample(rng, -0.2, 0.2)            # contrast
        mean = x.mean()
        x = (x - mean) * c + mean + b
        sat = 1.0 + _sample(rng, -0.2, 0.2)          # saturation
        gray = x.mean(axis=-1, keepdims=True)
        x = gray + (x - gray) * sat
        out = np.clip(x, 0, 255).astype(np.uint8)
        out = _hsv_shift(out, int(_sample(rng, -0.2, 0.2) * 180 / 2), 0, 0)   # hue ±0.2
    if rng.random() < 0.1:
        out = _rain(out, rng)
    if rng.random() < 0.25:
        out = _iso_noise(out, rng)
    return out


def apply_rgb_augmentation(
    img: np.ndarray, methods: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    """按顺序应用一组光度增强（对应 BoxDreamer 的 ``apply_rgb_augmentation``）。"""
    out = img
    for m in methods:
        if m == "dark":
            out = apply_dark_aug(out, rng)
        elif m == "mobile":
            out = apply_mobile_aug(out, rng)
        else:
            raise ValueError(f"unknown rgb aug method: {m!r} (expected 'dark' or 'mobile')")
    return out


# --------------------------------------------------------------------------- #
# 遮挡合成
# --------------------------------------------------------------------------- #
def mask_bbox_region(
    img: np.ndarray,
    bbox_xywh: Sequence[float],
    ratio_range: Tuple[float, float],
    rng: np.random.Generator,
    source: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对应 BoxDreamer 的 ``random_mask_image_with_bbox``。

    在物体 bbox 内随机取一个 ``bbox边长 × ratio`` 的矩形，用别处的像素替换掉。
    物体因此被"挖掉一块"——而角点标签**不变**（amodal 监督）。

    Args:
        img:         ``[H, W, 3]`` RGB uint8（已裁剪到网络输入尺寸）
        bbox_xywh:   目标物体的 2D 框（与 img 同一坐标系）
        ratio_range: ``(min, max)``，取 ``ratio ~ U(min, max)``
        source:      替换像素的来源图；``None`` 则从**本图其它位置**随机抠一块
    """
    h, w = img.shape[:2]
    lo, hi = ratio_range
    ratio = float(rng.uniform(lo, hi))
    if ratio <= 1e-3:
        return img

    x, y, bw, bh = [float(v) for v in bbox_xywh]
    mw = max(1, int(bw * ratio))
    mh = max(1, int(bh * ratio))

    # 在 bbox 内随机放这个矩形（照抄 BoxDreamer 的 randint 范围）
    x0 = int(rng.uniform(x, max(x, x + bw - mw)))
    y0 = int(rng.uniform(y, max(y, y + bh - mh)))
    x1, y1 = x0 + mw, y0 + mh

    # 裁到图像内
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(w, x1), min(h, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return img

    out = img.copy()
    if source is None:
        # 从本图随机位置抠一块同尺寸的贴上去（避开目标区域）
        for _ in range(8):
            sx = int(rng.integers(0, max(1, w - mw)))
            sy = int(rng.integers(0, max(1, h - mh)))
            if not (sx < x1 and sx + mw > x0 and sy < y1 and sy + mh > y0):
                break
        patch = img[sy:sy + (iy1 - iy0), sx:sx + (ix1 - ix0)]
        if patch.shape[:2] != (iy1 - iy0, ix1 - ix0):
            patch = cv2.resize(patch, (ix1 - ix0, iy1 - iy0))
    else:
        patch = cv2.resize(source, (ix1 - ix0, iy1 - iy0))
    out[iy0:iy1, ix0:ix1] = patch
    return out


def paste_foreign_object(
    img: np.ndarray,
    obj_rgb: np.ndarray,
    obj_alpha: np.ndarray,
    dst_xy: Tuple[float, float],
    dst_wh: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """对应 BoxDreamer 的 ``random_paste_obj_with_bbox``：把一个别的物体贴到目标上。

    Args:
        img:        底图 ``[H, W, 3]``
        obj_rgb:    要贴的物体 RGB（任意尺寸，会被缩放到 ``dst_wh``）
        obj_alpha:  同尺寸的 alpha（``mask_visib`` 二值图）
        dst_xy:     贴的左上角（与 img 同坐标系）
        dst_wh:     贴的目标尺寸
    """
    h, w = img.shape[:2]
    dw, dh = int(max(2, dst_wh[0])), int(max(2, dst_wh[1]))
    dx, dy = int(dst_xy[0]), int(dst_xy[1])

    rgb = cv2.resize(obj_rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
    a = cv2.resize(obj_alpha, (dw, dh), interpolation=cv2.INTER_NEAREST)
    a = (a > 0).astype(np.float32)[..., None]

    # 目标区域（裁到图内），源图对应子块也一起裁
    sx0, sy0 = max(0, -dx), max(0, -dy)
    dx0, dy0 = max(0, dx), max(0, dy)
    dx1, dy1 = min(w, dx + dw), min(h, dy + dh)
    if dx1 <= dx0 or dy1 <= dy0:
        return img
    sub_rgb = rgb[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    sub_a = a[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    if sub_rgb.shape[:2] != (dy1 - dy0, dx1 - dx0):
        return img

    out = img.copy()
    dst = out[dy0:dy1, dx0:dx1].astype(np.float32)
    out[dy0:dy1, dx0:dx1] = (dst * (1 - sub_a) + sub_rgb.astype(np.float32) * sub_a).astype(np.uint8)
    return out
