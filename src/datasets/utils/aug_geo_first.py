"""重几何轻外观增强 —— 只扰动【外观】，几何一步不动。

这条原则的由来（见 docs/RESULTS.md）
-----------------------------------
v1 的模型在真实视频上崩，而目标的**外观**和训练数据差很多：
参考照片里屏幕是熄的（深蓝无纹理），视频里屏幕**亮着白色 AprilTag 棋盘格**，
还多了腕带底座。屏幕是正面最大的高对比度面，它的图案一变，
网络赖以定位的纹理线索就全废了。

**但 8 个角点是「机身 3D 盒」的角点 —— 所有附件都不改变它。**
所以这是**域差异**，不是标签错。办法就是让外观变得**不可依赖**，
逼网络只靠**形状**。

⚠️ 关键区分（v3 那次我搞反了）
----------------------------
    "让外观不可依赖"  ->  扰动【外观】          ->  ✅ 对，几何完整保留
    "把图弄难"        ->  遮挡【物体的一部分】  ->  ❌ 错，这是破坏几何

v3 照抄 BoxDreamer 的 obj_mask_ratio 0~40%，是**删几何信息**，
中位误差从 1.44 炸到 8.35。**遮挡增强在单图设定下是有害的**
（BoxDreamer 是 6 帧，遮掉的在别的帧里还看得见）。

本模块只做外观：颜色 / 亮度 / 对比度 / gamma / 色调 / 模糊 / 噪声 / 灰度。
**刻意不做**：遮挡、贴别的物体、任何几何变形、任何会裁掉物体边界的操作。
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["augment_image_geo_first"]


def _motion_blur(img: np.ndarray, ksize: int, rng: np.random.Generator) -> np.ndarray:
    ksize = int(max(3, ksize | 1))
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    ang = rng.uniform(0, np.pi)
    c = ksize // 2
    for t in np.linspace(-c, c, ksize * 3):
        x = int(round(c + t * np.cos(ang)))
        y = int(round(c + t * np.sin(ang)))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s <= 0:
        return img
    return cv2.filter2D(img, -1, kernel / s, borderType=cv2.BORDER_REFLECT)


def _hsv_shift(img: np.ndarray, dh: int, ds: int, dv: int) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_image_geo_first(
    image: np.ndarray,
    color_jitter: float = 0.22,
    blur_prob: float = 0.35,
    noise_std: float = 0.030,
    grayscale_prob: float = 0.18,
    seed: int | None = None,
) -> np.ndarray:
    """重几何轻外观：只扰动外观，几何一步不动。

    Args:
        color_jitter: 白平衡的**暖/冷**偏移幅度（相关三通道，不是逐通道独立）。
        blur_prob:    高斯/运动模糊概率。
        noise_std:    ISO 噪声强度（相对 255）。
        grayscale_prob: 灰度化概率 —— 彻底废掉颜色线索。
        seed:         仅用于测试复现。

    为什么白平衡用"相关偏移"而不是"逐通道独立增益"
    ----------------------------------------------
    逐通道独立增益会渲出洋红/亮绿这种**现实中不存在**的色偏。
    那不是在模拟相机，而是在**把图弄坏** —— 模型只会学到"这图没法看"，
    而不是"颜色不重要，该看形状"。真实相机的白平衡漂移是**色温方向**的：
    偏暖时 R↑B↓，偏冷时 R↓B↑，G 基本不动。
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    img = image.astype(np.float32)

    # ---- ① 白平衡：暖/冷方向的相关偏移（真实相机的行为）----
    if color_jitter > 0:
        t = rng.uniform(-1.0, 1.0)                    # -1 冷 / +1 暖
        g_r = 1.0 + color_jitter * t
        g_b = 1.0 - color_jitter * t
        img[..., 0] *= g_r
        img[..., 2] *= g_b

    # ---- ② 整体曝光增益（±20%）----
    if rng.random() < 0.80:
        img *= 1.0 + rng.uniform(-0.20, 0.20)

    # ---- ③ 灰度化（概率触发）—— 把颜色线索彻底废掉 ----
    if grayscale_prob > 0 and rng.random() < grayscale_prob:
        g = img.mean(axis=-1, keepdims=True)
        img = np.repeat(g, 3, axis=-1)

    # ---- ④ 亮度 / 对比度 ----
    if rng.random() < 0.85:
        b = rng.uniform(-0.22, 0.15) * 255.0
        c = 1.0 + rng.uniform(-0.35, 0.35)
        m = float(img.mean())
        img = (img - m) * c + m + b

    # ---- ⑤ gamma（暗部/亮部非线性）----
    if rng.random() < 0.30:
        gam = float(rng.uniform(0.60, 1.60))
        img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gam)

    # ---- ⑥ 饱和度偏移（不转色相，避免不真实）----
    if rng.random() < 0.30:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        img = _hsv_shift(img_u8, 0, int(rng.uniform(-35, 35)), 0).astype(np.float32)

    img_u8 = np.clip(img, 0, 255).astype(np.uint8)

    # ---- ⑦ 模糊 / 运动模糊（手持视频的主要退化）----
    if blur_prob > 0 and rng.random() < blur_prob:
        if rng.random() < 0.5:
            k = int(rng.choice([3, 5, 7]))
            img_u8 = cv2.GaussianBlur(img_u8, (k, k), 0)
        else:
            img_u8 = _motion_blur(img_u8, int(rng.integers(3, 13)), rng)

    # ---- ⑧ ISO / 传感器噪声 ----
    if noise_std > 0:
        x = img_u8.astype(np.float32)
        x = x + rng.normal(0.0, noise_std * 255.0, x.shape).astype(np.float32)
        img_u8 = np.clip(x, 0, 255).astype(np.uint8)

    return img_u8
