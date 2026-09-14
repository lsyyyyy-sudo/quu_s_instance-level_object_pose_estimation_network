"""第 5 步：把训练好的网络跑在**目标视频**上，输出 8 个角点。

流程（对应 photo_of_the_project.png 的最后两步）::

    视频帧 ──[框]──▶ 裁剪 256x256 ──[网络]──▶ 8 角点热图 ──▶ 8 个 2D 角点
                                                        └─▶ 投回原图坐标

**关于"框"**：我们的网络只负责「给定裁剪图 → 8 个角点」（训练时用的是 GT 的
`bbox_visib`）。真系统里这个框由**检测器**提供；本脚本先用**手工标注的框**
（`BOXES`）做演示，这是第 5 步最小可用的形态。
见 docs/TRAINING.md：本项目不需要相机内参，所以后面不跑 PnP。

用法::

    python scripts/demo_video.py +demo_out=<dir> [demo_frames=0,1200,2841]
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.bop_pbr import crop_and_resize  # noqa: E402
from src.models.utils.prediction_utils import predict_corners_and_pose  # noqa: E402

ROOT = r"D:\AAA_Projects\psd_zju3dv_coding_exam"
VID = os.path.join(ROOT, "head_left_rgb_raw.mp4(1)", "head_left_rgb_raw.mp4")

# 手工标注的框（原图坐标 x0,y0,x1,y1）。视频 3248x2464。
BOXES = {
    0:    [(1002, 1022, 1668, 1592), (2062, 992, 2738, 1508)],
    2841: [(2360, 1660, 2900, 2260)],
}


def box_to_xywh(b):
    x0, y0, x1, y1 = b
    return [x0, y0, x1 - x0, y1 - y0]


def crop_to_orig(corner_crop: np.ndarray, bbox_xywh, out_size: int, crop_scale: float):
    """把裁剪图坐标系里的角点还原到原图坐标（与 crop_and_resize 完全逆运算）。"""
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(max(w, h) * float(crop_scale), 1.0)
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    scale = float(out_size) / side
    return corner_crop / scale + np.array([x0, y0], dtype=np.float64)


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = str(cfg.get("demo_out", os.path.join(ROOT, "data", "results", "stage5_test")))
    os.makedirs(out_dir, exist_ok=True)

    # ---- 模型 ----
    model = instantiate(cfg.model, _recursive_=False)
    ckpt_path = cfg.model.pretrained_ckpt
    print(f"[demo] checkpoint = {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[demo] missing={len(missing)} unexpected={len(unexpected)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    inner = model.model
    print(f"[demo] device={device}  extraction={inner.extraction}  "
          f"image_size={inner.image_size} heatmap_size={inner.heatmap_size}")

    size = inner.image_size
    crop_scale = float(cfg.datamodule.crop_scale)
    K_dummy = np.array([[800.0, 0, size / 2], [0, 800.0, size / 2], [0, 0, 1]])

    cap = cv2.VideoCapture(VID)
    frames = [int(s) for s in str(cfg.get("demo_frames", "0,2841")).split(",")]

    panels = []
    records = []
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"[demo] frame {fi} read failed")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for bi, b in enumerate(BOXES.get(fi, [])):
            xywh = box_to_xywh(b)
            crop, _ = crop_and_resize(rgb, K_dummy, xywh, size, crop_scale=crop_scale)
            t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = t.to(device)
            with torch.no_grad():
                out = model({"image": t})
                preds = predict_corners_and_pose(
                    out["pred_heatmap"],
                    torch.zeros(1, 8, 3, device=device),   # 不需要 bbox_3d（不跑 PnP）
                    torch.from_numpy(K_dummy).float().unsqueeze(0).to(device),
                    image_size=size, heatmap_size=inner.heatmap_size,
                    method=inner.extraction, beta=inner.soft_argmax_beta, k=inner.topk,
                    heatmap_range=inner.heatmap_range, solve_pose=False,
                )
            c_crop = preds["corner_2d"][0].cpu().numpy().astype(np.float64)
            c_orig = crop_to_orig(c_crop, xywh, size, crop_scale)
            records.append({"frame": fi, "box_index": bi, "box": list(b),
                            "corner_orig": c_orig.round(1).tolist()})
            print(f"[demo] frame {fi} box{bi}: "
                  f"corner x[{c_orig[:,0].min():.0f},{c_orig[:,0].max():.0f}] "
                  f"y[{c_orig[:,1].min():.0f},{c_orig[:,1].max():.0f}]")

            # ---- 面板：原图 + 框 + 角点 (左)  |  裁剪图 + 角点 (右) ----
            sc = 1000.0 / rgb.shape[1]
            base = Image.fromarray(rgb).resize(
                (1000, int(rgb.shape[0] * sc)), Image.LANCZOS)
            d = ImageDraw.Draw(base)
            d.rectangle([b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc],
                        outline=(0, 255, 255), width=3)
            for k in range(8):
                ox, oy = c_orig[k] * sc
                d.line([ox - 7, oy, ox + 7, oy], fill=(255, 40, 40), width=4)
                d.line([ox, oy - 7, ox, oy + 7], fill=(255, 40, 40), width=4)
                d.text((ox + 8, oy - 8), str(k), fill=(255, 220, 60))
            # 8 点连线，方便看框形
            for k in range(4):
                d.line([c_orig[k][0] * sc, c_orig[k][1] * sc,
                        c_orig[(k + 1) % 4][0] * sc, c_orig[(k + 1) % 4][1] * sc],
                       fill=(255, 120, 0), width=2)
                d.line([c_orig[4 + k][0] * sc, c_orig[4 + k][1] * sc,
                        c_orig[4 + (k + 1) % 4][0] * sc, c_orig[4 + (k + 1) % 4][1] * sc],
                       fill=(255, 120, 0), width=2)

            cim = Image.fromarray(crop).resize((size * 2, size * 2), Image.LANCZOS)
            dc = ImageDraw.Draw(cim)
            for k in range(8):
                ox, oy = c_crop[k] * 2
                dc.line([ox - 8, oy, ox + 8, oy], fill=(255, 40, 40), width=3)
                dc.line([ox, oy - 8, ox, oy + 8], fill=(255, 40, 40), width=3)
                dc.text((ox + 9, oy - 9), str(k), fill=(255, 220, 60))

            ph = max(base.height, cim.height)
            panel = Image.new("RGB", (base.width + cim.width + 24, ph + 22), (14, 14, 16))
            dd = ImageDraw.Draw(panel)
            dd.text((4, 4), f"frame {fi}  box{bi}  (cyan) + predicted 8 corners (red)",
                    fill=(235, 235, 235))
            dd.text((base.width + 16, 4), f"crop {size}->{size*2}  corners", fill=(235, 235, 235))
            panel.paste(base, (0, 20))
            panel.paste(cim, (base.width + 12, 20))
            panels.append(panel)

    cap.release()

    if panels:
        W = max(p.width for p in panels)
        H = sum(p.height for p in panels) + 12 * (len(panels) + 1)
        sheet = Image.new("RGB", (W + 24, H), (10, 10, 12))
        y = 12
        for p in panels:
            sheet.paste(p, (12, y))
            y += p.height + 12
        outp = os.path.join(out_dir, "stage5_corners.jpg")
        sheet.save(outp, quality=92)
        print(f"[demo] wrote {outp}  {sheet.size}")

    with open(os.path.join(out_dir, "stage5_corners.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[demo] wrote {len(records)} records")


if __name__ == "__main__":
    main()
