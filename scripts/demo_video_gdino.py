"""第 5 步（完整版）：Grounding DINO 出框 -> 我们的网络出 8 角点。

这是 **BoxDreamer 自己的流程**：
    refs/BoxDreamer/src/demo/ov_det.py  ->  Grounding DINO（transformers 路径）出框
    （它那边还会接 SAM 2 跨帧传播掩码；我们只需要框，所以省掉 SAM）

之前我手搓的"暗色阈值 / 光流跟踪"是这套东西的劣质替代 —— 实测检测框明显更准。

检测结果会缓存到 `<out>/gdino_boxes.json`，重跑不用再检测。

    python scripts/demo_video_gdino.py +demo_out=<dir>
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
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.bop_pbr import crop_and_resize  # noqa: E402
from src.models.utils.prediction_utils import predict_corners_and_pose  # noqa: E402

ROOT = r"D:\AAA_Projects\psd_zju3dv_coding_exam"
VID = os.path.join(ROOT, "head_left_rgb_raw.mp4(1)", "head_left_rgb_raw.mp4")
MODEL_ID = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
PROMPTS = ["camera.", "action camera."]
DET_W = 1600


def detect_boxes(cap, frames, device, box_thr=0.25, txt_thr=0.20):
    print(f"[gdino] loading {MODEL_ID} on {device}")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    out = {}
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        s = DET_W / rgb.shape[1]
        small = cv2.resize(rgb, (DET_W, int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(small)
        boxes, scores = [], []
        for prompt in PROMPTS:
            inputs = proc(images=pil, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                o = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                o, inputs.input_ids, threshold=box_thr, text_threshold=txt_thr,
                target_sizes=[(small.shape[0], small.shape[1])])[0]
            for b, sc in zip(res["boxes"], res["scores"]):
                boxes.append([float(v) / s for v in b.tolist()])
                scores.append(float(sc))
        # 去重（同一目标可能被两个 prompt 各检一次）
        keep = []
        for b, sc in sorted(zip(boxes, scores), key=lambda t: -t[1]):
            if all(_iou(b, k[0]) < 0.5 for k in keep):
                keep.append((b, sc))
        out[fi] = [{"box": [round(v, 1) for v in b], "score": round(sc, 3)} for b, sc in keep]
        print(f"[gdino] frame {fi}: {len(keep)} boxes  "
              f"scores={[round(sc, 3) for _, sc in keep]}")
    return out


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def crop_to_orig(c, xywh, size, crop_scale):
    x, y, w, h = [float(v) for v in xywh]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(max(w, h) * float(crop_scale), 1.0)
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    return c / (float(size) / side) + np.array([x0, y0])


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = str(cfg.get("demo_out", os.path.join(ROOT, "data", "results", "stage5_test")))
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "gdino_boxes.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = [int(s) for s in str(cfg.get("demo_frames",
                                          "0,60,120,200,400,700,1000,1400,1800,2200,2600,2841")).split(",")]
    cap = cv2.VideoCapture(VID)

    if os.path.isfile(cache):
        det = {int(k): v for k, v in json.load(open(cache, encoding="utf-8")).items()}
        print(f"[gdino] 用缓存 {cache}（{len(det)} 帧）")
    else:
        det = detect_boxes(cap, frames, device)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in det.items()}, f, indent=2)
        print(f"[gdino] wrote {cache}")

    # ---- 网络 ----
    model = instantiate(cfg.model, _recursive_=False)
    ckpt = torch.load(cfg.model.pretrained_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    inner = model.model
    size = inner.image_size
    crop_scale = float(cfg.datamodule.crop_scale)
    K = np.array([[800.0, 0, size / 2], [0, 800.0, size / 2], [0, 0, 1]])
    print(f"[corner] device={device} extraction={inner.extraction}")

    panels, records = [], []
    for fi in frames:
        if fi not in det:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for bi, d in enumerate(det[fi]):
            x0, y0, x1, y1 = d["box"]
            xywh = [x0, y0, x1 - x0, y1 - y0]
            if xywh[2] < 40 or xywh[3] < 40:
                continue
            crop, _ = crop_and_resize(rgb, K, xywh, size, crop_scale=crop_scale)
            t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                o = model({"image": t})
                pr = predict_corners_and_pose(
                    o["pred_heatmap"], torch.zeros(1, 8, 3, device=device),
                    torch.from_numpy(K).float().unsqueeze(0).to(device),
                    image_size=size, heatmap_size=inner.heatmap_size,
                    method=inner.extraction, beta=inner.soft_argmax_beta, k=inner.topk,
                    heatmap_range=inner.heatmap_range, solve_pose=False)
            c_crop = pr["corner_2d"][0].cpu().numpy().astype(np.float64)
            c_orig = crop_to_orig(c_crop, xywh, size, crop_scale)
            records.append({"frame": fi, "det_index": bi, "score": d["score"],
                            "box": d["box"], "corner_orig": c_orig.round(1).tolist()})

            Z = 2
            cim = Image.fromarray(crop).resize((size * Z, size * Z), Image.LANCZOS)
            dd = ImageDraw.Draw(cim)
            for a, b2 in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]:
                dd.line([c_crop[a][0] * Z, c_crop[a][1] * Z,
                         c_crop[b2][0] * Z, c_crop[b2][1] * Z], fill=(255, 140, 0), width=2)
            for a, b2 in [(0, 4), (1, 5), (2, 6), (3, 7)]:
                dd.line([c_crop[a][0] * Z, c_crop[a][1] * Z,
                         c_crop[b2][0] * Z, c_crop[b2][1] * Z], fill=(0, 200, 255), width=2)
            for k in range(8):
                ox, oy = c_crop[k] * Z
                dd.line([ox - 8, oy, ox + 8, oy], fill=(255, 40, 40), width=3)
                dd.line([ox, oy - 8, ox, oy + 8], fill=(255, 40, 40), width=3)
                dd.text((ox + 9, oy - 9), str(k), fill=(255, 235, 80))
            hdr = Image.new("RGB", (cim.width, 20), (16, 16, 20))
            ImageDraw.Draw(hdr).text(
                (4, 3), f"f{fi} det{bi} score={d['score']} box={int(xywh[2])}x{int(xywh[3])}",
                fill=(235, 235, 235))
            p = Image.new("RGB", (cim.width, cim.height + 20), (16, 16, 20))
            p.paste(hdr, (0, 0)); p.paste(cim, (0, 20))
            panels.append(p)

    cap.release()
    if panels:
        cols = 6
        rows = (len(panels) + cols - 1) // cols
        pw, ph = panels[0].size
        sheet = Image.new("RGB", (cols * pw + (cols + 1) * 8, rows * ph + (rows + 1) * 8), (10, 10, 12))
        for i, p in enumerate(panels):
            r, c = divmod(i, cols)
            sheet.paste(p, (8 + c * (pw + 8), 8 + r * (ph + 8)))
        outp = os.path.join(out_dir, "stage5_gdino_corners.jpg")
        sheet.save(outp, quality=92)
        print(f"[out] wrote {outp}  {sheet.size}  ({len(panels)} panels)")
    with open(os.path.join(out_dir, "stage5_gdino_corners.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[out] {len(records)} records")


if __name__ == "__main__":
    main()
