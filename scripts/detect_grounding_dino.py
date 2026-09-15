"""用 Grounding DINO 在待测视频里自动检测目标物体（DJI Action 4）。

这是 **BoxDreamer 自己的做法**（`refs/BoxDreamer/src/demo/ov_det.py`）：
文本提示 -> 开放词汇检测器出框（它那边还会接 SAM 2 跨帧传播掩码）。
这里走 `ov_det.py` 里的 **transformers 后备路径**
（`AutoProcessor` + `AutoModelForZeroShotObjectDetection`），
避开原版 `groundingdino` 在 Windows 上的编译坑。

    python scripts/detect_grounding_dino.py +det_out=<dir> +det_frames='0,60,...'
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ROOT = r"D:\AAA_Projects\psd_zju3dv_coding_exam"
VID = os.path.join(ROOT, "head_left_rgb_raw.mp4(1)", "head_left_rgb_raw.mp4")

MODEL_ID = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
# 视频里就是"手腕上的运动相机"，多试几个提示词挑最好的
PROMPTS = ["camera.", "action camera.", "digital camera.", "video camera."]
DET_W = 1600          # 先缩到这么宽再检测（原图 3248 太大，DINO 内部也会缩）


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = str(cfg.get("det_out", os.path.join(ROOT, "data", "results", "stage5_test")))
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gdino] model={MODEL_ID} device={device}")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    print("[gdino] loaded")

    frames = [int(s) for s in str(cfg.get("det_frames", "0,60,120,400")).split(",")]
    box_thr = float(cfg.get("det_box_thr", 0.25))
    txt_thr = float(cfg.get("det_text_thr", 0.20))

    cap = cv2.VideoCapture(VID)
    all_records = {}

    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        s = DET_W / rgb.shape[1]
        small = cv2.resize(rgb, (DET_W, int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(small)

        print("=" * 70)
        print(f"frame {fi}")
        best = None
        for prompt in PROMPTS:
            inputs = proc(images=pil, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=box_thr, text_threshold=txt_thr,
                target_sizes=[(small.shape[0], small.shape[1])])[0]
            n = len(res["boxes"])
            if n:
                sc = [round(float(x), 3) for x in res["scores"]]
                print(f"  {prompt:18s} {n} boxes  scores={sc}")
            else:
                print(f"  {prompt:18s} 0 boxes")
            if best is None or n > len(best[1]):
                best = (prompt, [b.tolist() for b in res["boxes"]],
                        [float(x) for x in res["scores"]])

        prompt, boxes, scores = best
        # 缩回原图坐标
        boxes = [[b[0] / s, b[1] / s, b[2] / s, b[3] / s] for b in boxes]
        all_records[fi] = {"prompt": prompt,
                           "boxes": [[round(v, 1) for v in b] for b in boxes],
                           "scores": [round(v, 3) for v in scores]}

        im = Image.fromarray(rgb)
        sc2 = 1500.0 / im.width
        im = im.resize((1500, int(im.height * sc2)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        for k, (b, sc) in enumerate(zip(boxes, scores)):
            d.rectangle([b[0] * sc2, b[1] * sc2, b[2] * sc2, b[3] * sc2],
                        outline=(255, 60, 60), width=4)
            d.rectangle([b[0] * sc2, b[1] * sc2, b[0] * sc2 + 200, b[1] * sc2 + 18],
                        fill=(0, 0, 0))
            d.text((b[0] * sc2 + 4, b[1] * sc2 + 3), f"#{k} {sc:.2f} {prompt}",
                   fill=(255, 220, 60))
        d.rectangle([0, 0, 420, 18], fill=(0, 0, 0))
        d.text((4, 3), f"frame {fi}  prompt='{prompt}'  box_thr={box_thr}",
               fill=(255, 255, 120))
        p = os.path.join(out_dir, f"gdino_f{fi:05d}.jpg")
        im.save(p, quality=90)
        print(f"  -> {p}")

    cap.release()
    with open(os.path.join(out_dir, "gdino_detections.json"), "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"[gdino] {len(all_records)} frames written")


if __name__ == "__main__":
    main()
