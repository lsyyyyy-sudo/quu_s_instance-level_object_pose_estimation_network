"""第 5 步（多帧版）：从首帧验证过的框出发，用光流跟踪往前推，逐帧跑网络出角点。

为什么这么做：视频里没有检测器，而手工估坐标容易出错（之前就因为预览图二次缩放
读偏过）。首帧两台相机又大又清晰，框已经人工验证过；LK 光流 + 仿射估计能把框
稳定地推到后续帧。

    python scripts/demo_video_track.py +demo_out=<dir>
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

# 首帧人工验证过的两台相机框 (x0,y0,x1,y1)
SEED = {0: [(1002, 1022, 1668, 1592), (2062, 992, 2738, 1508)]}
# 额外单帧（片尾竖大拇指，左上腕一台）
EXTRA = {2841: [(2360, 1660, 2900, 2260)]}


def track_box(gray_prev, gray_cur, box, pts_prev=None, redetect=False):
    """用 LK 光流 + 仿射估计把框推一帧。

    性能要点：特征点只在必要时重检（`pts_prev` 复用），否则每帧都跑
    goodFeaturesToTrack 会慢到不可用（2841 帧实测跑不完）。
    """
    x0, y0, x1, y1 = box
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(gray_prev.shape[1] - 1, int(x1)); y1 = min(gray_prev.shape[0] - 1, int(y1))

    p0 = pts_prev
    if p0 is None or len(p0) < 8 or redetect:
        m = np.zeros(gray_prev.shape, np.uint8)
        m[y0:y1, x0:x1] = 255
        p0 = cv2.goodFeaturesToTrack(gray_prev, mask=m, maxCorners=200,
                                     qualityLevel=0.01, minDistance=10)
        if p0 is None or len(p0) < 8:
            return box, None, False
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_cur, p0, None)
    if p1 is None:
        return box, None, False
    ok = st.ravel() == 1
    a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
    if len(a) < 8:
        return box, None, False
    M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                       ransacReprojThreshold=3.0)
    if M is None:
        return box, None, False
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    tp = (M @ np.hstack([pts, np.ones((4, 1), np.float32)]).T).T
    nx0, ny0 = tp.min(axis=0)
    nx1, ny1 = tp.max(axis=0)
    h, w = gray_cur.shape
    nb = (int(np.clip(nx0, 0, w - 2)), int(np.clip(ny0, 0, h - 2)),
          int(np.clip(nx1, 1, w - 1)), int(np.clip(ny1, 1, h - 1)))
    if nb[2] - nb[0] < 30 or nb[3] - nb[1] < 30:
        return box, None, False
    # 用变换后的点作为下一帧的输入，省掉重检
    return nb, b.reshape(-1, 1, 2).astype(np.float32), True


def box_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


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

    model = instantiate(cfg.model, _recursive_=False)
    ckpt = torch.load(cfg.model.pretrained_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    inner = model.model
    size = inner.image_size
    crop_scale = float(cfg.datamodule.crop_scale)
    K = np.array([[800.0, 0, size / 2], [0, 800.0, size / 2], [0, 0, 1]])
    print(f"[track] device={device} extraction={inner.extraction}")

    targets = [int(s) for s in str(cfg.get("demo_frames", "0,40,80,120,160")).split(",")]
    cap = cv2.VideoCapture(VID)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- 跟踪：从 0 号帧推到每个目标帧 ----
    # 用半分辨率做跟踪：像素少 4 倍，速度大约 4 倍，精度对推框足够。
    SC = 0.5
    boxes_by_frame = {0: list(SEED[0])}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, prev = cap.read()
    gp = cv2.cvtColor(cv2.resize(prev, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY)
    cur = [(b[0] * SC, b[1] * SC, b[2] * SC, b[3] * SC) for b in boxes_by_frame[0]]
    pts = [None] * len(cur)
    want = sorted(set(t for t in targets if 0 < t < n_total))
    last = max(want) if want else 1
    for fi in range(1, last + 1):
        ok, fr = cap.read()
        if not ok:
            break
        gc = cv2.cvtColor(cv2.resize(fr, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA),
                          cv2.COLOR_BGR2GRAY)
        new, npts = [], []
        for bi, b in enumerate(cur):
            redetect = (fi % 120 == 0)          # 每 120 帧重检一次，防止点跑光
            nb, pb, good = track_box(gp, gc, b, pts[bi], redetect=redetect)
            new.append(nb)
            npts.append(pb)
            if not good and fi in want:
                print(f"[track] frame {fi} box{bi}: 跟踪退化，沿用上一帧")
        cur, pts = new, npts
        gp = gc
        if fi in want:
            boxes_by_frame[fi] = [(b[0] / SC, b[1] / SC, b[2] / SC, b[3] / SC)
                                  for b in cur]
    print(f"[track] 得到 {len(boxes_by_frame)} 个帧的框")

    # ---- 逐帧推理 ----
    panels, records = [], []
    for fi in targets:
        boxes = boxes_by_frame.get(fi, EXTRA.get(fi, []))
        if not boxes:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for bi, b in enumerate(boxes):
            xywh = box_to_xywh(b)
            crop, _ = crop_and_resize(rgb, K, xywh, size, crop_scale=crop_scale)
            t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                out = model({"image": t})
                preds = predict_corners_and_pose(
                    out["pred_heatmap"], torch.zeros(1, 8, 3, device=device),
                    torch.from_numpy(K).float().unsqueeze(0).to(device),
                    image_size=size, heatmap_size=inner.heatmap_size,
                    method=inner.extraction, beta=inner.soft_argmax_beta, k=inner.topk,
                    heatmap_range=inner.heatmap_range, solve_pose=False)
            c_crop = preds["corner_2d"][0].cpu().numpy().astype(np.float64)
            c_orig = crop_to_orig(c_crop, xywh, size, crop_scale)
            records.append({"frame": fi, "box_index": bi, "box": list(b),
                            "corner_orig": c_orig.round(1).tolist()})

            # 面板：裁剪图（放大 2.5x）+ 角点
            Z = 2
            cim = Image.fromarray(crop).resize((size * Z, size * Z), Image.LANCZOS)
            d = ImageDraw.Draw(cim)
            order = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]
            for a, bb in order:
                col = (255, 140, 0) if (a < 4 and bb < 4) or (a >= 4 and bb >= 4) else (0, 200, 255)
                d.line([c_crop[a][0] * Z, c_crop[a][1] * Z,
                        c_crop[bb][0] * Z, c_crop[bb][1] * Z], fill=col, width=2)
            for k in range(8):
                ox, oy = c_crop[k] * Z
                d.line([ox - 8, oy, ox + 8, oy], fill=(255, 40, 40), width=3)
                d.line([ox, oy - 8, ox, oy + 8], fill=(255, 40, 40), width=3)
                d.text((ox + 9, oy - 9), str(k), fill=(255, 235, 80))
            hdr = Image.new("RGB", (cim.width, 20), (16, 16, 20))
            ImageDraw.Draw(hdr).text((4, 3), f"frame {fi}  box{bi}  {b[2]-b[0]}x{b[3]-b[1]}",
                                     fill=(235, 235, 235))
            p = Image.new("RGB", (cim.width, cim.height + 20), (16, 16, 20))
            p.paste(hdr, (0, 0)); p.paste(cim, (0, 20))
            panels.append(p)

    cap.release()
    if panels:
        cols = 5
        rows = (len(panels) + cols - 1) // cols
        pw, ph = panels[0].size
        sheet = Image.new("RGB", (cols * pw + (cols + 1) * 8, rows * ph + (rows + 1) * 8), (10, 10, 12))
        for i, p in enumerate(panels):
            r, c = divmod(i, cols)
            sheet.paste(p, (8 + c * (pw + 8), 8 + r * (ph + 8)))
        outp = os.path.join(out_dir, "stage5_tracked.jpg")
        sheet.save(outp, quality=93)
        print(f"[track] wrote {outp}  {sheet.size}")
    with open(os.path.join(out_dir, "stage5_tracked.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[track] {len(records)} records")


if __name__ == "__main__":
    main()
