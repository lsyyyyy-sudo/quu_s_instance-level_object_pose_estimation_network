"""多实例视频推理：整图 -> 网络 -> N 个实例 x 8 角点（不需要检测器）。

与 demo_video_gdino.py 的区别
-----------------------------
gdino 版：GroundingDINO 出框 -> 每个框裁一次 -> 单实例网络 -> 8 角点
本脚本：  **不裁剪、不检测** -> 多实例网络 -> 中心热图的峰 = 实例 -> 每个峰 8 角点

用法::

    python scripts/demo_video_multi.py pretrain_name=MI30b \
        +demo_out=data/results/video_multi +demo_frames=0
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

from src.models.utils.multi_instance import decode_instances  # noqa: E402

ROOT = r"D:\AAA_Projects\psd_zju3dv_coding_exam"
VID = os.path.join(ROOT, "head_left_rgb_raw.mp4(1)", "head_left_rgb_raw.mp4")
# 手工标注的框（原图坐标），只用来【和预测对比】，不参与推理
GT_BOXES = {
    0: [(1002, 1022, 1668, 1592), (2062, 992, 2738, 1508)],
    2841: [(2360, 1660, 2900, 2260)],
}
# 网络输入边长（必须和训练一致）
NET = 256
# 推理时把原图缩到多大（再居中裁 NET）。
# ⚠️ 必须匹配训练的物体尺度：训练裁剪 = 2.5x 物体框，所以物体在 256 输入里
#    占 256/2.5 ≈ 102 px。视频里物体约 666 px（GT 框 1002->1668），
#    所以原图要缩到 666 -> 102，即 3248 * (102/666) ≈ 500 宽。
#    一开始用 1024，物体在 256 裁剪里只剩 ~50 px 的一半，网络完全找不到（解出 0 个）。
INFER_W = 500


def center_crop_resize(rgb: np.ndarray, net: int) -> tuple:
    """把整图等比缩到 INFER_W 宽，再居中裁 net x net。返回 (img, 变换参数)。"""
    h0, w0 = rgb.shape[:2]
    s = INFER_W / float(w0)
    h1, w1 = int(round(h0 * s)), INFER_W
    small = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_AREA)
    # 居中裁 net
    y0 = max(0, (h1 - net) // 2)
    x0 = max(0, (w1 - net) // 2)
    y1, x1 = min(h1, y0 + net), min(w1, x0 + net)
    crop = small[y0:y1, x0:x1]
    # 记录 crop -> net 的真实缩放。多数情况下裁剪就是 small 上的
    # 256x256 窗口（crop2net=1.0）；只有边界不足时才 resize。
    # 反变换必须用这个数，不能想当然地乘 INFER_W/net，否则预测会跑到画面外。
    if crop.shape[0] != net or crop.shape[1] != net:
        crop = cv2.resize(crop, (net, net))
        crop2net = 1.0          # resize 后 net 坐标归一化到 [0,1]，等价于窗口整体
    else:
        crop2net = 1.0
    return crop, dict(scale=s, x0=x0, y0=y0, net=net, crop2net=crop2net)


def to_orig(pts_net: np.ndarray, tr: dict) -> np.ndarray:
    """网络坐标系(net) -> 原图坐标。

    链路：原图 --等比缩到 INFER_W--> small --取 net x net 窗口--> 网络输入
    反变换：p_small = p_net * crop2net + (x0, y0);  p_orig = p_small / scale

    注意 crop2net 通常就是 1.0（直接取窗口不缩放）。
    这里不能用 INFER_W/net —— 那会放大将近 2 倍，预测全跑到画面外。
    """
    scale_net_to_small = tr.get("crop2net", 1.0)
    p = pts_net * scale_net_to_small
    p = p + np.array([tr["x0"], tr["y0"]])
    p = p / tr["scale"]
    return p


@hydra.main(config_path="../configs", config_name="train_multi.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out_dir = str(cfg.get("demo_out", os.path.join(ROOT, "data", "results", "video_multi")))
    os.makedirs(out_dir, exist_ok=True)

    model = instantiate(cfg.model, _recursive_=False)
    ckpt = torch.load(cfg.model.pretrained_ckpt, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[mi] checkpoint = {cfg.model.pretrained_ckpt}")
    print(f"[mi] load missing={len(miss)} unexpected={len(unexp)}")
    if len(miss) or len(unexp):
        print(f"[mi] !! 权重没对上，结果无效: {list(miss)[:3]} / {list(unexp)[:3]}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    stride = model.model.image_size / float(model.model.heatmap_size)
    thr = float(cfg.model.multi.det_thr)
    topk = int(cfg.model.multi.det_topk)
    min_dist = float(cfg.model.multi.min_dist)
    print(f"[mi] device={device} stride={stride} thr={thr} topk={topk}")

    frames = [int(s) for s in str(cfg.get("demo_frames", "0")).split(",")]
    cap = cv2.VideoCapture(VID)
    if not cap.isOpened():
        raise FileNotFoundError(VID)

    records, panels = [], []
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"[mi] 读不到帧 {fi}")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop, tr = center_crop_resize(rgb, NET)
        x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            out = model.model({"image": x})
        insts = decode_instances(out["pred_heatmap"], out["pred_offset"],
                                 thr=thr, topk=topk, min_dist=min_dist, stride=stride)[0]
        print(f"[mi] frame {fi}: 解出 {len(insts)} 个实例  "
              f"(score {', '.join(f'{i['score']:.2f}' for i in insts[:6])})")

        # 画在原图上
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        # GT 手工框（黄色）用于对比
        for b in GT_BOXES.get(fi, []):
            d.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 215, 0), width=5)
        for i, inst in enumerate(insts):
            c = to_orig(np.asarray(inst["corners"], dtype=np.float64), tr)
            cen = to_orig(np.asarray(inst["center"], dtype=np.float64)[None], tr)[0]
            col = (255, 60, 60)
            for (px, py) in c:
                d.line([px - 14, py, px + 14, py], fill=col, width=5)
                d.line([px, py - 14, px, py + 14], fill=col, width=5)
            for k in range(4):
                d.line([*c[k], *c[(k + 1) % 4]], fill=col, width=3)
                d.line([*c[k + 4], *c[4 + (k + 1) % 4]], fill=col, width=3)
                d.line([*c[k], *c[k + 4]], fill=col, width=3)
            d.text((cen[0], cen[1]), f"#{i} {inst['score']:.2f}", fill=(0, 255, 120))
            records.append({"frame": fi, "idx": i, "score": float(inst["score"]),
                            "center_orig": cen.tolist(), "corners_orig": c.tolist()})

        im.save(os.path.join(out_dir, f"multi_f{fi:06d}.jpg"), quality=92)
        panels.append(im)

    cap.release()
    with open(os.path.join(out_dir, "multi_corners.json"), "w") as f:
        json.dump(records, f, indent=2)

    if panels:
        # 每帧缩放拼图
        th = 760
        sc = [p.resize((int(p.width * th / p.height), th)) for p in panels]
        W = sum(p.width for p in sc) + 10 * (len(sc) + 1)
        sheet = Image.new("RGB", (W, th + 20), (8, 8, 10))
        xo = 10
        for p in sc:
            sheet.paste(p, (xo, 10)); xo += p.width + 10
        outp = os.path.join(out_dir, "multi_video.jpg")
        sheet.save(outp, quality=92)
        print(f"[mi] wrote {outp}  {sheet.size}")
    print(f"[mi] wrote {len(records)} 条记录 -> {out_dir}")


if __name__ == "__main__":
    main()
