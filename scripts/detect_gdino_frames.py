"""在**任意视频**上跑 GroundingDINO 出框（可移植版）。

为什么新写一个
-------------
原来的 ``scripts/detect_grounding_dino.py`` 把项目根目录和视频路径**硬编码**成
Windows 路径（``ROOT = r"D:\\..."``、``VID = os.path.join(ROOT, "head_left_rgb_raw.mp4(1)", ...)``），
在 Linux / 实例上**根本没法用**。本脚本改成纯命令行参数，两边都能跑。

顺带做两件原脚本没做的事：
  1. **内存护栏**：可选 ``--det-w`` 控制送入检测器的宽度；无卡模式（cgroup 2 GB）
     下这是能不能跑起来的关键。启动时打印 RSS，结束时打印峰值。
  2. **提示词对照**：多个 ``--prompt`` 各自跑一遍，报告每个提示词出几个框，
     方便挑最好的那个（真实场景里"camera"和"action camera"结果差很多）。

用法::

    # 抽 3 帧试水（无卡模式建议 --det-w 800）
    python scripts/detect_gdino_frames.py --video target.mp4 \
        --frames 0,60,120 --out det_out --det-w 800 \
        --prompt "camera." --prompt "action camera."

    # 正式跑一批帧
    python scripts/detect_gdino_frames.py --video target.mp4 \
        --frames 0,300,600,900 --out det_out --det-w 1600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", default="0", help="逗号分隔的帧号，如 0,60,120")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 时只跑前 N 帧")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-tiny"))
    ap.add_argument("--prompt", action="append", default=None,
                    help="可重复；缺省 ["+"camera., action camera."+"]")
    ap.add_argument("--det-w", type=int, default=800,
                    help="送入检测器的宽度（原图 3248 太大；无卡模式建议 800）")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.20)
    ap.add_argument("--vis-w", type=int, default=1500, help="可视化输出宽度")
    args = ap.parse_args()

    prompts = args.prompt or ["camera.", "action camera."]
    frames = [int(s) for s in str(args.frames).split(",") if s.strip() != ""]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    os.makedirs(args.out, exist_ok=True)

    print(f"[gdino] 模型={args.model}  det_w={args.det_w}  帧={frames}")
    print(f"[gdino] 提示词={prompts}")
    print(f"[gdino] 载入前 RSS={_rss_mb():.0f} MB", flush=True)

    import cv2
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    print(f"[gdino] 载入完成 {time.time()-t0:.0f}s  device={device}  RSS={_rss_mb():.0f} MB", flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[gdino] ❌ 打不开视频 {args.video}", file=sys.stderr)
        return 2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[gdino] 视频共 {total} 帧  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", flush=True)

    records = {}
    peak = _rss_mb()
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"[gdino] 帧 {fi} 读不到，跳过")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        s = args.det_w / rgb.shape[1]
        small = cv2.resize(rgb, (args.det_w, int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(small)

        print("=" * 66)
        print(f"帧 {fi}")
        best = None
        for prompt in prompts:
            t = time.time()
            inputs = proc(images=pil, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=args.box_thr,
                text_threshold=args.text_thr,
                target_sizes=[(small.shape[0], small.shape[1])])[0]
            n = len(res["boxes"])
            sc = [round(float(x), 3) for x in res["scores"]]
            print(f"  {prompt:18s} {n} 框  {sc}   ({time.time()-t:.0f}s)")
            if best is None or n > len(best[1]):
                best = (prompt, [b.tolist() for b in res["boxes"]],
                        [float(x) for x in res["scores"]])
            del inputs, out, res
        peak = max(peak, _rss_mb())

        prompt, boxes, scores = best
        boxes_full = [[b[0] / s, b[1] / s, b[2] / s, b[3] / s] for b in boxes]  # 缩回原图坐标
        records[str(fi)] = {"prompt": prompt,
                            "boxes": [[round(v, 1) for v in b] for b in boxes_full],
                            "scores": [round(v, 3) for v in scores]}

        im = Image.fromarray(rgb)
        sc2 = args.vis_w / im.width
        im = im.resize((args.vis_w, int(im.height * sc2)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        for k, (b, sc) in enumerate(zip(boxes_full, scores)):
            d.rectangle([b[0] * sc2, b[1] * sc2, b[2] * sc2, b[3] * sc2],
                        outline=(255, 60, 60), width=4)
            d.rectangle([b[0] * sc2, b[1] * sc2, b[0] * sc2 + 190, b[1] * sc2 + 18], fill=(0, 0, 0))
            d.text((b[0] * sc2 + 4, b[1] * sc2 + 3), f"#{k} {sc:.2f}", fill=(255, 220, 60))
        d.rectangle([0, 0, 460, 18], fill=(0, 0, 0))
        d.text((4, 3), f"frame {fi}  prompt='{prompt}'  det_w={args.det_w}", fill=(255, 255, 120))
        p = os.path.join(args.out, f"gdino_f{fi:05d}.jpg")
        im.save(p, quality=90)
        print(f"  -> {p}")

    cap.release()
    with open(os.path.join(args.out, "gdino_detections.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n[gdino] 写出 {len(records)} 帧 -> {args.out}/gdino_detections.json")
    print(f"[gdino] RSS 峰值 = {peak:.0f} MB")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
