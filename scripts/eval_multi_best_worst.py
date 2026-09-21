"""多实例模型的测试集可视化：最好/最差各若干张。

每个面板：
  黄框 = GT 实例（数据集的 inst_corners 外接框）
  红框 = 预测实例（decode 出来的角点外接框）
  标题 = 实例匹配情况 + 平均角点误差

按"平均角点误差"排序，取最好和最差各 n 张。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.utils.multi_instance import decode_instances  # noqa: E402

ROOT = r"D:\AAA_Projects\psd_zju3dv_coding_exam"


def draw_panel(img_chw, gt_corners, preds, match_dist, tag):
    a = img_chw.detach().cpu().numpy()
    if a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    a = np.clip(a, 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
    d = ImageDraw.Draw(im)

    def box(c, col, w, with_pts=False):
        q = c * 2.0
        x0, y0 = q[:, 0].min(), q[:, 1].min()
        x1, y1 = q[:, 0].max(), q[:, 1].max()
        d.rectangle([x0, y0, x1, y1], outline=col, width=w)
        if with_pts:
            for (x, y) in q:
                d.line([x - 4, y, x + 4, y], fill=col, width=2)
                d.line([x, y - 4, x, y + 4], fill=col, width=2)

    for c in gt_corners:
        box(c, (255, 215, 0), 3)

    # 匹配 + 误差
    errs, n_match = [], 0
    gt_used = set()
    for p in preds:
        pc = p["corners"]
        pcen = np.asarray(p["center"])
        best, bd = None, match_dist
        for gi, gc in enumerate(gt_corners):
            if gi in gt_used:
                continue
            dd = float(np.linalg.norm(gc.mean(axis=0) - pcen))
            if dd < bd:
                best, bd = gi, dd
        if best is not None:
            gt_used.add(best)
            n_match += 1
            errs.append(float(np.linalg.norm(pc - gt_corners[best], axis=-1).mean()))
        box(pc, (255, 60, 60), 2, with_pts=True)

    e = float(np.mean(errs)) if errs else float("nan")
    bar = Image.new("RGB", (im.width, 20), (0, 0, 0))
    bd = ImageDraw.Draw(bar)
    bd.text((3, 4), f"{tag}  GT {len(gt_corners)} / pred {len(preds)} / match {n_match}"
                     f"   err {e:.1f}px",
            fill=(235, 235, 235))
    out = Image.new("RGB", (im.width, im.height + 20), (10, 10, 10))
    out.paste(bar, (0, 0)); out.paste(im, (0, 20))
    return out, e


@hydra.main(config_path="../configs", config_name="train_multi.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    n_each = int(cfg.get("bw_n", 4))
    out_path = str(cfg.get("bw_out", os.path.join(ROOT, "data", "results", "mi_best_worst",
                                                  "MIGEO7.png")))

    m = instantiate(cfg.model, _recursive_=False)
    ck = torch.load(cfg.model.pretrained_ckpt, map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["state_dict"], strict=False)
    print(f"[bw] load missing={len(miss)} unexpected={len(unexp)}")
    if len(miss) or len(unexp):
        print("[bw] !! 权重没对上，结果无效")
        return
    m = m.cuda().eval()   # 忘了这句 -> 权重在 CPU、batch 在 GPU -> 设备不一致
    stride = m.model.image_size / float(m.model.heatmap_size)
    thr = float(cfg.model.multi.det_thr)
    topk = int(cfg.model.multi.det_topk)
    min_dist = float(cfg.model.multi.min_dist)

    dm = instantiate(cfg.datamodule)
    dm.setup("validate")
    loader = dm.val_dataloader()

    recs = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = m.model(batch)
            preds_all = decode_instances(out["pred_heatmap"], out["pred_offset"],
                                         thr=thr, topk=topk, min_dist=min_dist,
                                         stride=stride)
            for b in range(batch["image"].shape[0]):
                gts = [batch["inst_corners"][b, i].cpu().numpy()
                       for i in range(batch["inst_valid"].shape[1])
                       if float(batch["inst_valid"][b, i]) > 0]
                panel, e = draw_panel(batch["image"][b].cpu(), gts, preds_all[b],
                                      20.0, f"s{bi}_{b}")
                recs.append((e, panel, len(gts), len(preds_all[b])))
            if len(recs) >= 160:
                break

    valid = [r for r in recs if not np.isnan(r[0])]
    print(f"[bw] 共 {len(recs)} 个样本，其中有匹配的 {len(valid)} 个")
    if valid:
        n_gt = sum(r[2] for r in recs)
        n_pr = sum(r[3] for r in recs)
        n_m = len(valid)
        print(f"[bw] GT 实例 {n_gt}  预测 {n_pr}  匹配 {n_m}  "
              f"recall {n_m/max(n_gt,1):.3f}  precision {n_m/max(n_pr,1):.3f}")
        print(f"[bw] 匹配上的平均角点误差 中位 {np.median([r[0] for r in valid]):.2f} px")
        valid.sort(key=lambda r: r[0])
        print("[bw] 最好 4 个: " + " ".join(f"{r[0]:.1f}" for r in valid[:4]))
        print("[bw] 最差 4 个: " + " ".join(f"{r[0]:.1f}" for r in valid[-4:]))
        sel = valid[:n_each] + valid[-n_each:][::-1]
    else:
        sel = recs[: 2 * n_each]

    if not sel:
        print("[bw] 没有可画的样本")
        return
    cols = n_each
    pw, ph = sel[0][1].size
    rows = (len(sel) + cols - 1) // cols
    sheet = Image.new("RGB", (pw * cols + 6 * (cols + 1), ph * rows + 6 * (rows + 1)),
                      (8, 8, 10))
    for i, (_e, p, _a, _b) in enumerate(sel):
        sheet.paste(p, (6 + (i % cols) * (pw + 6), 6 + (i // cols) * (ph + 6)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, quality=92)
    print(f"[bw] wrote {out_path}  {sheet.size}")


if __name__ == "__main__":
    main()
