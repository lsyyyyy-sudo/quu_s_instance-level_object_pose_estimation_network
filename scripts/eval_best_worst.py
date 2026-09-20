"""按逐样本误差排序，导出测试集里【最好】和【最差】的样本对照图。

用法::

    python scripts/eval_best_worst.py pretrain_name=train_v1 \
        +bw_out=data/results/best_worst/v1.png [+bw_n=4]

每个面板：GT 角点（黄色 +）vs 预测（红色 ×），标题给出该样本 8 个角点的
平均像素误差和 visib_fract。**上排最好、下排最差**，用同一尺度看差距。
"""

from __future__ import annotations

import os
import sys

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.utils.prediction_utils import predict_corners_and_pose  # noqa: E402

K = 8


def panel(img_chw: torch.Tensor, gt: np.ndarray, pred: np.ndarray,
          err: np.ndarray, visib: float, tag: str):
    """img_chw: [3,H,W] 已归一化；gt/pred: [8,2] 在 256 坐标系。"""
    a = img_chw.detach().cpu().numpy()
    if a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    a = np.clip(a, 0, 1)
    if a.max() <= 1.0:
        a = (a * 255).astype(np.uint8)
    else:
        a = a.astype(np.uint8)
    im = Image.fromarray(a).convert("RGB")
    im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
    d = ImageDraw.Draw(im)

    for arr, col, w in ((gt, (255, 220, 0), 2), (pred, (255, 60, 60), 2)):
        p = arr * 2.0
        for (x, y) in p:
            d.line([x - 5, y, x + 5, y], fill=col, width=w)
            d.line([x, y - 5, x, y + 5], fill=col, width=w)
        # 连成 BB8 的盒子线框，方便看形状
        for i in range(4):
            d.line([*p[i], *p[(i + 1) % 4]], fill=col, width=1)
            d.line([*p[i + 4], *p[4 + (i + 1) % 4]], fill=col, width=1)
            d.line([*p[i], *p[i + 4]], fill=col, width=1)

    bar = Image.new("RGB", (im.width, 18), (0, 0, 0))
    bd = ImageDraw.Draw(bar)
    bd.text((2, 3), f"{tag}  err {err.mean():.1f}px  visib {visib:.2f}",
            fill=(235, 235, 235))
    out = Image.new("RGB", (im.width, im.height + 18), (10, 10, 10))
    out.paste(bar, (0, 0))
    out.paste(im, (0, 18))
    return out


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.get("seed", 42))
    n_each = int(cfg.get("bw_n", 4))

    model = instantiate(cfg.model, _recursive_=False)
    ckpt_path = cfg.model.pretrained_ckpt
    print(f"[bw] checkpoint = {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[bw] load: missing={len(missing)} unexpected={len(unexpected)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    inner = model.model

    dm = instantiate(cfg.datamodule)
    dm.setup("validate")
    loader = dm.val_dataloader()

    recs = []          # (mean_err, img, gt, pred, err, visib)
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            preds = predict_corners_and_pose(
                out["pred_heatmap"], batch["bbox_3d"], batch["cam_K"],
                image_size=inner.image_size, heatmap_size=inner.heatmap_size,
                method=inner.extraction, beta=inner.soft_argmax_beta, k=inner.topk,
                heatmap_range=inner.heatmap_range, solve_pose=False,
            )
            pred = preds["corner_2d"].cpu().numpy()
            gt = batch["corner_2d"].cpu().numpy()
            d = np.linalg.norm(pred - gt, axis=-1)
            vf = (batch["visib_fract"].cpu().numpy()
                  if "visib_fract" in batch else np.ones(len(pred)))
            for b in range(pred.shape[0]):
                recs.append((float(d[b].mean()), batch["image"][b].cpu().clone(),
                             gt[b].copy(), pred[b].copy(), d[b].copy(), float(vf[b])))

    recs.sort(key=lambda r: r[0])
    print(f"[bw] 共 {len(recs)} 个样本")
    print(f"     最好 {n_each} 个: " +
          " ".join(f"{r[0]:.1f}" for r in recs[:n_each]))
    print(f"     最差 {n_each} 个: " +
          " ".join(f"{r[0]:.1f}" for r in recs[-n_each:]))

    best = [panel(r[1], r[2], r[3], r[4], r[5], f"BEST#{i+1}") for i, r in enumerate(recs[:n_each])]
    worst = [panel(r[1], r[2], r[3], r[4], r[5], f"WORST#{i+1}")
             for i, r in enumerate(recs[-n_each:][::-1])]

    cols = n_each
    pw, ph = best[0].size
    sheet = Image.new("RGB", (pw * cols + 6 * (cols + 1), (ph + 6) * 2 + 6), (8, 8, 10))
    for i, p in enumerate(best):
        sheet.paste(p, (6 + i * (pw + 6), 6))
    for i, p in enumerate(worst):
        sheet.paste(p, (6 + i * (pw + 6), 6 + ph + 6))

    out_path = str(cfg.get("bw_out", "best_worst.png"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    print(f"[bw] wrote {out_path}  {sheet.size}")
    print("[bw] 图例: 黄+ 带线框 = GT 盒子,  红x 带线框 = 预测盒子")


if __name__ == "__main__":
    main()
