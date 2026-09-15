"""评估 8 角点预测（2D）——本项目交付物的正确评价方式。

**为什么不是 ADD / 旋转 / 平移**：本项目的交付物是 **2D 的 8 个角点**，不是 6D 位姿。
ADD、旋转误差、平移误差都需要**相机内参**去跑 PnP，而本项目不需要内参
（见 docs/TRAINING.md §5.1）。拿 BOP 的 ``ADD < 0.1×直径`` 去判一个角点任务
是用错的标准：位姿对小物体的 2D 误差极其敏感，会把"角点其实还行"读成"完全失败"。

**这里用的指标**（都以 2D 角点为对象）：

* ``err_norm``   ：像素误差 ÷ GT 2D 框对角线（消掉物体在画面里的大小差异）
* 分位数        ：median / p90 / p95 —— 均值会被少数崩掉的角点带偏，必须看分位
* ``pck@t``      ：误差 < t×对角线 的角点占比
* 逐角点拆解    ：8 个角点里哪几个最难（被遮挡的、贴边的）
* 失败率        ：误差 > 0.1×对角线 的角点占比 —— 部署时最关心这个
* 框 IoU         ：预测 8 点与 GT 8 点各自外接框的 IoU

用法::

    python scripts/eval_corners.py exp_name=eval_corners pretrain_name=train_v1 \
        use_pretrained=true datamodule.max_val_samples=1000
"""

from __future__ import annotations

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

from src.models.utils.prediction_utils import predict_corners_and_pose  # noqa: E402

K = 8


def _draw_panel(img_chw: torch.Tensor, gt: np.ndarray, pred: np.ndarray, err: np.ndarray):
    """左：GT(黄圈) vs 预测(红十字)；右：误差放大 5 倍的可视化。"""
    rgb = (img_chw.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    p1 = Image.fromarray(rgb.copy())
    d1 = ImageDraw.Draw(p1)
    for k in range(K):
        gx, gy = gt[k]
        px, py = pred[k]
        d1.line([gx - 4, gy, gx + 4, gy], fill=(255, 255, 0), width=2)
        d1.line([gx, gy - 4, gx, gy + 4], fill=(255, 255, 0), width=2)
        d1.line([px - 3, py - 3, px + 3, py + 3], fill=(255, 70, 70), width=2)
        d1.line([px - 3, py + 3, px + 3, py - 3], fill=(255, 70, 70), width=2)

    # 右：把预测点相对 GT 的偏差放大 5 倍画出来（否则小误差看不出来）
    p2 = Image.fromarray(rgb.copy())
    d2 = ImageDraw.Draw(p2)
    for k in range(K):
        gx, gy = gt[k]
        ex, ey = pred[k] - gt[k]
        d2.line([gx - 4, gy, gx + 4, gy], fill=(255, 255, 0), width=1)
        d2.line([gx, gy - 4, gx, gy + 4], fill=(255, 255, 0), width=1)
        d2.line([gx, gy, gx + ex * 5, gy + ey * 5], fill=(255, 70, 70), width=2)
    panel = Image.new("RGB", (w * 2 + 10, h + 18), (18, 18, 22))
    dd = ImageDraw.Draw(panel)
    dd.text((2, 2), f"GT(yellow +) vs pred(red x)   mean err {err.mean():.1f}px", fill=(235, 235, 235))
    dd.text((w + 12, 2), "deviation x5", fill=(235, 235, 235))
    panel.paste(p1, (0, 16))
    panel.paste(p2, (w + 10, 16))
    return panel


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.get("seed", 42))

    model = instantiate(cfg.model, _recursive_=False)
    ckpt_path = cfg.model.pretrained_ckpt
    print(f"[eval] checkpoint = {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[eval] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("       missing[:5] =", list(missing)[:5])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    inner = model.model  # CornerPoseModel

    dm = instantiate(cfg.datamodule)
    dm.setup("validate")
    loader = dm.val_dataloader()
    print(f"[eval] val batches = {len(loader)}  batch_size = {dm.batch_size}")

    # 累积器
    per_corner_err = np.zeros(K)          # 每个角点的平均像素误差
    per_corner_n = 0
    errs_px = []                          # 每个角点的像素误差
    errs_norm = []                        # 归一化误差
    ious = []
    visib = []                            # 每个角点所属样本的 visib_fract
    n_samples = 0
    panels = []
    max_panels = int(os.environ.get("EVAL_PANELS", "8"))

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            preds = predict_corners_and_pose(
                out["pred_heatmap"], batch["bbox_3d"], batch["cam_K"],
                image_size=inner.image_size, heatmap_size=inner.heatmap_size,
                method=inner.extraction, beta=inner.soft_argmax_beta, k=inner.topk,
                heatmap_range=inner.heatmap_range, solve_pose=False,
            )
            pred = preds["corner_2d"].cpu().numpy()      # [B,8,2]
            gt = batch["corner_2d"].cpu().numpy()        # [B,8,2]

            d = np.linalg.norm(pred - gt, axis=-1)       # [B,8] 像素误差
            xy_min = gt.min(axis=1, keepdims=True)       # [B,1,2]
            xy_max = gt.max(axis=1, keepdims=True)
            diag = np.linalg.norm(xy_max - xy_min, axis=-1)   # [B,1]
            dn = d / np.clip(diag, 1e-6, None)

            per_corner_err += d.sum(axis=0)
            per_corner_n += d.shape[0]
            errs_px.append(d.reshape(-1))
            errs_norm.append(dn.reshape(-1))
            if "visib_fract" in batch:
                visib.append(np.repeat(batch["visib_fract"].cpu().numpy(), K))
            n_samples += d.shape[0]

            for b in range(pred.shape[0]):
                pb = pred[b]
                gb = gt[b]
                bx0, by0 = pb.min(axis=0); bx1, by1 = pb.max(axis=0)
                gx0, gy0 = gb.min(axis=0); gx1, gy1 = gb.max(axis=0)
                iw = max(0.0, min(bx1, gx1) - max(bx0, gx0))
                ih = max(0.0, min(by1, gy1) - max(by0, gy0))
                inter = iw * ih
                union = (bx1 - bx0) * (by1 - by0) + (gx1 - gx0) * (gy1 - gy0) - inter
                ious.append(inter / union if union > 0 else 0.0)

                if len(panels) < max_panels and bi % 3 == 0 and b == 0:
                    panels.append(_draw_panel(batch["image"][b].cpu(), gb, pb, d[b]))

    d_all = np.concatenate(errs_px)
    dn_all = np.concatenate(errs_norm)
    ious = np.asarray(ious)

    def pct(a, q):
        return float(np.percentile(a, q))

    print()
    print("=" * 74)
    print(f"8 角点预测评估   样本 {n_samples}   角点 {d_all.size}")
    print("=" * 74)
    print("【像素误差】（256x256 裁剪坐标系）")
    print(f"  mean {d_all.mean():7.2f}   median {pct(d_all,50):7.2f}   "
          f"p90 {pct(d_all,90):7.2f}   p95 {pct(d_all,95):7.2f}   max {d_all.max():8.1f}")
    print()
    print("【归一化误差】（÷ GT 2D 框对角线）")
    print(f"  mean {dn_all.mean():7.4f}  median {pct(dn_all,50):7.4f}  "
          f"p90 {pct(dn_all,90):7.4f}  p95 {pct(dn_all,95):7.4f}")
    print()
    print("【PCK】误差 < t×对角线 的角点占比")
    ts = (0.02, 0.05, 0.10, 0.15, 0.20)
    for t in ts:
        print(f"  PCK@{t:<5} {100*np.mean(dn_all < t):6.2f}%")
    # AUC：把 PCK 曲线在 [0, t_max] 上积分再除以 t_max，汇总成一个数（越大越好，参考 BOP 的 AUC）
    grid = np.linspace(0.0, 0.20, 201)
    pck_curve = np.array([float(np.mean(dn_all < g)) for g in grid])
    auc = float(np.trapz(pck_curve, grid) / 0.20)
    print(f"  --> PCK-AUC(0~0.20) = {auc:.4f}   （越大越好；随机预测约 0）")
    print()
    print("【失败率】")
    for t in (0.05, 0.10, 0.20):
        print(f"  误差 > {t:<5} 对角线的角点占比 {100*np.mean(dn_all > t):6.2f}%")
    print()
    print("【逐角点】平均像素误差（角点编号 0-7，见 src/models/utils/box_utils.py 的 BB8_BITS）")
    avg = per_corner_err / max(per_corner_n, 1)
    order = np.argsort(-avg)
    for k in range(K):
        bar = "#" * int(avg[k] / max(avg.max(), 1e-6) * 40)
        mark = "  <- 最难" if k == order[0] else ("  <- 最易" if k == order[-1] else "")
        print(f"  corner {k}: {avg[k]:7.2f} px  {bar}{mark}")
    print()
    print("【框 IoU】预测 8 点外接框 vs GT 8 点外接框")
    print(f"  mean {ious.mean():.4f}  median {np.median(ious):.4f}  "
          f"min {ious.min():.4f}  (<0.5 的样本 {100*np.mean(ious<0.5):.2f}%)")

    # ---- 按遮挡程度分层（检验 DATA-16 假设：误差是否集中在被遮挡的样本上）----
    if visib:
        v = np.concatenate(visib)
        print()
        print("【按 visib_fract 分层】这一栏直接检验「训练集缺遮挡 -> 遮挡处崩」的假设")
        print(f"  {'bin':>12} {'角点数':>7} {'占比':>7} {'中位误差':>10} {'p90':>9} "
              f"{'失败率>0.1':>11}")
        bins = [(0.0, 0.5, "0.0~0.5"), (0.5, 0.8, "0.5~0.8"),
                (0.8, 0.95, "0.8~0.95"), (0.95, 1.01, "0.95~1.0")]
        for lo, hi, lab in bins:
            m = (v >= lo) & (v < hi)
            if m.sum() == 0:
                continue
            sub = dn_all[m]
            print(f"  {lab:>12} {int(m.sum()):>7} {100*m.mean():>6.1f}% "
                  f"{np.median(sub):>10.4f} {np.percentile(sub,90):>9.4f} "
                  f"{100*np.mean(sub > 0.1):>10.2f}%")
    print("=" * 74)

    if panels:
        pw, ph = panels[0].size
        cols = 2
        rows = (len(panels) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * pw + (cols + 1) * 8, rows * ph + (rows + 1) * 8), (12, 12, 14))
        for i, p in enumerate(panels):
            r, c = divmod(i, cols)
            sheet.paste(p, (8 + c * (pw + 8), 8 + r * (ph + 8)))
        out_path = cfg.get("eval_vis_out", "eval_corners.png")
        sheet.save(out_path)
        print(f"[eval] 可视化 -> {os.path.abspath(out_path)}  {sheet.size}")


if __name__ == "__main__":
    main()
