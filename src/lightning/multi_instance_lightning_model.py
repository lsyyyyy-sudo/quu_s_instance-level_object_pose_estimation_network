"""多实例版的 LightningModule。

与单实例版的区别
----------------
单实例（``CornerPoseLightningModule``）：8 张角点热图，逐角点算误差，
    假设一张裁剪图里只有一个物体。
多实例（本文件）：中心热图 + 角点偏移，**一张图里的 N 个实例全部监督、
    全部评测**。评测时把预测实例与 GT 实例做匈牙利匹配（按中心距离），
    这样"预测几个"和"预测得准不准"都能单独看出来。

为什么单独写一个而不是在原来那个里加分支
------------------------------------------
原来的 ``_shared_step`` 深度绑定了"逐角点热图 + 单实例指标"的管线，
塞进多实例会变成两套逻辑纠缠。分开写更清楚，也不会动到已验证的单实例路径。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch import nn

from src.models.CornerPoseModel import CornerPoseModel
from src.models.utils.multi_instance import decode_instances
from src.loss.utils.multi_instance_loss import multi_instance_loss


class MultiInstanceLightningModel(pl.LightningModule):
    """多实例训练/验证模块。"""

    def __init__(
        self,
        modules: DictConfig,
        loss: DictConfig,
        opt: DictConfig,
        metrics: Optional[DictConfig] = None,
        vis: Optional[DictConfig] = None,
        multi: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()
        # ⚠️ 必须吃 **kwargs。
        # configs/model/*.yaml 里除了 modules/loss/opt/metrics/vis/multi，
        # 还有 resume_ckpt / pretrained_ckpt 等【给训练入口用】的键，
        # hydra.utils.instantiate(..., _recursive_=False) 会把它们全传进来。
        # 逐个加参数会一直打地鼠（resume_ckpt -> pretrained_ckpt -> ...）。
        self.extra_cfg = dict(kwargs)
        self.resume_ckpt = kwargs.get("resume_ckpt")
        self.pretrained_ckpt = kwargs.get("pretrained_ckpt")
        self.save_hyperparameters(ignore=["modules", "loss", "opt", "metrics", "vis", "multi"])
        self.model = CornerPoseModel(modules)
        self.loss_cfg = loss
        self.opt_cfg = opt
        m = multi or {}
        self.geo_weight = float(m.get("geo_weight", 1.0))
        self.corner_weight = float(m.get("corner_weight", 1.0))
        self.center_weight = float(m.get("center_weight", 1.0))
        self.det_thr = float(m.get("det_thr", 0.3))
        self.det_topk = int(m.get("det_topk", 30))
        self.min_dist = float(m.get("min_dist", 3.0))
        self.match_dist = float(m.get("match_dist", 20.0))   # 裁剪像素

    # ------------------------------------------------------------------ #
    def forward(self, data):
        return self.model(data)

    # ------------------------------------------------------------------ #
    def _shared_step(self, batch: Dict, stage: str) -> Dict:
        out = self.model(batch)
        losses = multi_instance_loss(
            out["pred_heatmap"], out["pred_offset"], batch,
            image_size=self.model.image_size, heatmap_size=self.model.heatmap_size,
            center_weight=self.center_weight, corner_weight=self.corner_weight,
            geo_weight=self.geo_weight,
        )
        metrics = {}
        if stage != "train":
            metrics = self._eval_instances(out, batch)
        return {"losses": losses, "metrics": metrics}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _eval_instances(self, out: Dict, batch: Dict) -> Dict[str, float]:
        """解码实例并与 GT 匹配，统计三个可解释的指标。

        返回的指标刻意分开，避免"用聚合数字掩盖问题"：
          recall    有多少 GT 实例被找到了
          precision 找到的实例里有多少是真的
          err_matched 匹配上的那些实例，平均角点误差（裁剪像素）
          geo_resid   匹配上的实例，共点残差中位数（几何合法性，越小越好）
        """
        stride = self.model.image_size / float(self.model.heatmap_size)
        preds = decode_instances(
            out["pred_heatmap"], out["pred_offset"],
            thr=self.det_thr, topk=self.det_topk,
            min_dist=self.min_dist, stride=stride,
        )
        from src.loss.utils.geo_consistency import concurrence_residual

        n_gt = n_pred = n_match = 0
        errs: List[float] = []
        resids: List[float] = []

        gt_corners = batch["inst_corners"]          # [B,N,8,2]
        gt_centers = batch["inst_centers"]          # [B,N,2]
        gt_valid = batch["inst_valid"]              # [B,N]

        for b in range(len(preds)):
            gts = [i for i in range(gt_valid.shape[1]) if gt_valid[b, i] > 0]
            ps = preds[b]
            n_gt += len(gts)
            n_pred += len(ps)

            used_gt = set()
            for p in ps:
                pc = torch.tensor(p["corners"])                     # [8,2]
                pcen = torch.tensor(p["center"])
                best, best_d = None, self.match_dist
                for gi in gts:
                    if gi in used_gt:
                        continue
                    d = float(torch.linalg.norm(gt_centers[b, gi] - pcen))
                    if d < best_d:
                        best, best_d = gi, d
                if best is not None:
                    used_gt.add(best)
                    n_match += 1
                    e = float(torch.linalg.norm(pc - gt_corners[b, best], dim=-1).mean())
                    errs.append(e)
                    resids.append(float(concurrence_residual(pc[None])[0]))

        m = {
            "n_gt": float(n_gt), "n_pred": float(n_pred), "n_match": float(n_match),
            "recall": n_match / max(n_gt, 1),
            "precision": n_match / max(n_pred, 1),
            "err_matched": float(np.mean(errs)) if errs else float("nan"),
            "geo_resid": float(np.median(resids)) if resids else float("nan"),
        }
        return m

    # ------------------------------------------------------------------ #
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        r = self._shared_step(batch, "train")
        L = r["losses"]
        bs = batch["image"].shape[0]
        self.log("train/loss", L["loss"], on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        for k in ("center", "corner", "geo"):
            self.log(f"train/loss_{k}", L[k], on_step=False, on_epoch=True, batch_size=bs)
        return L["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        r = self._shared_step(batch, "val")
        L, M = r["losses"], r["metrics"]
        bs = batch["image"].shape[0]
        self.log("val/loss", L["loss"], on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        for k in ("center", "corner", "geo"):
            self.log(f"val/loss_{k}", L[k], on_step=False, on_epoch=True, batch_size=bs)
        for k, v in M.items():
            if k.startswith("n_"):
                continue
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, batch_size=bs,
                     prog_bar=(k in ("recall", "err_matched")))
        # ⚠️ 必须补记 val/corner_err_px：
        # configs/callbacks/default.yaml 里的 ModelCheckpoint 监控这个键，
        # 找不到会直接抛异常终止训练（实测在第 4 个 epoch 崩掉，白等 1 小时）。
        # 多实例下的对应量是【匹配上的实例的平均角点误差】，语义一致。
        # ⚠️ nan 会**静默不记录**，checkpoint 同样崩。所以用哨兵值兜底：
        # 没有任何匹配时记一个很大的有限值（=裁剪图对角线），语义上表示全错。
        e = M.get("err_matched")
        if e is None or math.isnan(e):
            e = float(self.model.image_size) * 1.5   # 384，远大于任何正常误差
        self.log("val/corner_err_px", e, on_step=False, on_epoch=True, batch_size=bs)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.opt_cfg.optimizer, params=self.model.parameters()
        )
        scheduler = hydra.utils.instantiate(self.opt_cfg.scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
