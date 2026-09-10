"""PyTorch Lightning 模块：训练循环、指标与可视化。

对应 BoxDreamer/src/lightning/BoxDreamer_lightning_model.py（``PL_BoxDreamer``）。

职责边界（与 BoxDreamer 一致）：
- 网络结构在 ``src/models/CornerPoseModel.py``（纯 nn.Module）
- 数据在 ``src/datamodules/``
- 本文件只管「怎么训练 / 怎么评估」
"""

from typing import Dict, List, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from src.lightning.utils.metrics import CornerPoseMetrics
from src.lightning.utils.vis import make_vis_panel, to_chw_float
from src.models.CornerPoseModel import CornerPoseModel
from src.models.utils.box_utils import project_bbox3d
from src.models.utils.prediction_utils import predict_corners_and_pose
from src.utils.log import INFO


class PL_CornerPose(pl.LightningModule):
    """单图 8 角点热图网络的 LightningModule。

    Args 与 ``configs/model/heatmap.yaml`` 一一对应（hydra 用 ``_recursive_=False``
    实例化，因此子配置 ``loss`` / ``opt`` / ``metrics`` / ``vis`` 原样传进来）。
    """

    def __init__(
        self,
        modules: DictConfig,
        loss: DictConfig,
        opt: DictConfig,
        metrics: DictConfig,
        vis: DictConfig,
        resume_ckpt: Optional[str] = None,
        pretrained_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.model = CornerPoseModel(modules)
        self.loss_fn = hydra.utils.instantiate(loss)

        self.opt_cfg = opt
        self.metrics_cfg = metrics
        self.vis_cfg = vis
        self.resume_ckpt = resume_ckpt
        self.pretrained_ckpt = pretrained_ckpt

        metrics_plain = OmegaConf.to_container(metrics, resolve=True) if isinstance(
            metrics, DictConfig
        ) else (metrics or {})
        self.val_metrics = CornerPoseMetrics(metrics_plain)
        self.test_metrics = CornerPoseMetrics(metrics_plain)

        self._val_vis: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # 前向 / 公共步骤
    # ------------------------------------------------------------------ #
    def forward(self, data):
        return self.model(data)

    def _shared_step(self, batch: Dict, stage: str) -> Dict:
        out = self.forward(batch)

        losses = self.loss_fn(
            out["pred_heatmap"],
            batch["heatmap"],
            pred_offset=out.get("pred_offset"),
            corner_2d=batch["corner_2d"],
            image_size=self.model.image_size,
            heatmap_size=self.model.heatmap_size,
        )

        preds = predict_corners_and_pose(
            out["pred_heatmap"],
            batch["bbox_3d"],
            batch["cam_K"],
            image_size=self.model.image_size,
            heatmap_size=self.model.heatmap_size,
            method=self.model.extraction,
            beta=self.model.soft_argmax_beta,
            solve_pose=bool(self.metrics_cfg.get("solve_pose", True)),
        )
        return {"out": out, "losses": losses, "preds": preds}

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        result = self._shared_step(batch, "train")
        losses = result["losses"]

        bs = batch["image"].shape[0]
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("train/loss_focal", losses["loss_focal"], on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)
        return losses["loss"]

    # ------------------------------------------------------------------ #
    # 验证
    # ------------------------------------------------------------------ #
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        result = self._shared_step(batch, "val")
        losses, preds = result["losses"], result["preds"]

        bs = batch["image"].shape[0]
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=bs, sync_dist=True)

        self.val_metrics.update(
            preds["corner_2d"], batch["corner_2d"], batch["bbox_3d"],
            pose_pred=preds["poses"], pose_gt=batch["pose_gt"],
        )

        if self._should_vis():
            self._collect_vis(batch, preds)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(self.val_metrics, "val")
        self.val_metrics.reset()
        self._flush_vis("val")

    # ------------------------------------------------------------------ #
    # 测试
    # ------------------------------------------------------------------ #
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        result = self._shared_step(batch, "test")
        losses, preds = result["losses"], result["preds"]

        bs = batch["image"].shape[0]
        self.log("test/loss", losses["loss"], on_step=False, on_epoch=True,
                 batch_size=bs, sync_dist=True)

        self.test_metrics.update(
            preds["corner_2d"], batch["corner_2d"], batch["bbox_3d"],
            pose_pred=preds["poses"], pose_gt=batch["pose_gt"],
        )

    def on_test_epoch_end(self) -> None:
        self._log_metrics(self.test_metrics, "test")
        self.test_metrics.reset()

    # ------------------------------------------------------------------ #
    # 指标 / 可视化
    # ------------------------------------------------------------------ #
    def _log_metrics(self, metrics: CornerPoseMetrics, stage: str) -> None:
        values = metrics.compute()
        if not values:
            return
        for key, value in values.items():
            self.log(f"{stage}/{key}", value, on_step=False, on_epoch=True, sync_dist=True)
        summary = "  ".join(f"{k}={v:.4f}" for k, v in values.items())
        INFO(f"[{stage}] {summary}")

    def _should_vis(self) -> bool:
        if not bool(self.vis_cfg.get("enable", False)):
            return False
        every = int(self.vis_cfg.get("every_n_epochs", 1))
        return every > 0 and (self.current_epoch % every == 0)

    def _collect_vis(self, batch: Dict, preds: Dict) -> None:
        limit = int(self.vis_cfg.get("num_samples", 4))
        already = len(self._val_vis)
        if already >= limit:
            return

        bs = min(limit - already, batch["image"].shape[0])
        pose_corner_2d = None
        if preds.get("poses") is not None:
            pose_corner_2d = project_bbox3d(
                batch["bbox_3d"], batch["cam_K"], preds["poses"]
            )

        for i in range(bs):
            panel = make_vis_panel(
                batch["image"][i],
                corner_2d=preds["corner_2d"][i],
                corner_2d_gt=batch["corner_2d"][i],
                pose_corner_2d=None if pose_corner_2d is None else pose_corner_2d[i],
            )
            self._val_vis.append(to_chw_float(panel))

    def _flush_vis(self, stage: str) -> None:
        if not self._val_vis:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_image"):
            for i, panel in enumerate(self._val_vis):
                experiment.add_image(f"{stage}/vis_{i}", panel, self.current_epoch)
        else:
            INFO(
                f"Collected {len(self._val_vis)} visualization panels, "
                "but the active logger does not support add_image; skipping."
            )
        self._val_vis = []

    # ------------------------------------------------------------------ #
    # 优化器
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.opt_cfg.optimizer, params=self.model.parameters()
        )
        scheduler = hydra.utils.instantiate(self.opt_cfg.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    # ------------------------------------------------------------------ #
    # 权重加载
    # ------------------------------------------------------------------ #
    def load_pretrained_params(self, path: str) -> None:
        """从 Lightning checkpoint 里取出网络权重（去掉 ``model.`` 前缀）。"""
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        prefix = "model."
        stripped = {
            k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
        }
        if not stripped:
            stripped = state

        missing, unexpected = self.model.load_state_dict(stripped, strict=False)
        INFO(
            f"Loaded pre-trained weights from {path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
