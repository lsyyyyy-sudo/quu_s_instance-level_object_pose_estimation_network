"""评估指标累积器。

对应 BoxDreamer/src/lightning/utils/ 下的 metrics 部分。

不依赖 torchmetrics，自己攒 list 到 epoch 末尾再算平均 —— 样本量不大，
这样最直观也最好 debug。
"""

from typing import Dict, List, Optional

import torch

from src.models.utils.pose_utils import (
    add_error,
    adds_error,
    pose_error_5cm5deg,
    rotation_angle_deg,
)


class CornerPoseMetrics:
    """累积并计算 2D 角点误差与 6D 位姿误差。

    指标口径：
    - ``corner_err_px``：8 个角点的平均 L2 像素误差
    - ``pck@t``        ：误差 < t × (GT 2D 框对角线) 的角点占比
    - ``add`` / ``adds``：模型点集在预测/真值位姿下的平均距离
      （本项目用 3D 包围盒的 8 个角点代替稠密 mesh 点，绝对值与 BOP 官方不完全可比）
    - ``rot_err_deg`` / ``trans_err_mm``
    - ``acc_5cm5deg``  ：位姿命中率
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.pck_thresholds = list(cfg.get("corner_pck_thresholds", [0.05, 0.10, 0.15]))
        self.symmetric = bool(cfg.get("add_symmetric", False))
        self.solve_pose = bool(cfg.get("solve_pose", True))
        self.pose_unit_scale = float(cfg.get("pose_unit_scale", 1.0))
        self.trans_threshold = 50.0 * self.pose_unit_scale

        self.reset()

    def reset(self):
        self._corner_err: List[torch.Tensor] = []
        self._pck_hits: Dict[float, List[torch.Tensor]] = {t: [] for t in self.pck_thresholds}
        self._add: List[torch.Tensor] = []
        self._rot: List[torch.Tensor] = []
        self._trans: List[torch.Tensor] = []
        self._acc: List[torch.Tensor] = []
        self._n_pose_valid = 0
        self._n_total = 0

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update(
        self,
        corner_2d_pred: torch.Tensor,
        corner_2d_gt: torch.Tensor,
        bbox_3d: torch.Tensor,
        pose_pred: Optional[torch.Tensor] = None,
        pose_gt: Optional[torch.Tensor] = None,
    ):
        """``corner_2d_*``: ``[B, 8, 2]``，``bbox_3d``: ``[B, 8, 3]``，``pose_*``: ``[B, 4, 4]``。"""
        B = corner_2d_gt.shape[0]
        self._n_total += B

        diff = torch.linalg.norm(corner_2d_pred - corner_2d_gt, dim=-1)   # [B, 8]
        self._corner_err.append(diff.mean(dim=-1))

        xy_min = corner_2d_gt.amin(dim=1)
        xy_max = corner_2d_gt.amax(dim=1)
        diag = torch.linalg.norm(xy_max - xy_min, dim=-1, keepdim=True).clamp(min=1e-6)
        for t in self.pck_thresholds:
            self._pck_hits[t].append((diff < t * diag).float().mean(dim=-1))

        if not self.solve_pose or pose_pred is None or pose_gt is None:
            return

        valid = pose_pred[:, 3, 3].abs() > 1e-8
        if valid.sum() == 0:
            return

        pp = pose_pred[valid]
        pg = pose_gt[valid]
        bb = bbox_3d[valid]
        self._n_pose_valid += int(valid.sum())

        self._add.append(
            adds_error(pp, pg, bb) if self.symmetric else add_error(pp, pg, bb)
        )
        self._rot.append(rotation_angle_deg(pp[:, :3, :3], pg[:, :3, :3]))
        self._trans.append(torch.linalg.norm(pp[:, :3, 3] - pg[:, :3, 3], dim=-1))
        self._acc.append(
            pose_error_5cm5deg(
                pp, pg, bb,
                translation_threshold_mm=self.trans_threshold,
                symmetric=self.symmetric,
            ).float()
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not self._corner_err:
            return out

        corner_err = torch.cat(self._corner_err)
        out["corner_err_px"] = float(corner_err.mean())

        for t in self.pck_thresholds:
            hits = self._pck_hits[t]
            if hits:
                out[f"pck@{t:g}"] = float(torch.cat(hits).mean())

        if self._add:
            out["add"] = float(torch.cat(self._add).mean())
            out["rot_err_deg"] = float(torch.cat(self._rot).mean())
            out["trans_err_mm"] = float(torch.cat(self._trans).mean())
            out["acc_5cm5deg"] = float(torch.cat(self._acc).mean())
            out["pose_valid_ratio"] = (
                float(self._n_pose_valid) / float(self._n_total) if self._n_total else 0.0
            )

        return out
