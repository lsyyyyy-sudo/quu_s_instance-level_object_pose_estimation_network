"""Instance-level object pose estimation network 主体。

对应 BoxDreamer/src/models/BoxDreamerModel.py。

结构上刻意保持一致：
- 纯 ``nn.Module``，**不含任何训练逻辑**（训练循环在 ``src/lightning/`` 里）
- ``_load_config`` / ``_initialize_modules`` 两段式初始化，模块由配置驱动
- ``forward(data)`` 吃一个 batch dict，返回预测 dict

与 BoxDreamer 的差异（因为输入不一样）：
- 输入是**单张 RGB**，不是「参考图 + 查询图」，所以没有 T 维、没有跨视角 matcher
- 输出是 ``[B, 8, h, w]`` 的角点热图（BoxDreamer 还要额外回归相机位姿/射线）
"""

from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.models.modules.backbone import build_backbone
from src.models.modules.decoder import HeatmapDecoder
from src.models.utils.prediction_utils import predict_corners_and_pose


class CornerPoseModel(nn.Module):
    """单图 -> 8 角点热图。

    Args:
        config: ``configs/model/heatmap.yaml`` 里的 ``modules`` 子树
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self._load_config(config)
        self._initialize_modules(config)

    # ------------------------------------------------------------------ #
    # 配置
    # ------------------------------------------------------------------ #
    def _load_config(self, module_configs: DictConfig):
        module_configs = OmegaConf.create(module_configs) if not isinstance(
            module_configs, DictConfig
        ) else module_configs

        self.encoder_cfg = module_configs.get("encoder", {})
        self.decoder_cfg = module_configs.get("decoder", {})
        self.task_cfg = module_configs.get("task", {})

        self.num_keypoints = int(self.task_cfg.get("num_keypoints", 8))
        self.image_size = int(self.task_cfg.get("image_size", 256))
        self.heatmap_size = int(self.task_cfg.get("heatmap_size", 64))
        self.bbox_representation = str(self.task_cfg.get("bbox_representation", "heatmap"))
        self.corner_order = str(self.task_cfg.get("corner_order", "bb8"))
        self.extraction = str(self.task_cfg.get("extraction", "soft_argmax"))
        self.soft_argmax_beta = float(self.task_cfg.get("soft_argmax_beta", 100.0))

        if self.num_keypoints != 8:
            raise ValueError(
                f"This implementation assumes 8 bbox corners, got num_keypoints={self.num_keypoints}"
            )

    # ------------------------------------------------------------------ #
    # 模块
    # ------------------------------------------------------------------ #
    def _initialize_modules(self, module_configs: DictConfig):
        self.encoder = build_backbone(self.encoder_cfg)
        self.decoder = HeatmapDecoder(
            in_channels_list=list(self.encoder.out_channels),
            hidden_dim=int(self.decoder_cfg.get("hidden_dim", 128)),
            num_keypoints=self.num_keypoints,
            heatmap_size=self.heatmap_size,
            upsample_mode=str(self.decoder_cfg.get("upsample_mode", "bilinear")),
            offset=bool(self.decoder_cfg.get("offset", False)),
        )

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(self, data) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: batch dict，至少要有 ``image`` ``[B, 3, H, W]``
        Returns:
            ``{"pred_heatmap": [B, K, h, w], "pred_offset": [B, 2K, h, w] or None}``
        """
        image = data["image"] if isinstance(data, dict) else data
        feats = self.encoder(image)
        out = self.decoder(feats)
        return {"pred_heatmap": out["heatmap"], "pred_offset": out.get("offset")}

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        data,
        bbox_3d: torch.Tensor = None,
        K: torch.Tensor = None,
        solve_pose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """端到端推理：图 -> 角点热图 -> 2D 角点 -> PnP 位姿。

        Args:
            data:     batch dict（用其中的 ``image``）
            bbox_3d:  ``[B, 8, 3]``，默认从 ``data['bbox_3d']`` 取
            K:        ``[B, 3, 3]``，默认从 ``data['cam_K']`` 取
            solve_pose: 是否跑 PnP
        """
        self.eval()
        out = self.forward(data)

        if bbox_3d is None:
            bbox_3d = data["bbox_3d"]
        if K is None:
            K = data["cam_K"]

        return predict_corners_and_pose(
            out["pred_heatmap"],
            bbox_3d,
            K,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            method=self.extraction,
            beta=self.soft_argmax_beta,
            solve_pose=solve_pose,
        )
