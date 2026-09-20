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
from src.models.utils.data_processing import heatmap_value_range
from src.models.utils.prediction_utils import DEFAULT_TOPK, predict_corners_and_pose


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

        # 热图风格决定取值区间与输出激活方式
        self.heatmap_style = str(self.task_cfg.get("heatmap_style", "boxdreamer"))
        if self.heatmap_style not in ("boxdreamer", "centernet"):
            raise ValueError(
                f"Unknown heatmap_style: {self.heatmap_style!r} "
                "(expected 'boxdreamer' or 'centernet')"
            )
        self.heatmap_range = heatmap_value_range(self.heatmap_style)

        # 角点提取方式：BoxDreamer 官方用 topk（top-20 平均）
        self.extraction = str(self.task_cfg.get("extraction", "topk"))
        self.soft_argmax_beta = float(self.task_cfg.get("soft_argmax_beta", 100.0))
        self.topk = int(self.task_cfg.get("topk", DEFAULT_TOPK))

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
            # 多实例模式：主头输出 1 通道【中心热图】（峰数 = 实例数），
            # 8 个角点从 offset 分支（16 通道）读出。见 heatmap_head.py。
            center_heatmap=bool(self.decoder_cfg.get("center_heatmap", False)),
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

        ``pred_heatmap`` 已经做过激活：
        - ``heatmap_style='boxdreamer'``：``2 * sigmoid(x) - 1``，取值 ``[-1, 1]``
          （对齐 BoxDreamer 的 ``betr.py``）
        - ``heatmap_style='centernet'``：保持 logits，交给 focal loss 自己 sigmoid
        """
        image = data["image"] if isinstance(data, dict) else data
        feats = self.encoder(image)
        out = self.decoder(feats)

        heatmap = out["heatmap"]
        # 多实例模式：中心热图必须保持 **logits**，因为 focal loss 自己会 sigmoid。
        # （boxdreamer 的 2*sigmoid-1 是给逐角点热图用的，套在中心热图上会毁掉 focal）
        if self.decoder_cfg.get("center_heatmap", False):
            return {"pred_heatmap": heatmap, "pred_offset": out.get("offset"),
                    "center_heatmap_mode": True}
        if self.heatmap_style == "boxdreamer":
            heatmap = 2.0 * torch.sigmoid(heatmap) - 1.0

        return {"pred_heatmap": heatmap, "pred_offset": out.get("offset")}

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
            k=self.topk,
            heatmap_range=self.heatmap_range,
            solve_pose=solve_pose,
        )
