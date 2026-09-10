"""热图解码头：多尺度特征 -> 8 通道角点热图。

对应 BoxDreamer 里 ``modules/decoder`` 的角点预测部分。
BoxDreamer 是 12 层 transformer decoder 做跨视角注意力；本项目是**单图**，
因此退化成 FPN 式的卷积解码器，结构上更接近 CenterNet / SimpleBaseline。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapDecoder(nn.Module):
    """FPN 式自顶向下融合 + 热图头。

    Args:
        in_channels_list: 主干各 stage 的输出通道，例如 resnet18 为 ``[64,128,256,512]``
        hidden_dim:       融合后的通道数
        num_keypoints:    角点个数（本项目为 8）
        heatmap_size:     输出热图边长
        upsample_mode:    上采样方式
        offset:           是否额外输出亚像素偏移 ``[B, 2K, h, w]``
    """

    def __init__(
        self,
        in_channels_list: List[int],
        hidden_dim: int = 128,
        num_keypoints: int = 8,
        heatmap_size: int = 64,
        upsample_mode: str = "bilinear",
        offset: bool = False,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.upsample_mode = upsample_mode
        self.offset = offset

        # 1x1 侧向连接，把所有 stage 压到 hidden_dim
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_channels_list]
        )
        # 每个尺度融合后再 3x3 平滑
        self.smooth = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1) for _ in in_channels_list]
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_keypoints, kernel_size=1),
        )

        if offset:
            self.offset_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, num_keypoints * 2, kernel_size=1),
            )

    def _upsample_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode=self.upsample_mode, align_corners=False)

    def forward(self, feats: List[torch.Tensor]):
        """
        Args:
            feats: 主干输出的多尺度特征，**从高分辨率到低分辨率**（stride 4 -> 32）
        Returns:
            dict: ``{"heatmap": [B, K, h, w], "offset": [B, 2K, h, w] or None}``
        """
        if len(feats) != len(self.lateral):
            raise ValueError(
                f"Expected {len(self.lateral)} feature levels, got {len(feats)}"
            )

        laterals = [conv(f) for conv, f in zip(self.lateral, feats)]

        # 自顶向下：低分辨率上采样后与高分辨率相加
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + self._upsample_like(laterals[i], laterals[i - 1])

        fused = self.smooth[0](laterals[0])

        heatmap = self.head(fused)
        if heatmap.shape[-1] != self.heatmap_size:
            heatmap = F.interpolate(
                heatmap,
                size=(self.heatmap_size, self.heatmap_size),
                mode=self.upsample_mode,
                align_corners=False,
            )

        offset: Optional[torch.Tensor] = None
        if self.offset:
            offset = self.offset_head(fused)
            if offset.shape[-1] != self.heatmap_size:
                offset = F.interpolate(
                    offset,
                    size=(self.heatmap_size, self.heatmap_size),
                    mode=self.upsample_mode,
                    align_corners=False,
                )

        return {"heatmap": heatmap, "offset": offset}
