"""DINOv2 主干（图里的 "ViT" 分支）。

通过 ``torch.hub`` 拉 ``facebookresearch/dinov2``，**首次运行需要联网**。
输出同样是多尺度特征列表，接口与 ``ResNetBackbone`` 一致，因此解码头可以复用。

注：BoxDreamer 把 DINOv2 源码 vendor 进了 ``src/models/sources/DINOv2/``；
本项目选择走 torch.hub，避免引入一大坨第三方代码。
如果后面需要离线部署，把权重下载到本地再改 ``ckpt_path`` 即可。
"""

from typing import List, Sequence

import torch
import torch.nn as nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# model_type -> (patch_size, embed_dim, out_indices 对应的「层深」)
_DINOV2_SPECS = {
    "dinov2_vits14": (14, 384),
    "dinov2_vitb14": (14, 768),
    "dinov2_vitl14": (14, 1024),
}


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT 特征提取器。

    Args:
        model_type: ``dinov2_vits14`` / ``dinov2_vitb14`` / ``dinov2_vitl14``
        freeze:     是否冻结（ViT 通常先冻结着用）
        ckpt_path:  本地权重路径；为 None 时走 torch.hub 在线加载
        out_indices: 取哪几个 transformer block 的输出（默认最后 4 层做多尺度）
    """

    def __init__(
        self,
        model_type: str = "dinov2_vitb14",
        freeze: bool = True,
        ckpt_path: str = None,
        out_indices: Sequence[int] = (8, 9, 10, 11),
    ):
        super().__init__()
        if model_type not in _DINOV2_SPECS:
            raise ValueError(
                f"Unsupported dinov2 model_type={model_type!r}; "
                f"available: {sorted(_DINOV2_SPECS)}"
            )
        self.patch_size, self.embed_dim = _DINOV2_SPECS[model_type]
        self.out_indices = tuple(out_indices)

        if ckpt_path:
            self.net = torch.hub.load(
                "facebookresearch/dinov2", model_type, pretrained=False
            )
            state = torch.load(ckpt_path, map_location="cpu")
            self.net.load_state_dict(state.get("model", state), strict=False)
        else:
            self.net = torch.hub.load("facebookresearch/dinov2", model_type)

        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

        # 多尺度特征在通道维拼接，通道数是 embed_dim 的 len(out_indices) 倍
        self.out_channels: List[int] = [self.embed_dim] * len(self.out_indices)

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            image: ``[B, 3, H, W]``，H/W 需能被 ``patch_size`` 整除
        Returns:
            4 个 ``[B, embed_dim, H/p, W/p]`` 的特征图
        """
        x = self.preprocess(image)
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        # 取指定 block 的输出
        feats = self.net.get_intermediate_layers(x, n=self.out_indices, reshape=True)
        return [f for f in feats]
