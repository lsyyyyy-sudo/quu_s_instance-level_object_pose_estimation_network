"""ResNet 主干（图里的 "ResNet" 分支）。

输出 stride 4/8/16/32 的多尺度特征，供热图解码头做 FPN 式融合。
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torchvision

# model_type -> (构造函数, 预训练权重枚举, 各 stage 输出通道)
_RESNET_SPECS = {
    "resnet18": (
        torchvision.models.resnet18,
        torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        [64, 128, 256, 512],
    ),
    "resnet34": (
        torchvision.models.resnet34,
        torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
        [64, 128, 256, 512],
    ),
    "resnet50": (
        torchvision.models.resnet50,
        torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        [256, 512, 1024, 2048],
    ),
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetBackbone(nn.Module):
    """torchvision ResNet 的多尺度特征提取器。

    Args:
        model_type:   ``resnet18`` / ``resnet34`` / ``resnet50``
        pretrained:   是否加载 ImageNet 预训练权重
        freeze:       是否冻结主干（只用它当特征提取器）
        out_indices:  取哪几个 stage，默认 ``(0,1,2,3)`` 即 layer1..layer4
    """

    def __init__(
        self,
        model_type: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ):
        super().__init__()
        if model_type not in _RESNET_SPECS:
            raise ValueError(
                f"Unsupported resnet model_type={model_type!r}; "
                f"available: {sorted(_RESNET_SPECS)}"
            )
        ctor, weights_enum, channels = _RESNET_SPECS[model_type]
        weights = weights_enum if pretrained else None
        net = ctor(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.stages = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        self.out_indices = tuple(out_indices)
        self.out_channels: List[int] = [channels[i] for i in self.out_indices]

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """ImageNet 归一化。输入 ``[B, 3, H, W]``，取值 ``[0, 1]``。"""
        return (image - self.mean) / self.std

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            image: ``[B, 3, H, W]``，取值 ``[0, 1]``
        Returns:
            按 ``out_indices`` 选出的特征列表，stride 分别为 4 / 8 / 16 / 32
        """
        x = self.preprocess(image)
        x = self.stem(x)
        feats: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                feats.append(x)
        return feats
