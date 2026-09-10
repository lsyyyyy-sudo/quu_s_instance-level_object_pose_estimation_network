"""角点热图的 focal loss（CenterNet / CornerNet 风格）。

对应 BoxDreamer/src/loss/utils/focal_loss.py。

关键点：GT 上的「峰」是唯一正样本，峰周围的**负样本按与峰的距离衰减权重**
（``(1 - gt) ** beta``），避免网络把高斯峰的裙边也当成背景狠压。
"""

import torch
import torch.nn as nn


def focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """逐元素 focal loss 后按正样本数归一化。

    Args:
        pred: ``[B, K, H, W]`` **logits**（内部会 sigmoid）
        gt:   ``[B, K, H, W]`` 高斯峰标签，取值 ``[0, 1]``，峰顶为 1
        alpha: 正样本项的 ``(1 - pred) ** alpha`` 衰减指数
        beta:  负样本项的 ``(1 - gt) ** beta`` 衰减指数

    Returns:
        标量 loss
    """
    pred = torch.sigmoid(pred).clamp(eps, 1.0 - eps)

    pos_mask = (gt >= 1.0).float()
    neg_mask = 1.0 - pos_mask
    neg_weights = (1.0 - gt).pow(beta)

    pos_loss = -torch.log(pred) * (1.0 - pred).pow(alpha) * pos_mask
    neg_loss = -torch.log(1.0 - pred) * pred.pow(alpha) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    loss = pos_loss.sum() + neg_loss.sum()
    return loss / num_pos.clamp(min=1.0)


class FocalLoss(nn.Module):
    """``nn.Module`` 包装，方便放进配置里。"""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return focal_loss(pred, gt, alpha=self.alpha, beta=self.beta, eps=self.eps)
