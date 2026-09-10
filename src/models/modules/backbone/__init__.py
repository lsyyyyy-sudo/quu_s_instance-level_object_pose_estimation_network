"""主干网络工厂。"""

from omegaconf import DictConfig

from src.models.modules.backbone.resnet import ResNetBackbone
from src.models.modules.backbone.vit import DINOv2Backbone

__all__ = ["ResNetBackbone", "DINOv2Backbone", "build_backbone"]


def build_backbone(encoder_cfg: DictConfig):
    """按 ``configs/model/heatmap.yaml`` 的 ``modules.encoder`` 构造主干。

    配置形如::

        encoder:
          name: 'resnet'            # 'resnet' | 'dinov2'
          resnet:
            ckpt_path: null
            cfg: {model_type: 'resnet18', pretrained: true, freeze: false}
          dinov2:
            ckpt_path: null
            cfg: {model_type: 'dinov2_vitb14', freeze: true}
    """
    name = str(encoder_cfg.get("name", "resnet")).lower()

    if name == "resnet":
        cfg = encoder_cfg.get("resnet", {})
        sub = cfg.get("cfg", {})
        return ResNetBackbone(
            model_type=sub.get("model_type", "resnet18"),
            pretrained=sub.get("pretrained", True),
            freeze=sub.get("freeze", False),
            out_indices=tuple(sub.get("out_indices", (0, 1, 2, 3))),
        )

    if name == "dinov2":
        cfg = encoder_cfg.get("dinov2", {})
        sub = cfg.get("cfg", {})
        return DINOv2Backbone(
            model_type=sub.get("model_type", "dinov2_vitb14"),
            freeze=sub.get("freeze", True),
            ckpt_path=cfg.get("ckpt_path", None),
        )

    raise ValueError(f"Unknown encoder name: {name!r} (expected 'resnet' or 'dinov2')")
