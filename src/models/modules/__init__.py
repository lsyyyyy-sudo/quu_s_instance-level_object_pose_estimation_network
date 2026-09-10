"""网络子模块。

对应 BoxDreamer/src/models/modules/。相比 BoxDreamer 砍掉了两个多视角专有组件：

- ``matcher/``   —— 参考图与查询图的跨视角特征匹配（我们只有单图）
- ``tracker/``   —— CoTracker 时序跟踪（我们不做视频跟踪）
"""

from src.models.modules.backbone import build_backbone
from src.models.modules.decoder import HeatmapDecoder

__all__ = ["build_backbone", "HeatmapDecoder"]
