"""Lightning 层（对应 BoxDreamer/src/lightning/）。

模块名与类名刻意不同（``corner_pose_lightning_model`` / ``PL_CornerPose``），
与 BoxDreamer 的 ``BoxDreamer_lightning_model`` / ``PL_BoxDreamer`` 保持一致——
否则 ``from .mod import Class`` 会遮蔽同名子模块，导致
``import src.lightning.PL_CornerPose`` 拿到类而不是模块。
"""

from src.lightning.corner_pose_lightning_model import PL_CornerPose

__all__ = ["PL_CornerPose"]
