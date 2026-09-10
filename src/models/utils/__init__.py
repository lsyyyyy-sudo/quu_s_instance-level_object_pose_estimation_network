"""模型内部工具（对应 BoxDreamer/src/models/utils/）。

- ``box_utils``        : 3D 包围盒角点定义、投影、PnP 解位姿
- ``pose_utils``       : 位姿运算与 ADD / ADD-S / 5cm5° 指标
- ``prediction_utils`` : 热图 -> 2D 角点 -> 位姿
- ``data_processing``  : 热图监督构造与 batch 工具
"""
