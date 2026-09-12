"""数据集（对应 BoxDreamer/src/datasets/）。"""

from src.datasets.bop_pbr import BOPPBRDataset, crop_and_resize

__all__ = ["BOPPBRDataset", "crop_and_resize"]
