"""LightningDataModule（对应 BoxDreamer/src/datamodules/BoxDreamer_datamodule.py）。"""

import os
from typing import Optional, Sequence, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from src.datasets.bop_pbr import BOPPBRDataset


class CornerPoseDataModule(pl.LightningDataModule):
    """把 BOP PBR 数据集包装成 Lightning 需要的形式。

    验证集切分策略：
    - ``val_split == train_split``：用固定随机种子从训练集里切 ``val_ratio`` 出来，
      保证每次运行切分一致，避免验证指标抖动。
    - 否则：从 ``val_split`` 单独构建。
    """

    def __init__(
        self,
        dataset_root: Union[str, Sequence[str]],
        train_split: str = "train_pbr",
        val_split: str = "train_pbr",
        val_ratio: float = 0.05,
        obj_ids: Sequence[int] = (1,),
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 256,
        heatmap_size: int = 64,
        heatmap_style: str = "boxdreamer",
        sigma: float = 2.0,
        crop_scale: float = 1.4,
        use_gt_crop: bool = True,
        augment: bool = True,
        aug_color_jitter: float = 0.3,
        aug_blur_prob: float = 0.3,
        aug_noise_std: float = 0.02,
        aug_random_crop_jitter: float = 0.1,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = 512,
        shuffle_train: bool = True,
        seed: int = 42,
        min_px_visib: int = 64,
        min_visib_fract: float = 0.10,
        obj_mask_ratio: Optional[Sequence[float]] = None,
        crop_use_bbox_obj: bool = False,
        obj_paste_prob: float = 0.0,
        rgb_augmethods: Optional[Sequence[str]] = None,
        multi_instance: bool = False,
        multi_crop_scale: float = 2.5,
        max_instances: int = 8,
        center_sigma: float = 2.0,
        instance_min_visib: float = 0.10,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset_root = dataset_root
        self.train_split = train_split
        self.val_split = val_split
        self.val_ratio = float(val_ratio)
        self.obj_ids = list(obj_ids)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self._image_size = int(image_size)
        self._heatmap_size = int(heatmap_size)
        self._heatmap_style = str(heatmap_style)
        self._sigma = float(sigma)

        self._train_kwargs = dict(
            image_size=self._image_size,
            heatmap_size=self._heatmap_size,
            heatmap_style=self._heatmap_style,
            sigma=self._sigma,
            crop_scale=float(crop_scale),
            use_gt_crop=bool(use_gt_crop),
            obj_ids=self.obj_ids,
            augment=bool(augment),
            aug_color_jitter=float(aug_color_jitter),
            aug_blur_prob=float(aug_blur_prob),
            aug_noise_std=float(aug_noise_std),
            aug_random_crop_jitter=float(aug_random_crop_jitter),
            min_px_visib=int(min_px_visib),
            min_visib_fract=float(min_visib_fract),
            obj_mask_ratio=None if obj_mask_ratio is None else list(obj_mask_ratio),
            crop_use_bbox_obj=bool(crop_use_bbox_obj),
            obj_paste_prob=float(obj_paste_prob),
            rgb_augmethods=None if rgb_augmethods is None else list(rgb_augmethods),
            multi_instance=bool(multi_instance),
            multi_crop_scale=float(multi_crop_scale),
            max_instances=int(max_instances),
            center_sigma=float(center_sigma),
            instance_min_visib=float(instance_min_visib),
            aug_seed=int(seed),
        )
        self._max_train_samples = max_train_samples
        self._max_val_samples = max_val_samples
        self._shuffle_train = bool(shuffle_train)
        self._seed = int(seed)

        self.data_train: Optional[BOPPBRDataset] = None
        self.data_val: Optional[BOPPBRDataset] = None
        self.data_test: Optional[BOPPBRDataset] = None

    # ------------------------------------------------------------------ #
    def _make_dataset(self, split: str, augment: bool, max_samples: Optional[int]) -> BOPPBRDataset:
        kwargs = dict(self._train_kwargs)
        kwargs["augment"] = augment
        return BOPPBRDataset(
            dataset_root=self.dataset_root,
            split=split,
            max_samples=max_samples,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        # dataset_root 可能是单个路径，也可能是【列表】（多路径合并训练）。
        # Hydra 传进来的列表是 ListConfig，不能直接喂给 os.path.isdir，
        # 否则报 "stat: path should be string ... not ListConfig"。
        roots = self.dataset_root
        if isinstance(roots, (str, bytes)):
            roots = [roots]
        else:
            roots = [str(r) for r in roots]
        missing = [r for r in roots if not os.path.isdir(r)]
        if missing:
            raise FileNotFoundError(
                "dataset_root 不存在: " + ", ".join(missing) + "\n"
                "请先用 HCCEPose 的 s2_p1_gen_pbr_data.py 渲染数据，"
                "或用 datamodule.dataset_root=<你的路径> 覆盖配置。"
            )

        if self.data_train is not None and self.data_val is not None:
            return

        if self.val_split == self.train_split:
            # 同一 split 上建两份独立数据集：一份开增强（训练）、一份关（验证）。
            # 两份的样本顺序一致，因此可以用同一套下标切分。
            train_full = self._make_dataset(self.train_split, augment=True, max_samples=None)
            val_full = self._make_dataset(self.train_split, augment=False, max_samples=None)

            n_total = len(train_full)
            n_val = max(1, int(round(n_total * self.val_ratio))) if n_total > 1 else 0

            import random

            indices = list(range(n_total))
            random.Random(self._seed).shuffle(indices)

            val_idx = sorted(indices[:n_val])
            train_idx = sorted(indices[n_val:])

            if self._max_val_samples is not None and len(val_idx) > self._max_val_samples:
                val_idx = val_idx[: self._max_val_samples]
            if self._max_train_samples is not None and len(train_idx) > self._max_train_samples:
                train_idx = train_idx[: self._max_train_samples]

            self.data_train = Subset(train_full, train_idx)
            self.data_val = Subset(val_full, val_idx)
            self.data_test = self.data_val
        else:
            self.data_train = self._make_dataset(
                self.train_split, augment=True, max_samples=self._max_train_samples
            )
            self.data_val = self._make_dataset(
                self.val_split, augment=False, max_samples=self._max_val_samples
            )
            self.data_test = self.data_val

    # ------------------------------------------------------------------ #
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self._shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
