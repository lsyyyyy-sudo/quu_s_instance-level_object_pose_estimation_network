"""BOP 格式合成渲染数据集。

读取 HCCEPose 的 BlenderProc 脚本（``s2_p1_gen_pbr_data.py``）产出的标准 BOP 结构::

    <dataset_root>/
    ├── models/
    │   └── models_info.json
    └── train_pbr/
        ├── 000000/
        │   ├── rgb/000000.png
        │   ├── depth/000000.png
        │   ├── mask_visib/000000_000001.png
        │   ├── scene_camera.json
        │   ├── scene_gt.json
        │   └── scene_gt_info.json
        └── 000001/ ...

每个样本产出：裁剪缩放后的 RGB、对应的相机内参、物体坐标系下的 8 个 3D 角点、
GT 位姿，以及监督用的 2D 角点与高斯热图。

对应 BoxDreamer/src/datasets/ 里的各数据集实现（这里只有 BOP PBR 一种）。
"""

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.utils.box_utils import bbox3d_from_models_info, project_points
from src.models.utils.data_processing import make_heatmap_target


# HCCEPose 的渲染脚本用的是 ``color_file_format = "JPEG"``，
# 所以实际产物可能是 .jpg；这里把常见扩展名都试一遍，避免第一次渲染完就报找不到文件。
RGB_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def resolve_rgb_path(scene_dir: str, stem: str) -> Optional[str]:
    """在 ``<scene_dir>/rgb/`` 下找 ``<stem>.<ext>``，返回第一个存在的路径。

    Returns:
        找到的完整路径；都不存在则返回 ``None``。
    """
    rgb_dir = os.path.join(scene_dir, "rgb")
    for ext in RGB_EXTENSIONS:
        path = os.path.join(rgb_dir, f"{stem}{ext}")
        if os.path.isfile(path):
            return path
    return None


# --------------------------------------------------------------------------- #
# 几何：裁剪 + 缩放
# --------------------------------------------------------------------------- #
def crop_and_resize(
    image: np.ndarray,
    K: np.ndarray,
    bbox_xywh: Sequence[float],
    out_size: int,
    crop_scale: float = 1.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 2D 框裁剪并缩放到正方形，同时同步更新相机内参。

    Args:
        image:      ``[H, W, 3]`` RGB uint8
        K:          ``[3, 3]``
        bbox_xywh:  ``[x, y, w, h]``
        out_size:   输出边长
        crop_scale: 裁剪框相对物体框的外扩系数

    Returns:
        ``(image_out, K_out)``，``image_out`` 为 ``[out_size, out_size, 3]``
    """
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * float(crop_scale)
    side = max(side, 1.0)

    x0 = cx - side / 2.0
    y0 = cy - side / 2.0
    scale = float(out_size) / side

    M = np.array([[scale, 0.0, -x0 * scale], [0.0, scale, -y0 * scale]], dtype=np.float32)
    image_out = cv2.warpAffine(
        image,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    K_out = K.astype(np.float64).copy()
    K_out[0, 0] *= scale
    K_out[1, 1] *= scale
    K_out[0, 2] = (K[0, 2] - x0) * scale
    K_out[1, 2] = (K[1, 2] - y0) * scale
    return image_out, K_out


# --------------------------------------------------------------------------- #
# 数据增强
# --------------------------------------------------------------------------- #
def augment_image(
    image: np.ndarray,
    color_jitter: float = 0.3,
    blur_prob: float = 0.3,
    noise_std: float = 0.02,
) -> np.ndarray:
    """贴合真实视频域差异的增强：颜色抖动 + 模糊 + 噪声。

    真实数据是头戴相机拍的手持视频，存在白平衡漂移、运动模糊与传感器噪声，
    这三点是缩小 sim-to-real 差距最直接的手段。
    """
    img = image.astype(np.float32)

    if color_jitter > 0:
        for c in range(3):
            gain = 1.0 + random.uniform(-color_jitter, color_jitter)
            bias = random.uniform(-color_jitter, color_jitter) * 32.0
            img[..., c] = img[..., c] * gain + bias

    if blur_prob > 0 and random.random() < blur_prob:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if noise_std > 0:
        img = img + np.random.randn(*img.shape).astype(np.float32) * noise_std * 255.0

    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class BOPPBRDataset(Dataset):
    """BOP PBR 渲染数据集（单物体、单图输入）。

    Args:
        dataset_root: 数据集根目录
        split:        子目录名，如 ``train_pbr``
        obj_ids:      只取这些物体 id 的标注
        image_size:   网络输入边长
        heatmap_size: 监督热图边长
        sigma:        高斯峰标准差
        crop_scale:   裁剪外扩系数
        use_gt_crop:  True 用 GT 框裁剪；False 用整图（更接近真实部署）
        augment:      是否做数据增强
        max_samples:  限制样本数（调试用）
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train_pbr",
        obj_ids: Sequence[int] = (1,),
        image_size: int = 256,
        heatmap_size: int = 64,
        heatmap_style: str = "boxdreamer",
        sigma: float = 2.0,
        crop_scale: float = 1.4,
        use_gt_crop: bool = True,
        augment: bool = False,
        aug_color_jitter: float = 0.3,
        aug_blur_prob: float = 0.3,
        aug_noise_std: float = 0.02,
        aug_random_crop_jitter: float = 0.1,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.obj_ids = list(obj_ids)
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.heatmap_style = str(heatmap_style)
        self.sigma = float(sigma)
        self.crop_scale = float(crop_scale)
        self.use_gt_crop = bool(use_gt_crop)
        self.augment = bool(augment)
        self.aug_color_jitter = float(aug_color_jitter)
        self.aug_blur_prob = float(aug_blur_prob)
        self.aug_noise_std = float(aug_noise_std)
        self.aug_random_crop_jitter = float(aug_random_crop_jitter)

        models_info_path = os.path.join(dataset_root, "models", "models_info.json")
        if not os.path.isfile(models_info_path):
            raise FileNotFoundError(
                f"models_info.json not found: {models_info_path}\n"
                "先跑 HCCEPose 的 s1_p3_obj_infos.py 生成它（见 README 的流水线说明）。"
            )
        with open(models_info_path, "r", encoding="utf-8") as f:
            self.models_info = json.load(f)

        # 缓存每个物体的 3D 包围盒角点 [8, 3]
        self.bbox3d: Dict[int, torch.Tensor] = {
            int(o): bbox3d_from_models_info(self.models_info, int(o)) for o in self.obj_ids
        }

        self.samples = self._build_index()
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[: int(max_samples)]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found under {os.path.join(dataset_root, split)} "
                f"for obj_ids={self.obj_ids}. 渲染数据是否已经生成？"
            )

    # ------------------------------------------------------------------ #
    def _build_index(self) -> List[Tuple[str, int, int]]:
        """扫描 split 目录下所有 scene，建立 (scene_dir, frame_id, gt_idx) 索引。"""
        split_dir = os.path.join(self.dataset_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples: List[Tuple[str, int, int]] = []
        for scene_name in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene_name)
            gt_path = os.path.join(scene_dir, "scene_gt.json")
            if not os.path.isfile(gt_path):
                continue
            with open(gt_path, "r", encoding="utf-8") as f:
                scene_gt = json.load(f)
            for frame_id, annotations in scene_gt.items():
                for gt_idx, ann in enumerate(annotations):
                    if int(ann["obj_id"]) in self.obj_ids:
                        samples.append((scene_dir, int(frame_id), gt_idx))
        return samples

    def _load_scene_json(self, scene_dir: str, name: str) -> dict:
        path = os.path.join(scene_dir, name)
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        scene_dir, frame_id, gt_idx = self.samples[index]
        stem = f"{frame_id:06d}"

        scene_camera = self._load_scene_json(scene_dir, "scene_camera.json")
        scene_gt = self._load_scene_json(scene_dir, "scene_gt.json")
        scene_gt_info = self._load_scene_json(scene_dir, "scene_gt_info.json")

        cam = scene_camera[str(frame_id)]
        K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)

        ann = scene_gt[str(frame_id)][gt_idx]
        obj_id = int(ann["obj_id"])
        R = np.array(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t = np.array(ann["cam_t_m2c"], dtype=np.float64).reshape(3)

        pose = torch.eye(4, dtype=torch.float32)
        pose[:3, :3] = torch.from_numpy(R).float()
        pose[:3, 3] = torch.from_numpy(t).float()

        bbox_3d = self.bbox3d[obj_id]  # [8, 3]

        image_path = resolve_rgb_path(scene_dir, stem)
        if image_path is None:
            raise FileNotFoundError(
                f"RGB not found for frame {stem} under {os.path.join(scene_dir, 'rgb')} "
                f"(tried {', '.join(RGB_EXTENSIONS)})"
            )
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---- 物体 2D 框：优先用 scene_gt_info 的 bbox_visib，缺失则用投影角点算 ----
        bbox = self._get_bbox(scene_gt_info, frame_id, gt_idx, bbox_3d, K, pose, image.shape)

        if self.augment and self.aug_random_crop_jitter > 0:
            bbox = self._jitter_bbox(bbox, image.shape)

        if self.use_gt_crop:
            image, K = crop_and_resize(
                image, K, bbox, self.image_size, crop_scale=self.crop_scale
            )
        else:
            # 整图缩放到 image_size，内参按 x/y 两个方向分别缩放。
            # 注意图像被各向异性缩放，针孔模型依然精确（fx、fy 独立缩放即可）。
            h0, w0 = image.shape[:2]
            sx = self.image_size / float(w0)
            sy = self.image_size / float(h0)
            image = cv2.resize(
                image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
            K = K.copy()
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy

        if self.augment:
            image = augment_image(
                image,
                color_jitter=self.aug_color_jitter,
                blur_prob=self.aug_blur_prob,
                noise_std=self.aug_noise_std,
            )

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        K_t = torch.from_numpy(K.astype(np.float32))

        # ---- 2D 角点标签：把 3D 角点用 GT 位姿投影到（已裁剪的）图像上 ----
        corner_2d = project_points(bbox_3d.unsqueeze(0), K_t.unsqueeze(0), pose.unsqueeze(0))[0]

        heatmap = make_heatmap_target(
            corner_2d.unsqueeze(0),
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
            style=self.heatmap_style,
        )[0]

        return {
            "image": image_t,                 # [3, H, W]
            "cam_K": K_t,                     # [3, 3]
            "bbox_3d": bbox_3d,               # [8, 3] 物体坐标系
            "corner_2d": corner_2d,           # [8, 2] 图像像素
            "heatmap": heatmap,               # [8, h, w]，取值随 heatmap_style
            "pose_gt": pose,                  # [4, 4]
            "obj_id": torch.tensor(obj_id, dtype=torch.long),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
            "image_path": image_path,
        }

    # ------------------------------------------------------------------ #
    def _get_bbox(self, scene_gt_info, frame_id, gt_idx, bbox_3d, K, pose, image_shape) -> np.ndarray:
        """取物体 2D 框 ``[x, y, w, h]``。"""
        info = None
        if scene_gt_info:
            entries = scene_gt_info.get(str(frame_id), [])
            if gt_idx < len(entries):
                info = entries[gt_idx]

        if info is not None:
            for key in ("bbox_visib", "bbox_obj"):
                if key in info and info[key] is not None:
                    bbox = np.array(info[key], dtype=np.float64)
                    if bbox[2] > 1 and bbox[3] > 1:
                        return bbox

        # 回退：直接把 3D 角点投到图上取外接框
        corner_2d = project_points(bbox_3d.unsqueeze(0), torch.from_numpy(K).float().unsqueeze(0),
                                   torch.from_numpy(pose.numpy()).float().unsqueeze(0))[0].numpy()
        x0, y0 = corner_2d.min(axis=0)
        x1, y1 = corner_2d.max(axis=0)
        return np.array([x0, y0, max(x1 - x0, 1.0), max(y1 - y0, 1.0)], dtype=np.float64)

    def _jitter_bbox(self, bbox: np.ndarray, image_shape) -> np.ndarray:
        """随机扰动裁剪框，模拟检测器给出的框不完美。"""
        x, y, w, h = bbox
        j = self.aug_random_crop_jitter
        cx = x + w / 2.0 + random.uniform(-j, j) * w
        cy = y + h / 2.0 + random.uniform(-j, j) * h
        scale = 1.0 + random.uniform(-j, j) * 2.0
        w2, h2 = w * scale, h * scale
        return np.array([cx - w2 / 2.0, cy - h2 / 2.0, w2, h2], dtype=np.float64)
