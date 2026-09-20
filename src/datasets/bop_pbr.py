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

from src.datasets.utils.aug_boxdreamer import (  # noqa: E402
    apply_rgb_augmentation,
    mask_bbox_region,
    paste_foreign_object,
)
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
        min_px_visib: int = 64,
        min_visib_fract: float = 0.10,
        obj_mask_ratio: Optional[Sequence[float]] = None,
        crop_use_bbox_obj: bool = False,
        obj_paste_prob: float = 0.0,
        rgb_augmethods: Optional[Sequence[str]] = None,
        aug_seed: int = 0,
        multi_instance: bool = False,
        multi_crop_scale: float = 2.5,
        max_instances: int = 8,
        center_sigma: float = 2.0,
        instance_min_visib: float = 0.10,
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
        # 可见性过滤：完全（或几乎完全）被挡住的实例不能用。
        # 裁剪按 bbox_visib 做，像素可见数为 0 时那个框是退化的（0 大小），
        # 裁出来是一块无关的背景，却配着一个"正确"的角点标签 —— 纯噪声。
        # 料箱堆叠之后这类实例占比很高（实测 25%），必须滤掉。
        self.min_px_visib = int(min_px_visib)
        self.min_visib_fract = float(min_visib_fract)

        # ---- BoxDreamer 式增强（遮挡合成 + 光度）----
        # 为什么需要：训练集里 95% 实例完全可见（DATA-16），模型在遮挡处失败率 45%
        # （完全可见只有 0.79%）；且没有真实相机的光度特性。详见
        # src/datasets/utils/aug_boxdreamer.py 的模块文档。
        self.obj_mask_ratio = (
            None if obj_mask_ratio is None
            else (float(obj_mask_ratio[0]), float(obj_mask_ratio[1]))
        )
        # 裁剪用 bbox_obj（amodal 全物体框）还是 bbox_visib（可见部分框）。
        # 默认 False 保持v1 的历史行为；开启它才和 amodal 角点标签语义一致。
        self.crop_use_bbox_obj = bool(crop_use_bbox_obj)
        self.obj_paste_prob = float(obj_paste_prob)
        self.rgb_augmethods = list(rgb_augmethods) if rgb_augmethods else []
        # 每个 worker 进程用自己的 rng，避免所有 worker 抽到同一串随机数
        self._rng = np.random.default_rng(aug_seed + (os.getpid() % 100000))

        # ---- 多实例模式 ----
        # 动机：目标视频里往往有多个同类相机，而单实例裁剪只监督一个，
        # 模型只能靠"目标在裁剪图中心"这个位置先验去猜（实测 88.3% 的裁剪图
        # 里混进了其他实例）。多实例模式下裁剪放大到 multi_crop_scale，
        # **框内所有实例都监督**，"哪个是目标"这个问题就不存在了。
        self.multi_instance = bool(multi_instance)
        self.multi_crop_scale = float(multi_crop_scale)
        self.max_instances = int(max_instances)
        self.center_sigma = float(center_sigma)
        self.instance_min_visib = float(instance_min_visib)
        if self.multi_instance:
            # 多实例必须用放大后的裁剪，否则框里只有一个物体
            self.crop_scale = self.multi_crop_scale

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
        n_skip_px = n_skip_frac = 0
        for scene_name in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene_name)
            gt_path = os.path.join(scene_dir, "scene_gt.json")
            if not os.path.isfile(gt_path):
                continue
            with open(gt_path, "r", encoding="utf-8") as f:
                scene_gt = json.load(f)
            # scene_gt_info.json 里的 px_count_visib / visib_fract 用来做可见性过滤
            scene_gti = self._load_scene_json(scene_dir, "scene_gt_info.json")
            for frame_id, annotations in scene_gt.items():
                frame_info = scene_gti.get(str(frame_id), [])
                for gt_idx, ann in enumerate(annotations):
                    if int(ann["obj_id"]) not in self.obj_ids:
                        continue
                    if 0 <= gt_idx < len(frame_info):
                        info = frame_info[gt_idx]
                        if int(info.get("px_count_visib", 1 << 30)) < self.min_px_visib:
                            n_skip_px += 1
                            continue
                        if float(info.get("visib_fract", 1.0)) < self.min_visib_fract:
                            n_skip_frac += 1
                            continue
                    samples.append((scene_dir, int(frame_id), gt_idx))
        if n_skip_px or n_skip_frac:
            print(f"[bop_pbr] 可见性过滤：丢掉 {n_skip_px} 个 px_count_visib<{self.min_px_visib}"
                  f"、{n_skip_frac} 个 visib_fract<{self.min_visib_fract}，"
                  f"保留 {len(samples)} 个样本")
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

        # ---- BoxDreamer 式「物体贴物体」遮挡合成：先把兄弟实例抠出来备用 ----
        # 必须在裁剪**之前**取，因为兄弟实例在裁剪图里可能只露出一部分。
        foreign = None
        if self.augment and random.random() < self.obj_paste_prob:
            foreign = self._pick_foreign_object(scene_dir, stem, frame_id, gt_idx, image.shape)

        if self.augment and self.aug_random_crop_jitter > 0:
            bbox = self._jitter_bbox(bbox, image.shape)

        if self.use_gt_crop:
            # 保留原图引用：遮挡合成要用兄弟实例在原图里的像素。
            # crop_and_resize 返回的是新数组（warpAffine），不会改动原图。
            image_orig = image
            image, K = crop_and_resize(
                image, K, bbox, self.image_size, crop_scale=self.crop_scale
            )
        else:
            image_orig = image
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
            # ---- ① 先做遮挡合成，② 再统一做光度增强 ----
            # 顺序有讲究：光度增强要作用在**合成后**的图上，否则贴上去的物体
            # 和底图的光照/噪声分布不一致，网络能靠这一点作弊区分。
            if foreign is not None:
                image = self._paste_foreign(image, foreign, bbox, image_orig)

            if self.obj_mask_ratio and self.obj_mask_ratio[1] > 0:
                xywh_crop = self._bbox_in_crop(bbox, self.image_size)
                image = mask_bbox_region(image, xywh_crop, self.obj_mask_ratio, self._rng)

            if self.rgb_augmethods:
                image = apply_rgb_augmentation(image, self.rgb_augmethods, self._rng)

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

        # ---- 可见性（评估时按遮挡程度分层用）----
        # 训练不需要它，但评估时必须能回答"误差是不是集中在被遮挡的样本上"，
        # 这是 DATA-16（训练集缺遮挡）假设的直接检验。
        _info = None
        _entries = scene_gt_info.get(str(frame_id), []) if scene_gt_info else []
        if gt_idx < len(_entries):
            _info = _entries[gt_idx]
        visib_fract = float(_info.get("visib_fract", 1.0)) if _info else 1.0
        px_count_visib = int(_info.get("px_count_visib", -1)) if _info else -1

        out = {
            "image": image_t,                 # [3, H, W]
            "cam_K": K_t,                     # [3, 3]
            "bbox_3d": bbox_3d,               # [8, 3] 物体坐标系
            "corner_2d": corner_2d,           # [8, 2] 图像像素
            "heatmap": heatmap,               # [8, h, w]，取值随 heatmap_style
            "pose_gt": pose,                  # [4, 4]
            "obj_id": torch.tensor(obj_id, dtype=torch.long),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
            "image_path": image_path,
            "visib_fract": torch.tensor(visib_fract, dtype=torch.float32),
            "px_count_visib": torch.tensor(px_count_visib, dtype=torch.long),
        }

        # ---- 多实例标签 ----
        # 监督裁剪框内的【所有】实例：每个实例一个中心峰 + 8 个角点。
        # 这样"哪个是目标"不再是问题 —— 全部都是目标。
        if self.multi_instance:
            insts = self._collect_crop_instances(
                scene_gt, scene_gt_info, frame_id, K, self.image_size, bbox
            )
            n = len(insts)
            ctr_hm = np.zeros((1, self.heatmap_size, self.heatmap_size), dtype=np.float32)
            corners = np.zeros((self.max_instances, 8, 2), dtype=np.float32)
            centers = np.zeros((self.max_instances, 2), dtype=np.float32)
            valid = np.zeros((self.max_instances,), dtype=np.float32)
            inst_visib = np.zeros((self.max_instances,), dtype=np.float32)

            scale = self.heatmap_size / float(self.image_size)
            yy, xx = np.mgrid[0:self.heatmap_size, 0:self.heatmap_size]
            for i, d in enumerate(insts):
                corners[i] = d["corners"]
                centers[i] = d["center"]
                valid[i] = 1.0
                inst_visib[i] = d["visib"]
                # 中心热图：在该实例中心放一个高斯峰（取 max 防止重叠处叠加爆炸）
                gx = d["center"][0] * scale
                gy = d["center"][1] * scale
                g = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2)
                           / (2.0 * self.center_sigma ** 2)).astype(np.float32)
                ctr_hm[0] = np.maximum(ctr_hm[0], g)

            out.update({
                "center_heatmap": torch.from_numpy(ctr_hm),          # [1, h, w]
                "inst_corners": torch.from_numpy(corners),           # [N, 8, 2]
                "inst_centers": torch.from_numpy(centers),           # [N, 2] 裁剪像素
                "inst_valid": torch.from_numpy(valid),               # [N]
                "inst_visib": torch.from_numpy(inst_visib),          # [N]
                "n_instances": torch.tensor(n, dtype=torch.long),
            })
        return out


    # ------------------------------------------------------------------ #
    def _collect_crop_instances(self, scene_gt, scene_gt_info, frame_id,
                                K_crop, image_size, crop_box):
        """收集【裁剪框内】的所有实例，返回裁剪坐标系下的角点与中心。

        Args:
            K_crop:     已按裁剪调整过的内参（crop_and_resize 的返回值）
            crop_box:   裁剪用的 [x, y, w, h]（原图坐标），用于把中心判进框内

        Returns:
            list of dict: {"corners": [8,2], "center": [2], "visib": float,
                           "obj_id": int, "gt_idx": int}
            corners/center 都在**裁剪后的 image_size x image_size** 坐标系里。
        """
        anns = scene_gt.get(str(frame_id), [])
        infos = scene_gt_info.get(str(frame_id), []) if scene_gt_info else []
        K_t = torch.from_numpy(K_crop.astype(np.float32)).unsqueeze(0)

        out = []
        for j, ann in enumerate(anns):
            obj_id = int(ann["obj_id"])
            if obj_id not in self.bbox3d:
                continue                    # 只关心我们建模的物体
            R = np.array(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t = np.array(ann["cam_t_m2c"], dtype=np.float64).reshape(3)
            pose = torch.eye(4, dtype=torch.float32)
            pose[:3, :3] = torch.from_numpy(R).float()
            pose[:3, 3] = torch.from_numpy(t).float()

            info = infos[j] if j < len(infos) else {}
            visib = float(info.get("visib_fract", 1.0))
            if visib < self.instance_min_visib:
                continue                    # 几乎完全被挡的实例不要（噪声标签）

            c2 = project_points(self.bbox3d[obj_id].unsqueeze(0),
                                K_t, pose.unsqueeze(0))[0].numpy()   # [8,2]
            if not np.all(np.isfinite(c2)):
                continue
            center = c2.mean(axis=0)

            # 中心必须落在裁剪图内（留 2% 边距），否则这个实例只是擦边
            m = 0.02 * image_size
            if not (m <= center[0] <= image_size - m and m <= center[1] <= image_size - m):
                continue

            out.append({
                "corners": c2.astype(np.float32),
                "center": center.astype(np.float32),
                "visib": visib,
                "obj_id": obj_id,
                "gt_idx": j,
            })

        # 按"离画面中心的距离"排序：目标实例通常最靠中间，放第一个便于对照
        cx = image_size / 2.0
        out.sort(key=lambda d: (d["center"][0] - cx) ** 2 + (d["center"][1] - cx) ** 2)
        return out[: self.max_instances]

    # ------------------------------------------------------------------ #
    def _get_bbox(self, scene_gt_info, frame_id, gt_idx, bbox_3d, K, pose, image_shape) -> np.ndarray:
        """取物体 2D 框 ``[x, y, w, h]``。"""
        info = None
        if scene_gt_info:
            entries = scene_gt_info.get(str(frame_id), [])
            if gt_idx < len(entries):
                info = entries[gt_idx]

        if info is not None:
            # ⚠️ 用哪个框裁剪，是个**必须和标签语义一致**的选择。
            # 标签是 3D 盒 8 个角点的投影（**amodal**，含被挡住的部分），
            # 所以按道理该用 bbox_obj（全物体框）。用 bbox_visib（可见部分框）
            # 在物体被严重遮挡时会退化：框偏移、变小、甚至部分跑到画面外
            # （实测 v2 有 bbox_visib=[-512, 0, 78, 39] 的样本），
            # 裁出来是无关背景，标签却还是这个物体的角点 —— 纯噪声样本。
            # v1 侥幸没暴露：它 95.3% 的实例 bbox_visib == bbox_obj；
            # v2 加料箱造遮挡后这个潜伏 bug 立刻引爆（见 docs/RESULTS.md）。
            order = ("bbox_obj", "bbox_visib") if self.crop_use_bbox_obj \
                else ("bbox_visib", "bbox_obj")
            for key in order:
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

    # ------------------------------------------------------------------ #
    # BoxDreamer 式「物体贴物体」遮挡合成
    # ------------------------------------------------------------------ #
    def _bbox_in_crop(self, bbox_xywh, out_size: int) -> list:
        """把原图坐标的 ``[x,y,w,h]`` 换算到裁剪图坐标系（与 crop_and_resize 同一变换）。"""
        x, y, w, h = [float(v) for v in bbox_xywh]
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(max(w, h) * float(self.crop_scale), 1.0)
        x0, y0 = cx - side / 2.0, cy - side / 2.0
        s = float(out_size) / side
        return [(x - x0) * s, (y - y0) * s, w * s, h * s]

    def _pick_foreign_object(self, scene_dir, stem, frame_id, gt_idx, image_shape):
        """在同一帧里随机挑一个**别的实例**，抠出它的物体像素当作遮挡源。

        为什么不跨帧取：同帧的兄弟实例本来就在同一个场景里（同一料箱、同一光照），
        贴上去最自然，而且不用额外读整张图。

        Returns:
            ``(bbox_xywh, mask_bool)`` 或 ``None``。``mask_bool`` 是原图尺寸的布尔掩码。
        """
        gti = self._load_scene_json(scene_dir, "scene_gt_info.json")
        entries = gti.get(str(frame_id), [])
        # 挑一个可见像素足够多、且不是自己的实例
        cands = [
            i for i, e in enumerate(entries)
            if i != gt_idx and int(e.get("px_count_visib", 0)) >= 200
            and e.get("bbox_visib", [-1, -1, -1, -1])[2] > 20
        ]
        if not cands:
            return None
        j = int(self._rng.choice(cands))
        mask_path = os.path.join(scene_dir, "mask_visib", f"{stem}_{j:06d}.png")
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        mask = m > 0
        if mask.sum() < 200:
            return None
        return np.array(entries[j]["bbox_visib"], dtype=np.float64), mask

    def _paste_foreign(self, image_crop: np.ndarray, foreign, target_bbox,
                       src_image: np.ndarray) -> np.ndarray:
        """把兄弟实例的物体像素贴到**目标的位置**上，形成真实遮挡。"""
        sib_bbox, sib_mask = foreign
        ys, xs = np.where(sib_mask)
        if len(xs) == 0:
            return image_crop
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        if src_image is None or src_image.shape[:2] != sib_mask.shape:
            return image_crop
        rgb = src_image[y0:y1, x0:x1]
        alpha = (sib_mask[y0:y1, x0:x1].astype(np.uint8)) * 255

        # 贴到目标 bbox 在裁剪图里的位置，尺寸随机 0.6~1.1 倍并小幅偏移
        tx, ty, tw, th = self._bbox_in_crop(target_bbox, self.image_size)
        k = float(self._rng.uniform(0.6, 1.1))
        dw, dh = tw * k, th * k
        dx = tx + float(self._rng.uniform(-0.15, 0.15)) * tw
        dy = ty + float(self._rng.uniform(-0.15, 0.15)) * th
        return paste_foreign_object(image_crop, rgb, alpha, (dx, dy), (dw, dh), self._rng)
