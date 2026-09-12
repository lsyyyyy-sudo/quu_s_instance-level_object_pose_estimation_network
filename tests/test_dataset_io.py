"""数据集 IO 的自检用例（不需要真实数据，用临时目录构造 BOP 结构）。"""

import json
import os

import numpy as np
import pytest

from src.datasets.bop_pbr import RGB_EXTENSIONS, resolve_rgb_path


def _make_scene(tmp_path, stem="000000", ext=".png"):
    scene_dir = tmp_path / "train_pbr" / "000000"
    (scene_dir / "rgb").mkdir(parents=True)
    rgb_path = scene_dir / "rgb" / f"{stem}{ext}"
    rgb_path.write_bytes(b"not a real image")
    return str(scene_dir), str(rgb_path)


# --------------------------------------------------------------------------- #
def test_resolve_rgb_path_finds_png(tmp_path):
    scene_dir, rgb_path = _make_scene(tmp_path, ext=".png")
    assert resolve_rgb_path(scene_dir, "000000") == rgb_path


def test_resolve_rgb_path_finds_jpg(tmp_path):
    """HCCEPose 的渲染脚本用 color_file_format='JPEG'，产物可能是 .jpg。"""
    scene_dir, rgb_path = _make_scene(tmp_path, ext=".jpg")
    assert resolve_rgb_path(scene_dir, "000000") == rgb_path


def test_resolve_rgb_path_finds_jpeg(tmp_path):
    scene_dir, rgb_path = _make_scene(tmp_path, ext=".jpeg")
    assert resolve_rgb_path(scene_dir, "000000") == rgb_path


def test_resolve_rgb_path_prefers_png(tmp_path):
    """两种扩展名都存在时，优先 .png。"""
    scene_dir, png_path = _make_scene(tmp_path, ext=".png")
    (tmp_path / "train_pbr" / "000000" / "rgb" / "000000.jpg").write_bytes(b"x")
    assert resolve_rgb_path(scene_dir, "000000") == png_path


def test_resolve_rgb_path_returns_none_when_missing(tmp_path):
    scene_dir, _ = _make_scene(tmp_path, ext=".png")
    assert resolve_rgb_path(scene_dir, "999999") is None


def test_resolve_rgb_path_returns_none_without_rgb_dir(tmp_path):
    scene_dir = tmp_path / "train_pbr" / "000000"
    scene_dir.mkdir(parents=True)
    assert resolve_rgb_path(str(scene_dir), "000000") is None


def test_rgb_extensions_cover_case_variants():
    """大小写变体都要覆盖（不同渲染后端输出的扩展名大小写不一致）。"""
    assert ".png" in RGB_EXTENSIONS and ".PNG" in RGB_EXTENSIONS
    assert ".jpg" in RGB_EXTENSIONS and ".JPG" in RGB_EXTENSIONS


# --------------------------------------------------------------------------- #
def test_dataset_reports_missing_dataset_root(tmp_path):
    """dataset_root 不存在时应给出可读的报错，而不是下游的 KeyError。"""
    from src.datasets.bop_pbr import BOPPBRDataset

    with pytest.raises(FileNotFoundError, match="models_info.json"):
        BOPPBRDataset(dataset_root=str(tmp_path / "nope"), split="train_pbr")


def test_dataset_builds_index_and_labels(tmp_path):
    """用最小 BOP 结构跑通 __getitem__：索引 -> 裁剪 -> 投影 -> 热图。"""
    from src.datasets.bop_pbr import BOPPBRDataset

    root = tmp_path / "dji"
    scene = root / "train_pbr" / "000000"
    (root / "models").mkdir(parents=True)
    (scene / "rgb").mkdir(parents=True)

    # models_info.json：边长 70.5 x 44.2 x 32.8 mm 的盒子
    (root / "models" / "models_info.json").write_text(
        json.dumps({
            "1": {
                "min_x": -35.25, "min_y": -22.1, "min_z": -16.4,
                "max_x": 35.25, "max_y": 22.1, "max_z": 16.4,
                "size_x": 70.5, "size_y": 44.2, "size_z": 32.8,
                "diameter": 89.4,
            }
        }),
        encoding="utf-8",
    )

    # 一张 400x400 的假图（JPEG 扩展名，顺便验证扩展名回退）
    img = (np.random.rand(400, 400, 3) * 255).astype(np.uint8)
    import cv2
    cv2.imwrite(str(scene / "rgb" / "000000.jpg"), img)

    K = [500.0, 0.0, 200.0, 0.0, 500.0, 200.0, 0.0, 0.0, 1.0]
    (scene / "scene_camera.json").write_text(
        json.dumps({"0": {"cam_K": K, "depth_scale": 0.1}}), encoding="utf-8"
    )
    (scene / "scene_gt.json").write_text(
        json.dumps({"0": [{
            "obj_id": 1,
            "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "cam_t_m2c": [0.0, 0.0, 400.0],
        }]}),
        encoding="utf-8",
    )
    (scene / "scene_gt_info.json").write_text(
        json.dumps({"0": [{"bbox_visib": [150, 150, 100, 100]}]}), encoding="utf-8"
    )

    ds = BOPPBRDataset(
        dataset_root=str(root),
        split="train_pbr",
        obj_ids=[1],
        image_size=64,
        heatmap_size=16,
        heatmap_style="boxdreamer",
    )
    assert len(ds) == 1

    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["heatmap"].shape == (8, 16, 16)
    assert item["bbox_3d"].shape == (8, 3)
    assert item["corner_2d"].shape == (8, 2)
    assert item["pose_gt"].shape == (4, 4)
    assert item["image_path"].endswith(".jpg")
    # boxdreamer 风格的热图取值在 [-1, 1]
    assert float(item["heatmap"].min()) >= -1.0 - 1e-5
    assert abs(float(item["heatmap"].max()) - 1.0) < 1e-3
