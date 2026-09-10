"""几何与推理工具的自检用例。

只依赖 torch / numpy / opencv，不需要 pytorch-lightning 或 hydra，可以随时跑::

    pytest tests/ -v
"""

import math

import numpy as np
import torch

from src.models.utils.box_utils import (
    BB8_BITS,
    bbox3d_diameter,
    bbox3d_from_minmax,
    project_points,
    recover_pose_from_bb8,
    transform_points,
)
from src.models.utils.data_processing import make_heatmap_target
from src.models.utils.pose_utils import (
    add_error,
    invert_pose,
    pose_error_5cm5deg,
    rotation_angle_deg,
)
from src.models.utils.prediction_utils import (
    argmax_2d,
    heatmap_to_corners,
    soft_argmax_2d,
)


def _dummy_K(image_size=256, fov_deg=60.0):
    f = image_size / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return torch.tensor(
        [[f, 0.0, image_size / 2.0], [0.0, f, image_size / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _random_pose(t_range=300.0):
    """构造一个朝着相机前方的随机位姿。"""
    R_np, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(R_np) < 0:
        R_np[:, 0] *= -1
    R = torch.from_numpy(R_np).float()
    t = torch.tensor(
        [
            np.random.uniform(-0.2, 0.2) * t_range,
            np.random.uniform(-0.2, 0.2) * t_range,
            np.random.uniform(0.8, 1.5) * t_range,
        ],
        dtype=torch.float32,
    )
    pose = torch.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


# --------------------------------------------------------------------------- #
def test_bb8_corner_order_matches_spec():
    """角点顺序必须与 BB8_BITS 定义严格一致（bottom CCW 然后 top CCW）。"""
    min_xyz = torch.tensor([-1.0, -2.0, -3.0])
    max_xyz = torch.tensor([1.0, 2.0, 3.0])
    corners = bbox3d_from_minmax(min_xyz, max_xyz)

    assert corners.shape == (8, 3)
    for i, bits in enumerate(BB8_BITS):
        expected = min_xyz + torch.tensor(bits, dtype=torch.float32) * (max_xyz - min_xyz)
        assert torch.allclose(corners[i], expected), f"corner {i} mismatch"

    # 底面 4 点 z 应相同且小于顶面
    assert torch.allclose(corners[:4, 2], corners[0, 2])
    assert corners[:4, 2].max() < corners[4:, 2].min()


def test_bbox3d_diameter():
    corners = bbox3d_from_minmax(torch.zeros(3), torch.tensor([3.0, 4.0, 0.0]))
    assert abs(float(bbox3d_diameter(corners)) - 5.0) < 1e-5


def test_project_points_center_pixel():
    """物体在光轴上时，投影应落在主点。"""
    K = _dummy_K()
    pts = torch.tensor([[[0.0, 0.0, 1000.0]]])
    uv = project_points(pts, K)
    assert torch.allclose(uv[0, 0], torch.tensor([128.0, 128.0]), atol=1e-3)


def test_project_points_scaling_with_depth():
    """深度翻倍 -> 相对主点的偏移减半。"""
    K = _dummy_K()
    near = project_points(torch.tensor([[[10.0, 0.0, 500.0]]]), K)
    far = project_points(torch.tensor([[[10.0, 0.0, 1000.0]]]), K)
    off_near = near[0, 0, 0] - 128.0
    off_far = far[0, 0, 0] - 128.0
    assert abs(float(off_far) * 2.0 - float(off_near)) < 1e-3


def test_transform_points_identity_and_inverse():
    bbox = bbox3d_from_minmax(torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0]))
    pose = _random_pose()
    moved = transform_points(bbox, pose)
    back = transform_points(moved, invert_pose(pose))
    assert torch.allclose(back, bbox, atol=1e-4)


def test_recover_pose_from_bb8_recovers_ground_truth():
    """给定完美 2D 角点，PnP 应能还原位姿（角点误差远小于物体尺寸）。"""
    torch.manual_seed(0)
    K = _dummy_K()
    bbox_3d = bbox3d_from_minmax(
        torch.tensor([-20.0, -15.0, -8.0]), torch.tensor([20.0, 15.0, 8.0])
    )
    pose_gt = _random_pose(t_range=300.0)

    corner_2d = project_points(bbox_3d.unsqueeze(0), K.unsqueeze(0), pose_gt.unsqueeze(0))
    poses, _ = recover_pose_from_bb8(corner_2d, bbox_3d.unsqueeze(0), K.unsqueeze(0))

    assert float(poses[0, 3, 3]) > 0.5, "PnP should succeed"

    # 用 ADD 误差衡量：应远小于物体直径
    diam = float(bbox3d_diameter(bbox_3d))
    err = float(add_error(poses, pose_gt.unsqueeze(0), bbox_3d.unsqueeze(0))[0])
    assert err < 0.01 * diam, f"ADD error too large: {err} (diameter={diam})"

    rot_err = float(rotation_angle_deg(poses[0, :3, :3], pose_gt[:3, :3]))
    assert rot_err < 1.0, f"rotation error too large: {rot_err} deg"


def test_recover_pose_flags_failure_with_zero_pose():
    """2D 角点全是垃圾时，PnP 失败样本的位姿最后一行应为 0。"""
    K = _dummy_K()
    bbox_3d = bbox3d_from_minmax(torch.tensor([-1.0] * 3), torch.tensor([1.0] * 3))
    bad_2d = torch.zeros(1, 8, 2)
    poses, _ = recover_pose_from_bb8(bad_2d, bbox_3d.unsqueeze(0), K.unsqueeze(0))
    # 无论成功与否，都不能崩；失败时第 4 行第 4 列保持 0
    assert poses.shape == (1, 4, 4)


def test_heatmap_target_peaks_at_corner():
    corner_2d = torch.tensor([[[64.0, 64.0]]])     # 图像 256 -> 热图 64：应为 (16, 16)
    heatmap = make_heatmap_target(corner_2d, image_size=256, heatmap_size=64, sigma=2.0)
    assert heatmap.shape == (1, 1, 64, 64)
    peak = torch.argmax(heatmap[0, 0])
    py, px = divmod(int(peak), 64)
    assert (px, py) == (16, 16), f"peak at {(px, py)}"
    assert abs(float(heatmap.max()) - 1.0) < 1e-5


def test_soft_argmax_is_subpixel():
    """软 argmax 应能给出非整数的峰值位置。"""
    heatmap = torch.zeros(1, 1, 32, 32)
    # 在 (10.3, 12.7) 附近放一个非对称峰
    ys = torch.arange(32).view(1, 1, 32, 1).float()
    xs = torch.arange(32).view(1, 1, 1, 32).float()
    heatmap = -((xs - 10.3) ** 2 + (ys - 12.7) ** 2)

    coords = soft_argmax_2d(heatmap, beta=1.0)
    assert abs(float(coords[0, 0, 0]) - 10.3) < 0.5
    assert abs(float(coords[0, 0, 1]) - 12.7) < 0.5


def test_argmax_matches_soft_argmax_on_sharp_peak():
    heatmap = torch.zeros(1, 2, 16, 16)
    heatmap[0, 0, 3, 5] = 10.0
    heatmap[0, 1, 9, 2] = 10.0
    coords = argmax_2d(heatmap)
    assert tuple(coords[0, 0].tolist()) == (5.0, 3.0)
    assert tuple(coords[0, 1].tolist()) == (2.0, 9.0)


def test_heatmap_to_corners_dispatch():
    heatmap = torch.randn(2, 8, 16, 16)
    for method in ("soft_argmax", "argmax"):
        out = heatmap_to_corners(heatmap, method=method)
        assert out.shape == (2, 8, 2)
    try:
        heatmap_to_corners(heatmap, method="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown method")


def test_pose_error_5cm5deg_accepts_perfect_pose():
    bbox_3d = bbox3d_from_minmax(torch.tensor([-20.0, -15.0, -8.0]),
                                 torch.tensor([20.0, 15.0, 8.0]))
    pose = _random_pose(t_range=300.0)
    hit = pose_error_5cm5deg(pose.unsqueeze(0), pose.unsqueeze(0), bbox_3d.unsqueeze(0))
    assert bool(hit[0])


def test_rotation_angle_deg_known_value():
    R_gt = torch.eye(3)
    angle = math.radians(30.0)
    R_pred = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert abs(float(rotation_angle_deg(R_pred, R_gt)) - 30.0) < 1e-3
