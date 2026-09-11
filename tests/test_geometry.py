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
    to_zero_one,
    topk_2d,
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


def _projected_box_corners_2d(img_size=256, seed=0, n=2, fill=0.6):
    """投影一个真实 3D 包围盒，得到像数据集里那样的 8 个 2D 角点 ``[n, 8, 2]``。

    比"均匀随机点"更贴近现实：真实的 8 个角点都分布在物体 2D 轮廓的边界上，
    不会出现某个角点几乎压在物体 2D 中心的情况。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    bbox = bbox3d_from_minmax(
        torch.tensor([-35.0, -22.0, -16.0]), torch.tensor([35.0, 22.0, 16.0])
    )
    K = _dummy_K(img_size)

    out = []
    for _ in range(n):
        pose = _random_pose(t_range=300.0)
        uv = project_points(bbox.unsqueeze(0), K.unsqueeze(0), pose.unsqueeze(0))[0]

        # 把投影结果整体平移+缩放到画面中央，保证角点都在热图范围内
        uv = uv - uv.mean(dim=0, keepdim=True)
        extent = float(uv.max(dim=0).values.sub(uv.min(dim=0).values).max())
        uv = uv * (fill * img_size / max(extent, 1e-6)) + img_size / 2.0
        out.append(uv)
    return torch.stack(out, dim=0)


# --------------------------------------------------------------------------- #
# centernet 风格热图
# --------------------------------------------------------------------------- #
def test_centernet_heatmap_peaks_at_corner():
    corner_2d = torch.tensor([[[64.0, 64.0]]])     # 图像 256 -> 热图 64：应为 (16, 16)
    heatmap = make_heatmap_target(
        corner_2d, image_size=256, heatmap_size=64, sigma=2.0, style="centernet"
    )
    assert heatmap.shape == (1, 1, 64, 64)
    peak = torch.argmax(heatmap[0, 0])
    py, px = divmod(int(peak), 64)
    assert (px, py) == (16, 16), f"peak at {(px, py)}"
    assert abs(float(heatmap.max()) - 1.0) < 1e-5
    # centernet 风格取值范围是 [0, 1]
    assert float(heatmap.min()) >= 0.0


# --------------------------------------------------------------------------- #
# BoxDreamer 风格热图（尺度自适应 + [-1, 1]）
# --------------------------------------------------------------------------- #
def test_boxdreamer_heatmap_range_and_peak():
    """BoxDreamer 风格：峰值归一化到 1，整体映射到 [-1, 1]。"""
    corners = _projected_box_corners_2d(seed=0, n=1)
    heatmap = make_heatmap_target(corners, 256, 64, style="boxdreamer")

    assert heatmap.shape == (1, 8, 64, 64)
    assert abs(float(heatmap.max()) - 1.0) < 1e-4, "峰值应归一化到 1"
    assert float(heatmap.min()) >= -1.0 - 1e-5, "下界应不低于 -1"
    assert float(heatmap.min()) < -0.5, "背景应接近 -1"

    # 每个通道的峰值应落在该角点的投影位置附近
    for k in range(8):
        cx = float(corners[0, k, 0]) * 64 / 256
        cy = float(corners[0, k, 1]) * 64 / 256
        peak = torch.argmax(heatmap[0, k])
        py, px = divmod(int(peak), 64)
        assert abs(px - cx) <= 1.5 and abs(py - cy) <= 1.5, (
            f"corner {k}: peak=({px},{py}) expected≈({cx:.1f},{cy:.1f})"
        )


def test_min_scale_prevents_degenerate_delta():
    """回归测试：角点与物体 2D 中心重合时，热图不能退化成 δ。

    退化时背景全部下溢为 0，top-k 会出现大量并列，取出的位置基本是随机的
    （实测平均偏 14~18 px）。`min_scale` 就是为此加的下限。
    """
    corners = torch.tensor([[[128.0, 128.0], [128.0, 128.0]]])   # 重合 -> s_i = 0
    heatmap = make_heatmap_target(corners, 256, 64, style="boxdreamer")

    peak = float(heatmap[0, 0].max())
    assert abs(peak - 1.0) < 1e-4
    n_tied = int((heatmap[0, 0] >= peak - 1e-6).sum())
    assert n_tied == 1, f"峰值应唯一，实际有 {n_tied} 个并列像素"

    coords = topk_2d(heatmap, k=20, heatmap_range="minus_one_to_one")
    err = float(torch.linalg.norm(coords[0, 0] - torch.tensor([32.0, 32.0])))
    assert err < 1.0, f"topk 应落在角点上，实际偏了 {err:.2f} px"


def test_boxdreamer_heatmap_is_scale_adaptive():
    """离物体 2D 中心越远的角点，热图越宽（这就是尺度自适应 σ 的作用）。

    衰减是 ``exp(-d / scale)``，半高宽 = ``scale·ln2``，其中
    ``scale = (s_i / 10)²``，``s_i`` 是角点 i 到 8 角点均值的像素距离。
    """
    corners = _projected_box_corners_2d(seed=1, n=1)
    heatmap = make_heatmap_target(corners, 256, 64, style="boxdreamer")

    centers_hm = corners[0] * 64 / 256
    obj_center = centers_hm.mean(dim=0)
    dis = torch.linalg.norm(centers_hm - obj_center, dim=-1)   # [8]
    assert float(dis.max()) > float(dis.min()) * 1.5, "测试样本本身的 dis 差异不够大"

    # 量每个通道在其峰值右侧 2 像素处的取值：热图越宽，该值越接近 1
    widths = torch.zeros(8)
    for k in range(8):
        peak = torch.argmax(heatmap[0, k])
        py, px = divmod(int(peak), 64)
        x2 = min(px + 2, 63)
        widths[k] = heatmap[0, k, py, x2]

    i_min, i_max = int(dis.argmin()), int(dis.argmax())
    assert float(widths[i_max]) > float(widths[i_min]), (
        f"远离中心的角点热图应该更宽: widths[min]={widths[i_min]:.4f}, "
        f"widths[max]={widths[i_max]:.4f}"
    )


def test_unknown_heatmap_style_raises():
    try:
        make_heatmap_target(torch.zeros(1, 8, 2), 256, 64, style="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown heatmap style")


# --------------------------------------------------------------------------- #
# 角点提取
# --------------------------------------------------------------------------- #
def test_soft_argmax_is_subpixel():
    """软 argmax 应能给出非整数的峰值位置。"""
    heatmap = torch.zeros(1, 1, 32, 32)
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


def test_to_zero_one_handles_both_ranges():
    hm = torch.tensor([[[[-1.0, 0.0], [0.5, 1.0]]]])
    out = to_zero_one(hm, "minus_one_to_one")
    assert torch.allclose(out, torch.tensor([[[[0.0, 0.5], [0.75, 1.0]]]]))
    assert torch.allclose(to_zero_one(hm, "zero_to_one"), hm)


def test_topk_extraction_on_boxdreamer_heatmap():
    """topk 提取应落在峰值附近（BoxDreamer 官方做法）。"""
    corners = _projected_box_corners_2d(seed=2, n=1)
    heatmap = make_heatmap_target(corners, 256, 64, style="boxdreamer")

    coords = topk_2d(heatmap, k=20, heatmap_range="minus_one_to_one")
    assert coords.shape == (1, 8, 2)

    gt_hm = corners[0] * (64 / 256)
    err = torch.linalg.norm(coords[0] - gt_hm, dim=-1)
    assert float(err.mean()) < 2.0, f"topk mean error too large: {err.mean()}"


def test_soft_argmax_beats_argmax_on_boxdreamer_heatmap():
    """soft-argmax(beta=25) 在干净热图上应优于 argmax（这是 fine_beta 的实测依据）。"""
    corners = _projected_box_corners_2d(seed=3, n=4)
    heatmap = make_heatmap_target(corners, 256, 64, style="boxdreamer")
    gt_hm = corners * (64 / 256)

    sm = soft_argmax_2d(to_zero_one(heatmap, "minus_one_to_one"), beta=25.0)
    am = argmax_2d(heatmap)

    err_sm = float(torch.linalg.norm(sm - gt_hm, dim=-1).mean())
    err_am = float(torch.linalg.norm(am - gt_hm, dim=-1).mean())
    assert err_sm < err_am, f"soft_argmax {err_sm:.3f} should beat argmax {err_am:.3f}"


def test_heatmap_to_corners_dispatch():
    heatmap = torch.rand(2, 8, 16, 16)
    for method in ("soft_argmax", "argmax", "topk"):
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
