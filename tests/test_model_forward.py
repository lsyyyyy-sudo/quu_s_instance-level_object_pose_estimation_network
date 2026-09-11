"""网络前向与损失的自检用例（不需要任何数据集）。

用 ``pretrained=False`` 避免下载权重，输入张量也很小，整体几秒内跑完。
目的是保证「配置 -> 模型 -> 损失 -> 推理后处理」这条链路是通的。
"""

import torch
from omegaconf import OmegaConf

from src.loss.loss import CornerHeatmapLoss
from src.models.CornerPoseModel import CornerPoseModel
from src.models.utils.box_utils import bbox3d_from_minmax
from src.models.utils.data_processing import make_heatmap_target
from src.models.utils.prediction_utils import predict_corners_and_pose

IMAGE_SIZE = 64
HEATMAP_SIZE = 16
NUM_KEYPOINTS = 8


def _model_cfg(heatmap_style="boxdreamer", extraction="topk"):
    return OmegaConf.create(
        {
            "encoder": {
                "name": "resnet",
                "resnet": {
                    "ckpt_path": None,
                    "cfg": {
                        "model_type": "resnet18",
                        "pretrained": False,   # 测试里不联网
                        "freeze": False,
                        "out_indices": [0, 1, 2, 3],
                    },
                },
            },
            "decoder": {
                "hidden_dim": 32,
                "num_keypoints": NUM_KEYPOINTS,
                "heatmap_size": HEATMAP_SIZE,
                "upsample_mode": "bilinear",
                "offset": False,
            },
            "task": {
                "num_keypoints": NUM_KEYPOINTS,
                "bbox_representation": "heatmap",
                "corner_order": "bb8",
                "image_size": IMAGE_SIZE,
                "heatmap_size": HEATMAP_SIZE,
                "heatmap_style": heatmap_style,
                "extraction": extraction,
                "topk": 20,
                "soft_argmax_beta": 25.0,
            },
        }
    )


def _fake_batch(batch_size=2, heatmap_style="boxdreamer"):
    torch.manual_seed(0)
    bbox_3d = bbox3d_from_minmax(
        torch.tensor([-20.0, -15.0, -8.0]), torch.tensor([20.0, 15.0, 8.0])
    ).unsqueeze(0).repeat(batch_size, 1, 1)

    f = IMAGE_SIZE / (2.0 * torch.tan(torch.deg2rad(torch.tensor(60.0)) / 2.0))
    K = torch.tensor(
        [[f, 0.0, IMAGE_SIZE / 2.0], [0.0, f, IMAGE_SIZE / 2.0], [0.0, 0.0, 1.0]]
    ).repeat(batch_size, 1, 1)

    image = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    # 角点放在中间区域，避免贴边导致热图被截断
    corner_2d = torch.rand(batch_size, NUM_KEYPOINTS, 2) * IMAGE_SIZE * 0.6 + IMAGE_SIZE * 0.2
    heatmap = make_heatmap_target(
        corner_2d, IMAGE_SIZE, HEATMAP_SIZE, style=heatmap_style
    )

    return {
        "image": image,
        "cam_K": K,
        "bbox_3d": bbox_3d,
        "corner_2d": corner_2d,
        "heatmap": heatmap,
    }


# --------------------------------------------------------------------------- #
# 前向
# --------------------------------------------------------------------------- #
def test_model_forward_output_shape():
    model = CornerPoseModel(_model_cfg())
    batch = _fake_batch(2)
    out = model(batch)

    assert "pred_heatmap" in out
    assert out["pred_heatmap"].shape == (2, NUM_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE)


def test_boxdreamer_style_output_is_in_minus_one_to_one():
    """BoxDreamer 风格：输出经过 2*sigmoid(x)-1，取值落在 [-1, 1]。"""
    model = CornerPoseModel(_model_cfg(heatmap_style="boxdreamer"))
    out = model(_fake_batch(2))

    assert model.heatmap_range == "minus_one_to_one"
    h = out["pred_heatmap"].detach()
    assert float(h.min()) >= -1.0 - 1e-6
    assert float(h.max()) <= 1.0 + 1e-6


def test_centernet_style_output_stays_logits():
    """centernet 风格不激活，输出应等于解码头原始输出（logits）。

    不能断言"一定超出 [-1,1]" —— 随机初始化的网络输出范围是任意的。
    """
    model = CornerPoseModel(_model_cfg(heatmap_style="centernet"))
    batch = _fake_batch(2, heatmap_style="centernet")

    assert model.heatmap_style == "centernet"
    assert model.heatmap_range == "zero_to_one"

    out = model(batch)
    raw = model.decoder(model.encoder(batch["image"]))["heatmap"]
    assert torch.allclose(out["pred_heatmap"], raw), "centernet 风格不应该做输出激活"


def test_unknown_heatmap_style_raises():
    try:
        CornerPoseModel(_model_cfg(heatmap_style="nope"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown heatmap_style")


def test_model_backbone_channels_match_decoder():
    model = CornerPoseModel(_model_cfg())
    assert list(model.encoder.out_channels) == [64, 128, 256, 512]
    assert len(model.decoder.lateral) == 4


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def test_coarse_and_fine_loss_backward():
    model = CornerPoseModel(_model_cfg())
    loss_fn = CornerHeatmapLoss()
    batch = _fake_batch(2)

    out = model(batch)
    losses = loss_fn(
        out["pred_heatmap"],
        batch["heatmap"],
        corner_2d=batch["corner_2d"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        heatmap_range=model.heatmap_range,
    )

    for key in ("loss", "loss_coarse", "loss_fine"):
        assert key in losses, f"missing {key}"
        assert torch.isfinite(losses[key]), f"{key} is not finite"
    assert float(losses["loss"].detach()) > 0.0

    losses["loss"].backward()
    grad_norm = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0.0, "no gradient flowed into the backbone"


def test_fine_term_actually_contributes():
    """fine_weight>0 时总损失必须比只算粗损失大（否则 fine 项没生效）。"""
    torch.manual_seed(0)
    batch = _fake_batch(2)
    pred = batch["heatmap"] * 0.5      # 随便一个有梯度的预测

    with_fine = CornerHeatmapLoss(heatmap_loss="smooth_l1", fine_weight=2.0)
    without_fine = CornerHeatmapLoss(heatmap_loss="smooth_l1", fine_weight=0.0)

    kwargs = dict(
        corner_2d=batch["corner_2d"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        heatmap_range="minus_one_to_one",
    )
    l_with = float(with_fine(pred, batch["heatmap"], **kwargs)["loss"])
    l_without = float(without_fine(pred, batch["heatmap"], **kwargs)["loss"])

    assert float(with_fine(pred, batch["heatmap"], **kwargs)["loss_fine"]) > 0.0
    assert l_with > l_without


def test_fine_loss_prefers_correct_corners():
    """角点预测越准，fine 项越小。"""
    torch.manual_seed(0)
    corners = torch.rand(1, NUM_KEYPOINTS, 2) * IMAGE_SIZE * 0.6 + IMAGE_SIZE * 0.2
    gt_heatmap = make_heatmap_target(corners, IMAGE_SIZE, HEATMAP_SIZE, style="boxdreamer")

    loss_fn = CornerHeatmapLoss(heatmap_loss="smooth_l1", fine_weight=1.0)

    # 「好」的预测：直接用 GT 热图（soft-argmax 应该能落回原角点）
    good = gt_heatmap.clone()
    # 「坏」的预测：把热图整体平移 4 个热图像素（= 16 个图像像素）
    bad = torch.roll(gt_heatmap, shifts=4, dims=-1)

    kwargs = dict(
        corner_2d=corners,
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        heatmap_range="minus_one_to_one",
    )
    l_good = float(loss_fn(good, gt_heatmap, **kwargs)["loss_fine"])
    l_bad = float(loss_fn(bad, gt_heatmap, **kwargs)["loss_fine"])

    assert l_good < l_bad, f"expected {l_good} < {l_bad}"
    assert l_good < 3.0, f"perfect heatmap should give a small fine loss, got {l_good}"


def test_focal_penalizes_wrong_prediction_more():
    """预测越离谱，focal 粗损失越大。"""
    gt = torch.zeros(1, 1, 16, 16)
    gt[0, 0, 8, 8] = 1.0
    loss_fn = CornerHeatmapLoss(heatmap_loss="focal", fine_weight=0.0)

    good = torch.full_like(gt, -4.0)
    good[0, 0, 8, 8] = 6.0
    bad = torch.full_like(gt, 4.0)   # 到处都说是正样本

    l_good = float(loss_fn(good, gt)["loss"])
    l_bad = float(loss_fn(bad, gt)["loss"])
    assert l_good < l_bad, f"expected {l_good} < {l_bad}"


def test_smooth_l1_coarse_loss_matches_torch():
    """粗损失在 smooth_l1 模式下应等价于 nn.SmoothL1Loss(mean)。"""
    torch.manual_seed(0)
    pred = torch.randn(2, 8, 16, 16)
    gt = torch.randn(2, 8, 16, 16)
    loss_fn = CornerHeatmapLoss(heatmap_loss="smooth_l1", heatmap_weight=1.0, fine_weight=0.0)
    got = float(loss_fn(pred, gt)["loss_coarse"])
    expected = float(torch.nn.functional.smooth_l1_loss(pred, gt, beta=1.0))
    assert abs(got - expected) < 1e-6


def test_unknown_loss_type_raises():
    for kwargs in ({"heatmap_loss": "nope"}, {"fine_loss": "nope"}):
        try:
            CornerHeatmapLoss(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def test_predict_returns_corners_and_pose():
    model = CornerPoseModel(_model_cfg())
    batch = _fake_batch(2)

    preds = predict_corners_and_pose(
        model(batch)["pred_heatmap"],
        batch["bbox_3d"],
        batch["cam_K"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        method=model.extraction,
        k=model.topk,
        heatmap_range=model.heatmap_range,
        solve_pose=True,
    )

    assert preds["corner_2d"].shape == (2, NUM_KEYPOINTS, 2)
    assert preds["poses"].shape == (2, 4, 4)
    assert float(preds["corner_2d"].min()) >= 0.0
    assert float(preds["corner_2d"].max()) <= IMAGE_SIZE


def test_predict_without_pose():
    model = CornerPoseModel(_model_cfg())
    batch = _fake_batch(1)
    preds = predict_corners_and_pose(
        model(batch)["pred_heatmap"],
        batch["bbox_3d"],
        batch["cam_K"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        method=model.extraction,
        k=model.topk,
        heatmap_range=model.heatmap_range,
        solve_pose=False,
    )
    assert preds["poses"] is None
