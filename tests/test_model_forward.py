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


def _model_cfg():
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
                "extraction": "soft_argmax",
                "soft_argmax_beta": 100.0,
            },
        }
    )


def _fake_batch(batch_size=2):
    torch.manual_seed(0)
    bbox_3d = bbox3d_from_minmax(
        torch.tensor([-20.0, -15.0, -8.0]), torch.tensor([20.0, 15.0, 8.0])
    ).unsqueeze(0).repeat(batch_size, 1, 1)

    f = IMAGE_SIZE / (2.0 * torch.tan(torch.deg2rad(torch.tensor(60.0)) / 2.0))
    K = torch.tensor(
        [[f, 0.0, IMAGE_SIZE / 2.0], [0.0, f, IMAGE_SIZE / 2.0], [0.0, 0.0, 1.0]]
    ).repeat(batch_size, 1, 1)

    image = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    corner_2d = torch.rand(batch_size, NUM_KEYPOINTS, 2) * IMAGE_SIZE
    heatmap = make_heatmap_target(corner_2d, IMAGE_SIZE, HEATMAP_SIZE, sigma=2.0)

    return {
        "image": image,
        "cam_K": K,
        "bbox_3d": bbox_3d,
        "corner_2d": corner_2d,
        "heatmap": heatmap,
    }


# --------------------------------------------------------------------------- #
def test_model_forward_output_shape():
    model = CornerPoseModel(_model_cfg())
    batch = _fake_batch(2)
    out = model(batch)

    assert "pred_heatmap" in out
    assert out["pred_heatmap"].shape == (2, NUM_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE)


def test_model_backbone_channels_match_decoder():
    """resnet18 的 4 个 stage 应该是 [64, 128, 256, 512]。"""
    model = CornerPoseModel(_model_cfg())
    assert list(model.encoder.out_channels) == [64, 128, 256, 512]
    assert len(model.decoder.lateral) == 4


def test_loss_is_finite_and_backward_works():
    model = CornerPoseModel(_model_cfg())
    loss_fn = CornerHeatmapLoss()
    batch = _fake_batch(2)

    out = model(batch)
    losses = loss_fn(out["pred_heatmap"], batch["heatmap"])

    assert torch.isfinite(losses["loss"])
    assert float(losses["loss"].detach()) > 0.0

    losses["loss"].backward()
    grad_norm = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0.0, "no gradient flowed into the backbone"


def test_focal_loss_penalizes_wrong_prediction_more():
    """预测越离谱，loss 应该越大。"""
    gt = torch.zeros(1, 1, 16, 16)
    gt[0, 0, 8, 8] = 1.0
    loss_fn = CornerHeatmapLoss()

    good = torch.full_like(gt, -4.0)
    good[0, 0, 8, 8] = 6.0
    bad = torch.full_like(gt, 4.0)   # 到处都说是正样本

    l_good = float(loss_fn(good, gt)["loss"])
    l_bad = float(loss_fn(bad, gt)["loss"])
    assert l_good < l_bad, f"expected {l_good} < {l_bad}"


def test_predict_returns_corners_and_pose():
    model = CornerPoseModel(_model_cfg())
    batch = _fake_batch(2)

    preds = predict_corners_and_pose(
        model(batch)["pred_heatmap"],
        batch["bbox_3d"],
        batch["cam_K"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        method="soft_argmax",
        solve_pose=True,
    )

    assert preds["corner_2d"].shape == (2, NUM_KEYPOINTS, 2)
    assert preds["poses"].shape == (2, 4, 4)
    # 预测的角点应该落在图像范围内（soft-argmax 天然有界）
    assert float(preds["corner_2d"].detach().min()) >= 0.0
    assert float(preds["corner_2d"].detach().max()) <= IMAGE_SIZE


def test_predict_without_pose():
    model = CornerPoseModel(_model_cfg())
    batch = _fake_batch(1)
    preds = predict_corners_and_pose(
        model(batch)["pred_heatmap"],
        batch["bbox_3d"],
        batch["cam_K"],
        image_size=IMAGE_SIZE,
        heatmap_size=HEATMAP_SIZE,
        solve_pose=False,
    )
    assert preds["poses"] is None
