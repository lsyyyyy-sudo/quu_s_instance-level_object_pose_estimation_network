"""Hydra 配置装配的自检用例。

验证 ``configs/`` 能否正确组合，以及 ``run.py`` 使用的那条实例化路径
（``hydra.utils.instantiate(cfg.model, _recursive_=False)``）能否真的建出
LightningModule。这一层最容易因为配置改动而悄悄坏掉。

注意：配置里用了 ``${hydra:runtime.cwd}``，所以必须在 ``initialize()`` 上下文内
解析，并且把组合出的配置装进 ``HydraConfig``，否则插值会报
``HydraConfig was not set``。
"""

from typing import Callable

import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = "../configs"


def _with_config(config_name: str, fn: Callable[[DictConfig], object]):
    """在 hydra 上下文里组合配置、装好 HydraConfig，再交给 ``fn``。"""
    with initialize(version_base="1.3", config_path=CONFIG_DIR):
        cfg = compose(config_name=config_name, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return fn(cfg)


def _lightning_module(cfg: DictConfig):
    """按 run.py 的路径实例化；主干关掉预训练权重避免联网。"""
    OmegaConf.set_struct(cfg, False)
    cfg.model.modules.encoder.resnet.cfg.pretrained = False
    cfg.model.modules.decoder.hidden_dim = 16
    return hydra.utils.instantiate(cfg.model, _recursive_=False)


# --------------------------------------------------------------------------- #
def test_train_config_composes():
    def check(cfg):
        assert cfg.mode == "train"
        assert cfg.num_keypoints == 8
        assert cfg.image_size == 256
        assert cfg.heatmap_size == 64
        assert "_target_" in cfg.model
        assert "_target_" in cfg.trainer
        assert "_target_" in cfg.datamodule
        return True

    assert _with_config("train.yaml", check)


def test_test_config_composes():
    def check(cfg):
        assert cfg.mode == "test"
        assert cfg.use_pretrained is True
        return True

    assert _with_config("test.yaml", check)


def test_model_group_is_fully_resolved():
    """配置里我们自己写的 ${...} 插值应能全部解析成具体数值。

    注意不要对**整个** cfg 做 ``resolve=True``：hydra 自带的
    ``hydra.sweep.subdir`` 引用了只在 multirun 下存在的 ``hydra.job.num``，
    在单次运行里解析必然报 ``MissingMandatoryValue``。
    """

    def check(cfg):
        model = OmegaConf.to_container(cfg.model, resolve=True)
        assert model["modules"]["decoder"]["num_keypoints"] == 8
        assert model["modules"]["decoder"]["heatmap_size"] == 64
        assert model["modules"]["task"]["image_size"] == 256
        # ${hydra:runtime.cwd} 也要能落到具体路径
        assert "checkpoints" in model["resume_ckpt"]

        datamodule = OmegaConf.to_container(cfg.datamodule, resolve=True)
        assert datamodule["image_size"] == 256
        assert datamodule["heatmap_size"] == 64

        trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
        assert trainer["precision"] == cfg.precision
        assert trainer["max_epochs"] == cfg.max_epochs
        return True

    assert _with_config("train.yaml", check)


def test_heatmap_style_is_shared_between_model_and_datamodule():
    """热图风格必须是**同一个源**，否则 GT 和网络输出的量程会对不上。"""

    def check(cfg):
        assert cfg.heatmap_style == "boxdreamer"
        assert cfg.model.modules.task.heatmap_style == cfg.heatmap_style
        assert cfg.datamodule.heatmap_style == cfg.heatmap_style
        return True

    assert _with_config("train.yaml", check)


def test_loss_config_matches_boxdreamer_recipe():
    """默认损失应是 BoxDreamer 的配方：SmoothL1 粗损失 + λ=2.0 的细损失。"""

    def check(cfg):
        loss = OmegaConf.to_container(cfg.model.loss, resolve=True)
        assert loss["heatmap_loss"] == "smooth_l1"
        assert loss["fine_weight"] == 2.0      # 论文里的 λ
        assert loss["fine_beta"] == 25.0       # 实测最优的 soft-argmax 温度
        return True

    assert _with_config("train.yaml", check)


def test_task_extraction_settings():
    def check(cfg):
        task = OmegaConf.to_container(cfg.model.modules.task, resolve=True)
        assert task["extraction"] == "topk"    # BoxDreamer 官方做法
        assert task["topk"] == 20
        return True

    assert _with_config("train.yaml", check)


def test_instantiate_lightning_module_from_config():
    def check(cfg):
        module = _lightning_module(cfg)
        assert module.__class__.__name__ == "PL_CornerPose"
        assert module.model.num_keypoints == 8
        assert hasattr(module, "loss_fn")
        assert "optimizer" in module.opt_cfg
        return True

    assert _with_config("train.yaml", check)


def test_instantiate_optimizer_from_config():
    def check(cfg):
        module = _lightning_module(cfg)
        optimizers = module.configure_optimizers()

        assert "optimizer" in optimizers
        assert "lr_scheduler" in optimizers
        # CosineAnnealingLR 的 T_max 应该被 ${max_epochs} 解析成整数
        assert optimizers["lr_scheduler"]["scheduler"].T_max == cfg.max_epochs
        return True

    assert _with_config("train.yaml", check)


def test_backbone_and_decoder_wired_together():
    def check(cfg):
        module = _lightning_module(cfg)
        assert list(module.model.encoder.out_channels) == [64, 128, 256, 512]
        assert len(module.model.decoder.lateral) == 4
        return True

    assert _with_config("train.yaml", check)
