"""Hydra 统一入口，负责组装 Trainer / LightningModule / DataModule / Callbacks / Logger。

用法::

    python run.py --config-name=train.yaml
    python run.py --config-name=test.yaml exp_name=<实验名>

结构对齐 BoxDreamer/run.py：所有组件都由 Hydra 从 ``configs/`` 里实例化，
本文件只做编排，不写任何模型或数据逻辑。
"""

import argparse
import os
import sys
import traceback
from typing import List

import cv2
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase

from src.utils.customize.template_utils import log_hparams_to_all_loggers
from src.utils.log import ERROR, INFO, WARNING, finish, print_key_configs

cv2.setNumThreads(0)
os.environ["HYDRA_FULL_ERROR"] = "1"


def handle(config: DictConfig):
    """按配置组装并运行训练 / 测试。"""
    print_key_configs(config)

    if "seed" in config:
        seed_everything(config["seed"], workers=True)

    # ---- LightningModule（注意 _recursive_=False，子配置原样传进去）----
    model: LightningModule = hydra.utils.instantiate(config["model"], _recursive_=False)

    # ---- LightningDataModule ----
    datamodule = hydra.utils.instantiate(config["datamodule"])

    # ---- Callbacks ----
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # ---- Loggers ----
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for key in config["logger"]["in_use"]:
            logger.append(hydra.utils.instantiate(config["logger"][key]))

    # ---- Trainer ----
    trainer: Trainer = hydra.utils.instantiate(
        config["trainer"], callbacks=callbacks, logger=logger
    )

    log_hparams_to_all_loggers(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    resume_path = config.model.get("resume_ckpt", None)
    pretrain_path = config.model.get("pretrained_ckpt", None)

    try:
        if config.mode == "train":
            resumed = False
            if resume_path is not None and config.get("resume", False):
                if os.path.exists(resume_path):
                    INFO(f"Resuming training from checkpoint {resume_path}.")
                    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_path)
                    resumed = True
                else:
                    WARNING(f"Checkpoint not found at {resume_path}; training from scratch.")

            if not resumed:
                pretrained_used = False
                if pretrain_path is not None and config.get("use_pretrained", False):
                    if os.path.exists(pretrain_path):
                        INFO(f"Loading pre-trained weights from {pretrain_path}.")
                        model.load_pretrained_params(pretrain_path)
                        pretrained_used = True
                    else:
                        WARNING(f"Pre-trained checkpoint not found at {pretrain_path}.")
                INFO("Starting training" + (" from pre-trained weights." if pretrained_used else " from scratch."))
                trainer.fit(model=model, datamodule=datamodule)

        elif config.mode == "test":
            if pretrain_path is None:
                raise ValueError(
                    "Testing requires model.pretrained_ckpt "
                    "(checkpoints/${pretrain_name}/last.ckpt)."
                )
            if not os.path.exists(pretrain_path):
                raise FileNotFoundError(f"Pre-trained checkpoint not found: {pretrain_path}")
            INFO(f"Loading pre-trained weights from {pretrain_path} for testing.")
            model.load_pretrained_params(pretrain_path)
            trainer.test(model=model, datamodule=datamodule)

        else:
            raise ValueError(f"Invalid mode: {config.mode}. Valid modes are 'train' and 'test'.")

    except Exception as e:  # noqa: BLE001 - 顶层兜底，附带完整 traceback
        ERROR(f"An error occurred during training/testing: {e}\n{traceback.format_exc()}")
        raise

    finish(config=config, model=model, datamodule=datamodule, trainer=trainer)


@hydra.main(config_path="configs/", config_name="train.yaml", version_base="1.3")
def main(config: DictConfig):
    try:
        handle(config)
    except Exception as e:  # noqa: BLE001
        ERROR(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-name", default=None, help="Hydra 配置名（也可直接传给 hydra）")
    parser.parse_known_args()

    main()
