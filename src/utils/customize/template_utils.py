"""项目自定义的通用模板工具（对应 BoxDreamer/src/utils/customize/template_utils.py）。"""

from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase

from src.utils.log import DEBUG, WARNING


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """从配置实例化所有带 ``_target_`` 的 callback。"""
    callbacks: List[Callback] = []
    if not callbacks_cfg:
        WARNING("No callback configs found; skipping.")
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig.")
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            DEBUG(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """按 ``in_use`` 列表实例化 logger。"""
    loggers: List[LightningLoggerBase] = []
    if not logger_cfg:
        WARNING("No logger configs found; skipping.")
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig.")
    if "in_use" not in logger_cfg:
        WARNING("logger.in_use missing; skipping.")
        return loggers
    for key in logger_cfg["in_use"]:
        if key not in logger_cfg:
            WARNING(f"logger.{key} not defined; skipping.")
            continue
        DEBUG(f"Instantiating logger <{key}>")
        loggers.append(hydra.utils.instantiate(logger_cfg[key]))
    return loggers


def log_hparams_to_all_loggers(config, model, datamodule, trainer, callbacks=None, logger=None):
    """把关键超参写进各个 logger（wandb / tensorboard 的 hparams tab）。"""
    if not logger:
        return
    params = OmegaConf.to_container(config, resolve=True)
    for lg in logger:
        try:
            lg.log_hyperparams(params)
        except Exception as e:  # noqa: BLE001 - 某些 logger 不支持，忽略即可
            WARNING(f"Logger {type(lg).__name__} does not support log_hyperparams: {e}")
