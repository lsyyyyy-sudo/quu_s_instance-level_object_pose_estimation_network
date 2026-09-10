"""终端日志工具（对应 BoxDreamer/src/utils/log.py）。"""

import sys

try:  # rich 不是硬依赖
    from rich.console import Console

    _console = Console()
except Exception:  # pragma: no cover
    _console = None


def _print(message: str, style: str = "", file=sys.stdout):
    if _console is not None:
        _console.print(message, style=style)
    else:
        print(message, file=file, flush=True)


def INFO(message: str, **kwargs):
    _print(f"[INFO] {message}", style="green")


def WARNING(message: str, **kwargs):
    _print(f"[WARN] {message}", style="yellow")


def ERROR(message: str, **kwargs):
    _print(f"[ERROR] {message}", style="red", file=sys.stderr)


def DEBUG(message: str, **kwargs):
    _print(f"[DEBUG] {message}", style="dim")


def print_key_configs(config, keys=("mode", "exp_name", "seed", "image_size", "heatmap_size")):
    """开跑前把关键配置打出来，方便对日志。"""
    INFO("=" * 60)
    INFO(f"experiment : {config.get('exp_name', '<unset>')}")
    for key in keys:
        if key in config:
            INFO(f"{key:<11}: {config[key]}")
    INFO("=" * 60)


def finish(config=None, model=None, datamodule=None, trainer=None):
    """收尾。对应 BoxDreamer 的同名钩子，用于优雅退出。"""
    INFO("All done. Exiting.")
