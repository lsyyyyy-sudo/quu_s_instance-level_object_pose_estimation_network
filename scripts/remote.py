"""在远程 Linux 实例（AutoDL 等）上执行命令、传文件。

凭据从环境变量读，**不写进代码、不进仓库**：

    AUTODL_HOST   例如 region-1.example-cloud.com
    AUTODL_PORT   例如 12345
    AUTODL_USER   例如 root
    AUTODL_PASS   登录密码

用法::

    python scripts/remote.py run "nvidia-smi"
    python scripts/remote.py run --timeout 3600 "pip install -r requirements.txt"
    python scripts/remote.py put local.jpg /root/autodl-tmp/local.jpg
    python scripts/remote.py get /root/autodl-tmp/out.glb ./out.glb
    python scripts/remote.py ls /root/autodl-tmp

为什么用它而不是直接 `ssh`：Windows 的 OpenSSH 客户端**无法从 stdin 喂密码**，
而且这里需要长时间流式看输出（装依赖、下模型）。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，远端输出里的 ✓ / emoji / 中文会直接让 print 抛
# UnicodeEncodeError（把整个工具搞崩）。统一改成 UTF-8 + replace。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import paramiko
except ImportError:
    sys.exit("需要 paramiko： pip install paramiko")

# AutoDL 的学术加速脚本；clone GitHub / 下 HuggingFace 权重前必须 source
NETWORK_TURBO = "/etc/network_turbo"


def _env(name: str, required: bool = True) -> str:
    value = os.environ.get(name, "")
    if required and not value:
        sys.exit(
            f"缺少环境变量 {name}。先设置：\n"
            f"  $env:AUTODL_HOST='...'; $env:AUTODL_PORT='...'; "
            f"$env:AUTODL_USER='root'; $env:AUTODL_PASS='...'"
        )
    return value


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=_env("AUTODL_HOST"),
        port=int(_env("AUTODL_PORT")),
        username=_env("AUTODL_USER"),
        password=_env("AUTODL_PASS"),
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: float | None = None,
        turbo: bool = False, quiet: bool = False) -> int:
    """执行命令并**实时**把 stdout/stderr 打到本地终端，返回远端退出码。

    Args:
        turbo: 为 True 时在命令前加 `source /etc/network_turbo`
        timeout: 秒；None 表示不超时（大下载要设 None）
    """
    if turbo:
        command = f"source {NETWORK_TURBO} 2>/dev/null; {command}"

    chan = client.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(f"bash -lc {_shquote(command)}")

    start = time.time()
    while True:
        got = False
        while chan.recv_ready():
            data = chan.recv(65536).decode("utf-8", errors="replace")
            if not quiet:
                sys.stdout.write(data)
                sys.stdout.flush()
            got = True
        while chan.recv_stderr_ready():
            data = chan.recv_stderr(65536).decode("utf-8", errors="replace")
            if not quiet:
                sys.stderr.write(data)
                sys.stderr.flush()
            got = True

        if chan.exit_status_ready() and not got:
            # 通道还有残留在内核缓冲里时再多读一轮
            if not chan.recv_ready() and not chan.recv_stderr_ready():
                break
        if timeout is not None and (time.time() - start) > timeout:
            chan.close()
            print(f"\n[remote] ⏱ 超时 {timeout}s，已中断", file=sys.stderr)
            return 124
        if not got:
            time.sleep(0.05)

    code = chan.recv_exit_status()
    chan.close()
    return code


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


# --------------------------------------------------------------------------- #
def cmd_run(args) -> int:
    if args.file:
        # 从文件读命令：避开 Windows shell 的引号/换行转义地狱
        command = Path(args.file).read_text(encoding="utf-8")
    elif args.command == "-":
        command = sys.stdin.read()
    else:
        command = args.command

    client = connect()
    try:
        code = run(client, command, timeout=args.timeout, turbo=args.turbo)
        print(f"\n[remote] exit={code}")
        return code
    finally:
        client.close()


def cmd_put(args) -> int:
    client = connect()
    try:
        sftp = client.open_sftp()
        sftp.put(args.local, args.remote)
        print(f"[remote] 上传 {args.local} -> {args.remote}")
        sftp.close()
        return 0
    finally:
        client.close()


def cmd_get(args) -> int:
    client = connect()
    try:
        sftp = client.open_sftp()
        Path(args.local).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(args.remote, args.local)
        print(f"[remote] 下载 {args.remote} -> {args.local}")
        sftp.close()
        return 0
    finally:
        client.close()


def cmd_ls(args) -> int:
    client = connect()
    try:
        code = run(client, f"ls -la {_shquote(args.path)}")
        return code
    finally:
        client.close()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="远程实例操作")
    sub = ap.add_subparsers(dest="action", required=True)

    p = sub.add_parser("run", help="执行命令")
    p.add_argument("command", nargs="?", default=None,
                   help="要执行的命令；传 '-' 表示从 stdin 读")
    p.add_argument("--file", default=None,
                   help="从本地文件读命令（推荐：避开 shell 转义问题）")
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--turbo", action="store_true", help="命令前 source /etc/network_turbo")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("put", help="上传文件")
    p.add_argument("local")
    p.add_argument("remote")
    p.set_defaults(func=cmd_put)

    p = sub.add_parser("get", help="下载文件")
    p.add_argument("remote")
    p.add_argument("local")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("ls", help="列目录")
    p.add_argument("path", nargs="?", default=".")
    p.set_defaults(func=cmd_ls)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
