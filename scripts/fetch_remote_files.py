"""按清单从远端批量取文件，支持**断点续传 + 尺寸校验 + 重试**。

为什么单独写一个（而不是循环调 `remote.py get`）
-----------------------------------------------
一次实测：36 个大文件（每个约 140 MB）循环取，**取到第 18 个时远端关机了**，
后 18 个全失败。循环版有三个问题暴露出来：
  1. **没有任何校验** —— 传了一半的文件也算"存在"，下次会跳过它，永久坏掉。
     跨机传大文件必须比大小（本项目已经栽过一次：截断的 ckpt 让 torch.load 报
     `PytorchStreamReader failed reading zip archive`）。
  2. **不记住失败项** —— 得人肉从日志里扒出缺哪些。
  3. **一个连接一个文件** —— 每个文件都重新握手，慢且容易触发限流。

所以本工具：
  · 先**一次 SSH** 把所有远端文件的大小拿回来（`stat`），再逐个传；
  · 每个文件传完**比对大小**，不符就删掉重试（最多 `--retries` 次）；
  · 已存在且大小正确的自动跳过（**这就是断点续传**）；
  · 结束时打印缺哪些，并以退出码 1 结束，便于接自动化。

命名规则：`<父目录名>__<文件名>`（扁平），因为清单里是 8 个不同克隆下的
`checkpoints/<实验名>/last*.ckpt`，扁平化后不带路径也能一眼看出是哪个实验。

用法::

    python scripts/fetch_remote_files.py --list <清单.txt> --out <本地目录>
    python scripts/fetch_remote_files.py --list ... --out ... --dry-run   # 只报缺口
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from remote import connect, run  # noqa: E402  （复用同一套凭据读取与实时输出）


def remote_sizes(client, paths: list[str]) -> dict[str, int]:
    """一次 SSH 拿回所有远端文件的大小（-1 表示不存在）。"""
    # 用 xargs 分批，避免命令行过长
    quoted = " ".join("'" + p.replace("'", "'\\''") + "'" for p in paths)
    cmd = f"for f in {quoted}; do if [ -f \"$f\" ]; then stat -c '%s %n' \"$f\"; else echo \"MISSING $f\"; fi; done"
    out: dict[str, int] = {}
    buf: list[str] = []

    class _Cap:
        def write(self, s):  # noqa: ANN001
            buf.append(s)

        def flush(self):
            pass

    # remote.run 直接写 sys.stdout，这里临时换掉以捕获输出
    real = sys.stdout
    sys.stdout = _Cap()  # type: ignore[assignment]
    try:
        run(client, cmd, quiet=False)
    finally:
        sys.stdout = real

    for line in "".join(buf).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("MISSING "):
            out[line[len("MISSING "):]] = -1
        else:
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                out[parts[1]] = int(parts[0])
    return out


def local_name(remote_path: str) -> str:
    parts = remote_path.rstrip("/").split("/")
    return f"{parts[-2]}__{parts[-1]}" if len(parts) >= 2 else parts[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="一行一个远端路径")
    ap.add_argument("--out", required=True, help="本地输出目录（扁平命名 <父目录>__<文件名>）")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=4.0, help="重试前的等待秒数")
    ap.add_argument("--dry-run", action="store_true", help="只比对大小，报告缺口")
    args = ap.parse_args()

    paths = [l.strip() for l in open(args.list, encoding="utf-8") if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    print(f"清单 {len(paths)} 个文件 -> {args.out}")

    client = connect()
    try:
        sizes = remote_sizes(client, paths)
    finally:
        client.close()

    missing_remote = [p for p in paths if sizes.get(p, -1) < 0]
    if missing_remote:
        print(f"⚠️ 远端已不存在 {len(missing_remote)} 个：")
        for p in missing_remote[:8]:
            print(f"    {p}")

    todo = []
    for p in paths:
        sz = sizes.get(p, -1)
        if sz < 0:
            continue
        dst = os.path.join(args.out, local_name(p))
        if os.path.isfile(dst) and os.path.getsize(dst) == sz:
            continue
        todo.append((p, dst, sz))

    print(f"远端存在 {sum(1 for p in paths if sizes.get(p, -1) >= 0)} 个；"
          f"需要下载 {len(todo)} 个；已完成 {len(paths) - len(todo) - len(missing_remote)} 个")

    if args.dry_run:
        for p, dst, sz in todo:
            print(f"  缺 {sz/1e6:8.1f} MB  {p}")
        return 0 if not todo else 1

    good = bad = 0
    for i, (p, dst, sz) in enumerate(todo, 1):
        for attempt in range(1, args.retries + 1):
            if os.path.isfile(dst):
                os.remove(dst)
            client = connect()
            try:
                run(client, f"echo ready", quiet=True)   # 探活
                sftp = client.open_sftp()
                sftp.get(p, dst)
                sftp.close()
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(todo)}] 第 {attempt} 次失败 {os.path.basename(dst)}: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                time.sleep(args.sleep)
                continue
            finally:
                client.close()

            got = os.path.getsize(dst) if os.path.isfile(dst) else -1
            if got == sz:
                print(f"[{i}/{len(todo)}] OK {got:>12,} B  {os.path.basename(dst)}")
                good += 1
                break
            print(f"[{i}/{len(todo)}] 大小不符 {got} != {sz}（第 {attempt} 次），重试")
            time.sleep(args.sleep)
        else:
            print(f"[{i}/{len(todo)}] FAILED {os.path.basename(dst)}")
            bad += 1

    print(f"\n完成：成功 {good} / 失败 {bad} / 共 {len(todo)}")
    if bad:
        print("以下文件仍缺失：")
        for p, dst, sz in todo:
            if not (os.path.isfile(dst) and os.path.getsize(dst) == sz):
                print(f"  {p}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
