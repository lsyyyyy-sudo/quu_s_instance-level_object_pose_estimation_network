"""仓库体检：链接、体积、卫生、以及「不该公开的东西有没有公开」。

为什么需要：这个仓库是 **public** 的，而本项目里有两类东西**天然不该进去** ——
项目方提供的原始素材、以及人类自己的笔记。已经踩过一次（见
`docs/TROUBLESHOOTING.md` 的 `GIT-01`）：`gitignore` 只影响之后的提交，
一旦推上去就得改写历史才能清干净。所以这个检查要能**经常跑、几十秒出结果**。

检查项：
  1. 本地与 origin 是否同步
  2. 已跟踪文件：按顶层目录统计数量/体积；列出最大的 15 个
  3. ⚠️ 敏感文件是否被跟踪（人类笔记、项目方素材、图床式截图、凭据）
  4. Markdown 里的相对链接/图片引用是否都能解析到实际文件
  5. 文档索引（README 的文档表）与实际 `docs/*.md` 是否对得上
  6. 仓库体积（含 .git 历史）

退出码：有问题返回 1，便于接进 CI。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 明确不该出现在 public 仓库里的路径特征
SENSITIVE = [
    (r"HUMAN_LOG", "人类自己的实验笔记"),
    (r"ASSIGNMENT", "题目原文"),
    (r"联想截图|screenshot|Screenshot|Snipaste|微信截图|QQ图片", "个人截图工具产出的图"),
    (r"\.env$|credential|secret|password", "凭据"),
]
# 这些大文件类型不该入库
BIG_EXT = (".ckpt", ".pth", ".pt", ".mp4", ".avi", ".zip", ".tar", ".gz", ".ply", ".glb")


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout.strip()


def tracked() -> list[str]:
    out = sh("git", "-c", "core.quotepath=false", "ls-files")
    return [l for l in out.splitlines() if l.strip()]


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n/1:.1f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} GB"


def main() -> int:
    problems: list[str] = []
    files = tracked()
    print(f"仓库: {ROOT}")
    print(f"已跟踪文件: {len(files)} 个\n")

    # ---- 1 同步状态 ----
    print("=" * 68)
    print("1. 本地 vs origin")
    head = sh("git", "rev-parse", "HEAD")
    origin = sh("git", "rev-parse", "origin/main")
    print(f"  HEAD   = {head[:12]}")
    print(f"  origin = {origin[:12]}")
    if head != origin:
        problems.append(f"本地与 origin 不一致（{head[:8]} vs {origin[:8]}）")
        print("  ⚠️ 未同步")
    else:
        print("  ✅ 一致")
    dirty = [l for l in sh("git", "status", "--porcelain").splitlines() if l.strip()]
    if dirty:
        print(f"  工作区有 {len(dirty)} 项未提交：")
        for l in dirty:
            print(f"    {l}")
    else:
        print("  ✅ 工作区干净")

    # ---- 2 体积分布 ----
    print()
    print("=" * 68)
    print("2. 体积分布（按顶层目录）")
    groups: dict[str, list[int]] = {}
    for f in files:
        p = os.path.join(ROOT, f)
        sz = os.path.getsize(p) if os.path.isfile(p) else 0
        groups.setdefault(f.split("/")[0], []).append(sz)
    for k in sorted(groups, key=lambda x: -sum(groups[x])):
        v = groups[k]
        print(f"  {k:<28} {len(v):>4} 个   {sum(v)/1e6:>8.2f} MB")

    print()
    print("  最大的 15 个已跟踪文件：")
    sized = sorted(((os.path.getsize(os.path.join(ROOT, f)), f) for f in files
                    if os.path.isfile(os.path.join(ROOT, f))), reverse=True)
    for sz, f in sized[:15]:
        print(f"    {sz/1e6:>8.2f} MB  {f}")

    # ---- 3 敏感文件 ----
    print()
    print("=" * 68)
    print("3. ⚠️ 敏感文件是否被跟踪")
    hits = []
    for f in files:
        for pat, why in SENSITIVE:
            if re.search(pat, f, re.IGNORECASE):
                hits.append((f, why))
    if hits:
        for f, why in hits:
            print(f"  🔴 {f}   （{why}）")
        problems.append(f"{len(hits)} 个敏感文件被 git 跟踪：{[h[0] for h in hits]}")
    else:
        print("  ✅ 没发现")
    print()
    print("  大文件类型是否入库：")
    bigs = [f for f in files if f.lower().endswith(BIG_EXT)]
    if bigs:
        for f in bigs:
            print(f"  🔴 {f}")
        problems.append(f"{len(bigs)} 个大文件入库")
    else:
        print("  ✅ 没有")
    # 检查 .gitignore 有没有覆盖这些模式的规则
    gi = os.path.join(ROOT, ".gitignore")
    gi_txt = open(gi, encoding="utf-8").read() if os.path.isfile(gi) else ""
    for pat, why in SENSITIVE[:3]:
        key = pat.split("|")[0].strip("^$")
        if key and key.lower() not in gi_txt.lower():
            print(f"  ⚠️ .gitignore 里没有与 “{key}”（{why}）相关的规则 → 下次 git add . 可能再带进来")

    # ---- 4 Markdown 链接 ----
    print()
    print("=" * 68)
    print("4. Markdown 相对链接 / 图片引用")
    md_files = [f for f in files if f.lower().endswith(".md")]
    broken = []
    for f in md_files:
        path = os.path.join(ROOT, f)
        try:
            txt = open(path, encoding="utf-8").read()
        except OSError:
            continue
        refs = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", txt) + \
               re.findall(r'<img[^>]+src="([^"]+)"', txt)
        for r in refs:
            r = r.split("#")[0].strip()
            if not r or r.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(path), r))
            if not os.path.exists(target):
                broken.append((f, r))
    if broken:
        for f, r in broken:
            print(f"  🔴 {f}  ->  {r}")
        problems.append(f"{len(broken)} 个 Markdown 链接指向不存在的文件")
    else:
        print(f"  ✅ {len(md_files)} 个 md 文件的相对链接全部可解析")

    # ---- 5 文档索引一致性 ----
    print()
    print("=" * 68)
    print("5. 文档索引 vs 实际 docs/")
    docs = sorted(f for f in files if f.startswith("docs/") and f.endswith(".md")
                  and "/" not in f[len("docs/"):])
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    missing_in_readme = [d for d in docs if os.path.basename(d) not in readme]
    print(f"  docs/ 下 {len(docs)} 个 md，README 里引用了 {len(docs) - len(missing_in_readme)} 个")
    if missing_in_readme:
        print("  ⚠️ README 未引用（**也可能是故意的** —— 比如人类自己的笔记就不该在 public README 里推广）:")
        for d in missing_in_readme:
            print(f"     {d}")

    # ---- 6 体积 ----
    print()
    print("=" * 68)
    print("6. 仓库体积")
    print("  " + sh("git", "count-objects", "-vH").replace("\n", "\n  "))
    if not os.path.isfile(os.path.join(ROOT, "LICENSE")):
        print("  ⚠️ 没有 LICENSE（public 仓库建议补一个）")

    # ---- 结论 ----
    print()
    print("=" * 68)
    if problems:
        print("❌ 发现 %d 类问题：" % len(problems))
        for p in problems:
            print(f"   · {p}")
        return 1
    print("✅ 没发现问题")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
