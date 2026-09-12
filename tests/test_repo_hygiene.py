"""仓库卫生检查。

**背景**：本项目的 ``.gitignore`` 曾经写过一条无锚点的 ``datasets/``，
它匹配**任意深度**的同名目录，把源码目录 ``src/datasets/`` 整个忽略掉了 ——
结果那个包从来没被提交，别人 clone 下来会直接
``ModuleNotFoundError: No module named 'src.datasets'``。
详见 ``docs/TROUBLESHOOTING.md`` 的 ``GIT-03``。

这个用例就是为了让这类"静默漏提交"再也藏不住。

为什么用 ``git check-ignore`` 而不是 ``git ls-files``
------------------------------------------------------
"未被跟踪"是个**临时状态**：刚写好的新文件本来就还没 ``git add``。
而"被忽略"才是真 bug。``git check-ignore`` 正好只报告**未跟踪且被忽略**的路径
（已被跟踪的文件不会报），所以判据是精确的。
"""

import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# 这些目录下的文件不检查：第三方克隆、虚拟环境、缓存、数据
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "refs", "data",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "build", "dist", "node_modules", "outputs", "checkpoints",
}

SOURCE_SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".cfg", ".json", ".sh")

# `.gitignore` 里必须锚定到仓库根目录的模式（无锚点会匹配任意深度）
PATTERNS_NEEDING_ANCHOR = {
    "data", "datasets", "dataset", "outputs", "output",
    "results", "preds", "checkpoints", "weights", "logs",
}

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git 不可用"
)


def _project_files() -> list:
    files = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        files.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    return sorted(files)


def _ignored(paths: list) -> list:
    """返回其中**未被跟踪且被 .gitignore 匹配**的路径。"""
    if not paths:
        return []
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(paths),
        capture_output=True,
        text=True,
    )
    # 0 = 有匹配；1 = 无匹配；其它 = 出错
    if result.returncode not in (0, 1):
        pytest.fail(f"git check-ignore 执行失败: {result.stderr}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
def test_project_files_were_found():
    """先确认扫描本身是有效的（否则下面的断言会变成永远通过的空测试）。"""
    files = _project_files()
    assert len(files) > 20, f"只扫到 {len(files)} 个文件，扫描逻辑可能有问题"
    assert any(f.startswith("src/") for f in files)
    assert any(f.startswith("configs/") for f in files)


def test_no_source_files_are_ignored():
    """源码 / 配置 / 脚本都不应被 .gitignore 吃掉。"""
    ignored = _ignored(_project_files())
    assert not ignored, (
        "以下文件被 .gitignore 忽略了，很可能是**无锚点**的目录模式误伤：\n  "
        + "\n  ".join(ignored)
        + "\n\n例如 `datasets/` 会匹配任意深度的同名目录（包括 src/datasets/），"
          "应该写成 `/datasets/`。"
    )


def test_known_victims_are_not_ignored():
    """具体盯住 GIT-03 的两个受害者：`src/datasets/` 和 `configs/model/vis/`。

    前者被 `.gitignore` 的无锚点 `datasets/` 吞掉，
    后者被无锚点的 `vis/` 吞掉 —— 而 `configs/model/heatmap.yaml` 里
    有 `- vis: default`，缺了它 hydra 的 defaults 组合会直接失败。
    """
    targets = [
        "src/datasets/__init__.py",
        "src/datasets/bop_pbr.py",
        "configs/model/vis/default.yaml",
    ]
    existing = [t for t in targets if (REPO_ROOT / t).is_file()]
    assert len(existing) == len(targets), (
        f"这些文件应该存在但找不到：{sorted(set(targets) - set(existing))}"
    )
    ignored = _ignored(existing)
    assert not ignored, f"这些文件被 .gitignore 忽略了：{ignored}"


def test_every_config_group_in_defaults_has_a_file():
    """`defaults:` 里引用的每个配置组，文件都必须存在。

    这条能直接拦住"配置文件被 gitignore 吞掉"导致的那类故障。
    """
    import re

    group_dir = REPO_ROOT / "configs"
    problems = []

    for yaml_path in group_dir.rglob("*.yaml"):
        text = yaml_path.read_text(encoding="utf-8")
        in_defaults = False
        for raw in text.splitlines():
            line = raw.rstrip()
            if re.match(r"^defaults:\s*$", line):
                in_defaults = True
                continue
            if in_defaults:
                stripped = line.strip()
                if not stripped.startswith("- "):
                    if stripped and not stripped.startswith("#"):
                        in_defaults = False
                    continue
                item = stripped[2:].strip()
                if item in ("_self_",) or ":" not in item:
                    continue
                group, _, option = item.partition(":")
                group, option = group.strip(), option.strip()
                candidate = yaml_path.parent / group / f"{option}.yaml"
                if not candidate.is_file():
                    problems.append(
                        f"{yaml_path.relative_to(REPO_ROOT)} 引用了 "
                        f"{group}: {option}，但 {candidate.relative_to(REPO_ROOT)} 不存在"
                    )

    assert not problems, "配置组文件缺失：\n  " + "\n  ".join(problems)


def test_gitignore_directory_patterns_are_anchored():
    """静态检查：容易误伤的目录模式必须带前导 `/`。"""
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    suspicious = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line.endswith("/") and not line.startswith("/") and "**" not in line:
            if line.rstrip("/") in PATTERNS_NEEDING_ANCHOR:
                suspicious.append(f"L{lineno}: {line}")

    assert not suspicious, (
        "以下 .gitignore 模式没有锚定到仓库根目录，会误伤任意深度的同名源码目录：\n  "
        + "\n  ".join(suspicious)
        + "\n在前面加 `/` 即可。"
    )
