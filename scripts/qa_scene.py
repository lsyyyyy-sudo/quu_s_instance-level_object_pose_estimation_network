"""场景质检门：渲完一个场景先打分，不合格就丢掉重渲。

为什么需要（见 docs/RESULTS.md）
--------------------------------
v2 用同一个脚本渲了 52 个场景，逐场景评分后发现：

    median 误差从  7.8 px  一直分布到  121 px

而 v1 的 25 个场景全都正常（整体 median 1.44 px）。**坏场景是成片出现的**，
但用 bbox 统计、亮度、姿态分布**都查不出来**（都做过，见 docs/RESULTS.md）。
与其继续找那个单一原因，不如**渲完就验货** —— 20 秒一个场景，
比事后 debug 便宜得多。

判据
----
用**参考模型**（默认 v1 的 checkpoint）在刚渲好的场景上跑一遍，看两个量：

``median_norm``   归一化中位误差（÷ GT 2D 框对角线）—— 抗重尾，看"大多数样本好不好"
``fail_rate``     误差 > 0.1 对角线的角点占比 —— 看尾巴有多脏

**只看 median**：mean 会被少量崩掉的样本主导，实测同一批渲染里
median 都是 2~20 px，mean 却能差 8 倍。所以要同时对两者设阈。

用法::

    python scripts/qa_scene.py --dataset-root <场景目录> [--ckpt train_v1]
    python scripts/qa_scene.py --dataset-root <目录> --gate   # 只输出 PASS/FAIL，退出码 0/1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 阈值：v1 的 25 个场景实测 median_norm 0.0054、fail_rate 2.2%，
# 这里放到 4 倍 / 10 倍作为"这个场景还算干净"的宽松上限。
DEFAULT_MAX_MEDIAN = 0.020
DEFAULT_MAX_FAIL = 0.25


def score(dataset_root: str, ckpt: str, workers: int = 2) -> dict:
    """跑一遍 eval_corners.py，把关键数字抠出来。"""
    log = os.path.join(tempfile.gettempdir(), "qa_scene.log")
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts", "eval_corners.py"),
        f"exp_name=qa_{os.path.basename(os.path.dirname(dataset_root))}",
        f"pretrain_name={ckpt}",
        f"+eval_vis_out={os.path.join(tempfile.gettempdir(), 'qa_scene.png')}",
        f"datamodule.dataset_root={dataset_root}",
        "datamodule.val_ratio=1.0",          # 20 帧的小场景也要全评
        "datamodule.max_val_samples=4096",
        "datamodule.augment=false",
        f"datamodule.num_workers={workers}",
        "datamodule.batch_size=16",
        "trainer.accelerator=auto",
        "trainer.devices=1",
    ]
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=False)
    txt = open(log, encoding="utf-8", errors="ignore").read()

    out = {"n_corners": 0, "median_norm": None, "mean_norm": None,
           "fail_0.1": None, "pck05": None}
    import re
    m = re.search(r"角点\s+(\d+)", txt)
    if m:
        out["n_corners"] = int(m.group(1))
    # 归一化那一段：两行，第一行 mean/median
    m = re.search(r"【归一化误差】[^\n]*\n\s*mean\s+([\d.]+)\s+median\s+([\d.]+)", txt)
    if not m:
        m2 = re.findall(r"mean\s+([\d.]+)\s+median\s+([\d.]+)", txt)
        if len(m2) >= 2:
            m = m2[1]
            out["mean_norm"], out["median_norm"] = float(m[0]), float(m[1])
    else:
        out["mean_norm"], out["median_norm"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"误差 > 0\.1[^\n]*?([\d.]+)%", txt)
    if m:
        out["fail_0.1"] = float(m.group(1)) / 100.0
    m = re.search(r"PCK@0\.05\s+([\d.]+)%", txt)
    if m:
        out["pck05"] = float(m.group(1)) / 100.0
    out["_log_tail"] = txt[-400:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="场景目录（含 models/ camera.json train_pbr/）")
    ap.add_argument("--ckpt", default="train_v1", help="参考模型名（checkpoints/<name>/last.ckpt）")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-median", type=float, default=DEFAULT_MAX_MEDIAN)
    ap.add_argument("--max-fail", type=float, default=DEFAULT_MAX_FAIL)
    ap.add_argument("--gate", action="store_true", help="只输出 PASS/FAIL 并按退出码返回")
    a = ap.parse_args()

    r = score(a.dataset_root, a.ckpt, a.workers)
    if r["median_norm"] is None:
        print(f"QA FAIL (无法解析评测输出)\n{r['_log_tail']}")
        return 1

    ok = (r["median_norm"] <= a.max_median) and \
         (r["fail_0.1"] is not None and r["fail_0.1"] <= a.max_fail)
    tag = "PASS" if ok else "FAIL"
    if a.gate:
        print(f"{tag} {a.dataset_root} median_norm={r['median_norm']:.4f} "
              f"fail={r['fail_0.1']:.3f} n={r['n_corners']}")
        return 0 if ok else 1

    print(f"=== QA {tag} ===")
    print(f"  {a.dataset_root}")
    print(f"  角点数            {r['n_corners']}")
    print(f"  归一化 median     {r['median_norm']:.4f}   (上限 {a.max_median})")
    print(f"  归一化 mean       {r['mean_norm']:.4f}" if r["mean_norm"] is not None else "")
    print(f"  失败率 >0.1       {r['fail_0.1']:.3f}    (上限 {a.max_fail})")
    print(f"  PCK@0.05          {r['pck05']:.3f}" if r["pck05"] is not None else "")
    print(f"  参考 v1(25 场景)  median_norm 0.0054  fail 0.022")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
