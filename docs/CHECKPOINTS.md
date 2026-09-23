# 产物台账：checkpoint / 日志 / 数据在哪

> 记录时间：2026-09-21（GEO8 训练被主动停止，准备关实例）
>
> ⚠️ **关机 ≠ 释放。**
> AutoDL 上 `/root/autodl-tmp` 是**数据盘**：**关机（停止实例）会保留**，
> **释放实例会全部销毁**。所有 checkpoint、conda 环境
> （`/root/autodl-tmp/envs/train`）、渲染数据（`/root/autodl-tmp/bop/`）
> 都在这个盘上，**都不在系统镜像里** —— 所以「保存镜像」救不了它们。
> 只关机就安全；要点释放之前，必须先确认下面「本地已备份」这一节覆盖了你要的东西。

---

## 1. 磁盘上有 8 个仓库克隆，checkpoint 是分散的

历史上为了并行跑实验，克隆了多份仓库，**每个克隆有自己的 `checkpoints/`**。
主仓库 `quu_s/checkpoints/` 里有一部分是**符号链接**，指向别的克隆
（`stat -c%s` 只返回几十字节，别误判成损坏；`torch.load` 正常）。

| 克隆 | git HEAD | 里面的 checkpoint |
|---|---|---|
| `quu_s` | `f59c893` | 主仓库，大部分实验（见下表） |
| `quu_s_GEO7` | `b8d3247` | **GEO7** ← 真实视频最好的模型 |
| `quu_s_geo` | `b8d3247` | **GEO1** |
| `quu_s_GEO2` | `b8d3247` | **GEO2** |
| `quu_s_B1` | `b8d3247` | **B1** |
| `quu_s_B2` | `c21194c` | **B2** |
| `quu_s_p1` | `623f2db` | **P1** |
| `quu_s_old` | `24d511b` | **OLD25**、`BASE → OLD25`（软链） |

**符号链接一览**（`quu_s/checkpoints/` 内）：

```
BASE -> quu_s_old/checkpoints/BASE/last.ckpt
GEO1 -> quu_s_geo/checkpoints/GEO1/last.ckpt
GEO2 -> quu_s_GEO2/checkpoints/GEO2/last.ckpt
GEO7 -> quu_s_GEO7/checkpoints/GEO7/last.ckpt
```

## 2. 各 checkpoint 是什么、指标多少

数据列：`epoch/step` 取自 checkpoint 内记录（即"验证最优"那一轮，不是训练停止那一轮）。

| 名字 | epoch/step | 数据 | 关键指标 | 说明 |
|---|---|---|---|---|
| **train_v1** | 244 / 72520 | v1（5000 样本） | **误差 4.12 px，PCK@0.05 96.88%**，框 IoU 0.961 | **合成集上最好的单实例模型**，所有对比的基准 |
| **GEO7** | 59 / 14100 | GEO7（3958 样本） | **真实视频平均 IoU 0.799**；合成集 39.21 px / PCK@0.05 48.64% | **真实视频上最好的模型**。注意它的合成集指标很差 —— 合成精度与真实表现排序相反 |
| **GEO8** | 94 / 22325 | GEO7 | 训练到 epoch 136 被停；epoch 89 时 50.62 px / 0.342 | **有增强那一臂**。同 epoch 落后 GEO7（45.64 px / 0.423）。用于关掉 `ALGO-11` |
| **MIGEO7** | 199 / 47000 | GEO7 | 合成集 **recall 0.828 / precision 0.978 / err 6.23 px** | **多实例**（中心热图 + 角点偏移）。真实帧上中心热图峰值仅 0.1729，不响应 |
| **GEO1** | 194 / 57525 | GEO1 | 16.37 px / PCK@0.05 79.28%；真实视频 IoU 0.605 | 对照点 |
| **E5_noaug_300ep** | 244 / 72275 | v1，无增强 300ep | 16.75 px | `ALGO-11` 的 E5 臂 |
| **GEO2** | 189 / 34770 | GEO2 | 数据本身不可用（48.6% 实例 visib<0.1） | 反面教材，留证 |
| **B1 / B2 / P1** | — | v1 | 消融/门禁实验 | 与 E6 并行跑 |
| **E0~E4** | 见 checkpoint | v1 | 损失函数消融 | `ALGO-12`：损失不是杠杆 |
| **E6_dinov2_224** | 189 / 56050 | v1 | 23.29 px，**比 ResNet18 更差** | 主干假设被否；356 MB（含冻结 DINOv2） |
| **E7_resnet_224** | 274 / 81125 | v1 | 输入 224 的对照 | — |
| **CMB** | 4 / 2650 | v1+GEO7 多路径 | 只跑了 4 轮就被误判为"卡死"而停 | 多根合并的产物；数据侧已验证 8927 样本正常 |
| **fit_/fitc_/eval_*** | — | — | 拟合实验与三种解码器对比 | `ALGO-05`：topk vs soft_argmax 差距可忽略 |
| **W0/W8/MSTEPS/SMOKE_*** | — | — | 冒烟/诊断用 | 可删 |

## 3. 数据在哪

| 数据 | 路径 | 量 |
|---|---|---|
| v1 | `/root/autodl-tmp/bop/v1/dji_action4_hybrid` | 500 帧 / 5000 实例 → 过滤后 4969 |
| GEO7 | `/root/autodl-tmp/bop/v3/GEO7/dji_action4_hybrid` | 700 帧 / 4200 实例 → 过滤后 3958 |
| GEO2 | `/root/autodl-tmp/bop/v3/GEO2/...` | 500 帧 / 6000 实例，**不可用** |
| 合并根 | `/root/autodl-tmp/bop/merged` | 符号链接布局，8927 样本 |
| 3D 模型 | `data/dji_action4_hybrid/models/`（也在各 bop 根下） | `models_info.json` + PLY |
| 渲染脚本 | `/root/autodl-tmp/bp_ws/`（**不在仓库里**，`data/` 是 gitignore） | BlenderProc 脚本 |

**渲染脚本必须单独备份** —— 它们在 `bp_ws/`，既不在仓库也不在 `data/results`。

## 4. 本地已备份（`data/results/remote_backup/`，共 547.7 MB）

| 文件 | 字节数 | 内容 |
|---|---|---|
| `results_small.tar.gz` | 65,930 | GEO8 的 metrics.csv + 曲线 + 关键日志（22 个条目） |
| `results_all.tar.gz` | 7,208,291 | 1279 个条目：7 个克隆的 outputs/configs + 135 个日志 |
| `results_scripts.tar.gz` | 849,965 | 198 个条目：**`bp_ws/` 全部渲染脚本 + 所有仓库外的 .py/.sh** |
| `train_v1_last.ckpt` | 141,098,264 | 合成集最好（4.12 px / 96.88%） |
| `GEO7_last.ckpt` | 141,098,328 | 真实视频最好（IoU 0.799） |
| `GEO8_last.ckpt` | 141,099,352 | 有增强那一臂（epoch 94） |
| `MIGEO7_last.ckpt` | 142,889,464 | 多实例（recall 0.828） |

全部字节数与远端**精确一致**（用 `stat -c%s` 逐个核对，避免再踩"下载被截断"
那个坑 —— 之前 66,912,256 / 142,889,464 的截断让 `torch.load` 报
`PytorchStreamReader failed reading zip archive`）。

**`results_scripts.tar.gz` 里最关键的三个文件**：
- `bp_ws/gen_pbr_data_demo.py`（35 KB）—— **生成全部训练数据的 BlenderProc 脚本**
- `bp_ws/preview_object.py`、`bp_ws/check_mesh.py`、`bp_ws/make_cc0textures.py`
- 各实验的 driver `.sh`

**没下载但仍留在数据盘上的**：`E0~E7`、`GEO1`、`GEO2`、`B1`、`B2`、`P1`、
`OLD25`、`CMB`、`fit*/fitc*/eval_*`、`MSTEPS`/`W0`/`W8`/`SMOKE_*`（共约 8.8 GB），
以及数据本身 `/root/autodl-tmp/bop`（1.7 GB）、
`dji_action4_hybrid_dataset.tar.gz`（357 MB）、`handoff.tar.gz`（248 MB）、
`qa_artifacts.tar.gz`（247 MB）、Blender 3.6（2.5 GB）、conda 环境（788 MB）。
**只关机会保留；释放前需要重新评估是否要下载。**

`hf_cache`（28 GB）是 HuggingFace 模型缓存，可重新下载，不必备份。


## 5. 恢复步骤（换实例/重开机后）

```bash
# 1. 代码：全部在 GitHub，最新 commit 见仓库 main
git clone https://github.com/lsyyyyy-sudo/quu_s_instance-level_object_pose_estimation_network.git
# 2. 环境：conda env 在 /root/autodl-tmp/envs/train（关机保留；释放需重建）
# 3. 数据：/root/autodl-tmp/bop/（关机保留）
# 4. checkpoint：把本地备份的 4 个 .ckpt 传回 quu_s/checkpoints/<名字>/last.ckpt
# 5. 续训：python run.py --config-name=train.yaml exp_name=GEO8 resume=true
#    （resume 读 checkpoints/${exp_name}/last.ckpt，见 configs/model/heatmap.yaml）
```

## 6. 遗留待办

- [ ] `ALGO-11` 收尾：GEO8（有增强）跑完的 A/B 离线对比没做完就被停了。
      已有同 epoch 证据：GEO8 落后 GEO7，增强是负收益。
- [ ] 真实视频仍**零标注** → "多实例解没解决"无法量化。优先做检测器框 + 人工修正，
      或 AprilTag 反投影拿真值。
- [ ] 域差距：真实 DJI **屏幕点亮 + 贴标签**，渲染的是干净模型。渲修改造未开始。
- [ ] `crop_use_bbox_obj` 建议改 `true`（标签是 amodal，裁剪也该用 amodal 框）。

---

## 7. 2026-09-23 补充：第二批备份（剩余 checkpoint）

### 已完成

| 文件 | 内容 |
|---|---|
| `INSTANCE_MANIFEST.txt` | ⭐ **实例完整清单**：磁盘 / 8 个克隆的 HEAD / 全部 59 个 checkpoint / 权重与下载 URL / 数据 / 渲染脚本 / md5 去重 |
| `bd_small.tar.gz` | 16.5 MB / 602 文件：BoxDreamer 工作树（含 dust3r/croco/GroundingDINO 子模块）+ 全部下载日志与探测输出 |
| `CKPT_DOWNLOAD_LIST.txt` | 去重后的待搬清单（36 条，一行一个远端路径） |
| `ckpt_misc/` | ✅ **36 个 / 4.94 GB 全部落地并通过校验** |

**去重省了一半**：59 个 ckpt 文件只有 **39 种内容**（`last.ckpt` 与 `last-v1.ckpt`
大多是同一份），排除已存的 4 个（`train_v1`/`GEO7`/`GEO8`/`MIGEO7`）后要搬 36 个。

**本地备份总计 5.49 GB**（36 个 ckpt + 4 个关键 ckpt + 3 个归档包 + 清单）。

### 校验结果（两道）

| 校验 | 结果 |
|---|---|
| 权威尺寸比对（`--dry-run`，逐文件比 `stat` 得到的远端大小） | ✅ **需要下载 0 个 / 已完成 36 个**，退出码 0 |
| **zip 结构完整性**（截断的文件读不出中央目录） | ✅ **36/36 完好，0 异常** |

### ⚠️ 中途被中断过一次，而且**新工具当场抓到一个被截断的文件**

第一次搬运（`foreach { remote.py get }` 循环）搬到第 18 个时**远端关机**，
后 18 个全失败，且循环自己还报 `exit 0`（详见 `ENV-22`）。

实例恢复后续传时，`fetch_remote_files.py` 报
**"需要下载 19 个；已完成 17 个"** —— 而我手工数是 18 已完成。
差的那 1 个正是 **`E6_dinov2_224/last-v1.ckpt`：本地 351,469,568 B，
远端 356,466,898 B，被截断了 5 MB**。`Test-Path` 判不出来（文件存在），
**只有比尺寸才发现** —— 这正是 `ENV-22` 记的那个坑，工具把它堵上了。

> 换句话说：如果没有尺寸校验，这个 truncated 的 ckpt 会被永久当成"已完成"，
> 直到某天 `torch.load` 报 `PytorchStreamReader failed reading zip archive`
> 才发现 —— 而那时原始文件可能已经不在了。

**续传命令**（以后再搬东西可复用；已完成的会自动跳过）：

```bash
python scripts/fetch_remote_files.py \
    --list data/results/remote_backup/CKPT_DOWNLOAD_LIST.txt \
    --out  data/results/remote_backup/ckpt_misc
# 只比对尺寸、报缺口，不下载：
python scripts/fetch_remote_files.py --list ... --out ... --dry-run
```


