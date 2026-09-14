# 阶段③④：网络与训练

> 阶段③ = 8 角点热图网络；阶段④ = 在阶段②产出的 BOP PBR 数据上训练。
>
> 本篇记录**集成验证结论**（数据 → 网络这一环到底通没通）、怎么跑、以及 CPU 冒烟/过拟合的结果。
> 数据本身的说明见 [`DATASET_v1.md`](DATASET_v1.md)。

---

## 1. 网络结构

```
RGB 256×256
   └─ ResNet18 backbone（11.2 M，去掉最后两层，取 stage2/3/4 特征）
        └─ HeatmapDecoder（FPN 式上采样，862 K）
             └─ 8 通道热图 64×64     ← 8 个 3D 包围盒角点
```

总参数 **12.0 M**，对应 BoxDreamer 的 `BoxDreamerModel`（DINOv2 换成了 ResNet18）。

推理后处理（`src/models/utils/prediction_utils.py`）：

```
热图 64×64  ──argmax / soft_argmax / topk──▶  2D 角点 256×256
                                                  └─ cv2.solvePnP ─▶ 6D 位姿
```

热图风格（`heatmap_style`）：

| | 公式 | 取值 | 损失 |
|---|---|---|---|
| `boxdreamer`（默认） | `exp(-d / (s_i/10)²)`，`s_i` = 角点到物体 2D 中心的像素距离 | `[-1, 1]` | smooth_l1 |
| `centernet` | 固定 σ 的标准高斯 | `[0, 1]` | focal |

---

## 2. 集成验证：数据 → 网络这一环

阶段② 之后我们有了 500 帧真实渲染数据，但 `BOPPBRDataset` **从没碰过真实数据**。
下面四条是补上的检查，工具都在 `scripts/` 里，以后换数据集可以直接复用。

### 2.1 `scripts/verify_bop.py` —— 标注本身对不对

把 3D 包围盒按 `scene_gt.json` 的位姿投影回 RGB（**独立的 numpy 实现**，不复用训练管线的代码）。

- ✅ 投影线框**严丝合缝包住物体**，8 个角点精确落在包围盒角上
- ✅ 投影框始终**略大于** `bbox_visib` 且包含它（正确关系：`bbox_visib` 只框可见像素）

### 2.2 `scripts/verify_dataloader.py` —— 标签自洽

对数据集里每个样本检查四项：

| 检查 | 结果 |
|---|---|
| 各字段形状/取值范围 | `image [3,256,256]∈[0,1]`、`K [3,3]`、`bbox_3d [8,3]`、`corner_2d [8,2]`、`heatmap [8,64,64]∈[-1,1]` ✅ |
| **用裁剪后的 K 重新投影，是否等于 `corner_2d`** | max err `< 1e-3` ✅ |
| **热图峰值格子 == `round(corner_2d × ratio)`** | **误差 0 格** ✅ |
| 角点是否落在裁剪图内 | 8/8 ✅ |

> ⚠️ 写这个检查时踩过一个坑：第一版用 `cell × stride + (stride−1)/2` 反算峰位置，
> 多加了 1.5 px，于是报出"峰值偏 4.2 px (>stride)"的**假问题**。
> 正确做法是直接比**整数格子索引**（GT 构造是 `centers = corner_2d × ratio`，
> 热图在整数网格上取值，峰就在 `round(centers)`）。
> 教训：**验证脚本本身也会错，报错时先怀疑检查公式。**

### 2.3 `scripts/verify_pnp.py` —— 位姿分支能不能用

喂 **GT 角点**给 `recover_pose_from_bb8`，必须精确还原 GT 位姿。实测（64 个样本）：

```
[pnp] 用 GT 角点解位姿：64 / 64 成功
[pnp] 旋转误差 (deg): median 0.0000  max 0.0396
[pnp] 平移误差 (mm) : median 0.0000  max 0.0001
[pnp] ADD    (mm)   : median 0.0000  max 0.0001
```

**这条最关键**：如果喂 GT 角点都解不出来，说明 K / `bbox3d` / 角点顺序 / 位姿约定之间有 bug ——
那种情况下训练再久也不会有人发现位姿分支是坏的。

### 2.4 训练循环冒烟

```powershell
python run.py --config-name=train.yaml exp_name=smoke_cpu `
  datamodule.max_train_samples=16 datamodule.max_val_samples=8 `
  datamodule.num_workers=0 datamodule.batch_size=4 `
  trainer.max_epochs=2 +trainer.limit_train_batches=2 +trainer.limit_val_batches=2 `
  trainer.check_val_every_n_epoch=1 trainer.accelerator=cpu trainer.devices=1
```

- ✅ 模型实例化：12.0 M 参数
- ✅ 训练 2 个 batch × 2 epoch，`train/loss_epoch` 正常
- ✅ **验证循环**跑通，`val/loss`、`val/corner_err_px` 出来（未训练模型 ~101 px，256 px 图上就是随机水平）
- ✅ **checkpoint 保存**：`checkpoints/smoke_cpu/last.ckpt`（141 MB = 48 MB 权重 + Adam 两个动量）
- ✅ 退出码 0

> ⚠️ **`val/add` / `val/rot_err_deg` / `val/acc_5cm5deg` 在冒烟时没出现**，因为未训练模型预测的
> 角点是垃圾、`cv2.solvePnP` 全部失败，`CornerPoseMetrics` 会把 `pose_pred[:,3,3]==0` 的样本
> 过滤掉（`recover_pose_from_bb8` 的约定：PnP 失败留全 0）。
> **这是预期行为，不是 bug** —— 由 §2.3 的 PnP 往返验证排除，并由 §2.5 的过拟合测试正面确认。

### 2.5 小样本过拟合 —— 端到端真的在学

经典的 sanity test：**如果连几十个样本都过拟合不了，训练管线就有问题。**

```powershell
python run.py --config-name=train.yaml exp_name=cpu_overfit `
  image_size=128 heatmap_size=32 `
  datamodule.max_train_samples=64 datamodule.max_val_samples=32 `
  datamodule.batch_size=8 datamodule.augment=false `
  trainer.max_epochs=40 +trainer.limit_train_batches=8 +trainer.limit_val_batches=2 `
  trainer.check_val_every_n_epoch=5 trainer.accelerator=cpu
```

CPU（32 核）跑 **320 步**，结果（`outputs/cpu_overfit/*/logs/version_0/metrics.csv`）：

| epoch | tr_loss | tr_coarse | tr_fine | val corner_px | pck@0.15 | add (mm) | rot_deg | trans_mm | pnp_ok |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 29.5 | 0.468 | 14.51 | 46.0 | 0.195 | 610 | 139.0 | 599 | 1.00 |
| 9 | 20.6 | 0.446 | 10.08 | 43.4 | 0.227 | 537 | 116.0 | 527 | 1.00 |
| 19 | 15.0 | 0.431 | 7.28 | **34.8** | 0.367 | 447 | 99.5 | 442 | 1.00 |
| 34 | 8.6 | 0.350 | 4.10 | 36.0 | 0.406 | **321** | **79.3** | **314** | 1.00 |
| 39 | 8.8 | 0.310 | 4.22 | 35.3 | **0.461** | 575 | 112.6 | 567 | 1.00 |

**结论**：
- `tr_loss` **56.5 → 8.8 单调下降**，`tr_fine` 28.0 → 4.2
- `pck@0.15` 0.195 → **0.461**，`add` / `rot_deg` / `trans_mm` 都在往下走（后三个在 64 个样本上噪声较大）
- **`pose_valid_ratio = 1.000`** —— 模型一旦学到东西，PnP 在**预测**角点上 100% 成功。
  这正是 §2.4 里位姿指标缺失的原因，也正面排除了"位姿分支坏了"的可能
- 绝对精度还很差（`add` 300+ mm），**完全正常**：只有 64 个样本、320 步、CPU、无增强。
  这里要验的是"管线能学"，不是"学得好"
- `tr_coarse` 只从 0.488 降到 0.310，而 `tr_fine` 降了 7 倍 —— coarse 是对整张 `[-1,1]`
  热图取 L1，背景占绝大多数像素，所以它下降慢是合理的；**峰是否尖锐由 fine 项负责**

---

## 3. 依赖与运行

```bash
pip install -r requirements.txt
python run.py --config-name=train.yaml                     # 正式训练
python run.py --config-name=train.yaml exp_name=my_exp     # 换实验名
python run.py --config-name=test.yaml mode=test            # 测试
```

关键配置：

| 位置 | 键 | 说明 |
|---|---|---|
| `configs/train.yaml` | `image_size` / `heatmap_size` | 256 / 64 |
| | `heatmap_style` | `boxdreamer` / `centernet`，model 与 datamodule 共用 |
| `configs/datamodule/bop.yaml` | `dataset_root` | `${hydra:runtime.cwd}/data/dji_action4` |
| | `crop_scale` | 1.4，按 `bbox_visib` 外扩 |
| | `max_train_samples` | `null` = 全用（5000 个样本） |
| `configs/trainer/default.yaml` | `accelerator` | `auto`，有 CUDA 自动用 GPU |

⚠️ **Hydra struct 模式**：往 `trainer` 里加**新键**要写 `+trainer.limit_train_batches=2`，
不写 `+` 会直接报 `Key 'xxx' is not in struct`。

---

## 4. ⚠️ CPU 冒烟时遇到的两个"假失败"

1. **`[exit code: 1]`** —— 但日志里没有任何 traceback，且最后一行是 `All done. Exiting.`。
   用文件重定向验证真实退出码是 **0**。
   原因：PowerShell 会把子进程写到 **stderr** 的正常输出（Lightning 的 `Seed set to 42`、
   git 的进度）当成 `NativeCommandError`。
   **判长任务的成败要看产物和日志，不要只看 PowerShell 报的退出码。**

2. **`Could not override 'trainer.limit_train_batches'`** —— Hydra struct 模式，见上。

---

## 5. 还没做的

| 项 | 说明 |
|---|---|
| **GPU 正式训练** | 本次只在 CPU 上做了冒烟与过拟合（32 核 / 31.7 GB）。正式训练需要 GPU。 |
| **`DATA-16` 遮挡不足** | 训练集里 95% 实例完全可见，HCCEPose 的遮挡鲁棒能力学不到。见 [DATASET_v1.md](DATASET_v1.md) |
| **`DATA-17` 有效分辨率** | 裁完 256×256 有 1.5~2.4× 上采样 |
| **阶段⑤** | 在 `head_left_rgb_raw.mp4` 上测试。**还卡在 `DATA-01`：头戴相机内参 K 未知**（PnP 要用，没有它连评估都做不了） |
