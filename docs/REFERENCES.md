# 参考资料索引

为编程考核准备的参考论文与开源代码。**结论：这两个仓库是"临摹字帖"，不是拿来直接跑的成品。**

> 论文 PDF 与两个第三方仓库的克隆**不入库**（见根目录 `.gitignore`），本地放在 `refs/`：
> `refs/papers/BoxDreamer_ICCV2025.pdf`、`refs/papers/HccePose_BF_ICCV2025.pdf`、
> `refs/BoxDreamer/`、`refs/HCCEPose/`。重新获取的命令见根目录 `README.md`。

---

## 1. 论文

| 文件 | 说明 |
|---|---|
| `BoxDreamer_ICCV2025.pdf` | BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation (ICCV 2025)。**这就是图里"8 个角点 + heatmap"思路的来源**。 |
| `HccePose_BF_ICCV2025.pdf` | HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation (ICCV 2025)。**只是被借用它的 Blender 渲染脚本**，方法本身跟我们的任务无关。 |

在线版：
- BoxDreamer 项目页 https://zju3dv.github.io/boxdreamer/ ，arXiv https://arxiv.org/abs/2504.07955
- HccePose arXiv https://arxiv.org/abs/2510.10177

---

## 2. BoxDreamer 代码（`BoxDreamer/`，commit `e7200e0`，2025-10-06）

### 重点看这几个文件

| 路径 | 为什么要看 |
|---|---|
| `src/models/BoxDreamerModel.py` | **主模型**。多视角 transformer：参考图 + 查询图 → 角点 heatmap。这是我们网络设计的直接参考。 |
| `src/models/utils/box_utils.py` | **8 个角点的定义与投影**、2D/3D 角点互转。这一份几乎可以照抄。 |
| `src/models/utils/pose_utils.py` | 角点 → 位姿（PnP 那一套）。 |
| `src/models/utils/prediction_utils.py` | 从 heatmap 里提角点（argmax / 期望）的逻辑。 |
| `src/loss/lossesV3.py` | 角点 heatmap 的 loss 设计。 |
| `src/loss/utils/focal_loss.py` | heatmap 常用的 focal loss。 |
| `src/models/modules/encoder/resnet.py` / `dinov2.py` | 可复用的 backbone（图里写的 "ViT/ResNet"）。 |
| `src/datasets/linemod.py`、`onepose.py`、`objaverse.py` | 数据加载 + 角点标签构造，看它们怎么从 GT 位姿算出 2D 角点标签。 |
| `configs/train.yaml`、`configs/model/transformer.yaml` | 超参、loss 权重、训练规模。 |
| `src/demo/` | 推理 demo（CLI + Gradio）。 |

### 关键差异（务必记住）

BoxDreamer 是 **model-free + 多视角**：推理时输入"若干张参考图 + 1 张查询图"，**不需要 3D 模型**。
我们的任务正相反：**步骤①已经给了 3D 模型，测试时只有单张 RGB**。
所以：
- **可以借鉴**：角点定义、heatmap 头、loss、从 heatmap 提点、角点→PnP 的后处理。
- **必须去掉**：参考图分支、跨视角 cross-attention、DUSt3R/VGG-SfM 的重建与匹配部分。

### 未拉取的部分（按需再拉）

- `three/dust3r/` → https://github.com/naver/dust3r （git submodule，当前为空）
- `three/GroundingDINO/` → https://github.com/IDEA-Research/GroundingDINO （git submodule，当前为空）
- 预训练权重：https://1drv.ms/u/s!Ap2hsgjizYNElLIwfl1m9d3V1yf_OA?e=9sixmD （88.6M 参数，训练于 Objaverse + OnePose）

这两个 submodule **只有跑官方 demo / 重建流程才需要**，看代码逻辑不需要。

```bash
# 需要时再执行
cd BoxDreamer && git submodule update --init --recursive
```

---

## 3. HCCEPose 代码（`HCCEPose/`，commit `f75cc4d`，2026-06-02）

**我们只用它的渲染脚本**，不用它的 HCCE 方法（front/back surface 那套）。

### 从头到尾的脚本流水线（文件名前缀就是阶段号）

| 脚本 | 作用 | 我们要不要 |
|---|---|---|
| `s1_p1_obj_rename_center.py` | mesh 居中 + 按 BOP 规范重命名为 `obj_000001.ply` | ✅ 要 |
| `s1_p2_obj_symmetry.py` | 用 KASAL 分析旋转对称性 | ⚠️ 看情况（DJI Action 4 基本不对称） |
| `s1_p3_obj_infos.py` | 生成 `models/models_info.json`（含 bbox 的 `min`/`max`，**8 个 3D 角点就来自这里**） | ✅ 要 |
| `s2_p0_download_cc0textures.py` | 下载 PBR 材质库（用 512 轻量版，600MB） | ✅ 要 |
| **`s2_p1_gen_pbr_data.py`** | **核心：BlenderProc 渲染 PBR 训练数据 → BOP `train_pbr/`** | ✅✅ **最关键的参考文件** |
| `s2_p1_gen_pbr_data.sh` | 反复调用上面的 py，规避内存泄漏 | ✅ 要 |
| `s3_p1_prepare_yolo_label.py` / `s3_p2_train_yolo.py` | 转 YOLO 标注 + 训 2D 检测器 | ❌ 不需要（instance-level，单物体） |
| `s4_p1_gen_bf_labels.py` | 生成 front/back surface 标签 | ❌ 不需要（HCCE 专有） |
| `s4_p2_train_bf_pbr.py` / `s4_p2_test_bf_pbr*.py` | 训练与 BOP 评测 | 🔍 可看评测写法 |
| **`s4_p3_test_mi10_bin_picking.py`** | **单图推理 demo，看 `Tester.predict()` 的接口** | 🔍 可看 |
| `s4_p3_test_mi10_bin_picking_video.py` | **视频推理 demo（和我们任务最像）** | ✅ 参考 |
| `s4_p3_test_mi10_bin_picking_RGBD_foundationpose.py` | 用 FoundationPose 做 RGB-D 精修 | 🔍 **没有真值时可以拿它当伪真值/裁判** |
| `s4_p3_test_mi10_bin_picking_RGBD_megapose.py` | MegaPose 精修 | 🔍 同上 |
| `s4_p3_test_mi10_bin_picking_RGBD_FP_vs_MP.py` | HccePose vs FoundationPose vs MegaPose 对比 | 🔍 同上 |

### 其他重要目录/文件

- `HccePose/` —— 核心包（`tester.py`、`bop_loader.py` 是入口）
- `Refinement/` —— FoundationPose / MegaPose 精修集成
- `bop_toolkit.zip` —— BOP 官方工具包（评测指标 ADD / ADD-S / VSD 都在里面）
- `blenderproc.zip` —— 它们改过的 BlenderProc
- `test_imgs/`、`test_videos/`、`test_imgs_RGBD/` —— 示例输入
- `README.md` / `README_CN.md` —— 环境依赖钉得很死，见下
- `requirements-inference.txt` —— 只跑推理的精简依赖

### 环境要求（⚠️ 和我们机器冲突的点）

官方要求：**Ubuntu + Python 3.10 + torch 2.8.0+cu128 + `bpy==3.6.0` + BlenderProc + EGL**。

我们当前：Windows 11、Python 3.12.7、`torch 2.12.0+cpu`（**无 CUDA**）、RTX 4060 Laptop 8GB。
→ BlenderProc 在 Windows 上大概率跑不通，渲染这一步需要 Linux/服务器，或换渲染方案。

---

## 4. 我们自己的任务目标（一句话）

用 `s2_p1_gen_pbr_data.py` 渲染 DJI Action 4 的合成数据 → 写一个"单张 RGB → 8 角点 heatmap"的实例级网络 → 在 `head_left_rgb_raw.mp4` 上测出 6D 位姿。

算法流程图：`../photo_of_the_project.png`
题目原文：本地保留（`docs/ASSIGNMENT.md`，不入库）
