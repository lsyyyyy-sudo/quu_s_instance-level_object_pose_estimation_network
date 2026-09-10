# Instance-level Object Pose Estimation Network

> 从单张 RGB 图像估计目标物体的 **6D 位姿**（3 自由度旋转 + 3 自由度平移）。
> 全流程自建：**三维模型 → 合成数据渲染 → 网络实现 → 训练 → 真实数据测试**。

<p align="center">
  <img src="photo_of_the_project.png" width="520" alt="算法流程图">
</p>

**当前状态：🚧 进行中**（详细进度见 [§6 Roadmap](#6-roadmap)）

---

## 1. 任务概述

给定一个**目标物体**（本项目的目标物体是 **DJI Action 4** 运动相机），在**只有单目 RGB 图像、且没有真实标注数据**的条件下，估计它在相机坐标系下的 6D 位姿。

整条流水线分五步：

| 阶段 | 内容 | 产出 |
|:---:|---|---|
| **①** | 给定目标物体的图片，用三维生成模型得到目标物体的三维模型 | 带纹理 mesh（BOP 格式 `obj_000001.ply` + `models_info.json`） |
| **②** | 使用 **HCCEPose** 中的 Blender 脚本渲染构造物体位姿估计的训练数据 | BOP 格式 `train_pbr/`：RGB + GT 6D 位姿 + 相机内参 |
| **③** | 实现 instance-level object pose estimation network（检测 8 个角点） | 网络代码（单图 RGB → 8 通道角点热图） |
| **④** | 在第二步渲染得到的数据上训练网络 | 权重 + 训练曲线（**纯合成数据训练**） |
| **⑤** | 在目标数据上进行测试 | 可视化结果 + 位姿精度评估 |

### 为什么用「8 个角点」作为中间表示

直接回归旋转量在图像平面上高度非线性，精度和泛化都差。本项目采用被广泛验证的**角点中介**方案：

```
单目 RGB ──▶ 网络 (ViT / ResNet) ──▶ 8 个角点的 Heatmap ──▶ 提取 8 个角点 ──▶ solvePnP ──▶ R, t
```

- 8 个角点 = 物体 **3D 包围盒**的 8 个顶点，在物体坐标系下坐标已知（来自 `models_info.json` 的 `min`/`max`）；
- 网络预测它们在图像上的 2D 投影热图，取 soft-argmax 得到亚像素坐标；
- 得到 2D–3D 对应后用 `cv2.solvePnP` 最小二乘解出 R, t，数值上远比直接回归稳定。

> ⚠️ 被遮挡的角点**同样需要监督**（amodal 预测）——目标物体在实际场景中经常被手部 / 衣袖遮挡。

### 与 BoxDreamer 的差异

网络设计参考 [BoxDreamer](https://zju3dv.github.io/boxdreamer/)，但**输入不同，因此结构不同**：

| | BoxDreamer | 本项目 |
|---|---|---|
| 输入 | 若干参考图 + 1 张查询图 | **单张** RGB 图 |
| 是否需要 3D 模型 | 不需要（model-free） | **需要**（阶段①提供） |
| 结构 | 多视角 transformer + 跨视角注意力 | **单图** backbone（ResNet/ViT）+ 热图解码器 |

因此可复用其**角点定义、热图头、loss 与后处理**，但需去掉参考图分支与跨视角注意力。

---

## 2. 仓库结构（规划）

```
.
├── README.md
├── .gitignore
├── docs/
│   └── REFERENCES.md          # 参考论文 / 代码的阅读索引
├── photo_of_the_project.png   # 算法流程图
│
├── configs/                   # 训练 / 模型 / 数据配置
├── src/
│   ├── datasets/              # 数据加载、角点标签生成
│   ├── models/                # 网络结构（backbone + heatmap head）
│   ├── loss/                  # 热图损失
│   ├── train.py               # 训练入口
│   ├── infer.py               # 单图 / 视频推理
│   └── utils/                 # 角点定义、投影、PnP、可视化
└── scripts/                   # 数据准备与渲染调用脚本
```

> 目录将随实现逐步补齐。

---

## 3. 数据与参考资料（**不入库**）

本仓库**不包含**原始数据与第三方代码，请自行获取：

| 资源 | 说明 | 获取方式 |
|---|---|---|
| `head_left_rgb_raw.mp4` | 目标数据：头戴左目相机第一视角视频，3248×2464 @ 29.97 fps，2850 帧（95 s），目标物体为腕戴的 DJI Action 4 | 由项目方提供（277 MB，超过 GitHub 单文件上限，故未入库） |
| BoxDreamer 论文 | ICCV 2025，8 角点热图思路来源 | [arXiv:2504.07955](https://arxiv.org/abs/2504.07955) |
| HccePose(BF) 论文 | ICCV 2025，仅借用其 Blender 渲染脚本 | [arXiv:2510.10177](https://arxiv.org/abs/2510.10177) |
| BoxDreamer 代码 | 网络设计参考 | `git clone https://github.com/zju3dv/BoxDreamer.git` |
| HCCEPose 代码 | 渲染脚本参考 | `git clone https://github.com/WangYuLin-SEU/HCCEPose.git` |

克隆到本地后统一放在 `refs/` 目录（已被 `.gitignore` 忽略）：

```bash
mkdir -p refs && cd refs
git clone --depth 1 https://github.com/zju3dv/BoxDreamer.git
git clone --depth 1 https://github.com/WangYuLin-SEU/HCCEPose.git
```

阅读顺序建议见 [`docs/REFERENCES.md`](docs/REFERENCES.md)。

---

## 4. 环境

| 用途 | 要求 |
|---|---|
| 训练 / 推理 | Python 3.10+，PyTorch（CUDA），OpenCV |
| 合成数据渲染 | Ubuntu + BlenderProc + `bpy` + EGL（⚠️ Windows 上通常无法运行） |

> 渲染环境与训练环境建议使用**独立**的 conda / venv，HCCEPose 的依赖钉得较死。

---

## 5. 评估方式

- **定性**：将预测位姿下物体的 3D 包围盒投影回图像，检查是否贴合；
- **定量**：与真值位姿比较，报告 **ADD / ADD-S**、**5cm5°** 以及**重投影误差**。

> 目标视频若缺少位姿真值，可用 RGB-D 精修方法（如 FoundationPose / MegaPose）的输出作为伪真值进行对比。

---

## 6. Roadmap

| # | 任务 | 状态 |
|:---:|---|:---:|
| 1 | 获取目标物体 3D 模型并转换为 BOP 格式 | ⬜ 未开始 |
| 2 | 搭建 BlenderProc 渲染环境，跑通合成数据生成 | ⬜ 未开始 |
| 3 | 从 GT 位姿生成 8 角点 2D 标签 | ⬜ 未开始 |
| 4 | 实现单图角点热图网络 | ⬜ 未开始 |
| 5 | 在渲染数据上训练（含域随机化与数据增强） | ⬜ 未开始 |
| 6 | 在目标视频上测试并可视化 | ⬜ 未开始 |
| 7 | 定量评估与误差分析 | ⬜ 未开始 |

---

## 7. 参考

```bibtex
@InProceedings{yu2025boxdreamer,
  title     = {BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation},
  author    = {Yu, Yuanhong and He, Xingyi and Zhao, Chen and Yu, Junhao and Yang, Jiaqi
               and Hu, Ruizhen and Shen, Yujun and Zhu, Xing and Zhou, Xiaowei and Peng, Sida},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}

@InProceedings{HccePose_BF,
  title     = {HccePose(BF): Predicting Front \& Back Surfaces to Construct Ultra-Dense
               2D-3D Correspondences for Pose Estimation},
  author    = {Wang, Yulin and Hu, Mengting and Li, Hongli and Luo, Chen},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

---

## 致谢

本项目的渲染流程参考 [HCCEPose](https://github.com/WangYuLin-SEU/HCCEPose)（基于 [BlenderProc](https://github.com/DLR-RM/BlenderProc)），网络设计参考 [BoxDreamer](https://github.com/zju3dv/BoxDreamer)，数据格式遵循 [BOP benchmark](https://bop.felk.cvut.cz/)。
