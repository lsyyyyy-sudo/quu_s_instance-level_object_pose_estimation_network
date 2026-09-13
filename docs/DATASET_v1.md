# v1 训练集存档（BOP PBR，DJI Osmo Action 4）

> 阶段① + 阶段② 的产物记录。**2026-09-14 生成。**
>
> 数据本身不在仓库里（`/data/` 已 gitignore）：远端保留原件，本地留了一份打包存档，
> 位置和校验见 §6。这份文档记录**造了什么、怎么造的、验收结论、已知缺陷**。

---

## 1. 一句话

用**生成式 3D 模型**（Hunyuan3D-2mv，4 视图）做出 DJI Osmo Action 4 的网格，
再用 **HCCEPose 的 BlenderProc 渲染脚本**渲成 **500 帧 BOP PBR 训练集**，
含 RGB / 深度 / 掩码 / 6D 位姿 / 包围盒标注。

---

## 2. 模型

| | |
|---|---|
| 来源 | `tencent/Hunyuan3D-2mv`，`hunyuan3d-dit-v2-mv-turbo`（形状）+ `hunyuan3d-paint-v2-0-turbo`（纹理） |
| 输入 | `data/raw/official/thumbs/{11,12,13,16}_m.jpg` = front / back / right / left（**4 视图**） |
| 形状 | 400,856 顶点 / 801,708 面 |
| 简化后 | 319,668 面（保纹理简化，`preserveboundary=True`） |
| 包围盒 | **69.94 × 44.87 × 33.09 mm**，官方 70.5 × 44.2 × 32.8 → 三轴 **99% / 102% / 101%** |
| 直径 | 81.04 mm（BOP `diameter`，凸包最大点距） |
| 物体坐标系 | `+X` 长轴 / `+Y` 上 / `+Z` **镜头方向**，原点在包围盒中心 → 见 [DATA.md](DATA.md) §5.5 |

**为什么形状和纹理用了不同的视图数**：`13/16` 是 ¾ 侧视而非正交侧视。
形状编码器容忍视角近似（4 视图让厚度误差从 +27% 降到 +1.5%），
但纹理投影不容忍（4 视图把图集糊成噪声）。所以**形状用 4 视图、纹理只用 front/back**。
详见 [MODEL_REGEN.md](MODEL_REGEN.md)。

---

## 3. 数据集

### 3.1 生成命令

```bash
# 远端，RTX 4090，GPU 模式
cd /root/autodl-tmp/bop/dji_action4_hybrid     # cwd 必须是数据集目录
export BP_DEVICE_TYPE=CUDA BP_RES=""           # BP_RES 留空 = 用 camera.json 的 1024x768
export BP_NUM_SCENES=25 BP_FRAMES=20           # 25 场景 x 20 帧 = 500 帧
export BP_NUM_OBJS=10                          # 每场景 10 个实例
export BP_SAMPLES=50 BP_NUM_WORKER=4
export BP_WRITE_BOP=1 BP_RADIUS_MIN=0.35 BP_RADIUS_MAX=0.55
/root/autodl-tmp/blender-3.6.0-linux-x64/blender --background \
    --python /root/autodl-tmp/bp_ws/gen_pbr_data_demo.py
```

耗时 **96 分钟**（5759 s）。GPU 利用率大部分时间是 0% —— 瓶颈在 GT mask
（pyrender 算 5000 张掩码），渲染本身只占小头。

### 3.2 结构

```
dji_action4_hybrid/
├── camera.json                          # fx=fy=800, cx=512, cy=384, 1024x768, depth_scale 0.1
├── models/
│   ├── obj_000001.ply                   # 319,668 面，文本 PLY，29.3 MB
│   ├── obj_000001.png                   # 2048² 纹理图集（已做 padding）
│   └── models_info.json                 # diameter / min_* / max_* / size_*
└── train_pbr/000000/
    ├── rgb/       000000.png ... 000499.png      1024x768
    ├── depth/     000000.png ... 000499.png      uint16，0.1mm/单位
    ├── mask/      {frame:06d}_{inst:06d}.png     全部像素
    ├── mask_visib/{frame:06d}_{inst:06d}.png     仅可见像素
    ├── scene_gt.json        每帧 [{cam_R_m2c, cam_t_m2c, obj_id}]
    ├── scene_camera.json    每帧 {cam_K, cam_R_w2c, cam_t_w2c, depth_scale}
    ├── scene_gt_info.json   每实例 {bbox_obj, bbox_visib, px_count_*, visib_fract}
    └── scene_gt_coco.json
```

### 3.3 规模

| 项 | 数量 |
|---|---|
| 帧 | **500** |
| 实例（object instances） | **5000** |
| `mask` / `mask_visib` | 各 5000 |
| 磁盘 | 373 MB（打包后） |

---

## 4. 验收结论

### ✅ 通过的

| 检查 | 结果 |
|---|---|
| 文件完整性 | `rgb = depth = scene_gt = scene_camera = scene_gt_info = 500`，`mask = mask_visib = instances = 5000`，无缺失 |
| 旋转矩阵 | `det ∈ [0.999999, 1.000001]`，正交性 `max‖RRᵀ − I‖ = 8.5e-7` |
| 平移范围 | 308 ~ 743 mm |
| 位姿/内参/尺寸自洽 | `scripts/verify_bop.py`：把 3D 包围盒按 GT 位姿投影回 RGB，**线框严丝合缝包住物体**；投影框始终略大于 `bbox_visib` 且包含它（正确关系） |
| 图像可读性 | 抽样全部 768×1024 uint8 正常 |

验收图：`data/results/stage2_render/`（`bbox_overlay.png`、`dataset_frames.png`、
`dataset_mask_overlay.png`）。

### ⚠️ 已知缺陷（都记在 TROUBLESHOOTING.md）

| 编号 | 问题 | 影响 | 状态 |
|---|---|---|---|
| `DATA-16` | **遮挡严重不足**：`visib_fract` 中位数 1.000，95% 实例完全可见，`mask` ≈ `mask_visib` | HCCEPose 的卖点就是遮挡鲁棒，训练集里没有遮挡就学不到 | 🟡 已定位：物理模拟把 10 个物体**摊平**了（撒在 ±0.15 m，房间墙在 ±2 m 兜不住）。修法：收紧采样范围 / 加低矮围栏 / 照 BoxDreamer 在线合成遮挡 |
| `DATA-14` | `depth/` 实际不可用：深度被量化成整数米（统计只有 2~3 个不同值） | RGB-D（FoundationPose 精化）路线 | 🟡 暂不修，训练不用深度 |
| `DATA-17` | 裁完**有效分辨率偏低**：`bbox_visib` 只有 95~156 px，网络输入 224 需上采样 1.5~2.4× | 精度上限 | 🟡 对比 BoxDreamer（一图一物、512²）我们是下采样 |

---

## 5. 和 BoxDreamer 训练数据的差异

见对话记录；要点：

- **角点顺序完全一致** —— 我们的 `BB8_BITS` 与 BoxDreamer `bbox_utils.py` 的 `bits` 逐位相同，
  数据可以直接喂进它的 corner loss
- **最本质的差异是"遮挡在哪一步产生"**：BoxDreamer 渲干净单体 + 训练时在线合成（SUN2012 背景 + 贴遮挡物）；
  我们把 clutter 烘进渲染 —— 更真实，但**失去了在线调遮挡强度的灵活性**（`DATA-16` 就是这个代价）
- **3D 框约定不同**：BoxDreamer 对 co3d/moped 用 **PCA 有向框**；BOP 是**轴对齐**框

---

## 6. 存放位置

| 位置 | 内容 |
|---|---|
| **远端** `/root/autodl-tmp/bop/dji_action4_hybrid/` | 原件（`/root/autodl-tmp` 关机保留，**释放即丢**） |
| 远端 `/root/autodl-tmp/dji_action4_hybrid_dataset.tar.gz` | 打包件，373 MB |
| **本地** `data/bop/dji_action4_hybrid_dataset.tar.gz` | 同上的下载副本（373,384,931 字节，11016 个条目） |
| 本地 `data/bop/dji_action4_hybrid_dataset.tar.gz.sha256` | SHA256 |
| 本地 `data/bop/gen_train.log` | 生成过程完整日志（56 KB） |
| 本地 `data/dji_action4_hybrid/models/` | 模型的 BOP 三件套 |
| 本地 `data/mesh/generated_hybrid/textured.glb` | 生成模型的原始 glb（25.5 MB） |
| 本地 `data/results/stage1_generation/` | 阶段① 的渲染/图集对照 |
| 本地 `data/results/stage2_render/` | 阶段② 的渲染与验收图 |

**归档 SHA256**

```
E45D227ABEEDDCDA6C07888A2F784D303B3D8123BF10B63D997AF636DB336F42
data/bop/dji_action4_hybrid_dataset.tar.gz
```

### 解包

```powershell
tar -xzf data/bop/dji_action4_hybrid_dataset.tar.gz -C data/bop/
# 得到 data/bop/dji_action4_hybrid/{camera.json, models/, train_pbr/}
```
