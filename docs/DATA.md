# 数据说明 / 收集清单

> 本项目**所有原始数据、模型、第三方素材都不入 git**
> （`data/` 在 `.gitignore` 里），仓库是 public。
> 本文档记录**放哪、放什么、从哪来、许可如何**。

---

## 1. 目录总览

```
data/                                    ← 整个目录已被 .gitignore 忽略
├── raw/                                 # 原始素材（只读，不要在这里改）
│   ├── target_video/                    # 测试数据：目标视频
│   ├── video_frames/                    # 从目标视频裁出的目标物体帧  ← 阶段① 主输入
│   ├── web_views/                       # 网络收集的多视角实拍图        ← 阶段① 补充输入
│   ├── official/                        # 官方 / 实拍图
│   │   ├── bare/                        #   ★ 不带壳 —— 用来建模
│   │   └── with_case/                   #   带壳 —— 不参与建模
│   └── calibration/                      # 相机标定：棋盘格图 + 内参结果
├── mesh/                                # 3D 模型相关
│   ├── generated/                        # 3D 生成模型的输出（原始）
│   └── reference/                        # 网上找到的现成模型（仅作几何对照）
├── cc0textures-512/                     # PBR 材质库（渲染用，约 600 MB）
└── dji_action4/                         # ★ BOP 数据集根 —— 渲染脚本的工作目录
    ├── camera.json                       # 【手写】相机内参，必须自己提供
    ├── models/
    │   ├── obj_000001.ply                # 目标物体 mesh（居中、mm、非二进制）
    │   └── models_info.json              # 由 s1_p3_obj_infos.py 生成
    └── train_pbr/                        # 【渲染产出】脚本自动创建
        ├── 000000/{rgb/,depth/,mask_visib/,scene_gt.json,scene_camera.json,scene_gt_info.json}
        └── 000001/ ...
```

`data/dji_action4/` 这个路径**必须**和 `configs/datamodule/bop.yaml` 里的
`dataset_root: ${hydra:runtime.cwd}/data/dji_action4` 一致；要换位置就改配置或命令行覆盖。

---

## 2. 每个目录放什么

| 目录 | 放什么 | 格式 / 命名 | 谁生成 |
|---|---|---|---|
| `raw/target_video/` | `head_left_rgb_raw.mp4` | 原文件即可 | 项目方 |
| `raw/video_frames/` | 目标物体在视频里清晰可见的帧 | `frame_000123.jpg`（只裁物体或整帧都行） | 从视频抽取 |
| `raw/web_views/` | 网络收集的多视角实拍图 | 建议按角度命名 `front.jpg` / `back.jpg` / `left.jpg` / `right.jpg` / `top.jpg` / `bottom.jpg` | 手工下载 |
| `raw/official/` | 官方 / 实拍图，**按 `bare/` 和 `with_case/` 分开** | 见 §4.3 | 手工收集 |
| `raw/calibration/` | 棋盘格照片 + 标定输出 | `calib_*.jpg` + `intrinsics.json` | 自拍 / 问 psd 要 |
| `mesh/generated/` | 3D 生成模型原始输出 | 各家格式（`.obj`/`.glb`/`.ply`） | TRELLIS 等 |
| `mesh/reference/` | 网上下载的现成模型 | 原样保存，另存来源链接 | 手工下载 |
| `cc0textures-512/` | PBR 材质库 | 解压后的目录 | 下载 |
| `dji_action4/models/` | **最终**的目标 mesh | `obj_000001.ply` + `models_info.json` | `s1_p1` / `s1_p3` |
| `dji_action4/` | `camera.json` | 见 §5.2 | **手写** |
| `dji_action4/train_pbr/` | 训练数据 | BOP 格式 | 渲染脚本 |

> **不要在 `raw/` 里改文件。** 需要处理就输出到 `mesh/` 或 `dji_action4/`，
> 这样原始素材永远可以重新走一遍流程。

---

## 3. 数据源清单

### 3.1 目标视频（已有）

| 项 | 值 |
|---|---|
| 文件 | `head_left_rgb_raw.mp4` |
| 位置 | 仓库根目录 `head_left_rgb_raw.mp4(1)/`（已 gitignore），建议挪到 `data/raw/target_video/` |
| 规格 | 3248 × 2464，29.97 fps，2850 帧（95 秒），H.264 |
| 内容 | 第一视角整理牛仔外套；目标物体（DJI Action 4）腕戴在双手手腕上 |
| 内参 | ⛔ **未知**（H.264 重编码丢了元数据） |

### 3.2 多视角实拍图

| 来源 | 链接 | 角度覆盖 | 许可 |
|---|---|---|---|
| **GIGAZINE 开箱实拍** ⭐ | [gigazine.net/.../osmo-action-4-unboxing](https://gigazine.net/gsc_news/en/20230802-osmo-action-4-unboxing/) | **六面全覆盖**（正/背/左/右/顶/底），作者逐张标注了角度 | 版权归 GIGAZINE，仅研究用 |
| unwire.hk 開箱評測 | [unwire.hk/2023/08/02/djiaction4](https://unwire.hk/2023/08/02/djiaction4/review-2/) | 多角度 | 版权归 unwire.hk |
| 日文开箱博客 | [kotora-photo.hatenablog.com](https://kotora-photo.hatenablog.com/entry/2024/09/15/200508) | 多角度 | 版权归作者 |

GIGAZINE 那组的具体文件名（完整图去掉 `_m` 后缀）：

| 文件 | 角度 |
|---|---|
| `11.JPG` | 正面（前屏 + 镜头） |
| `12.JPG` | 背面（后屏） |
| `13.JPG` | 左侧（电池仓盖） |
| `16.JPG` / `17.JPG` | 右侧（电源键 / USB-C） |
| `18.JPG` | 顶部（录制键） |
| `19.JPG` | 底部（磁吸底座凹槽） |

URL 形如：`https://i.gzn.jp/img/2023/08/02/osmo-action-4-unboxing/11.JPG`

### 3.3 官方产品图

| 来源 | 链接 | 用途 |
|---|---|---|
| DJI 官方产品页 | [dji.com/cn/osmo-action-4](https://www.dji.com/cn/osmo-action-4) | 几何参照（**渲染图非实拍**，光照理想化） |
| DJI 官方技术参数 | [dji.com/cn/osmo-action-4/specs](https://www.dji.com/cn/osmo-action-4/specs) | **尺寸/重量**（见 §4.1） |

### 3.4 现成 3D 模型（仅作几何对照）

| 来源 | 链接 | 备注 |
|---|---|---|
| MakerWorld | [dji-osmo-action4](https://makerworld.com.cn/zh/models/580600-dji-osmo-action4) | 免费，有多个版本 |
| MakerWorld | [二次创作版](https://makerworld.com.cn/zh/models/662195-da-jiang-osmo-action4-mo-xing) | 免费 |
| MakerWorld | [无滤镜款](https://makerworld.com.cn/zh/models/1235490-da-jiang-action4yun-dong-xiang-ji-wu-lu-jing-kuan) | 免费 |
| Cults3D | [搜索页](https://cults3d.com/zh/bi%C4%81oqi%C4%81n/dji+osmo+action+4) | 2.2k 结果，⚠️ **绝大多数是保护壳/支架，不是相机本体** |
| CGTrader | [Osmo Action 5 Pro](https://www.cgtrader.com/3d-models/electronics/video/dji-osmo-action-5-pro) | **付费**；Action 5 Pro 外形与 4 几乎相同 |
| free3d | [dji-osmo-action-4k-camera](https://free3d.com/3d-model/dji-osmo-action-4k-camera-8042.html) | 需要再试（抓取时连接失败） |

> ⚠️ 在 MakerWorld / Cults3D / Printables 搜 "DJI Action 4"，
> **绝大多数结果是打印配件**（保护壳、镜头盖、Gridfinity 收纳盒、水下外壳），
> 要筛选出**相机本体**的模型。

### 3.5 材质库

| 来源 | 大小 | 获取 |
|---|---|---|
| `cc0textures-512`（轻量版，**用这个**） | ~600 MB | [HuggingFace](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/main/cc0textures-512.zip) 或 `s2_p0_download_cc0textures.py` |
| `cc0textures`（全量版） | ~44 GB | 不需要 |

---

## 4. 阶段①：给 3D 生成模型的图片

### 4.1 官方尺寸（尺度的唯一来源，必须钉死）

来自 [DJI 官方技术参数](https://www.dji.com/cn/osmo-action-4/specs)：

| 项 | 值 |
|---|---|
| **长 × 宽 × 高** | **70.5 × 44.2 × 32.8 mm** |
| 重量 | 145 g |
| 前屏 | 1.4″ 320×320 |
| 后屏 | 2.25″ 360×640 |
| 镜头 FOV | 155°（f/2.8） |

生成的 mesh **必须缩放到这个尺寸**，否则渲染出的包围盒和真实对不上，
PnP 解出的平移会整体差一个比例因子。

### 4.2 需要几张图？（**不是越多越好**）

先把两种路线分清楚，别把需求搞混：

| 路线 | 需要几张 | 代表 |
|---|---|---|
| **3D 生成模型**（考核要求的这条） | **1 张就够**，2~4 张更好 | TRELLIS、Hunyuan3D-2、InstantMesh、One-2-3-45 |
| 摄影测量 / NeRF / 3DGS | **30~100+ 张** | COLMAP + NeRF |

3D 生成模型的原理是**从单图"想象"出看不见的部分**——靠大规模 3D 数据训出来的先验补全。
所以**一张正面图就能出一个完整 mesh**。

**推荐 2~4 张而不是 1 张**的理由：

- 单图模型对**看不见的那一面是"编"的**。Action 4 前屏和后屏完全不同，只给正面图背面会被瞎编
- 但我们真正要的是**包围盒的 8 个角点**，角点在盒子的极值处 ——
  只要整体轮廓对，细节编错影响不大
- 所以：**正面 + 背面两张**足够；想更稳就再加左右侧面

### 4.3 ⚠️ 必须区分「带壳」和「不带壳」

**这是比图片数量重要得多的问题。** 把带壳和不带壳的照片混在一起喂给 3D 生成模型会出问题：

- 带壳的照片里物体轮廓更大（多了外壳厚度）
- 模型会在两种轮廓之间折中，或者跟着多数走
- 出来的 mesh 尺寸就**不是 70.5 × 44.2 × 32.8 mm** 了
  → 包围盒错 → **PnP 解出的平移整体差一个比例因子**

**我们的目标是哪个很明确**：

| | 用不用 | 理由 |
|---|---|---|
| **不带壳的 Action 4 本体** | ✅ **用这个** | 官方尺寸 70.5×44.2×32.8 mm 就是**本体**尺寸；视频里腕戴的也是本体 + 磁吸底座 |
| 带壳 / 带保护框的 | ❌ **单独放，别混进去** | 轮廓尺寸不对 |

目录上就这么分：

```
data/raw/official/
├── bare/          # 不带壳（用来建模）★
└── with_case/     # 带壳（先搁着，不参与建模）
```

> 如果后面发现视频里那台其实装了某个薄壳，再改用 `with_case/`——
> 先核对视频帧，别猜。

### 4.4 挑图标准

不是"多收几张"，而是"挑对几张"。按这个优先级：

| 优先级 | 标准 | 说明 |
|:---:|---|---|
| 1 | **干净背景** | 白底/纯色最好，避免杂乱桌面 |
| 2 | **物体占满画面** | 3D 生成模型对抠图质量敏感 |
| 3 | **清晰无遮挡** | 不要有手、支架、腕带挡在主体上 |
| 4 | **角度互补** | 正面 + 背面；正面 + 侧面也行 |
| 5 | **光照均匀** | 避免强反光把屏幕/镜面打爆 |

**不满足就换一张**——4 张好图胜过 20 张烂图。

### 4.5 来源（按需取用，不是全都要）

| 优先级 | 来源 | 理由 |
|:---:|---|---|
| **1** | 从目标视频裁帧 | **同一个物理实例**：光照、白平衡、镜头特性、磨损一致 |
| **2** | 官方/实拍多角度图 | 无遮挡、背景干净 |
| **3** | 官方产品图 | 几何标准，但**是渲染图**，光照过于理想 |
| 4 | 现成 3D 模型 | 只作几何对照，**不用作输入**（纹理缺失，且绕过考核第①步） |

**为什么优先实拍**：3D 生成模型会从输入图里"学"纹理。用渲染图当输入，
生成的纹理自带理想化光照，和真实视频的域差异反而更大。

### 4.3 一个需要先定的问题

视频里的 DJI 是 **磁吸底座 + 腕带**装着的。要决定建模范围：

| 方案 | 说明 | 建议 |
|---|---|---|
| **A. 只建模相机本体** | 包围盒严格等于 70.5×44.2×32.8 mm；腕带/底座当作遮挡物 | ✅ **推荐** |
| B. 相机 + 底座 | 更接近视频里看到的整体，但"物体"边界和尺度都变模糊 | ❌ |

选 A 的理由：包围盒定义干净，符合 BOP / BoxDreamer 的约定；
而且腕带和手自然成为遮挡，正好对应考核说的"严重遮挡"场景。

---

## 5. 阶段②：BOP 数据集结构（渲染脚本的要求）

HCCEPose 的 `s2_p1_gen_pbr_data.py` **从 cwd 反推路径**，所以目录结构必须严格：

```python
current_dir    = os.path.abspath(os.getcwd())        # 必须 cd 到 data/dji_action4
bop_parent_path = os.path.dirname(current_dir)       # -> data/
bop_dataset_path = os.path.join(bop_parent_path, os.path.basename(current_dir))
```

### 5.1 mesh 要求（逐条从 BlenderProc 源码核实）

以下结论来自 `blenderproc.zip` 里的实际代码，不是猜的：

| 项 | 要求 | 源码依据 |
|---|---|---|
| **文件名** | `models/obj_000001.ply`（6 位补零） | `bop_toolkit_lib/dataset_params.py` L152：`'model_tpath': join(models_path, 'obj_{obj_id:06d}.ply')` —— **扩展名和命名都写死** |
| **目录** | `<dataset_root>/models/` | 同上 L142：`models_path = join(datasets_path, dataset_name, 'models')` |
| **必须是文本 PLY** | ⚠️ 不能是二进制 PLY | `loader/ObjectLoader.py` L56 用 `open(filepath, "r", encoding="latin-1")` **当文本读**，还要做字符串替换（L72-73：`property float texture_u` → `property float s`）。二进制 PLY 会直接崩 |
| **顶点法线 + 顶点坐标** | 要有 | `s1_p1_obj_rename_center.py` 保存时用的是 `save_vertex_normal=True, save_vertex_coord=True, binary=False` |
| **纹理（方式一）** | PLY 头部写 `comment TextureFile <文件名>`，纹理图与 PLY **同目录** | `ObjectLoader.py` L52/L60-68 |
| **纹理（方式二）** | 不写 TextureFile → 自动用**顶点色**（`map_vertex_color()`） | `ObjectLoader.py` L83-88 |
| **配套文件** | `models/models_info.json` 必须存在 —— **物体 id 列表就是从它的 key 来的** | `dataset_params.py` L155 + `BopLoader.py` L53 |
| **单位** | **mm**（脚本用 `mm2m=True` 转成米） | `s2_p1_gen_pbr_data.py` L185 |
| **位置** | 居中到原点 | `mesh.py` 无强制要求，但 8 角点/包围盒的对称性依赖它；用 `s1_p1` 处理 |

> 另外 `ObjectLoader.load_obj` 其实也支持 `.obj` / `.fbx` / `.glb` / `.gltf` / `.dae` / `.stl`，
> 但 **BOP 的模型路径模板把扩展名写死成 `.ply`**，
> 所以走 `load_bop_objs()` 这条路就必须是 `.ply`。
> （想用别的格式只能改 BlenderProc 源码或绕过 BOP loader，不建议。）

> `models_info.json` 里的 `min_x/y/z`、`max_x/y/z` **就是我们那 8 个 3D 角点的来源**。
>
> 对称性文件 `obj_000001_sym_type.json` 是**可选**的，DJI Action 4 不对称，可以不做。

### 5.2 `camera.json`（⚠️ 必须自己写）

脚本第 85–96 行：**如果数据集里没有 `camera.json`，它会自动写一个 LINEMOD 的内参**
（640×480，fx≈572，HFOV 58°）。不管它的话：

```
物体投影宽度 = 0.0705 m × 572 / 0.75 m ≈ 54 px    ← 太小，裁剪+上采样后很糊
真实视频里物体约 250 px
```

所以要在 `data/dji_action4/camera.json` 手写一份。格式（来自脚本里的默认值结构）：

```json
{
  "cx": 512.0,
  "cy": 384.0,
  "fx": 512.0,
  "fy": 512.0,
  "depth_scale": 0.1,
  "width": 1024,
  "height": 768
}
```

**取值原则**（不是"必须等于真实内参"，见下）：

| 参数 | 建议 | 理由 |
|---|---|---|
| `width` / `height` | **≥ 1024×768** | 保证裁剪前物体有足够像素（目标 150~250 px 宽） |
| `fx` / `fy` | 按 HFOV 90° 配：`f = width / 2` | **f 本身不影响训练效果**（裁剪+resize 会把它归一化掉） |
| `cx` / `cy` | 图像中心 | |

> **为什么 f 不重要**：在针孔模型下改变焦距等于对整图做相似变换，
> 而我们最后要裁剪到物体再 resize 到 256×256 —— 正好把这个缩放抵消掉。
> 裁剪归一化后，角点的相对布局只由 **物体尺寸 / 相机距离 (s/d)** 决定。
>
> **真正重要的是 s/d**：要把渲染脚本里的相机采样半径调成贴近真实
> （见 §5.3）。真实场景腕戴→头戴相机约 0.4~0.5 m，物体 70.5 mm，s/d ≈ 0.15。

### 5.3 渲染脚本里要改的硬编码值

| 位置 | 现值 | 改成 | 原因 |
|---|---|---|---|
| L119 | `num_scenes = (50 * 1)` | 先改成 `2` 试通 | 50 scene × 20 帧 = **1000 帧/次调用**，`.sh` 跑 42 次 = 42000 帧 |
| L227-229 | `radius_min=0.3, radius_max=1.2` | **`0.35 ~ 0.6`** | s/d 要贴近真实（0.15）；现值均值 0.75 m → s/d≈0.094，偏远 |
| L245 | `color_file_format = "JPEG"` | 保持即可 | ⚠️ 产物是 **`.jpg`**，数据集加载器已兼容（见 §6） |
| L226 | `while cam_poses < 20` | 第一次可保持 20 | 每 scene 帧数 |
| L206-208 | `radius_min=1, radius_max=1.5` | **不用改** | 这是**光源**位置，不是相机 |

> ⚠️ 注意区分：**L206 是光源采样，L227 才是相机采样。** 别改错。

### 5.4 渲染产出

```
data/dji_action4/train_pbr/000000/
├── rgb/000000.jpg            # 训练图像（注意是 .jpg）
├── depth/000000.png
├── mask_visib/000000_000001.png
├── scene_gt.json             # ★ GT 位姿 cam_R_m2c / cam_t_m2c
├── scene_camera.json         # ★ 相机内参 cam_K
└── scene_gt_info.json        # ★ 物体 2D 框 bbox_visib（用于裁剪）
```

标 ★ 的三个文件就是 `src/datasets/bop_pbr.py` 要读的全部内容。
**渲染一跑通，数据链路直接接上。**

---

## 6. 命名与格式约定

| 项 | 约定 |
|---|---|
| 物体 id | `obj_000001.ply`（6 位补零） |
| 帧号 | `000000.jpg`（6 位补零） |
| mask | `000000_000001.png`（帧号_物体号） |
| RGB 扩展名 | **`.png` 和 `.jpg` 都支持**（渲染脚本输出 `.jpg`） |
| 单位 | mesh 和 `cam_t_m2c` 都是 **mm** |
| 位姿方向 | BOP 约定：`cam_R_m2c` / `cam_t_m2c` = **m**odel **to** **c**amera |

---

## 7. ⚠️ 许可证注意事项

**仓库是 public 的**，所以：

| 来源 | 版权 | 能不能入库 |
|---|---|---|
| DJI 官方产品图 | 归 **DJI** | ❌ 不放 |
| GIGAZINE / unwire 照片 | 归各自媒体 | ❌ 不放 |
| MakerWorld / Cults3D 模型 | 各自 CC 许可（有的禁商用） | ❌ 不放 |
| CGTrader / Sketchfab | 付费或明确许可 | ❌ 不放 |
| 目标视频 | 项目方提供 | ❌ 不放（277 MB 也超限） |
| **我们自己渲染的 `train_pbr/`** | 我们生成 | ⚠️ 体积太大，也不放 |

**规则**：
1. 所有素材放 `data/`（已 gitignore）
2. **只把来源链接和获取方式写进文档**（就是本文档）
3. 需要分享数据集时，另找地方（网盘 / HuggingFace），不要塞进 git

---

## 8. 收集进度清单

| # | 项 | 目标位置 | 状态 |
|:---:|---|---|:---:|
| 1 | 目标视频 | `data/raw/target_video/` | ⬜ 挪过来（现在在仓库根） |
| 2 | 目标物体的干净实拍图（**2~4 张**，不带壳） | `data/raw/official/bare/` | 🟡 已有部分，需挑图 |
| 3 | 带壳的图 | `data/raw/official/with_case/` | 🟡 已有，**不参与建模** |
| 4 | 视频里目标物体的清晰帧 | `data/raw/video_frames/` | ⬜ |
| 5 | 相机标定（棋盘格 + 内参） | `data/raw/calibration/` | ⛔ **需向 psd 索要** |
| 6 | 3D 生成模型输出 | `data/mesh/generated/` | ⬜ 依赖 2 (4) |
| 7 | 现成模型（几何对照） | `data/mesh/reference/` | ⬜ 可选 |
| 8 | cc0textures-512 | `data/cc0textures-512/` | ⬜ 600 MB 下载 |
| 9 | `obj_000001.ply`（居中、mm） | `data/dji_action4/models/` | ⬜ 依赖 6 |
| 10 | `models_info.json` | `data/dji_action4/models/` | ⬜ `s1_p3` 生成 |
| 11 | `camera.json` | `data/dji_action4/` | ⬜ **手写** |
| 12 | `train_pbr/` 训练数据 | `data/dji_action4/` | ⬜ 依赖 9/10/11 + Linux 环境 |

> **收集重点不是「多」，是「挑对」。** 2~4 张干净、无遮挡、角度互补的**不带壳**图，
> 比 20 张混着带壳/杂乱背景的图有用得多（理由见 §4.2~§4.4）。
