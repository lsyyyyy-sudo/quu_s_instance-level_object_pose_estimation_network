# 换掉 3D 模型：为什么现在的形状不像，以及两条可行路线

> 结论先说：**现在的形状不像，主要不是因为生成模型差，而是因为只喂了 8 个可用视角里的 2 个。**
> 修这个只要 30 秒 GPU。VGGT 那条路我已经把环境搭好，但它解决的是另一个问题，且风险高得多。

---

## 1. 我们到底有多少张目标物体图

之前以为只有 `front` + `back` 两张，实际核对（MD5 + 分辨率）后发现**有 8 张不同的官方图**：

| 文件 | 视角 | 说明 |
|---|---|---|
| `raw/official/bare/front.JPG` | 正面 | 镜头在右、前置屏在左、`ACTION 4` 标 |
| `raw/official/bare/back.JPG` | 背面 | 大屏 |
| `raw/official/closeups/13.JPG` | 右后 ¾ | 可见顶部录制键 + 镜筒侧 |
| `raw/official/closeups/16.JPG` | 左前 ¾ | 可见 `QS` 电源键那一面 |
| `raw/official/closeups/18.JPG` | 顶部 | 录制键 + 麦克风孔 |
| `raw/official/closeups/19.JPG` | 底部 | 磁性卡扣 |
| `raw/official/closeups/14.JPG` | 侧面（**磁吸盖打开**） | ⚠️ 排除，非刚体状态 |
| `raw/official/closeups/17.JPG` | 底部（**接口盖打开**） | ⚠️ 排除 |

> `closeups/_full_11.JPG` 与 `bare/front.JPG` 的 MD5 **完全相同**，
> `_full_12.JPG` 与 `bare/back.JPG` 也相同 —— 这两张是重复的，不是新视角。
> `thumbs/*.jpg` 是 560 px 缩略图。`with_case/*` 是带壳的，视频里不带壳，用不上。

**上次只用了 `front` + `back`。** 生成模型必须凭空编出左、右、上、下四个面，
所以"整体形状不像"是必然的，而不是模型能力问题。

---

## 2. 路线 A（推荐先试）：`Hunyuan3D-2mv` 喂 4 视图

**成本：约 30 秒 GPU**。环境和权重都已经在 `/root/autodl-tmp` 里，脚本现成。

`/root/autodl-tmp/run_generate.py` 本来就有 `--views` 参数（默认 `["front","back"]`），
`Hunyuan3D-2mv` 的 `MVImageProcessorV2.view2idx = {'front':0,'left':1,'back':2,'right':3}`
最多接受 4 个视图。

### 2.1 实测结果（2026-09-14，RTX 4090）

用 `data/raw/official/thumbs/{11,12,13,16,18}_m.jpg`
（11=front、12=back、13=right、16=left、18=top）跑形状，三轴尺寸比（降序）：

| 版本 | 顶点 | 面 | 尺寸比 | 对比官方 `1.000 / 0.627 / 0.465` |
|---|---|---|---|---|
| 原 2 视图 | 461368 | 922696 | 1.000 / 0.676 / **0.592** | 第三轴**超出 27%**（就是"太厚"） |
| **4 视图** | 400856 | 801708 | 1.000 / 0.642 / **0.472** | 第二轴差 2.4%，第三轴差 **1.5%** |

**"整体形状不像"的确就是视图不够造成的**，喂 4 视图后比例基本对上了。

⚠️ **`left`/`right` 判反了也没关系**：`--views front left back right` 和
`--views front right back left` 的结果**逐位相同**（顶点数、面数、包围盒完全一致）。
所以官方图没有拍摄方位信息这件事不影响这条路线，不需要纠结。

```bash
# 有 GPU 时
cd /root/autodl-tmp
export OMP_NUM_THREADS=8
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy          # HuggingFace 必须【关掉】代理

python run_generate.py \
    --input-dir /root/autodl-tmp/input_t --ext jpg \
    --out-dir   /root/autodl-tmp/output_mv \
    --views front left back right
```

实测耗时：模型加载 ~50 s + 去背景 4×11 s + **扩散采样 10 s** + 体素解码 15 s
（形状共 ~28 s）；纹理阶段约 20 分钟。

### 2.2 ⚠️ 但纹理反而变差了 —— 侧视图不是"正交侧视"

4 视图的**形状**修好了，**纹理**却坏了：图集里满是灰色噪点，连 UV 岛内部都碎了，
渲染出来整体发灰（本该是黑色机身）。

原因：`13.JPG` / `16.JPG` 是 **¾ 侧视**，不是正交的左/右侧视。
- **形状编码器**容忍视角近似 —— 它只用轮廓和大致方位做条件；
- **纹理投影**要求视角准确 —— 它把每张图按假设的视角方向投到网格上，
  把 ¾ 图当成正侧视投，纹理就被拉花。

**解法：形状和纹理分开取。** 形状用 4 视图（比例才准），纹理只用
`front` + `back`（这两张是干净的正交面，2 视图那次纹理质量明显更好）：

```bash
python run_texture_mv.py \
    --shape /root/autodl-tmp/output_mv/shape.glb \
    --view-dir /root/autodl-tmp/output_mv \
    --views front back \
    --out /root/autodl-tmp/out_hybrid/textured.glb
```

（`run_texture_mv.py` 是本地的 `data/render_ws/run_texture_mv.py`；
原来那个 `run_texture.py` 把 shape 路径和视图都写死了。）

### 2.3 三种组合的对比（已实测）

| 组合 | 三轴尺寸比 | 纹理 |
|---|---|---|
| 2 视图形状 + 2 视图纹理（旧） | 1.000 / 0.676 / 0.592 | 深色机身、有 `ACTION 4` 字样，**厚度超 27%** |
| 4 视图形状 + 4 视图纹理 | **1.000 / 0.642 / 0.472** | ❌ 图集糊成灰噪声，渲染整体发灰 |
| **4 视图形状 + 2 视图纹理** ✅ | **1.000 / 0.642 / 0.472** | 深色机身、可见屏幕与 `ACTION 4`、红色录制键 |
| *官方参考* | *1.000 / 0.627 / 0.465* | |

**采用第三种。** 形状取 4 视图、纹理取 2 视图 —— 这两件事的最优输入不一样。

产物（本地，`data/` 下，不入库）：
- 模型：`data/mesh/generated_hybrid/textured.glb`
- BOP：`data/dji_action4_hybrid/models/`（319668 面，PLY 29.3 MB）
- 渲染：`data/results/stage1_generation/final_4view_hybrid/`

### 2.4 ⏳ 仍然没解决的：朝向

模型是**躺着**的（镜头朝上而不是朝前），渲染出来像"一个盒子顶着一个镜头鼓包"，
要脑补旋转才认得出是 Action 4。这就是 `TROUBLESHOOTING.md` 里的 `DATA-10`。

**这件事必须做**，不只是观感问题：BOP 的 8 个角点顺序定义在物体坐标系上，
朝向不确定 ⇒ 角点顺序没有物理意义 ⇒ 训练出来的热图学不到一致的对应关系。

做法（任一）：
1. 用 `front.JPG` 的朝向做基准，手工求一个旋转矩阵把镜头转到 +Z；
2. 对网格做 PCA，把最长轴对齐到 X、法向对齐到 Y/Z；
3. 在 BlenderProc 里给 `obj_000001.ply` 直接加一个固定旋转（但这样 `models_info.json` 的
   bbox 也要跟着改，等于改了模型）。

推荐 1 或 2，改在 `scripts/mesh_to_bop.py` 里，这样 `models_info.json` 自动一致。

---

## 3. 路线 B：VGGT 前馈重建（你要求的）

**成本：数小时 + GPU**。环境已在准备（`/root/autodl-tmp/VGGT` + `/root/autodl-tmp/envs/vggt`）。

### 为什么是 VGGT 而不是 3DGS

| | 3DGS | **VGGT** |
|---|---|---|
| 需要多少视图 | 几十~几百张、**有重叠** | **1 张起**，几张也行 |
| 需要位姿 | 要先跑 COLMAP | **自己前馈估计**（位姿+内参+点图+深度） |
| 输出 | 高斯球，**不是网格** | **点图 → 点云 → 网格** |
| 8 张产品图能不能用 | ❌ 特征匹配会失败 | 🟡 能跑，但见下面的风险 |

VCGGT 是 CVPR 2025 最佳论文，前馈网络，**几十秒到几分钟**出结果，不需要 COLMAP。
3DGS 在这个场景基本没戏：物体是**黑色亮面塑料**，几乎没有纹理可供特征匹配，
而且 8 张产品图之间没有重叠视角。

### 但要认真说清楚风险

1. **黑色亮面塑料是最难的输入**。VGGT 的几何来自学到的先验，不是特征匹配，
   所以纹理缺失不会直接崩——但点图在无纹理高光面上噪声会明显偏大。
2. **产品图不是采集序列**。8 张图的内参未知、物体在画面里的位置和尺度各不相同。
   VGGT 假设的是"同一台相机拍的同一场景"。
3. **输出是部分点云**。拍不到的背面/底面没有几何，需要泊松重建去补——补出来的部分是平滑的猜测。
4. **纹理还得另外做**。VGGT 只给几何；纹理要么用点云颜色转顶点色
   （`scripts/mesh_to_bop.py` 已支持无 `TextureFile` 的顶点色路径），要么做多视图投影贴图。

**所以路线 B 的期望值是：得到一个"比现在好、但比路线 A 粗糙"的网格。**
如果目标只是"形状像这个物体"，**先跑路线 A**。

---

## 4. 已经做好的准备（都不需要 GPU）

| 位置 | 内容 |
|---|---|
| `data/mesh/multiview/{front,back,left,right,top,bottom}.JPG` | 本地整理好的 6 视图 |
| `/root/autodl-tmp/input/*.JPG` | 已上传到远端 |
| `/root/autodl-tmp/VGGT/` | VGGT 源码（codeload tarball，64 MB） |
| `/root/autodl-tmp/envs/vggt/` | 复用 base 的 torch 2.5.1+cu124 的 venv |
| `/root/autodl-tmp/hf_cache/hub/models--facebook--VGGT-1B` | VGGT-1B 权重（下载中） |
| `data/render_ws/vggt_reconstruct.py` | 重建脚本（**未在 GPU 上验证过**，也未入仓库） |

---

## 5. 建议的执行顺序

1. **先跑路线 A**（30 秒 × 2 种左右标注），转成 BOP 格式后重渲染，看镜头位置对不对。
2. 如果形状仍然不对 → 跑路线 B（VGGT），用全部 6~8 张图，不需要视图标注。
3. 拿到新 mesh 后走既有流程：
   `scripts/mesh_to_bop.py`（`--max-faces 80000 --ply-precision 5`）
   → `scripts/repair_texture_atlas.py`
   → `gen_pbr_data_demo.py` 渲染。

⚠️ 无论走哪条，**朝向规范化（`DATA-10`）仍然没做**：
生成出来的 mesh 是"躺着"的（镜头朝上）。BOP 的 8 个角点顺序依赖一个确定的物体坐标系，
这一步不做，后面训练出来的角点是没有物理意义的。
