# 云端 3D 生成环境（AutoDL）

> 用 Hunyuan3D-2 从图片生成 DJI Action 4 的带纹理 mesh。
> 本文档是**可复现的完整记录** —— 实例释放后照着重来即可。

---

## 0. 为什么在云端做

本机是 **RTX 4060 Laptop 8GB / Windows**。Hunyuan3D-2 的纹理阶段需要 **16 GB 显存**，
而且 `custom_rasterizer` / `differentiable_renderer` 两个 CUDA 扩展要在 Linux 下用 nvcc 编译。
所以生成放在 AutoDL 的 4090 上做，**本机只做后处理**（mesh → BOP PLY）。

---

## 1. 实例配置

| 项 | 选择 | 理由 |
|---|---|---|
| **GPU** | RTX 4090 / 24 GB | 形状 6 GB + 纹理到 16 GB，24 GB 不用开 low_vram |
| **镜像** | 基础镜像 → PyTorch → **`PyTorch 2.5.1` / `Python 3.12` / `CUDA 12.4`** | 见下 |
| **数据盘** | 所有东西放 `/root/autodl-tmp` | 系统盘只有 30 GB，模型缓存 28 GB |

**为什么必须用 PyTorch 镜像而不是 Miniconda**：那两个 CUDA 扩展要编译，
需要 **nvcc + CUDA 头文件**。AutoDL 的 PyTorch 镜像自带完整工具链
（官方文档：*"平台内置的CUDA均带 .h 头文件，如有二次编译代码的需求更方便"*）。
实测镜像自带 `nvcc 12.4`，和 torch 的 `cu124` 正好对上。

**实测环境**（2026-09，供对照）：

```
GPU        NVIDIA GeForce RTX 4090  24564 MiB  driver 570.124.04
Python     3.12.3
torch      2.5.1+cu124   cuda 12.4   avail True
nvcc       12.4.131
磁盘       / 30G    /root/autodl-tmp 50G
```

---

## 2. ⚠️ 网络：GitHub 和 HuggingFace 的代理要求**相反**

这是本次最大的坑，浪费了最多时间。

| 目标 | 开 `source /etc/network_turbo`？ | 实测速度 |
|---|:---:|---|
| **HuggingFace**（经 hf-mirror） | ❌ **必须关** | 关：**14 MB/s** ／ 开：1.3 MB/s（**慢 11 倍**） |
| **GitHub** | ✅ **必须开** | 开：**127 MB/s** ／ 关：~20 KB/s |

AutoDL 的提示本身就写了：*"开启加速后对访问其他资源如 pip 源等会更慢"*。

**所以：**

```bash
# 下载 HuggingFace 模型（hf-mirror）
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1          # 见坑 3
unset http_proxy https_proxy          # ★ 关键：不要走代理

# 下载 GitHub 的东西
source /etc/network_turbo             # ★ 关键：要走代理
```

---

## 3. 完整步骤

### 3.1 拉代码（用 tarball，不用 git clone）

```bash
source /etc/network_turbo
cd /root/autodl-tmp
curl -fL -o h3d.tar.gz \
  https://codeload.github.com/Tencent-Hunyuan/Hunyuan3D-2/tar.gz/refs/heads/main
tar xzf h3d.tar.gz && mv Hunyuan3D-2-main Hunyuan3D-2
```

> **为什么不用 `git clone`**：实测 `git clone` 反复报
> `RPC failed; curl 18 transfer closed` / `early EOF`。
> tarball + 完整性校验 + 重试稳定得多（76 MB / 51 秒）。

### 3.2 装依赖

```bash
unset http_proxy https_proxy            # pip 走国内源，别走代理
cd /root/autodl-tmp/Hunyuan3D-2
pip install -r requirements.txt
pip install -e .
```

### 3.3 编译两个 CUDA 扩展（**纹理必需**）

```bash
cd hy3dgen/texgen/custom_rasterizer      && python setup.py install && cd ../../..
cd hy3dgen/texgen/differentiable_renderer && python setup.py install && cd ../../..
```

> ⚠️ 装完的模块名是 **`mesh_processor`**（不是 `differentiable_renderer`），别被名字误导。
> ⚠️ `import custom_rasterizer` 会报 `libc10.so: cannot open shared object file`
> —— **必须先 `import torch`**，因为 torch 才会把它的 lib 目录加进 loader 路径。
> 正常代码里 torch 总是先导入，不是问题。

### 3.4 下载模型（27 GB）

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy

hf download tencent/Hunyuan3D-2mv --include "hunyuan3d-dit-v2-mv-turbo/*"
hf download tencent/Hunyuan3D-2  --include "hunyuan3d-paint-v2-0-turbo/*"
hf download tencent/Hunyuan3D-2  --include "hunyuan3d-delight-v2-0/*"
```

下载量：形状 9.2 GB + 纹理 18 GB = **27 GB**。

> ⚠️ `--include` **只接受一个值**。写两个的话第二个会被当成"显式文件名"，
> 导致 `--include` 被忽略并报 404（`Ignoring --include since filenames have been explicitly set`）。
> 要多个模式就分多次命令。

### 3.5 rembg 抠图模型（1 GB，走 GitHub）

```bash
source /etc/network_turbo          # ★ GitHub，要开代理
mkdir -p /root/.rembg/models/bria-rmbg
curl -fL -C - -o /root/.rembg/models/bria-rmbg/bria-rmbg.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg-2.0.onnx
```

> 不开代理的话这一步会以 ~20 KB/s 爬行，预计 **14 小时**。开了代理 7 秒。

### 3.6 两个代码兼容性补丁

**(a) `trust_remote_code`**：新版 diffusers 拒绝执行本地 `custom_pipeline` 目录里的
`pipeline.py`，除非显式声明信任。

```python
# hy3dgen/texgen/utils/multiview_utils.py 第 34 行
pipeline = DiffusionPipeline.from_pretrained(
    multiview_ckpt_path,
    custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16,
    trust_remote_code=True)          # ← 加这个
```

**(b) `.bin` → `.safetensors`**：`transformers 5.x` 因为 CVE-2025-32434，
要求 `torch>=2.6` 才能 `torch.load` `.bin` 权重（用 safetensors 不受限）。
而 `hunyuan3d-paint-v2-0-turbo` 的 `text_encoder` 和 `vae` **只有 `.bin`**。

不去升级 torch（可能让已编译的 CUDA 扩展 ABI 不匹配），也不降级 transformers
（会牵动 huggingface_hub / tokenizers 一串依赖），**直接转格式**：

```python
import torch
from safetensors.torch import save_file
# text_encoder/pytorch_model.bin -> text_encoder/model.safetensors
# vae/diffusion_pytorch_model.bin -> vae/diffusion_pytorch_model.safetensors
sd = torch.load(bin_path, map_location="cpu", weights_only=True)   # torch 自己 load 不受限
save_file({k: v.contiguous().clone() for k, v in sd.items() if isinstance(v, torch.Tensor)},
          safe_path, metadata={"format": "pt"})
```

> 命名要照 transformers/diffusers 的约定：CLIP 文本编码器用 `model.safetensors`，
> 扩散模型组件用 `diffusion_pytorch_model.safetensors`。
> 可对照 `hunyuan3d-delight-v2-0`（它本身两种格式都有）。

### 3.7 跑生成

```bash
cd /root/autodl-tmp
unset http_proxy https_proxy
export HF_HOME=/root/autodl-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
python finish_all.sh
```

关键参数（对齐官方 `examples/shape_gen_multiview.py` 和 `gradio_app.py`）：

| 参数 | 值 |
|---|---|
| 形状模型 | `tencent/Hunyuan3D-2mv` → `hunyuan3d-dit-v2-mv-turbo`（1.1B） |
| 纹理模型 | `tencent/Hunyuan3D-2` → `hunyuan3d-paint-v2-0-turbo`（1.3B） |
| `num_inference_steps` | 50 |
| `octree_resolution` | 380 |
| `num_chunks` | 20000 |
| `seed` | 12345 |

**多视角输入格式**（`MVImageProcessorV2`）：

```python
images = {"front": img_front, "left": img_left, "back": img_back, "right": img_right}
```

`view2idx = {'front':0, 'left':1, 'back':2, 'right':3}`，代码按索引排序后拼接 ——
**接受任意子集**，所以只有 `front` + `back` 两张完全没问题。

---

## 4. 渲染脚本要的一个坑：无卡模式只有 1 个 CPU 核

| 模式 | GPU | CPU 核 | 能干什么 |
|---|:---:|:---:|---|
| **GPU 模式** | ✅ | 多核 | 全部 |
| **无卡模式** | ❌ | **1 核** | 只能装环境 / 下文件；**跑不了模型加载** |

无卡模式省钱，适合装依赖和下载；但**连 CPU 上加载模型都很慢**（实测 1 核跑
纹理 pipeline 加载会被掐断）。所以：**无卡模式配环境，切回 GPU 模式再跑推理。**

---

## 5. 实测结果

| 阶段 | 耗时 | 产出 |
|---|---|---|
| 抠图（rembg，2 张） | ~20 s | `front_cutout.png` / `back_cutout.png` |
| **形状生成** | **23.3 s** | `shape.glb`，44.6 万顶点 / 89.1 万面 |
| 纹理生成 | 待测 | `textured.glb` |

**形状的包围盒比例检查**（重要）：

| 轴 | 生成 | 官方 70.5 : 44.2 : 32.8（归一化） | 偏差 |
|---|---:|---:|---:|
| 长 | 1.996 | 1.000 | — |
| 高 | 1.349 | 0.627 | +8% |
| **厚** | **1.137** | **0.465** | **+23%** ⚠️ |

前两轴准，**厚度偏大 23%** —— 只有正面+背面两张图时，模型对"侧面有多厚"只能猜。
后面做 `mesh_to_bop.py` 缩放到官方尺寸时会**把厚度也一起压回去**，
但如果要多视角更准，应该补一张侧面图。

---

## 6. 一次性的本地侧工作

生成的 `textured.glb` 下载回来后，要转成 BlenderProc 能吃的 BOP 格式 PLY：

```powershell
.venv\Scripts\python.exe scripts\mesh_to_bop.py `
    --input data\mesh\generated\textured.glb `
    --out-dir data\dji_action4\models `
    --obj-id 1
```

脚本会：居中 → **等比例缩放到官方对角线** → 算法线 → 导出**文本 PLY** →
纹理放到同目录并补 `comment TextureFile` 头 → 打印各轴对比。
要求细节见 [`DATA.md`](DATA.md) §5.1（逐条从 BlenderProc 源码核实）。

---

## 7. 省时间的两条

1. **配好后在 AutoDL 控制台「保存镜像」** —— 下次直接开，不用重来这 30+ 分钟
2. 上传输入图用 JupyterLab 拖拽即可（两张 JPG 各 ~50 KB）

---

## 8. 下一步：阶段② 渲染环境

用 HCCEPose 改造过的 BlenderProc 把 `data/dji_action4/` 渲成 BOP PBR 训练集。
⚠️ **HCCEPose README 里的 `pip install bpy==3.6.0` 在 AutoDL 上走不通**（`bpy` 没有 cp312
wheel + `download.blender.org` 被 Cloudflare 挡），改用 **Blender 3.6.0 官方发行包**。

完整步骤、已经就位的东西、以及**当前接续点**见 → [`RENDER_SETUP.md`](RENDER_SETUP.md)。
踩坑记录见 `TROUBLESHOOTING.md` 的 `ENV-09`～`ENV-11`、`DATA-07`。

