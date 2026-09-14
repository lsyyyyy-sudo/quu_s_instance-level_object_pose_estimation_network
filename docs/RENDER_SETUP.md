# 阶段② 合成数据渲染环境（HCCEPose BlenderProc / AutoDL）

> **目标**：用 HCCEPose 改造过的 BlenderProc，把 `data/dji_action4/`（DJI Osmo Action 4 的
> 生成式 3D 模型）渲成 BOP PBR 训练集。
>
> **状态**：✅ **已跑通**（2026-09-13）。而且是在**无卡模式**下跑通的——无 GPU、**1 个 CPU 核**、
> **cgroup 内存上限只有 2 GB**。环境装好后还踩了三个坑：pip 依赖冲突（§5）、
> **110 MB 的 PLY 把 Blender 打挂**（§8）、生成模型纹理图集的空白噪声（§9）。
> 实测结果见 §10。

---

## 1. 为什么不能照抄 HCCEPose 的 README

HCCEPose README 的写法是：

```bash
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
```

这条在 AutoDL 上走不通，有**两个互相独立**的原因：

1. **`bpy` 没有 cp312 wheel。**
   AutoDL 这个镜像的 Python 是 3.12（conda base，只有 base 环境）。
   Blender 官方 PyPI 源上 `bpy` 只发到 **`cp310`**（3.4.0 / 3.5.0 / 3.6.0）和 **`cp311`**（4.1.0 起）；
   PyPI 上更少，只有 cp311 / cp313。
   → 在 3.12 上 `pip install bpy` 必然 `No matching distribution found`。

2. **`download.blender.org` 对 AutoDL 的出口 IP 弹 Cloudflare 人机验证。**
   就算按 README 建个 py3.10 环境，索引页返回的也是 `403` + `cf-mitigated: challenge`，
   pip 只会报 `from versions: none`（看起来像"包不存在"，其实是网络被挡）。
   `source /etc/network_turbo` **绕不过**（实测同样 403）。
   清华 TUNA、南大镜像都**只镜像 `release/ source/ demo/`，没有 `pypi/`**。
   → 所以 `bpy` pip wheel 这条路在 AutoDL 上是死的。

## 2. 采用的方案：直接用 Blender 官方发行包

Blender 发行包里**自带 Python 3.10，且 `bpy` 在 Blender 进程内天然可导入**——这正是
BlenderProc "bpy as a pip module" 想达到的效果，换成官方发行包更省事。

关键前提：**HCCEPose 把 `blenderproc/__init__.py` 里"只能在 blenderproc CLI 下运行"的守卫
改成了 `if True:`**（原版检查 `USING_BPY_PYTHON_MODULE` 环境变量）。所以 `import blenderproc`
可以在普通 Python 里直接跑——他们自己的 `s2_p1_gen_pbr_data.sh` 也是直接 `python xxx.py`。

于是整条链路变成：

```
清华 TUNA 下 blender-3.6.0-linux-x64.tar.xz（256 MB，17.5 MB/s）
        ↓
把 blenderproc + 依赖装进 <blender>/3.6/python/
        ↓
blender --background --python gen_pbr_data_demo.py     # cwd = 数据集目录
```

`blenderproc` 的 CLI / `SetupUtility.setup_pip` 都不需要，因为我们不触发它们
（只有 `command_line.py` / `run.py` / `Pipeline.py` / `AMASSLoader` / `URDFLoader` 会调）。

## 3. 远端已就位的东西（`/root/autodl-tmp/`）

| 路径 | 内容 | 状态 |
|---|---|---|
| `blender-3.6.0-linux-x64/` | Blender 3.6.0（内置 Python 3.10.12） | ✅ 已解压，md5 `1887f4123dd08e02e66c6c7e04a45922` 与官方一致 |
| `blender-3.6.0-linux-x64.tar.xz` | 发行包本体 | ✅ 269,076,760 B |
| `blenderproc.zip` | HCCEPose 改造过的 BlenderProc **2.5.0** 源码包（0.6 MB） | ✅ 已上传，**尚未解压部署** |
| `bp_ws/gen_pbr_data_demo.py` | 我们的适配版渲染脚本（见 §6） | ✅ 已上传 |
| `bp_ws/make_cc0textures.py` | 合成 cc0textures-512 材质库的脚本 | ✅ 已上传，**尚未运行** |
| `bop/dji_action4/camera.json` | 手写内参 1024×768 / fx=fy=800 / depth_scale 0.1 | ✅ 已上传 |
| `bop/dji_action4/models/obj_000001.ply` | BOP 网格（ASCII，110,595,493 B） | ✅ 已上传（128 s） |
| `bop/dji_action4/models/obj_000001.png` | 2048² 纹理图集（4.39 MB） | ✅ 已上传 |
| `bop/dji_action4/models/models_info.json` | BOP 元信息（diameter 80.88 mm） | ✅ 已上传 |
| `envs/bp310/` | conda Python 3.10.21 | ⚠️ 遗留物，方案改后**用不到了**，可删 |
| `Hunyuan3D-2/` `hf_cache/` `output/` | 阶段① 的 3D 生成环境 | ✅ 保留 |

系统侧 apt 已补（否则 EGL 起不来）：`libegl1 libgles2 libgl1 libglvnd0 libsm6 libxrender1
libxext6 libxi6 libxkbcommon0`。
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 存在 → EGL 会走 NVIDIA 而不是 Mesa 软渲染。

## 4. ⏭️ 剩余步骤（从这里接续）

> ### ⚠️ 2026-09-15 更新：**本机 Windows 也能跑，而且搭起来比 Linux 顺**
>
> 如果只是想要数据，**不必用 AutoDL 的 GPU 实例**——本地 Windows 机器更快也更便宜。
> 实测环境（RTX 4060 Laptop 8 GB / 24 物理核 / 31.7 GB RAM）：
>
> ```powershell
> # 1) 下 Blender 3.6.0 Windows（365 MB，blender.org 可达，无需镜像）
> #    https://download.blender.org/release/Blender3.6/blender-3.6.0-windows-x64.zip
> #    MD5 044DDB6ABDF5F7D247DBCC06495137DB
> # 2) 依赖装进 Blender 自带的 python（3.10.12，和 Linux 版同版本）
> $PY = "<blender>\3.6\python\bin\python.exe"
> & $PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
>     "numpy==1.26.4" "opencv-python==4.9.0.80" "Pillow==10.4.0" "imageio==2.37.2" `
>     "scipy==1.15.3" "scikit-image==0.24.0" "scikit-learn==1.5.2" "trimesh==4.2.2" `
>     "matplotlib==3.9.2" "h5py==3.11.0" "rich==13.9.4" "PyYAML==6.0.2" `
>     "progressbar2==4.5.0" "tqdm==4.67.1" "requests==2.32.3" "pytz==2024.2" `
>     "pypng==0.20220715.0" "plyfile==1.1"
> & $PY -m pip install -i <同上> "PyOpenGL==3.1.7" "freetype-py==2.5.1" "pyglet==2.1.16"
> & $PY -m pip install -i <同上> --no-deps "pyrender==0.1.45"
> # 3) 把 HCCEPose 的 blenderproc 2.5.0 拷进 site-packages
> # 4) 跑：<blender>\blender.exe --background --python gen_pbr_data_demo.py
> ```
>
> **为什么 Windows 反而少踩坑**：
> - `download.blender.org` 直接可下（Linux 那次返回 403 + Cloudflare challenge，只能走 TUNA）
> - **不需要 EGL** —— `pyrender` 在 Windows 用原生 WGL（走 pyglet），不用装 `libegl`
> - 31.7 GB 内存，不会像无卡模式那样被 2 GB cgroup 打成 OOM
> - 那两处 `pyrender`/`PyOpenGL` 的版本冲突，按 §5 的分开装一次就过
>
> ### ⚠️ Windows 上**必须** `BP_NUM_WORKER=0`（两个连环坑）
>
> **坑 1：`multiprocessing` spawn 会重新导入主脚本**
>
> ```
> File "gen_pbr_data_demo.py", line 41, in <module>
>     import bpy
> ModuleNotFoundError: No module named '_bpy'
> ```
>
> Linux 的 `multiprocessing` 用 `fork`（子进程复制父进程，不重新导入）；
> **Windows 只有 `spawn`，子进程会重新 import 主模块** —— 而子进程是普通 Python、
> 不在 Blender 内，于是模块级的 `import bpy` 直接失败。
> 注意光有 `if __name__ == "__main__":` 守卫**还不够**：守卫只保护它后面的代码，
> 模块级的 import 照样执行。必须把 `bpy` / `blenderproc` 的 import **也挪进守卫里**。
> （已改：`data/render_ws/gen_pbr_data_demo.py` 文件头有注释说明。）
>
> **坑 2：改完还是不行 —— 因为 `blenderproc` 自己也模块级 import bpy**
>
> ```
> File "blenderproc/__init__.py", line 24, in <module>
>   from .api import loader
> File "blenderproc/python/loader/AMASSLoader.py", line 10, in <module>
>     import bpy
> ModuleNotFoundError: No module named '_bpy'
> ```
>
> 子进程要 unpickle 工作函数就得 import `blenderproc`，而它 import `bpy`。
> **这是个死结，无法绕过。** 所以 Windows 上 `BP_NUM_WORKER>0` 的 pyrender
> 进程池**根本用不了**——脚本第 74 行的注释其实早就写了这点。
>
> ### 解法：**多进程分片**（`scripts/merge_bop_shards.py`）
>
> 既然一个 Blender 进程只能吃一个核（掩码串行），就同时跑 N 个：
>
> ```powershell
> # N 个分片目录，各自 BP_NUM_SCENES 份场景 + 不同 BP_SEED + BP_NUM_WORKER=0
> # 每个进程的 cwd 指向自己的分片目录，互不干扰
> # 跑完后合并：
> python scripts/merge_bop_shards.py --out <merged> <shard0> <shard1> <shard2> <shard3>
> ```
>
> 实测（4 片并行，52 场景 × 20 帧 = 1040 帧）：
>
> | | |
> |---|---|
> | 单进程 | 约 21 s/帧（含启动摊薄） |
> | 4 片并行 | GPU 利用率 **84%**，显存 4.3~4.8 GB / 8.2 GB |
> | 预计总时长 | **约 1.5 小时** |
>
> ⚠️ 分片数别开太大：Cycles 每实例约占 1.2 GB 显存，8 GB 的卡最多 5~6 片。

---

### 4.1 装依赖（**当前卡在这**，见 §5）

```bash
BP=/root/autodl-tmp/blender-3.6.0-linux-x64
PY=$BP/3.6/python/bin/python3.10
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache        # 别写满根分区
export OMP_NUM_THREADS=8                               # 见 §6

# ① 主体依赖
"$PY" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "numpy==1.26.4" "opencv-python==4.9.0.80" "Pillow==10.4.0" "imageio==2.37.2" \
  "scipy==1.15.3" "scikit-image==0.24.0" "scikit-learn==1.5.2" "trimesh==4.2.2" \
  "matplotlib==3.9.2" "h5py==3.11.0" "rich==13.9.4" "PyYAML==6.0.2" \
  "progressbar2==4.5.0" "tqdm==4.67.1" "requests==2.32.3" "pytz==2024.2" \
  "pypng==0.20220715.0" "plyfile==1.1"

# ② PyOpenGL 与 pyrender 分开装（原因见 §5）
"$PY" -m pip install "PyOpenGL==3.1.7" "freetype-py==2.5.1" "pyglet==2.1.16"
"$PY" -m pip install --no-deps "pyrender==0.1.45"
```

### 4.2 部署 HCCEPose 的 blenderproc 2.5.0

```bash
SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")
rm -rf /root/autodl-tmp/bp_src && mkdir -p /root/autodl-tmp/bp_src
cd /root/autodl-tmp/bp_src && unzip -q -o /root/autodl-tmp/blenderproc.zip
rm -rf "$SITE/blenderproc"
cp -r /root/autodl-tmp/bp_src/blenderproc "$SITE/blenderproc"
find "$SITE/blenderproc" -name '__pycache__' -type d -prune -exec rm -rf {} +
grep __version__ "$SITE/blenderproc/version.py"     # 应为 2.5.0
```

### 4.3 冒烟测试

```bash
cat > /tmp/smoke.py <<'EOF'
import sys, bpy, blenderproc as bproc
print("python:", sys.version.split()[0])
print("bpy:", bpy.app.version_string, "| blenderproc:", bproc.__version__)
print("write_bop:", hasattr(bproc.writer, "write_bop"),
      "| load_512_ccmaterials:", hasattr(bproc.loader, "load_512_ccmaterials"))
import cv2, imageio, h5py, png, scipy, skimage, sklearn, trimesh, rich, yaml, tqdm, requests, pytz, PIL, matplotlib, pyrender
print("SMOKE OK")
EOF
cd /root/autodl-tmp && "$BP/blender" --background --python /tmp/smoke.py 2>&1 | tail -30
```

### 4.4 造材质库 + 渲染

```bash
"$PY" /root/autodl-tmp/bp_ws/make_cc0textures.py /root/autodl-tmp/cc0textures-512

rm -rf /root/autodl-tmp/bop/dji_action4/train_pbr
cd /root/autodl-tmp/bop/dji_action4                      # cwd 必须是数据集目录
export PYOPENGL_PLATFORM=egl
export BP_CC0TEXTURES=/root/autodl-tmp/cc0textures-512
export BP_TEMP_DIR=/root/autodl-tmp/bproc_tmp
export BP_NUM_SCENES=1 BP_NUM_OBJS=3 BP_FRAMES=4 BP_SAMPLES=64 BP_NUM_WORKER=0 BP_SEED=0
time "$BP/blender" --background --python /root/autodl-tmp/bp_ws/gen_pbr_data_demo.py
```

产物落在 `/root/autodl-tmp/bop/dji_action4/train_pbr/000000/`：
`rgb/000000.png` `depth/` `mask/` `mask_visib/` `scene_camera.json` `scene_gt.json` `scene_gt_info.json`。

### 4.5 取回图像

```powershell
python scripts\remote.py get /root/autodl-tmp/bop/dji_action4/train_pbr/000000/rgb/000000.png data/render_ws/out/000000.png
```

---

## 5. ✅ 已解决：`pyrender` 把 `PyOpenGL` 钉死在 `==3.1.0`

**触发命令**：§4.1 里把 `pyrender==0.1.45` 和 `PyOpenGL==3.1.7` 写在同一条 pip 命令里。

**现象**（原样）：

```
The conflict is caused by:
    The user requested PyOpenGL==3.1.7
    pyrender 0.1.45 depends on PyOpenGL==3.1.0

Additionally, some packages in these conflicts have no matching distributions available for your environment:
    pyopengl

ERROR: Cannot install PyOpenGL==3.1.7 and pyrender==0.1.45 because these package versions have conflicting dependencies.
ERROR: ResolutionImpossible
```

**原因**：`pyrender 0.1.45`（2021 年最后一次发版）在 `setup.py` 里写的是
`PyOpenGL==3.1.0` —— **精确钉死**，不是 `>=`。而 3.1.0 只有 sdist、没有 wheel。
后半句 "no matching distributions" 是解析失败的副产物，不是"镜像里没有 PyOpenGL"。

**解法**：**分两步装**，让 pyrender 跳过依赖检查（§4.1 的 ②）。
pyrender 运行时用 PyOpenGL 3.1.7 没有问题，那个 pin 只是它当年保守。

**副作用**：pip 是**先解析后安装**，所以那条命令**一个包都没装上**。
Blender 自带 Python 目前只有升级过的 `pip 26.2.1`，依赖需要从 §4.1 重跑。

**为什么非要 pyrender**：`write_bop()` 的 GT mask / `scene_gt_info` 是**用 pyrender 离屏渲染**算的
（`BopWriterUtility.py` 里 `import pyrender` 都写在函数内，所以导入不报错、跑到渲染才炸）。
这个 mask 是后续裁图 + 角点监督的输入，不能省。
它需要 EGL，所幸 `10_nvidia.json` 在、`libEGL.so.1` 已装。

---

## 6. 这次踩到 / 要记住的点

- **`Utility.temp_dir` 默认是空字符串**。BlenderProc 只有 CLI 会调
  `SetupUtility.setup_utility_paths()` 给它赋值；我们绕过 CLI 直接 `import`，
  于是 `ObjectLoader.load_obj()` 里
  `os.path.join(Utility.temp_dir, model_name)` 会退化成**相对路径**，
  把改写后的 **110 MB PLY 直接吐到当前工作目录**。
  → 适配版脚本里显式设了 `Utility.temp_dir`（可用 `BP_TEMP_DIR` 覆盖）。
- **`OMP_NUM_THREADS` 在远端是个非法值**，Blender 一启动就 `libgomp: Invalid value for
  environment variable OMP_NUM_THREADS`。跑之前 `export OMP_NUM_THREADS=8`。
- **不要加 `--factory-startup`** 跑正式渲染（冒烟测试可以）。BlenderProc 的 `bproc.init()`
  自己会清场，多一个变量不如少一个。
- **`num_worker` 默认是 4**。`write_bop` 会用 `multiprocessing.Pool` fork 出 4 个 pyrender
  worker，每个子进程都要重新 `eglInitialize`。HCCEPose 自己的注释也警告过
  "先 num_worker=0 再开 Pool 可能因主进程已初始化 EGL 导致失败"。
  → 首跑用 `BP_NUM_WORKER=0`（单进程）先确认能出图。
- **`color_file_format` 默认已是 `"PNG"`**，HCCEPose 脚本里写的是 `"JPEG"`（得到 `.jpg`）。
  我们的 dataloader 两种都认，演示用 PNG 更清晰。
- **aliyun pip 镜像这次只有 ~1.2 MB/s**，换清华 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 明显更快。
- **`bpy` 的 wheel 与 Blender 版本必须对齐**：`bpy-3.6.0` ↔ Blender 3.6。HCCEPose 的
  blenderproc 是 **2.5.0**（对应 Blender 3.5），但 README 让装 3.6.0，二者混用是他们验证过的组合；
  ⚠️ **不要**图省事去装 PyPI 上的 `bpy 4.2/4.3`——Blender 4.x 把 Principled BSDF 的
  `"Specular"` 改名成了 `"Specular IOR Level"`，而渲染脚本第 199 行正好在设 `"Specular"`，
  会直接报错。

---

## 7. 为什么这份脚本不進仓库

`gen_pbr_data_demo.py` 是 HCCEPose 的 `s2_p1_gen_pbr_data.py` 的改写版，而后者改编自
**BlenderProc（GPL-3.0）**。为避免把 GPL 派生代码放进公开仓库，本地放在
`data/render_ws/`（`/data/` 已在 `.gitignore` 中），远端放 `/root/autodl-tmp/bp_ws/`。

改动清单（相对原脚本）：
1. 所有规模参数改成环境变量（原脚本硬编码 `num_scenes = (50 * 1)` = 1000 帧、
   固定堆 30 个实例）；
2. `while cam_poses < 20` 加了尝试次数上限，避免内参/半径不合适时**死循环**；
3. 显式设置 `Utility.temp_dir`（见 §6）；
4. `kasal-6d` 的两个 json 小函数内联，少一个依赖；
5. `color_file_format` 改 PNG、`num_worker` 可配。
   场景构造（房间平面、cc0 材质、刚体堆叠、shell 采样、BOP writer）**与原脚本一致**。

---

## 8. ⚠️ 无卡模式的内存上限只有 2 GB —— 这才是 Blender 被杀的真正原因

第一次跑渲染时，日志停在 `load_bop_objs` 中途，**没有任何 traceback**，`bash` 只报：

```
bash: line 19:  2649 Killed  "$BP/blender" --background --python .../gen_pbr_data_demo.py
```

⚠️ **别看 exit code**。我最初写的 `blender ... | tail -100` 后面 `echo $?` 拿到的是 **`tail` 的**
退出码（0），看起来像"成功但没产物"。改成重定向到日志文件后才看到真身：

```
=== blender exit code = 137 ===
=== memory.events = low 0 high 0 max 6246 oom 0 oom_kill 0 ===
```

- `exit 137` = SIGKILL
- `memory.events` 里 **`max 6246`** = cgroup **撞了 6246 次内存上限**
- **`/sys/fs/cgroup/memory.max = 2147483648`（2 GB）** ← 无卡模式的真实限额
  （`free -g` 显示 1 TB 是**宿主机**的，容器里完全不是这么回事）

为什么这么费内存：BlenderProc 的 `ObjectLoader.load_obj()` 处理带纹理的 PLY 时，
是**把整个文件当字符串读进来**，然后做**两次 `.replace()`**，再把改写后的内容写出去：

```python
ply_file_content = file.read()                                    # 110 MB str
new_ply_file_content = ply_file_content
new_ply_file_content = new_ply_file_content.replace(...)          # +110 MB
new_ply_file_content = new_ply_file_content.replace(...)          # +110 MB
bpy.ops.import_mesh.ply(filepath=tmp_ply_file)                    # 三个字符串此时都还活着
```

也就是说 110 MB 的 PLY 在导入期间有 **~330 MB 的字符串副本**同时存在，外加 Blender
自己的网格数据（64.6 万顶点 / 89.1 万面）和 8 套 cc0 材质（48 张贴图）。
（`check_mesh.py` 单独跑能过，是因为它没有先把这 48 张贴图加载进来。）

**解法：把网格简化掉。** 生成模型 89 万面对 BOP 训练数据毫无必要。给
`scripts/mesh_to_bop.py` 加了 `--max-faces`：

```powershell
.venv\Scripts\python.exe scripts\mesh_to_bop.py `
    --input data\mesh\generated\textured.glb `
    --out-dir data\dji_action4\models --obj-id 1 `
    --max-faces 80000
```

结果：**89.1 万面 → 13.4 万面**，文本 PLY **110.6 MB → 29.1 MB**，
导入耗时 **80.8 s → 18.6 s**，峰值 RSS **851 MiB**（2 GB 以内，稳）。

⚠️ 一个反直觉的点：MeshLab 的保纹理简化滤波器里，**`preserveboundary=True` 会把简化硬卡在
40.3%**（891184 → 358830 面），而且**重复跑完全不再下降**——因为网格在 UV 缝处是开放边界，
它拒绝坍缩这些边。实测四种组合：

| 参数 | 结果 |
|---|---|
| `preserveboundary=True, planarquadric=True` | 358830 面（40.3%），**跑第二遍无效** |
| `preserveboundary=False, planarquadric=True` | 177836 面（20.0%） |
| `preserveboundary=False, planarquadric=False` | 100955 面（11.3%） |
| 上面再加 `preservenormal=True`（采用） | **134434 面（15.1%）** |

`qualitythr` 在 0.1~1.0 之间对本网格没有任何影响。

## 9. ⚠️ 渲染出的黑色裂纹：不是纹理，是**网格被简化撕开了**

渲染出来后物体表面有**黑色锯齿裂纹**。排查绕了三圈，最后靠一次"删掉整类原因"的实验定位。
（完整版见 `TROUBLESHOOTING.md` 的 `DATA-08`。）

1. **假设一：简化破坏了 UV 缝** → KD 树量化：简化后顶点 UV 距原始 UV **最大只有 2.9 px**
   （2048² 图集），>2px 的只有 0.6%。**排除**。
2. **假设二：纹理采样本身坏了** → `data/render_ws/uv_splat.py` 把顶点按 UV 取色后做正交投影
   点云（不用 Cycles，秒出）。结果**模型完全正确**——机身、镜头、屏幕、`ACTION 4K` 字样、
   红色 DJI 标、磁性卡扣全在。**排除**。
3. **假设三：图集空白区的噪声被采到了** → UV 只覆盖图集的 61.13%，剩下 38.87% 是黑白噪点
   （这个是**真的**，见下）。写了 `scripts/repair_texture_atlas.py` 做 texture padding，
   空白区高频噪声 3.85% → 1.32%。**重渲染，裂纹一模一样。也不是根因。**
4. **✅ 决定性实验：把纹理整个拿掉。** `preview_object.py` 加了 `BP_FLAT_MATERIAL=1`，
   断开 Base Color 的纹理连线、换成纯灰。**裂纹完全一样** → **是几何问题。**

**真正的根因**：`ms.get_topological_measures()` 显示这个网格是
**38705 个 UV 岛拼起来的**（646420 顶点 / 891184 面，边界边 394694）。
各岛边界本来严丝合缝地贴在一起，所以拓扑上开放、视觉上密闭。
而 `preserveboundary=False` 允许坍缩边界边 → **各岛边界各自往里缩 → 岛间裂开缝**，
还多出 **1383 条非流形边**。从缝里看进去是物体内表面，全黑。

**解法**：改回 **`preserveboundary=True`**（`scripts/mesh_to_bop.py` 的默认值）。
代价是简化下限被卡在 40.3%（358830 面），于是再用 `--ply-precision 5`
把文本 PLY 从 59.91 MB 压到 **32.89 MB** 来抵消内存。

**顺带修掉的真问题**：`scripts/repair_texture_atlas.py` —— §8 说的 padding。
它不解决裂纹，但它是对的：图集空白区（`preserveboundary=True` 的网格下是 25.84%）
本身就是噪声，缩小采样时会拉偏颜色。pad 之后岛内像素一个没动（diff = 0.0）。

⚠️ **教训**：在"纹理"这一类里换了三种假设都不对，**下一件事应该是跳出这一类**，
而不是换第四种。`BP_FLAT_MATERIAL` 那一次渲染零歧义。

## 10. 实测结果（无卡模式，1 核 CPU）

| 项目 | 数值 |
|---|---|
| PLY 导入 | 18.6 s（简化后） |
| 刚体模拟 | 3 s 收敛（72 帧） |
| 相机采样 | 4 次尝试全部通过（0.35–0.55 m） |
| Cycles CPU 渲染 | 480×360 / 32 采样 ≈ **4.5 min/帧**（单核） |
| 峰值内存 | **851 MiB** / 2 GB |
| BOP 产物 | `train_pbr/000000/{rgb,depth}` 正常写出（本机 `BP_WRITE_BOP=0` 跳过了标注） |

**`pyrender` 在无 GPU 时也能用**：EGL 会自动回落到 Mesa/llvmpipe
（`/usr/share/glvnd/egl_vendor.d/` 里 `10_nvidia.json` 在前、`50_mesa.json` 在后），
实测 `pyrender.OffscreenRenderer(64, 64)` 直接成功 → `write_bop()` 的 GT mask 路径可行。

> ⚠️ **有 GPU 时不要照搬这些参数**。`BP_DEVICE_TYPE=CUDA`、`BP_RES` 留空用
> `camera.json` 的 1024×768、`BP_SAMPLES=50`（原脚本值）才是正式配置。
> 无卡模式这套只是为了在没卡的时候把管线跑通并出图。

