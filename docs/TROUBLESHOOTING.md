# 问题记录 / 踩坑日志

> 记录**每次运行遇到什么问题、怎么解决的**。
> 目的：同一个坑不踩第二次；汇报进展时能直接引用。

## 怎么维护这个文档

**每次开始一段新的工作（训练、推理、渲染、环境搭建），遇到问题就加一条**，先写现象，解决了再补原因和解法。

条目编号规则：`<类别>-<序号>`，类别前缀：

| 前缀 | 范围 |
|---|---|
| `ENV` | 环境、依赖、Python/pip/conda |
| `PS` | PowerShell / shell / 命令行 |
| `GIT` | Git、GitHub |
| `CFG` | Hydra / OmegaConf 配置 |
| `CODE` | 自己写的代码 bug |
| `ALGO` | 算法、数值、超参 |
| `DATA` | 数据集、渲染、标注 |

**新增条目模板**：

````markdown
### <类别>-<序号> 一句话描述问题

- **时间**：YYYY-MM-DD
- **触发命令**：`...`
- **现象**：完整的报错信息或异常表现（**原样贴，不要转述**）
- **诊断**：怎么定位到根因的（做过哪些验证）
- **原因**：根因
- **解法**：具体怎么修的，含命令/代码
- **影响文件**：`path/to/file.py`
- **教训**：下次怎么避免 / 有无更通用的规律
````

---

## 运行记录

| 日期 | 做了什么 | 命令 | 结果 |
|---|---|---|---|
| 2026-09-10 | 拉取 BoxDreamer / HCCEPose 代码与论文 | `git clone --depth 1 ...` | ✅ 成功（注意 exit code 是假失败，见 `PS-01`） |
| 2026-09-10 | 仓库初始化并 push | `git push -u origin main` | ✅ 成功 |
| 2026-09-10 | 撤回误提交的题目原文 | 删库重建 + `git push` | ✅ 旧 SHA 已 404（见 `GIT-02`） |
| 2026-09-10 | 搭 Hydra + Lightning 骨架 | `pytest tests/` | ✅ 42/42 通过 |
| 2026-09-10 | 端到端冒烟 | `python run.py --config-name=train.yaml` | ⚠️ 按预期停在"数据集不存在"，链路已通 |
| 2026-09-10 | 对齐 BoxDreamer 的热图/损失配方 | `pytest tests/` | ✅ 42/42 通过；实测修掉 fine_beta 与热图退化两个数值问题（`ALGO-01`、`ALGO-02`） |
| 2026-09-10 | 建数据目录骨架 + 数据说明文档 | `mkdir data/...` | ✅ 完成；顺带发现 `.gitignore` 吞掉整个 `src/datasets/`（`GIT-03`） |
| 2026-09-10 | 加仓库卫生检查 | `pytest tests/test_repo_hygiene.py` | ✅ 55/55 通过 |
| 2026-09-12 | 云端部署 Hunyuan3D-2（AutoDL 4090） | `finish_all.sh` | 🟡 环境全配好并验证；**形状已跑通**（23.3s / 44.6万顶点），纹理待 GPU 模式 |
| 2026-09-12 | 跑通全部第一阶段 3D 生成 | 形状 23.3s / 纹理 1314s | ✅ `shape.glb` + `textured.glb`(26.7 MB) + 2048² 纹理图集 |
| 2026-09-12 | glb → BOP 格式转换 | `scripts/mesh_to_bop.py` | ✅ `obj_000001.ply`（110 MB ASCII）+ 贴图 + `models_info.json`（对角线 80.88 mm） |
| 2026-09-12 | UV 抽查（担心贴图映射错） | 读 PLY 第 7/8 列 | ✅ u/v 覆盖 0–0.9996、48 万个不同值，贴图能对上 |
| 2026-09-13 | 搭 BlenderProc 渲染环境 | 清华 TUNA 下 Blender 3.6.0 + HCCEPose 的 blenderproc 2.5.0 | ✅ 环境装好（`ENV-10` 的 pip 冲突按拆分装解决） |
| 2026-09-13 | 无卡模式下渲染（无 GPU / 1 核 / 2 GB） | `blender --background --python gen_pbr_data_demo.py` | ⛔→✅ 先被 cgroup 2 GB OOM 杀掉（`ENV-12`），简化网格后跑通 |
| 2026-09-13 | **渲染出图** | `BP_DEVICE_TYPE=CPU BP_RES=480x360 BP_SAMPLES=32` | ✅ 4 帧，480×360，峰值 851 MiB，4.5 min/帧（单核） |
| 2026-09-13 | 网格简化 | `mesh_to_bop.py --max-faces 80000` | ✅ 89.1 万面 → 13.4 万面，PLY 110.6 MB → 29.1 MB，导入 80.8 s → 18.6 s |
| 2026-09-13 | 排查渲染黑条纹 | `uv_splat.py` + 图集掩码统计 | ✅ 定位为图集空白噪声（`DATA-08`），padding 缓解 |
| — | 完整 BOP 数据集渲染（含 mask/GT） | `BP_WRITE_BOP=1` | ⏳ 待有 GPU 时跑正式配置（1024×768 / 50 采样） |
| 2026-09-14 | 核对官方图，发现是 **8 张不同视角**而非 2 张 | MD5 + 分辨率去重 | ✅ `closeups/13,14,16,17,18,19` 都是独立视角（14/17 磁盖开着，排除） |
| 2026-09-14 | 用 4 视图重跑形状 | `run_generate.py --views front left back right` | ✅ 尺寸比 `1.000/0.676/0.592` → **`1.000/0.642/0.472`**（官方 `1.000/0.627/0.465`） |
| 2026-09-14 | 4 视图纹理 | 同一命令 | ❌ 图集糊成噪声（`DATA-13`：侧视图是 ¾ 不是正交） |
| 2026-09-14 | 混合：4视图形状 + 2视图纹理 | `run_texture_mv.py --views front back` | ✅ 纹理恢复正常，深色机身 + 屏幕 + `ACTION 4` |
| 2026-09-14 | GPU 渲染（RTX 4090 / OptiX） | `preview_object.py` 8 视角 800×600 64 采样 | ✅ **127 秒**（无卡模式 CPU 同样参数约 45 分钟，快约 21 倍） |
| 2026-09-14 | 建 VGGT 重建环境（备用路线） | 权重 9.4 GB | 🟡 环境就绪，未在 GPU 上验证；用户决定先走 Hunyuan3D-2 |
| 2026-09-14 | 确认物体坐标系（判定 DATA-10） | 沿 ±X/±Y/±Z 正交投影点云 | ✅ 坐标系本就规范：+Z=镜头、+Y=上。`DATA-10` 是虚警 |
| 2026-09-14 | 验证 `write_bop` 标注 | 1 场景 4 帧 1024×768 | ✅ rgb/depth/mask/mask_visib/scene_gt/scene_camera/scene_gt_info 全部产出 |
| 2026-09-14 | 掩码多进程 | `BP_NUM_WORKER=4` | ✅ Blender 里 fork + EGL 正常，无崩溃 |
| 2026-09-14 | 修"一半场景只放 1 个物体" | 改挑选逻辑 | ✅ 每帧 8 实例（`DATA-15`） |
| 2026-09-14 | 深度缺陷定位 | `len(np.unique(depth))` | 🟡 记录为 `DATA-14`，暂不修（训练不用深度） |
| 2026-09-14 | **生成正式训练集** | 25 场景 × 20 帧，1024×768，50 采样，10 物体/场景 | ✅ **完成**：500 帧 / 5000 实例 / 5000 掩码 / 373 MB（96 分钟） |
| 2026-09-14 | 数据集验收 | 完整性 + 旋转矩阵正交性 + 分布统计 | ✅ 结构全过（`det≈1`、`max\|RRᵀ−I\|=8.5e-7`）；⚠️ 查出**遮挡严重不足**（`DATA-16`） |
| 2026-09-14 | 归档 | 远端打包 373 MB 并下载到本地（含 SHA256） | ✅ `data/bop/dji_action4_hybrid_dataset.tar.gz` |
| 2026-09-14 | **阶段③④：数据→网络集成验证**（无卡模式，本地 CPU） | `scripts/verify_{bop,dataloader,pnp}.py` | ✅ 四条全过：标注投影吻合、热图峰**0 格**误差、PnP GT 往返 `rot<0.04°`、训练循环跑通 |
| 2026-09-14 | 训练循环冒烟 | `run.py` 2 epoch / CPU | ✅ 12.0M 参数、loss 正常、验证循环与 checkpoint 保存都通 |
| 2026-09-14 | **小样本过拟合 sanity test** | 64 样本 / 320 步 / CPU | ✅ **tr_loss 56.5→8.8**，`pck@0.15` 0.195→0.461，`pose_valid_ratio=1.00` |
| — | GPU 正式训练 | `python run.py --config-name=train.yaml` | ⏳ 待有卡；5000 样本 / 256px / 500 epoch |

---

## 问题归档

### ENV-01 全局 Python 缺训练框架，但不想重装 torch

- **时间**：2026-09-10
- **现象**：`import pytorch_lightning` → `ModuleNotFoundError`。检查发现全局已有
  `torch 2.12.0+cpu`、`torchvision 0.27.0+cpu`、`cv2 4.11.0`、`numpy 1.26.4`、`scipy`、`PIL`、`yaml`、`pytest`，
  但缺 `pytorch-lightning` / `hydra-core` / `omegaconf` / `einops`。
- **原因**：本机全局环境没装训练框架。
- **解法**：建一个**复用全局包**的 venv，只补缺的那几个：

  ```powershell
  python -m venv --system-site-packages .venv
  .\.venv\Scripts\python.exe -m pip install `
      "pytorch-lightning>=2.4" "hydra-core>=1.3" "omegaconf>=2.3" "einops>=0.8"
  ```

  验证：`.\.venv\Scripts\python.exe -c "import pytorch_lightning, hydra, omegaconf, einops, torch; print(torch.__version__)"`
  → `pl 2.6.6 / hydra 1.3.6 / omegaconf 2.3.1 / einops 0.8.2 / torch 2.12.0+cpu`
- **注意**：全局的 torch 是 **CPU 版**。要真正训练必须换成 CUDA 版：
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- **教训**：`--system-site-packages` 能省掉重装 torch（几百 MB），但两个环境会共享全局包，
  后续若要升级 torch 版本要注意别互相影响。`.venv/` 已在 `.gitignore` 中。

---

### ENV-02 下载 resnet18 预训练权重报哈希校验失败

- **时间**：2026-09-10
- **触发命令**：`python run.py --config-name=train.yaml`
- **现象**：

  ```
  RuntimeError('invalid hash value (expected "f37072fd",
                got "99129373af0ac86478ffe203e178c48cae228231ac1357fc01635c3947ab07a9")')
  ```

  前面还能看到进度条正常下完了 44.7 MB。
- **诊断**：
  1. 用 PowerShell 直接下同一个 URL 并算 sha256 →
     `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`，**与期望前缀一致**
     → 说明**网络和服务器都没问题**。
  2. 检查缓存目录 `~/.cache/torch/hub/checkpoints/` → **是空的**
     → 说明 torch 在哈希校验失败时把临时文件删了，没有留下可复用的部分。
  3. 单独调 `torch.hub.load_state_dict_from_url(url)` → **成功**
     → 确认是**那一次下载本身损坏**（偶发），不是系统性问题。
- **原因**：单次下载过程中内容被损坏（网络抖动 / 中间环节），torch 校验后拒绝使用。
- **解法**：直接重试即可（缓存目录为空，会自动重新下载）。或手动清缓存：

  ```powershell
  Remove-Item "$env:USERPROFILE\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth" -ErrorAction SilentlyContinue
  ```
- **教训**：**看到哈希不匹配先别怀疑网络**。用别的工具下同一个文件对比哈希，
  能一秒区分"链路问题"和"这一次下载损坏"。torch 校验失败会清掉临时文件，
  所以缓存目录空 = 需要重下，不存在"下了一半可以续"的情况。

---

### ENV-03 Windows 控制台中文日志乱码

- **时间**：2026-09-10
- **现象**：报错信息在终端里显示成
  `������ HCCEPose �� s2_p1_gen_pbr_data.py ��Ⱦ���ݣ�`
- **诊断**：
  ```python
  import sys, pathlib
  print(sys.stdout.encoding)                                  # -> gbk
  pathlib.Path("src/datamodules/corner_pose_datamodule.py").read_bytes().decode("utf-8")  # -> 正常
  ```
  → **源文件是合法 UTF-8，字符串内容完好**；是 Python 按 GBK 编码输出、而读取端按 UTF-8 解码导致的。
- **原因**：Windows 控制台/管道的默认编码是 GBK（代码页 936）。
- **解法**（二选一，跑之前设一次）：

  ```powershell
  $env:PYTHONUTF8=1          # 推荐
  $env:PYTHONIOENCODING="utf-8"
  ```
- **教训**：这是**纯显示问题，不是代码 bug**。排查编码问题时，
  先 `decode('utf-8')` 验证文件本身，再 `print(sys.stdout.encoding)` 看输出侧 —— 一秒定位是哪一端的问题。

---

### PS-01 `git clone` 在 PowerShell 里 exit code 1，但实际克隆成功

- **时间**：2026-09-10
- **触发命令**：`git clone --depth 1 <url> <dir> 2>&1 | Select-Object -Last 20`
- **现象**：任务以 `[exit code: 1]` 结束，stderr 里是：
  ```
  git : Cloning into '...'...
  + CategoryInfo          : NotSpecified: (Cloning into '...'...:String) [], RemoteException
  + FullyQualifiedErrorId : NativeCommandError
  ```
  但目标目录**已经完整克隆好了**（`git log -1` 正常，文件齐全）。
- **原因**：git 把进度信息写到 **stderr**，而 PowerShell 把原生命令写到 stderr 的内容包装成
  `NativeCommandError` 并改变退出码。**这是 PowerShell 的行为，不是 git 失败**。
- **解法**：
  - 判断成功与否用 `$LASTEXITCODE`，不要只看任务级的 exit code；
  - 或者把 stderr 合并：`git ... 2>&1 | Out-String`；
  - **最重要**：核对**产物**（目录是否存在、`git log -1` 是否正常）。
- **教训**：⚠️ **把"命令报错"和"命令没生效"分开判断**。
  这个坑如果没查产物，会误以为克隆失败而重复执行，浪费大量时间。

---

### PS-02 PowerShell 不支持 bash 的 here-string 重定向

- **时间**：2026-09-10
- **现象**：`git commit -F - <<< $msg` → `Missing file specification after redirection operator`
- **原因**：`<<<` 是 bash 语法，PowerShell 不支持。
- **解法**：把内容写到文件再引用：
  ```powershell
  $msg | Out-File -Encoding utf8 "$p\.git\COMMIT_MSG_TMP.txt"
  git -C $p commit -q -F "$p\.git\COMMIT_MSG_TMP.txt"
  Remove-Item "$p\.git\COMMIT_MSG_TMP.txt"
  ```
  （写到 `.git/` 里可以避免污染工作区、也不会被 `git add .` 扫到。）

---

### PS-03 临时目录里跑脚本报 `No module named 'src'`

- **时间**：2026-09-10
- **现象**：脚本放在 `$env:TEMP` 下执行，`from src.models.utils...` → `ModuleNotFoundError: No module named 'src'`，
  即使在项目根目录下执行也报错。
- **原因**：Python 的 `sys.path[0]` 是**脚本所在目录**，不是当前工作目录（cwd）。
- **解法**：
  ```powershell
  $env:PYTHONPATH = "D:\AAA_Projects\psd_zju3dv_coding_exam"
  ```
- **教训**：临时验证脚本要么写到项目根目录下，要么设 `PYTHONPATH`。
  调试时设 `PYTHONPATH` 更干净（不污染仓库）。

---

### GIT-01 大文件超过 GitHub 单文件硬限制

- **时间**：2026-09-10
- **现象**：目标数据视频 `head_left_rgb_raw.mp4` **277 MB**，超过 GitHub 单文件 **100 MB** 硬限制，push 必被拒。
  另外 `refs/` 里的第三方克隆共 **248 MB**（含各自的 `.git`）。
- **解法**：写进 `.gitignore`（视频目录 + `refs/`），README 里说明怎么重新获取。
  验证：`git check-ignore -v refs "head_left_rgb_raw.mp4(1)"` 两个都命中。
- **教训**：**克隆/下载任何东西之前先量体积**。
  `Get-ChildItem -Recurse -File | Where-Object Length -gt 50MB` 一行就能提前发现问题。

---

### GIT-02 ⚠️ `.gitignore` 删不掉已经提交的历史（重要）

- **时间**：2026-09-10
- **现象**：题目原文 `docs/ASSIGNMENT.md` 在第一次 push 时已经公开，
  之后即使 `git rm --cached` + 加 `.gitignore` + 重新提交，**旧的 commit SHA 直链仍然能读到它**：
  ```
  GET /repos/<owner>/<repo>/contents/docs/ASSIGNMENT.md?ref=9e219f3...  ->  HTTP 200，2008 bytes
  GET /repos/<owner>/<repo>/commits/9e219f3...                          ->  HTTP 200
  ```
- **原因**：GitHub 会保留**不可达对象**，只要知道 SHA 就能直连访问。
  `.gitignore` / `git rm --cached` 只影响**之后的提交**，不动历史。
- **解法**（本次采用，仓库是刚建的、0 fork / 0 star，无外部副本）：
  1. 本地彻底清干净：
     ```powershell
     git reflog expire --expire=now --all
     git gc --prune=now
     ```
  2. **在网页上删除仓库，再同名重建一个空仓库**（不要勾 README / .gitignore / license）
  3. `git remote set-url origin <新地址>` → `git push -u origin main`
  4. 验证：新仓库里 `contents/docs/ASSIGNMENT.md?ref=<任意 SHA>` → **404**
- **备选方案**：保留仓库，联系 GitHub Support 请求 purge 缓存视图（要等几天）。
- **教训**：**"不要提交某个文件"必须在第一次 commit 之前决定。**
  一旦推上去，唯一可靠的补救是删库重建（或 `git filter-repo` 重写全部历史 + force push，
  但对已有 fork 的仓库无效）。**敏感内容先 gitignore，再 git add。**

- 🔁 **2026-09-23 第二次复发**（同类，但是"私人笔记/截图"而不是"题目原文"）：
  `docs/HUMAN_LOG.md`（人类自己的实验笔记，开头就写着"本文件由人类所有"）
  + 4 张 `联想截图_*.png`（个人截图工具产出，共 4.6 MB，其中 3 张内容完全相同）
  **一直被公开在 `main` 上**。查明过程：写了 `scripts/check_repo.py` 做例行体检，
  第 3 项"敏感文件是否被跟踪"直接点名。
  - **处理**：本次采用 **A 方案**（`git rm --cached` + 补 `.gitignore`），
    **不做历史重写** —— 人类判断"内容不敏感"，重写全部 SHA 的代价不值得。
  - ⚠️ **所以旧副本仍可通过 commit SHA 直连读取**（正是本条目上半部分描述的情形）。
    要彻底清除才需要 `git filter-repo` + force push（仓库 0 fork，重写安全）。
  - ✅ **预防（这次真正补上的那一环）**：`scripts/check_repo.py` 把这件事变成
    **几十秒能跑完的例行检查** —— 六项：同步状态 / 体积分布 / **敏感文件是否被跟踪**
    （人类笔记、题目原文、截图工具产物、凭据）/ 大文件类型 / **Markdown 相对链接**
    / 文档索引一致性。有问题返回退出码 1，可接 CI。
    另外 `.gitignore` 补了 `*截图*.png`、`docs/HUMAN_LOG.md` 等规则，防止 `git add .` 复发。
  - **教训（比第一次更具体）**：⚠️ **`.gitignore` 只防"以后"，防不住"已经"；
    而"已经"这件事只能靠定期体检发现。** 光加规则不够 —— 本次就是因为
    `.gitignore` 里从来没有这两条规则，第 5 张截图已经在工作区里等着被 `git add .` 带走。

---

### GIT-03 ⚠️ `.gitignore` 里的无锚点目录模式静默吞掉了整个源码包（重要）

- **时间**：2026-09-10
- **现象**：改了 `src/datasets/bop_pbr.py` 之后，`git status` **完全看不到这个文件**。
  进一步查发现：

  ```
  $ git check-ignore -v src/datasets/bop_pbr.py
  .gitignore:91:datasets/    src/datasets/bop_pbr.py

  $ git ls-files src          # 输出里没有 src/datasets/ 任何文件
  ```

  也就是说 **`src/datasets/` 这个包从来没被提交过**。
  翻回骨架那次提交的文件列表核对，确实缺了它 —— 已经 push 到公开仓库了。

  顺着查下去，**还有第二个受害者**：

  ```
  $ git check-ignore -v configs/model/vis/default.yaml
  .gitignore:98:vis/    configs/model/vis/default.yaml
  ```

  `configs/model/heatmap.yaml` 里有 `- vis: default` 这条 defaults 组合，
  但那个被引用的配置文件根本没进仓库 → **别人 clone 下来 `python run.py` 会直接因为
  找不到配置组而报错**。本地一直能跑，只是因为文件在磁盘上还躺着。
- **原因**：`.gitignore` 里写了两条**无锚点**的模式：

  ```gitignore
  datasets/          # ← 没有前导 /
  vis/               # ← 也没有
  ```

  无前导斜杠的模式会匹配**任意深度**的同名目录，所以 `src/datasets/` 和
  `configs/model/vis/` 都被匹配。
  `.gitignore` 是**静默**的：它不会警告"你忽略了源码"，
  `git status` 也不会列出被忽略的东西，所以这个错误可以潜伏很久。

  同类的高危模式（本次一并排查）：

  | 模式 | 实际误伤 |
  |---|---|
  | `datasets/` | `src/datasets/` ← **真的踩了** |
  | `vis/` | `configs/model/vis/` ← **也真的踩了** |
  | `data/` | 任何 `*/data/` |
  | `logs/` `runs/` `weights/` `checkpoints/` | 未来的 `src/logs/` 等 |
  | `outputs/` `output/` `results/` `preds/` | 同上 |

- **诊断**：用 `git check-ignore` 批量扫一遍工作区里所有源码文件：

  ```powershell
  # 注意：已跟踪的文件不会被报告，所以这条只揪"未跟踪且被忽略"的
  git status --ignored --short | Select-String '\.py$'
  git ls-files src        # 和磁盘上的文件对比
  ```

- **解法**：凡是"根目录下的数据/输出目录"，模式前面**加 `/` 锚定**：

  ```gitignore
  /data/
  /datasets/
  /dataset/
  /outputs/
  /checkpoints/
  /weights/
  /logs/
  ```

  （`train_pbr/`、`test_pbr/` 故意保留无锚点，因为它们本来就嵌在 `data/` 下面。）

- **影响文件**：`.gitignore`、`src/datasets/`、`configs/model/vis/`（后两者补交）
- **后果**：修复前那个公开仓库 clone 下来 **`python run.py` 跑不起来**
  （缺 `configs/model/vis/default.yaml`，hydra 的 defaults 组合会失败），
  而且 `src/datasets` 整个包缺失。
- **防复发**：新增了 `tests/test_repo_hygiene.py`，4 个用例：
  1. 扫描所有源码/配置/脚本，用 `git check-ignore --stdin` 检查有没有被忽略的
  2. 专门盯 `src/datasets/` 这个包
  3. 静态检查 `.gitignore`，高危目录模式必须带前导 `/`
  4. 确认扫描本身有效（避免断言退化成"空集合永远通过"）

  用 `git check-ignore` 而不是 `git ls-files` 是关键：**"未跟踪"是临时状态**
  （刚写的文件本来就还没 add），而**"被忽略"才是真 bug**；
  `check-ignore` 只报告未跟踪且被忽略的路径，判据正好精确。
- **教训**：⚠️ **在 `.gitignore` 里写目录名，默认一定要加前导 `/`。**
  无锚点模式的作用域比你想象的广得多，而且它是静默的 ——
  最可怕的不是它忽略了东西，而是**你以为它没忽略**。
  凡是"我以为提交上去了"的时刻，都用 `git ls-files <路径>` 核实一遍。

---

### CFG-01 Hydra 警告 "Defaults list is missing `_self_`"

- **时间**：2026-09-10
- **现象**：
  ```
  UserWarning: In 'train.yaml': Defaults list is missing `_self_`.
  ```
- **原因**：`defaults:` 列表没有显式声明 `_self_`，配置组合顺序不明确。
  （BoxDreamer 的 `configs/train.yaml` 也有这个问题。）
- **解法**：在 `defaults` 列表**末尾**加 `- _self_`，表示"本文件的键优先于各配置组"。
- **影响文件**：`configs/train.yaml`、`configs/test.yaml`

---

### CFG-02 `hydra.job.num` 在单次运行时不存在，导致整份配置解析失败

- **时间**：2026-09-10
- **现象**：`OmegaConf.to_container(cfg, resolve=True)` 报
  ```
  InterpolationToMissingValueError: MissingMandatoryValue while resolving interpolation:
  Missing mandatory value: hydra.job.num
      full_key: hydra.sweep.subdir
  ```
- **诊断**：`hydra.job.num` 只在 **multirun**（`-m`）时才被赋值；
  而 hydra 默认配置里的 `hydra.sweep.subdir` 引用了它。
- **解法**：
  1. 删掉自己写的 `configs/hydra/default.yaml` 里的 `sweep` 覆盖（用 hydra 默认即可）；
  2. **不要对整份 cfg 做 `resolve=True`**，只解析自己的子树（见 `CFG-03`）。
- **影响文件**：`configs/hydra/default.yaml`
- **教训**：`${hydra:...}` 命名空间里的键**不是全都有值**，逐项确认再引用。

---

### CFG-03 pytest 里解析配置报 `HydraConfig was not set`

- **时间**：2026-09-10
- **现象**：直接跑 `pytest` 时，任何涉及 `${hydra:runtime.cwd}` 的解析都失败：
  ```
  InterpolationResolutionError: ValueError raised while resolving interpolation:
  HydraConfig was not set
  ```
  但 `python run.py --cfg job` 完全正常。
- **原因**：`${hydra:runtime.cwd}` 由 hydra 的**自定义 resolver** 解析，
  该 resolver 去读全局单例 `HydraConfig.instance()`。pytest 不在 hydra 的运行时上下文里，单例为空。
- **解法**：在测试里进 `initialize()` 上下文，并把组合出的配置装进 `HydraConfig`：
  ```python
  with initialize(version_base="1.3", config_path="../configs"):
      cfg = compose(config_name="train.yaml", return_hydra_config=True)
      HydraConfig.instance().set_config(cfg)
      return fn(cfg)          # 所有断言都在这个上下文里做
  ```
  注意：`initialize()` 是上下文管理器，**退出后 GlobalHydra 会被清理**，
  所以断言必须写在 `with` 块内（本项目用一个 `_with_config(cfg_name, fn)` 辅助函数统一处理）。
- **影响文件**：`tests/test_config_instantiation.py`

---

### CODE-01 模块名与类名同名，被 `__init__.py` 的重导出遮蔽

- **时间**：2026-09-10
- **现象**：
  ```
  AttributeError: type object 'PL_CornerPose' has no attribute 'PL_CornerPose'
  ```
  以及 hydra 实例化时报：
  ```
  InstantiationException: Error locating target 'src.lightning.PL_CornerPose.PL_CornerPose'
  ```
- **诊断**：文件是 `src/lightning/PL_CornerPose.py`，类也叫 `PL_CornerPose`，
  而 `src/lightning/__init__.py` 里有 `from src.lightning.PL_CornerPose import PL_CornerPose`。
  这行把**类**绑成了包的属性 `src.lightning.PL_CornerPose`，**覆盖了同名的子模块**。
  之后 `import src.lightning.PL_CornerPose as m` 拿到的是类而不是模块。
- **解法**：**让模块名与类名不同**，改名为
  `src/lightning/corner_pose_lightning_model.py`（类仍是 `PL_CornerPose`）。
  这正是 BoxDreamer 的命名方式：`BoxDreamer_lightning_model.py` / `PL_BoxDreamer`。
- **影响文件**：`src/lightning/__init__.py`、`configs/model/heatmap.yaml`（`_target_`）
- **教训**：Python 里**模块名不要和里面的类名相同**，尤其当包会 re-export 的时候。

---

### CODE-02 验证指标把带梯度的张量累积到 epoch 末尾 → 显存泄漏

- **时间**：2026-09-10（自查发现，未实际爆显存）
- **现象**：`validation_step` 里 `preds["corner_2d"]` 直接来自网络输出，
  带着 `grad_fn`。指标累加器把它 append 进 list，到 `on_validation_epoch_end` 才消费。
- **原因**：这些张量引用了整条计算图，只要 list 不释放，图就一直挂在显存里。
  一个 epoch 累积 N 个 batch 的计算图 → 训练若干轮后 OOM。
- **解法**：提取角点 + PnP 只用于评估，全部放进 `torch.no_grad()`：
  ```python
  with torch.no_grad():
      preds = predict_corners_and_pose(...)
  ```
- **影响文件**：`src/lightning/corner_pose_lightning_model.py`
- **教训**：**凡是"累积到 epoch 末尾再算"的量，累积前一定要 detach 或放进 no_grad。**

---

### ALGO-01 fine loss 的 soft-argmax 温度凭直觉设错，偏了 19 像素

- **时间**：2026-09-10
- **现象**：实现 BoxDreamer 的 fine loss 时，需要从预测热图里**可微地**取出角点坐标。
  我第一版把 soft-argmax 的温度设成 `beta=1.0`，理由是"热图归一化到 [0,1]，1.0 梯度最健康"。
  实测发现**平均偏 19 个热图像素**（= 76 个图像像素），完全不可用。
- **诊断**：写了个测量脚本，用 30 组真实投影角点生成 BoxDreamer 风格 GT 热图，
  对比不同 beta 下 soft-argmax 与真值的距离：

  | 温度 | 平均误差 | p95 | max |
  |---:|---:|---:|---:|
  | 1.0 | **19.0 px** | 28.9 | 32.3 |
  | 10.0 | 1.26 px | 2.35 | 10.2 |
  | **25.0** | **0.25 px** | 0.44 | 10.2 |
  | 100.0 | 0.38 px | 0.51 | 10.2 |
  | argmax | 0.74 px | 0.62 | **49.0** |
  | topk(20) | 0.76 px | 0.27 | **43.0** |

- **原因**：BoxDreamer 风格的热图**不是尖峰而是很宽的小丘**（`exp(-d/scale)`，
  `scale=(s/10)²` 通常只有几个像素）。beta 太小时，4096 个背景像素的权重加起来
  远大于峰值附近那几十个像素，soft-argmax 的结果被**拉向热图质心**而不是峰顶。
- **解法**：`fine_beta: 25.0`（实测最优）。写进 `configs/model/loss/default.yaml` 的注释里，
  并把测量结论一并留档，防止后人又"凭直觉"改小。
- **影响文件**：`src/loss/loss.py`、`configs/model/loss/default.yaml`、`configs/model/heatmap.yaml`
- **教训**：⚠️ **softmax 类操作的 temperature 不能靠直觉定，必须实测标定。**
  另外这张表还说明：`argmax` 和 `topk` 的**平均**误差不大，但**最坏情况能偏 40+ px**
  （角点热图被画面边缘截断时会跳到边界），做误差分析时要看 p95/max 而不只是 mean。

---

### ALGO-02 BoxDreamer 的尺度自适应热图在角点贴近物体中心时退化成 δ 函数

- **时间**：2026-09-10
- **现象**：写 `topk` 提取的测试时，发现有 2/8 个角点的提取位置偏了 **14~18 个热图像素**，
  而另外 6 个只偏 0.1 px。
- **诊断**：那 2 个角点恰好**离物体 2D 中心很近**。
  BoxDreamer 的尺度是 `scale = (s_i/10)²`，`s_i` 是角点到 8 角点均值的像素距离。
  `s_i` 很小时 `scale → 0`，`exp(-d/scale)` 在 `d ≥ 1` 处全部**下溢为 0**
  → 热图退化成一个单像素的 δ
  → 背景 4095 个像素**全部等于 0，完全并列**
  → `torch.topk` 只能返回那 1 个真峰 + 19 个**任意**的零值像素
  → 平均位置基本是随机的。
- **原因**：原式没有下限保护。真实投影包围盒的角点离中心不会太近（都在轮廓边界上），
  但**随机角点、异常姿态、退化投影**都会踩到这个边界情况。
- **解法**：加 `min_scale` 下限（默认 1 个热图像素），并补了回归测试
  `test_min_scale_prevents_degenerate_delta`：
  ```python
  scale_factor = ((dis / 10.0) ** 2).clamp(min=float(min_scale))
  ```
- **影响文件**：`src/models/utils/data_processing.py`、`tests/test_geometry.py`
- **教训**：**峰/标签类的目标函数一定要检查"是否会出现大量并列值"**。
  并列不只影响指标，还会让 `topk` / `argmax` 这类操作的结果变成随机的，
  表现为"训练不收敛但看不出报错"。

---

### ALGO-03 测试数据用"均匀随机角点"，既不真实又掩盖问题

- **时间**：2026-09-10
- **现象**：最初写的几何测试用 `torch.rand(2)*0.7+0.15` 生成 8 个 2D 角点。
  它既不反映真实的投影几何（角点应分布在物体轮廓边界上），
  又**恰好触发了 `ALGO-02` 的退化情况**，一开始还让我以为实现写错了。
- **解法**：改成**投影一个真实 3D 包围盒**得到角点：
  ```python
  uv = project_points(bbox_3d.unsqueeze(0), K.unsqueeze(0), pose.unsqueeze(0))[0]
  uv = uv - uv.mean(0, keepdim=True)
  uv = uv * (0.6 * img_size / extent) + img_size / 2    # 平移缩放到画面中央
  ```
  这样测试数据既真实，又顺带验证了 `project_points` 这条链路。
- **影响文件**：`tests/test_geometry.py`
- **教训**：**测试数据的分布要和真实数据一致**，否则测出来的问题是假的、
  真问题反而漏掉。合成测试数据时优先"走一遍真实链路"（投影 / 裁剪 / 缩放），
  而不是直接 `rand`。

---

### ALGO-04 初始化网络的输出范围不可断言

- **时间**：2026-09-10
- **现象**：写测试断言"centernet 风格（不激活）的输出一定超出 [-1,1]"，实际失败：
  ```
  assert (-0.566 < -1.0 or 0.685 > 1.0)
  ```
- **原因**：**随机初始化的网络输出范围是任意的**，不能假设它一定超出某个区间。
  这个断言想验证的是"没做激活"，但用值域来验证是错的。
- **解法**：改成直接对比解码头原始输出：
  ```python
  raw = model.decoder(model.encoder(batch["image"]))["heatmap"]
  assert torch.allclose(out["pred_heatmap"], raw)
  ```
- **影响文件**：`tests/test_model_forward.py`
- **教训**：**测试要断言"逻辑"，不要断言"没训练的网络恰好表现出来的数值"。**

---

### ENV-05 ⚠️ GitHub 和 HuggingFace 对代理的要求**相反**（本次最费时的坑）

- **时间**：2026-09-12
- **现象**：在 AutoDL 上按官方提示 `source /etc/network_turbo` 之后：
  - 下 HuggingFace 模型慢到 **1.3 MB/s**
  - 下 GitHub 的东西慢到 **~20 KB/s**（1 GB 的 rembg 模型预计要 14 小时）
- **诊断**：写了个 15 秒采样的测速脚本对比，**其中一个变体忘了 source**，
  结果发现规律完全反了：

  | 目标 | 开 turbo | 关 turbo |
  |---|---:|---:|
  | HuggingFace（hf-mirror） | 1.3 MB/s | **14 MB/s** |
  | GitHub | **127 MB/s** | ~20 KB/s |

  AutoDL 的提示其实写了：*"开启加速后对访问其他资源如 pip 源等会**更慢**"*。
- **解法**：**按目标分别设置**

  ```bash
  # HuggingFace / pip：关代理
  unset http_proxy https_proxy
  export HF_ENDPOINT=https://hf-mirror.com

  # GitHub：开代理
  source /etc/network_turbo
  ```
- **教训**：⚠️ **"开了加速"不等于"全局变快"**。
  遇到慢先做**分端点测速**，不要假设。测速时也要注意别把代理状态搞混
  （我那次"忘了 source"反而成了发现真相的契机）。

---

### ENV-06 `hf download` 报 401 Unauthorized（Xet 后端在 hf-mirror 上不可用）

- **时间**：2026-09-12
- **现象**：
  ```
  RuntimeError: Task error: File reconstruction error: CAS Client Error:
  Request error: HTTP status client error (401 Unauthorized),
  domain: https://cas-server.xethub.hf.co/v2/reconstructions/...
  ```
- **原因**：HuggingFace 新的 **Xet** 存储后端要连 `cas-server.xethub.hf.co`，
  **hf-mirror 不支持**这个域名，于是 401。
- **解法**：禁用 Xet，回退到普通 HTTP 下载：

  ```bash
  export HF_HUB_DISABLE_XET=1
  ```

- **附带**：Xet 写出的 `.incomplete` 文件普通 HTTP 下载不一定能复用，
  换后端时先清掉：`find "$HF_HOME" -name '*.incomplete' -delete`

---

### ENV-07 `hf download --include` 只接受**一个**值

- **时间**：2026-09-12
- **现象**：
  ```
  UserWarning: Ignoring `--include` since filenames have been explicitly set.
  Error: File not found in repository.
  URL: https://hf-mirror.com/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-delight-v2-0/%2A
  ```
- **原因**：`--include` 不是 nargs，**第二个模式被当成了"显式文件名"**，
  于是 `--include` 被整体忽略，然后去找一个名字就叫 `*` 的文件 → 404。
- **解法**：**一个模式一条命令**

  ```bash
  hf download tencent/Hunyuan3D-2 --include "hunyuan3d-paint-v2-0-turbo/*"
  hf download tencent/Hunyuan3D-2 --include "hunyuan3d-delight-v2-0/*"
  ```
- **教训**：CLI 报"忽略某参数"时，先想"是不是我把另一个参数写成了位置参数"。

---

### ENV-08 AutoDL **无卡模式只有 1 个 CPU 核**

- **时间**：2026-09-12
- **现象**：切到无卡模式后想在 CPU 上做纹理 pipeline 的加载 dry-run，
  跑到 `Loading pipeline components... 67%` 时连接被掐断（输出 `exit=-1`）。
- **诊断**：`nproc` → **1**；`free -h` → 内存 1 TB（**不缺内存**）。
  所以不是 OOM，是**单核太慢**导致进程被中断。
- **解法**：无卡模式只用来**装环境 + 下文件**；
  **任何模型加载/推理都必须切回 GPU 模式**。
- **教训**：省钱模式和可用能力要分清楚。花 30 秒 `nproc` / `free -h`
  确认资源，比盲目等半小时划算。

---

### CODE-03 本地 `custom_pipeline` 需要 `trust_remote_code=True`

- **时间**：2026-09-12
- **现象**：
  ```
  ValueError: The directory .../hunyuanpaint contains custom code in pipeline.py
  which must be executed to correctly load the model.
  Pass `trust_remote_code=True` to allow loading remote code modules.
  ```
- **原因**：新版 diffusers 对**本地目录**里的自定义 pipeline 也要求显式声明信任。
  注意报错说的是"remote code"，但实际是**本地路径**（`custom_pipeline=<本地目录>`），
  容易看错方向。
- **解法**：`hy3dgen/texgen/utils/multiview_utils.py` 第 34 行那个
  `DiffusionPipeline.from_pretrained(...)` 加 `trust_remote_code=True`。

---

### CODE-04 `transformers 5.x` 拒绝加载 `.bin` 权重（CVE-2025-32434）

- **时间**：2026-09-12
- **现象**：
  ```
  ValueError: Due to a serious vulnerability issue in `torch.load`, even with
  `weights_only=True`, we now require users to upgrade torch to at least v2.6 ...
  This version restriction does not apply when loading files with safetensors.
  ```
- **诊断**：装依赖时没钉版本，`transformers` 装到了 **5.17.0**
  （仓库是 2025 年初的，对应 4.4x）。而 `hunyuan3d-paint-v2-0-turbo` 的
  `text_encoder` 和 `vae` **只提供 `.bin`**，没有 safetensors。
- **两条路都被我否决了**：
  - 升级 torch 到 2.6 → 可能让**已编译的 CUDA 扩展 ABI 不匹配**（要重编）
  - 降级 transformers → 会牵动 `huggingface_hub` / `tokenizers` 一串依赖
- **解法**：**直接转格式**，不动任何依赖

  ```python
  import torch
  from safetensors.torch import save_file
  sd = torch.load(bin_path, map_location="cpu", weights_only=True)  # torch 自己 load 不受限
  save_file({k: v.contiguous().clone() for k, v in sd.items()
             if isinstance(v, torch.Tensor)}, safe_path, metadata={"format": "pt"})
  ```

  命名照约定：CLIP 文本编码器 → `model.safetensors`；
  扩散组件 → `diffusion_pytorch_model.safetensors`。
- **教训**：⚠️ **`pip install -r requirements.txt` 里写 `>=` 的包，会装到远超预期的
  新版本**。这次 `transformers` 从预期的 4.4x 跳到 5.17。
  复现老仓库时应该**钉住当年的大版本**。
  另外：这类"格式兼容问题"往往**转换产物**比**改动环境**代价小得多。

---

### PS-04 `pkill -f 'xxx'` 会把自己的 shell 也杀掉

- **时间**：2026-09-12
- **现象**：远程执行
  `pkill -f 'curl.*Hunyuan3D'; pkill -f 'git clone'` 之后命令返回 `exit=-1`、**毫无输出**。
- **原因**：我这条命令是通过 `bash -lc '...'` 跑的，
  **它自己的命令行里就包含 `git clone` 这个字符串**，
  于是 `pkill -f` 匹配到了自己，把自己杀了。
- **解法**：先列 PID 再按 PID 杀，或者让模式不匹配自身：

  ```bash
  ps -eo pid,cmd | grep -E 'pattern' | grep -v grep | awk '{print $1}' | xargs -r kill -9
  ```
- **教训**：`pkill -f` 是**全命令行匹配**，在脚本里很容易自杀。
  排查"命令莫名 exit=-1 且无输出"时，先怀疑这个。

---

### PS-05 PowerShell 会把 here-string / `$(( ))` / 嵌套引号拆碎

- **时间**：2026-09-12
- **现象**：反复出现这三类失败：
  - `@'...'@` 多行 here-string 当参数传给原生命令 → 被按换行**拆成多个参数**
  - `` `$((b-a)) `` → `b-a : The term 'b-a' is not recognized`
  - `python -c "...math.cos(x)..."` → 引号被吃掉，Python 收到残缺代码
- **原因**：PowerShell 到原生程序的参数传递规则和 bash 差太远，转义层级一多就崩。
- **解法**：**一律落盘成文件**，然后让工具从文件读。

  ```powershell
  $cmd = @'
  echo hello
  a=$((1+2)); echo $a
  '@
  $cmd | Out-File -Encoding utf8 "$env:TEMP\x.sh"
  & $py scripts\remote.py run --file "$env:TEMP\x.sh"
  ```
  `scripts/remote.py` 专门为此加了 `--file` 参数。
- **教训**：⚠️ **跨 shell 传复杂命令，永远走文件**。
  在 PowerShell 里拼引号是纯粹的浪费时间。

### ENV-09 ⚠️ `bpy` 在 AutoDL 上根本装不了（两个独立原因叠在一起）

- **时间**：2026-09-13
- **触发命令**：`pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/`
  （HCCEPose README 的原话）
- **现象**：
  1. 在镜像自带的 Python 3.12 上：`ERROR: Could not find a version that satisfies the
     requirement bpy==3.6.0 (from versions: none)`
  2. 建好 py3.10 conda 环境、加上官方源后**还是同一句**。
  3. 直接 curl 那个索引页：`http_code=403 size=61375`，正文是 `<title>Just a moment...</title>`，
     响应头 `cf-mitigated: challenge`、`server: cloudflare`。
- **诊断**：把"包不存在"和"页面拿不到"区分开——`pip index versions` 的 "from versions: none"
  在 403 和被 Cloudflare 挡的情况下**长一样**。curl 一看就知道是后者。
  然后逐个试镜像：`download.blender.org` 直连 403、`source /etc/network_turbo` 后**仍 403**；
  清华 TUNA `mirrors.tuna.tsinghua.edu.cn/blender/` 返回 200，但目录里**只有
  `release/ source/ demo/`，没有 `pypi/`**；南大镜像同样；USTC 404。
  PyPI/阿里云上的 `bpy` 只有 cp311 和 cp313。
- **原因**：两条互相独立的路都被堵死：
  ① Blender 官方的 `bpy` wheel 只覆盖到 cp310 / cp311，而 AutoDL 镜像是 py3.12；
  ② `download.blender.org` 对 AutoDL 的出口 IP 弹 Cloudflare 人机验证，代理也绕不过。
- **解法**：**放弃 `bpy` pip 模块这条路**，改用 Blender 官方发行包——它自带 Python 3.10，
  且 `bpy` 在 Blender 进程内天然可导入。从清华 TUNA 下
  `blender-3.6.0-linux-x64.tar.xz`（256 MB，实测 17.5 MB/s），md5 与官方一致；
  把 blenderproc 装进它的 `3.6/python/`，用
  `blender --background --python <脚本>` 运行。
  这条路能成立的关键是 **HCCEPose 把 `blenderproc/__init__.py` 的 CLI 守卫改成了 `if True:`**，
  所以 `import blenderproc` 在普通 Python 里就可用。完整步骤见
  [docs/RENDER_SETUP.md](RENDER_SETUP.md)。
- **影响文件**：`docs/RENDER_SETUP.md`（新增）
- **教训**：**"pip 找不到包"先怀疑网络，别先怀疑包名。**
  `pip index versions` 的报错对 403 / 超时 / 真不存在是一视同仁的，多花 5 秒 curl 一下
  能看到真相。另外：**PyPI 镜像 ≠ 上游源镜像**，国内镜像站通常只镜像官方"发行版"目录，
  不会镜像 `pip` 源。

---

### ENV-10 ⛔ `pyrender` 把 `PyOpenGL` 精确钉死在 `==3.1.0`，导致整条 pip 命令解析失败

- **时间**：2026-09-13
- **触发命令**：把 `"pyrender==0.1.45"` 和 `"PyOpenGL==3.1.7"` 写在**同一条** `pip install` 里
- **现象**：

  ```
  The conflict is caused by:
      The user requested PyOpenGL==3.1.7
      pyrender 0.1.45 depends on PyOpenGL==3.1.0

  Additionally, some packages in these conflicts have no matching distributions available for your environment:
      pyopengl

  ERROR: Cannot install PyOpenGL==3.1.7 and pyrender==0.1.45 because these package versions have conflicting dependencies.
  ERROR: ResolutionImpossible
  ```

- **诊断**：`pyrender` 最后一次发版是 2021 年，`setup.py` 里写的是
  `PyOpenGL==3.1.0`（**精确等号**，不是 `>=`），而 3.1.0 只有 sdist、没有 wheel。
  最后那句 "no matching distributions" 是解析失败后的**副产物**，不是"镜像里没有 PyOpenGL"——
  pip 其实已经成功下载过 `PyOpenGL-3.1.7-py3-none-any.whl`。
- **原因**：pyrender 的过度严格的依赖声明 + pip "先整体解析、再安装"的策略。
- **解法**：**拆成两条命令**，让 pyrender 跳过依赖检查：

  ```bash
  "$PY" -m pip install "PyOpenGL==3.1.7" "freetype-py==2.5.1" "pyglet==2.1.16"
  "$PY" -m pip install --no-deps "pyrender==0.1.45"
  ```

  运行时 PyOpenGL 3.1.7 与 pyrender 0.1.45 兼容，那个 pin 只是当年保守。
- **影响文件**：`docs/RENDER_SETUP.md` §4.1
- **教训**：**pip 是先解析后安装的**——`ResolutionImpossible` 意味着**这条命令一个包都没装上**，
  不要以为前面 "Downloading ..." 过的包已经落盘了。涉及"老包钉死依赖"时，
  把可疑的那一个拆出来用 `--no-deps` 单独装。

- **⚠️ 为什么不能干脆不要 pyrender**：`write_bop()` 的 GT mask 和 `scene_gt_info.json` 是
  **用 pyrender 离屏渲染算出来的**（`BopWriterUtility.py` 里 `import pyrender` 全写在函数体内，
  所以导入期不报错、跑到写标注才炸）。这份 mask 是后面裁图 + 角点监督的输入。

---

### ENV-11 远端 `OMP_NUM_THREADS` 是非法值，Blender 一启动就报 libgomp

- **时间**：2026-09-13
- **触发命令**：`/root/autodl-tmp/blender-3.6.0-linux-x64/blender --version`
- **现象**：`libgomp: Invalid value for environment variable OMP_NUM_THREADS`
- **诊断**：`--version` 明明打出了 `Blender 3.6.0`，所以不是 Blender 的问题；
  环境变量是镜像预置的，值不是合法整数。
- **原因**：镜像环境变量污染。
- **解法**：跑之前 `export OMP_NUM_THREADS=8`（Blender 渲染本身只吃 GPU，
  这个变量主要给 OIDN 降噪和物理模拟用）。
- **影响文件**：`docs/RENDER_SETUP.md` §4.4
- **教训**：镜像预置的环境变量不可信，**第一次跑就把 `export` 写进命令里**，
  别指望"应该没问题"。

---

### DATA-07 ⚠️ 绕过 blenderproc CLI 直接 `import` 时，`Utility.temp_dir` 是空字符串

- **时间**：2026-09-13
- **触发命令**：`blender --background --python gen_pbr_data_demo.py`
- **现象**：还没跑到（预先读源码发现）。
- **诊断**：`blenderproc/python/utility/Utility.py` 里 `temp_dir = ""`，
  全仓库只有 `SetupUtility.setup_utility_paths(temp_dir)` 会赋值，
  而它只被 `command_line.py`（即 `blenderproc` CLI）调用。
  我们对准的路径是 `import blenderproc` 直接用，**不经过 CLI**。
  于是 `ObjectLoader.load_obj()` 里这一行会退化：

  ```python
  tmp_ply_file = os.path.join(Utility.get_temporary_directory(), model_name)
  # Utility.temp_dir 是 "" -> 结果是相对路径 "obj_000001.ply"
  ```

  也就是把 BlenderProc **改写过的 110 MB PLY 直接吐进当前工作目录**（cwd = 数据集目录）。
- **原因**：BlenderProc 默认假定自己由 CLI 启动。
- **解法**：在适配版脚本里显式设置（可用环境变量覆盖）：

  ```python
  from blenderproc.python.utility.Utility import Utility
  temp_dir = os.environ.get("BP_TEMP_DIR") or os.path.join(tempfile.gettempdir(), "bproc_temp")
  os.makedirs(temp_dir, exist_ok=True)
  Utility.temp_dir = temp_dir
  ```

- **影响文件**：`data/render_ws/gen_pbr_data_demo.py`
- **教训**：**把一个工具当库用时，先找出它"只有 CLI 才会初始化"的全局状态。**
  这类状态不会报错，只会让路径悄悄退化成相对路径。

### ENV-12 ⚠️ 无卡模式的 cgroup 内存上限只有 2 GB，而且 exit code 被管道吃掉了

- **时间**：2026-09-13
- **触发命令**：
  `blender --background --python gen_pbr_data_demo.py 2>&1 | tail -100; echo "exit=$?"`
- **现象**：日志停在 `load_bop_objs` 中途，**没有任何 traceback**，`echo` 打出来的是 `exit=0`，
  看起来像是"跑成功了但没产物"。
- **诊断**：
  1. 把 `| tail -100` 去掉、改成 `> log 2>&1` 后立刻现形：
     `bash: line 19: 2649 Killed "$BP/blender" ...`，真实退出码 **137**。
  2. 查 cgroup：`/sys/fs/cgroup/memory.max = 2147483648`（**2 GB**），
     `memory.events` 里 **`max 6246`** —— 撞了 6246 次上限。
     ⚠️ `free -g` 显示的 1 TB 是**宿主机**的，容器里完全不是这么回事。
  3. 反推内存去向：`ObjectLoader.load_obj()` 处理带纹理 PLY 时把整个文件读成字符串再做
     **两次 `.replace()`**，110 MB 的 PLY 在导入期间有 **~330 MB 的字符串副本**同时存活，
     再加上 Blender 的网格数据（89.1 万面）和 8 套 cc0 材质（48 张贴图）。
- **原因**：容器内存上限远小于网格规模所需。
- **解法**：给 `scripts/mesh_to_bop.py` 加 `--max-faces`，做保纹理简化。
  89.1 万面 → **13.4 万面**，PLY **110.6 MB → 29.1 MB**，峰值 RSS 落到 851 MiB。
- **影响文件**：`scripts/mesh_to_bop.py`、`data/render_ws/gen_pbr_data_demo.py`
- **教训**：⚠️ **`cmd | tail` 之后再 `echo $?` 拿到的是 `tail` 的退出码。**
  长任务的输出**永远重定向到文件**再读，否则会得到一个漂亮的假 0。
  另外：容器里**先看 `/sys/fs/cgroup/memory.max`，别信 `free`**。

---

### ENV-13 `compute_color_from_texture_per_vertex` 报 "Source texture ... doesn't exists"

- **时间**：2026-09-13
- **触发命令**：pymeshlab 里加载 `.glb` 后调 `ms.compute_color_from_texture_per_vertex()`
- **现象**：`PyMeshLabException Failed to apply filter ... Source texture
  "D:/.../texture_0" doesn't exists`
- **诊断**：pymeshlab 从 GLB 里读出的纹理是**内嵌的**，在 mesh 上只留了个内部名字
  `texture_0`，磁盘上并没有这个文件；滤镜按**文件路径**去找，自然找不到。
  （`m.textures()` 返回 `{'texture_0': <Image>}`，说明图确实在内存里。）
- **原因**：pymeshlab 的纹理滤镜走文件路径，不走内存里的 image。
- **解法**：绕开这个滤镜——自己用 numpy + PIL 按顶点 UV 采样
  （见 `data/render_ws/uv_splat.py`），或者干脆改用 `data/render_ws/make_cc0textures.py`
  那种"凭空造图"的路子。**不**要去改 pymeshlab 的纹理路径。
- **影响文件**：`data/render_ws/uv_splat.py`
- **教训**：库的报错说"文件不存在"时，先想清楚**它在找哪个文件**——
  从容器格式（GLB）读出来的资源经常只存在于内存里。

---

### DATA-08 ⚠️ 渲染出的黑色条纹：一路怀疑纹理，最后发现是**几何被简化撕开了**

- **时间**：2026-09-13
- **触发命令**：Cycles 渲染 `data/dji_action4/models/obj_000001.ply` + `obj_000001.png`
- **现象**：物体表面出现**黑色锯齿裂纹**，沿表面拓扑连成片，约覆盖 15~25% 的面积。

- **诊断（四次假设、四次验证，前三次都是错的）**：

  1. **假设一：简化破坏了 UV 缝。** → 用 `scipy.spatial.cKDTree` 量化：简化后顶点 UV 距原始 UV
     **最大 2.9 px**（2048² 图集），>2 px 的仅 0.6%。**排除**。
     佐证：`vertices with >1 distinct wedge UV: 0 / 646420`——UV 缝处完全未焊接，
     wedge→vertex 转换无损。

  2. **假设二：纹理采样本身坏了。** → 写 `data/render_ws/uv_splat.py`：按顶点 UV 从图集取色，
     做正交投影点云（纯 numpy+PIL，**不需要 Cycles，秒出**）。
     结果：**模型完全正确**——机身、镜头、屏幕、`ACTION 4K` 字样、红色 DJI 标、磁性卡扣全在。
     顺带确定了 V 轴约定 `row = (1-v)*H`（翻转过来是纯噪声）。**排除**。

  3. **假设三：图集空白区的噪声被采样到了。** → 光栅化 UV 三角形成掩码：
     UV 覆盖图集 **61.13%**，剩下 **38.87% 是黑白噪点**（Hunyuan3D-2 把 UV 岛之间的
     未使用区域填成了噪声）。写了 `scripts/repair_texture_atlas.py` 做标准 texture padding
     （`distance_transform_edt(return_indices=True)` 最近岛像素填充），
     空白区高频噪声 3.85% → 1.32%，岛内像素 diff = 0.0。
     **重渲染 —— 裂纹一模一样。所以这也不是根因。**

  4. **决定性实验：把贴图整个拿掉。** 给 `preview_object.py` 加 `BP_FLAT_MATERIAL=1`，
     断开 Base Color 的纹理连线、换成纯灰。
     **裂纹完全一样** → **这是几何问题，和纹理无关。**

  5. **量拓扑**（`ms.get_topological_measures()`）：

     | | 顶点 | 面 | 边界边 | 连通分量 | 非流形边 |
     |---|---|---|---|---|---|
     | 原始 | 646420 | 891184 | 394694 | **38705** | 0 |
     | `preserveboundary=True` | 380243 | 358830 | 394694 | 38705 | 0 |
     | `preserveboundary=False` | 191975 | 134434 | 196432 | 39704 | **1383** |

     这个网格是 **38705 个 UV 岛拼起来的**（所以 64.6 万顶点里大部分是重复的，
     焊接的话只需约 44.6 万）。UV 岛的边界本来**严丝合缝地贴在一起**，
     所以拓扑上"开放"但视觉上密闭。
     `preserveboundary=False` 允许坍缩边界边 → **各岛的边界各自往里缩 → 岛与岛之间裂开缝**，
     还制造出 **1383 条非流形边**。从裂缝看进去就是物体内表面 —— 全黑。

- **原因**：`preserveboundary=False` 撕开了 UV 岛拼合而成的网格。
- **解法**：改回 `preserveboundary=True`（代价是简化下限卡在 40.3%，见 `DATA-09`），
  并用 `--ply-precision 5` 把文本 PLY 压小（59.91 MB → 32.89 MB）来抵消面数增加带来的内存。
  图集 padding 保留 —— 它本身是对的（虽然不解决这个问题）。
- **影响文件**：`scripts/mesh_to_bop.py`、`scripts/repair_texture_atlas.py`、
  `data/render_ws/preview_object.py`、`data/render_ws/uv_splat.py`
- **教训**：
  - ⚠️ **"做减法"之前先确认网格的拓扑结构。** 这个网格有 38705 个连通分量，
    任何"允许移动边界"的简化都会把它撕碎。**先 `get_topological_measures()` 再选参数。**
  - ⚠️ **分离变量要彻底。** 我前三次都在"纹理"这个大类里换着法子查，
    真正一锤定音的是**把纹理整个删掉**这一步——一次渲染，零歧义。
    当你在同一类原因里换了三种假设都不对时，**下一件事应该是跳出这一类**，而不是换第四种。
  - 点云 splat 这类**几秒钟的中间表示**极其值钱：它一次就排除了整整一类原因。

---

### DATA-12 生成模型的纹理图集 38.9% 是黑色噪点（真实存在，但不是黑条纹的原因）

- **时间**：2026-09-13
- **现象**：把网格 UV 三角形光栅化成掩码后统计：**UV 只覆盖图集的 61.13%**
  （`preserveboundary=True` 的网格是 74.16%），
  剩下那片空白是**黑白噪点** —— Hunyuan3D-2 的纹理网络把 UV 岛之间的未使用区域填成了噪声。
- **原因**：生成模型的纹理后处理没有做 texture padding。
- **解法**：`scripts/repair_texture_atlas.py`，标准 padding 流程：
  光栅化 UV 掩码 → `scipy.ndimage.distance_transform_edt(~mask, return_indices=True)`
  给每个空白像素取**最近 UV 岛像素**的颜色 → 只对空白区轻度模糊。
  实测空白区高频噪声 **3.85% → 1.32%**，**岛内像素 diff = 0.0（一个没动）**。
- **影响文件**：`scripts/repair_texture_atlas.py`
- **教训**：这是**真实但次要**的问题——它主要影响缩小采样时的颜色偏移，
  不是黑裂纹的原因（见 `DATA-08`）。**把它的优先级排对了，才不会在它上面耗掉半天。**
  根治要么重跑纹理生成（要 GPU），要么改用顶点色（`uv_splat.py` 已证明顶点色是干净的，
  代价是丢 2048² 细节、文字会糊）。

---

### DATA-09 MeshLab 保纹理简化：`preserveboundary=True` 会把简化卡在 40% 且无法再降

- **时间**：2026-09-13
- **触发命令**：
  `ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=80000, preserveboundary=True, planarquadric=True)`
- **现象**：891184 面只降到 **358830 面（40.3%）**，离目标 80000 差得远；
  **重复跑第二遍、第三遍面数一模一样**（`358830 → 358830 → 358830`）。
- **诊断**：逐组试参数（见下表）。重复跑无效说明不是"一次降不够"，
  而是有一批边**被硬性拒绝坍缩**。
- **原因**：网格在 UV 缝处是开放边界（**38705 个连通分量**，见 `DATA-08`），
  `preserveboundary=True` 拒绝坍缩边界边；加上 `planarquadric=True` 进一步限制，
  于是卡在 40.3%。
- **解法**：⚠️ 一开始为了多降 3 倍改成了 `preserveboundary=False`，
  **结果把网格撕开、渲染出黑裂纹**（`DATA-08` 的决定性实验定位到）。
  最终**回到 `preserveboundary=True`**，改用 `--ply-precision 5` 压缩文本 PLY 来控制内存。
  实测矩阵：

  | 参数 | 面数 | 代价 |
  |---|---|---|
  | `preserveboundary=True, planarquadric=True` | 358830（40.3%），**重跑无效** | 无（**最终采用**） |
  | `preserveboundary=False, planarquadric=True` | 177836（20.0%） | 撕开 UV 岛 → 黑裂纹（`DATA-08`） |
  | `preserveboundary=False, planarquadric=False` | 100955（11.3%） | 同上，更严重 |
  | 再加 `preservenormal=True` | 134434（15.1%） | 同上 |

  `qualitythr` 在 0.1~1.0 之间对本网格**毫无影响**。
  事后用 KD 树验证过：简化后顶点 UV 距原始 UV 最大只有 2.9 px，**UV 没被弄坏**。

- **解法**：**采用 `preserveboundary=True`**（`DATA-08` 证明关掉它会撕开网格）。
  代价是面数只能降到 40.3%（358830 面，PLY 59.91 MB），
  于是再用 `--ply-precision 5` 把文本 PLY 压到 **32.89 MB** 来抵消内存（`ENV-12`）。
- **影响文件**：`scripts/mesh_to_bop.py`（`--max-faces`、`--ply-precision`）
- **教训**：
  - **"简化没到目标值"时，先试着重复跑一遍**：
    一模一样 → 是硬性约束挡住了，调参数；逐次下降 → 是迭代次数不够。
    这一个动作就能把问题分类。
  - ⚠️ **"降得更多"不等于"更好"。** 关掉 `preserveboundary` 能多降 3 倍，
    但代价是把网格撕碎——**参数选择必须回到拓扑上去验证**，不能只看面数。

---

### DATA-13 ⚠️ 多视图生成：形状和纹理的**最优输入不是同一组**

- **时间**：2026-09-14
- **触发命令**：
  `python run_generate.py --views front left back right`（4 视图，形状+纹理一起跑）
- **现象**：
  - **形状大幅变好**：三轴尺寸比从 `1.000/0.676/0.592` 变成 `1.000/0.642/0.472`，
    对比官方 `1.000/0.627/0.465` —— 第三轴误差从 **+27% 降到 +1.5%**。
  - **纹理反而崩了**：图集里满是灰色噪点，连 UV 岛内部都碎了，渲染出来整体发灰
    （本该是黑色机身）。
- **诊断**：把 `13.JPG` / `16.JPG` 拿出来单看——它们是 **¾ 侧视**，不是正交侧视
  （镜筒分别偏向画面左/右，而不是正对侧面）。
- **原因**：形状编码器和纹理投影对视角精度的敏感度完全不同：
  - **形状**只用轮廓和大致方位做条件，视角近似无所谓；
  - **纹理**是把每张图**按假设的视角方向投影**到网格上，
    把 ¾ 图当成正侧视投，纹理自然被拉花。
- **解法**：**形状和纹理分开取**。形状用 4 视图，纹理只喂 `front` + `back`
  （这两张是干净的正面/背面）。见 `docs/MODEL_REGEN.md` §2.2，
  脚本 `data/render_ws/run_texture_mv.py`（原来那个 `run_texture.py` 把 shape
  路径和视图都写死了）。
- **影响文件**：`data/render_ws/run_texture_mv.py`、`docs/MODEL_REGEN.md`
- **教训**：
  - ⚠️ **一个模型的不同子模块对输入质量的容忍度可以差很多。**
    "多喂几张"这个直觉对形状成立、对纹理不成立。
    多视图生成要**分别调参**，不要一把梭。
  - **改输入前先看一眼每张图的几何含义**（是不是正交视图、有没有开盖）。
    `14`/`17` 两张磁吸盖是开着的，直接排除——那不是刚体状态。

---

### DATA-15 ⚠️ 单模型数据集下，原脚本有一半场景只放 1 个物体

- **时间**：2026-09-14
- **触发命令**：`gen_pbr_data_demo.py`，`BP_NUM_OBJS=8`
- **现象**：`scene_gt.json` 每帧只有 **1 个实例**，`mask_visib/` 每帧只有 1 张；
  但日志明明写着 `loaded 8 object instances`。同一份脚本在 `BP_NUM_OBJS=3` 时是 3 个/帧。
- **诊断**：
  1. 先怀疑 BOP writer 的 `ignore_dist_thres=10` 把物体滤掉了 →
     grep 日志里 `ignored obj` 出现 **0 次**；又单独跑了一次物理模拟，
     8 个物体模拟后全部落在原点 0.16 m 内，**没有一个是远的**。排除。
  2. 回去看挑选物体的那段（**原脚本的逻辑**）：

     ```python
     if rand_s > 0.5:
         idx_l = np.random.choice(models_ids, size=30, replace=True)
     else:
         idx_l = np.random.choice(models_ids, size=min(models_ids.shape[0], 30), replace=False)
     ```

     我们只有 **1 个模型**，所以 `min(1, 8) = 1` —— **`rand_s <= 0.5` 的那一半场景只放 1 个物体**。
     `BP_SEED=7` 正好落在这一半，于是 8 帧全是 1 个实例。
- **原因**：原脚本的 "multi-class object picking mode" 在单模型数据集上退化成"只放 1 个"。
  它默认数据集里有几十个不同类别，从里面不重复地挑。
- **解法**：改成"先不重复地取现有模型，再重复铺满到 `num_objs` 个实例"：

  ```python
  n_distinct = min(models_ids.shape[0], num_objs)
  chosen = np.random.choice(models_ids, size=n_distinct, replace=False)
  idx_l = np.tile(chosen, int(np.ceil(num_objs / n_distinct)))[:num_objs]
  ```

  修改后退化成 1 个的情况消失（实测 3 个场景 × 4 帧全部是 8 个实例/帧）。
- **影响文件**：`data/render_ws/gen_pbr_data_demo.py`
- **教训**：
  - ⚠️ **借用别人的脚本时，要专门检查"在只有 1 个类别的数据集上会怎样"。**
    `min(n_models, n_objs)` 这种写法的退化行为非常隐蔽：不报错、数量对不上，
    而且**只在一半的场景里发生**（取决于一个随机分支），很容易当成偶发。
  - **随机分支要和种子一起看。** 同一份配置换个 seed 就从小数据变成大数据。

---

### DATA-14 ⚠️ 训练数据的 `depth/` 实际不可用（深度被量化成整数米）

- **时间**：2026-09-14
- **触发命令**：`gen_pbr_data_demo.py`（`BP_WRITE_BOP=1`）
- **现象**：写出的 `depth/*.png` 是 uint16，但**整张图只有 2~4 个不同值**
  （`0 / 10000 / 20000 / 30000`，按 `depth_scale=0.1` 换算即 `0/1/2/3 米`），
  物体在 0.4 m 处的结果直接变成 0。加日志确认：

  ```
  [render] depth[0]: shape (768, 1024) dtype uint8 min 0.0 max 3.0 unique 4
  ```

  **`bproc.renderer.render()` 返回的 `data["depth"]` 本身就是 uint8。**
- **诊断**：
  - `BopWriterUtility` 的换算是对的：`depth_mm = 1000 * depth; /depth_scale`。
    问题在上游——`depth` 已经是整数米了。
  - `enable_depth_output()` 确实设了 `output_file.format.file_format = "OPEN_EXR"`，
    但**没有设 `color_depth`**。
  - 读回来走 `load_output_file()` → `BlenderUtility.load_image()`
    → `imageio.imread(exr)[:, :, :num_channels]`；
    `trim_redundant_channels()` 只取第 0 通道、**不改类型**（有 docstring 保证）。
  - 试过在合成器里找 `CompositorNodeOutputFile` 把 `color_depth` 改成 `"32"`，
    **没匹配到节点**，没生效。
- **原因**：未完全定位到具体那一行。候选是 EXR 写入时用了 8 位、或
  `imageio.imread` 对这张 EXR 返回了整型。
- **解法**：**暂时不修**。理由是：
  - 我们的网络是 **RGB → 8 角点热图**，不读深度；
  - BOP 的任何**训练**指标也不消费训练集深度（VSD 用的是**测试集**深度）。
  如果以后要做 RGB-D（HCCEPose 的 FoundationPose 精化路径）就**必须**修，
  修的方向：绕开 `render()` 的深度加载 —— 自己挂一个 File Output 节点把 EXR
  写到可控目录，再用 `OpenEXR`/`imageio` 直接读，或者干脆用 pyrender
  （mask 那条路径已经在用）按 GT 位姿重算深度。
- **影响文件**：`data/render_ws/gen_pbr_data_demo.py`（脚本里已写明这段注释）
- **教训**：**"文件写出来了"不等于"数据是对的"。**
  BOP 这份数据集里 `rgb/ scene_gt.json scene_camera.json mask_visib/` 全部正确、
  路径齐全，但 `depth/` 是废的——只有**看数值分布**才发现（`len(np.unique(...))` 一行）。
  凡是"看起来应该连续"的量（深度、置信度、点图），落盘后都该查一下 unique 数。

---

### DATA-16 ⚠️ 训练集里遮挡严重不足：95% 的实例完全可见

- **时间**：2026-09-14
- **触发命令**：数据集验收脚本（统计 `scene_gt_info.json` 的 `visib_fract` 分布）
- **现象**：500 帧 / 5000 个实例跑完后统计：

  ```
  visib_fract:  median 1.000   min 0.000   max 1.000
                完全可见(=1.0) 占 95.0%      严重遮挡(<0.2) 仅 0.9%
  mask 像素数 > mask_visib 的实例: 0 / 59 采样
  ```

  也就是 **`mask/` 和 `mask_visib/` 几乎完全一样**——物体之间基本没有互相遮挡。
- **诊断**：文件完整性、旋转矩阵正交性（`max|RRᵀ−I| = 8.5e-7`）、内参、尺寸全部正常，
  所以不是标注错了，是**场景本身的分布问题**。
- **原因**：`sample_pose_func` 把 10 个物体撒在 `x,y ∈ ±0.15 m`、`z ∈ [0, 0.6]`，
  掉到平地之后被互相推开、**摊成一层**；房间墙在 `±2 m`，根本兜不住它们。
  物体 70×45 mm，10 个在 0.09 m² 的区域里本该有 ~35% 覆盖率，但物理模拟把它们铺开了。
  HCCEPose 原脚本就是这个行为（他们甚至撒 30 个，摊得更开）。
- **解法**（2026-09-15 实施，**部分修复**）：给渲染脚本加了**料箱**（4 面低矮围栏）
  + `randombin` 装载模式。详见下面「三次试错」。
- **影响文件**：`data/render_ws/gen_pbr_data_demo.py`
- **教训**：
  - ⚠️ **"数据造出来了"不等于"数据分布是对的"。**
    500 帧、5000 实例、文件齐全、旋转矩阵正交——所有"结构检查"都过，
    但**关键量的分布**（这里是 `visib_fract`）才是决定它能不能训出目标能力的东西。
    凡是"声称能解决 X"的方法（HCCEPose 的卖点就是遮挡鲁棒），
    **都要在训练集里统计 X 的分布**，而不是假设它会自然出现。
  - 这条也解释了为什么对比 BoxDreamer 有意义：它渲干净单体 + **训练时在线合成遮挡**，
    遮挡强度是可控参数；我们把 clutter 烘进渲染，一旦物理把它们摊平，就**没有回头路**。

#### 修复过程：三次试错才拿到能用的装载方式

| 方案 | 结果 |
|---|---|
| ① 收紧撒点范围 + 料箱 + 保留 `sample_poses` | ❌ **卡死**：`sample_poses` 要求初始摆放**互不相交**，而 16 个 70×45 mm 物体塞进 16 cm 箱子里**不存在这种解** → 卡在重试里十几分钟一帧不出，日志刷满 `completely inside` 警告 |
| ② 叠柱子 + 自由落体塌落 | ❌ **弹出箱子**：从 0.57 m 高的柱子掉下来，实测 16 个里 **10 个落到箱外**（`\|xy\|` 最大 0.19 vs 箱内半宽 0.08），最后还是摊平 |
| ③ 网格摆两层 + 上层错开半行 | ❌ **零遮挡**：上层全落进下层间隙、变单层平铺；而且 yaw-only 让 12 个物体姿态完全相同（都镜头朝上），姿态多样性也没了 |
| ④ **箱内随机撒 + `uniformSO3`，绕开相交检查** | ✅ 有遮挡了，且保住姿态多样性 |

**④ 的实测结果**（2 场景 × 3 帧 × 12 实例 = 72）：

```
visib_fract 分布 = [0.0 × 16,  0.04, 0.05, 0.35, 0.45,  1.0 × 52]
  完全可见 72.2%  |  明显遮挡(0.3~0.8) 2.8%  |  完全不可见 25.0%
```

#### ⚠️ 但仍然不够，而且这里有根本性的限制

过滤掉不可见实例后，**明显遮挡只有约 4%**（修复前 ~0.9%，好了 4 倍但仍很低）。

**根本原因**：我们只有**一个物体类别**，尺寸完全相同。两个一样的平板叠起来只有两种结局——
**并排（零遮挡）或正对齐（完全盖死）**，"部分遮挡"在物理上很难自然产生。
HCCEPose 的真实场景里箱中有多个不同尺寸的类别，小的压在大上面才有部分遮挡。

**正解是 BoxDreamer 的做法：训练时在线合成遮挡**（`occlusion_objs` —— 把别的物体
paste 到图上）。它同样只有单类别渲染，靠 paste 造遮挡。我们的数据集带 `mask_visib`，
完全可以做：随机取几帧的物体像素（用 mask 做 alpha），随机缩放贴到目标物体上。
**遮挡强度就成为可控参数，而且连续。** 列为待办。

---

### DATA-18 ⚠️ HCCEPose 的 `mask/` 和 `mask_visib/` 内容完全相同

- **时间**：2026-09-14
- **现象**：逐实例对比两个目录的掩码像素数，**48/48 完全相等**。最极端的一个实例
  `visib_fract = 0.068`（`px_count_all = 6665`，`px_count_visib = 452`），
  但 `mask/*.png` 里只有 **452** 个像素 —— 和 `mask_visib/` 一模一样。

  `mask/` 按 BOP 约定应该是 **amodal（全部像素，含被遮挡部分）**，这里写成了可见掩码。

- **诊断**：`scene_gt_info.json` 里的 `px_count_all` / `px_count_visib` 是**正确**的
  （所以 `visib_fract` 可信），说明 pyrender 那边算了两份，只是落盘时 `mask/` 写错了。
- **原因**：HCCEPose 用 pyrender 换掉了 BlenderProc 原本的 mask 计算，这条路径上 `mask/` 的写入有 bug。
- **解法**：**不改**（那是第三方代码）。对我们的影响：`bbox_visib` 取自 `scene_gt_info.json`，
  是**正确**的，而训练只用 `bbox_visib` + `scene_gt.json`，所以不受影响。
  但要记住 **`mask/` 不可信，只能用 `mask_visib/`**；将来做训练时遮挡 paste 增强时也只能用后者。
- **影响文件**：无（记录用）
- **教训**：**两个文件名不同不代表内容不同。** 发现"遮挡好像为零"时，
  99% 的人会信 `mask/`，但真正可信的是 `scene_gt_info.json` 里算出来的数值。

---

### DATA-19 ⚠️ 数据集缺可见性过滤，完全不可见的实例会变成噪声样本

- **时间**：2026-09-15
- **现象**：`BOPPBRDataset._build_index()` 把 `scene_gt.json` 里**所有**实例都收进来，
  没有任何可见性判断。加了料箱之后有 **25% 的实例 `visib_fract = 0`**（完全被盖住）。
- **为什么是 bug**：裁剪是按 `bbox_visib` 做的。可见像素为 0 时那个框退化成 0 大小，
  裁出来是一块**无关的背景**，却配着一个"正确"的角点标签 —— 纯噪声，
  而且会以 25% 的比例污染训练集。
- **解法**：加 `min_px_visib`（默认 64）和 `min_visib_fract`（默认 0.10）两个过滤阈值，
  从 `scene_gt_info.json` 读。实测在箱装数据上丢掉 16 个 `px_count_visib<64`
  + 2 个 `visib_fract<0.1`，保留 54/72，并打印过滤统计。
- **影响文件**：`src/datasets/bop_pbr.py`、`src/datamodules/corner_pose_datamodule.py`、
  `configs/datamodule/bop.yaml`
- **教训**：**"标注里有"不等于"能用"。** BOP 的 `scene_gt.json` 会记录**所有**实例，
  其中一部分在任何实际意义上都不可见。**索引阶段就该按可见性筛掉**，
  而不是指望后面的裁剪/网络去消化它们。

---

### CODE-07 ⚠️ 把 `metrics.csv` 的**行数**当成步数，训练耗时估错了 10 倍

- **时间**：2026-09-15
- **现象**：测训练吞吐时数 `metrics.csv` 的行数增长：`57 行 / 60 秒`，
  于是算出 "0.95 步/秒 → 每 epoch 5.6 分钟 → 500 epoch 要 46 小时"，
  并据此建议**砍掉正在跑的渲染**给训练让 CPU。
- **真相**：`metrics.csv` 的 `train/loss_step` 是 **每 10 步记一行**
  （`configs/trainer/default.yaml` 里 `log_every_n_steps: 10`）。
  所以要读 **`step` 列**，不是数行数。实测 `step` 列 60 秒涨 **570**：

  ```
  9.50 步/秒  ->  每 epoch(313 步) 33 秒  ->  500 epoch 约 4.6 小时
  ```

  （冒烟测试里 Lightning 自己报的 `11.39it/s` 一直是对的，我没信它反而信了自己算的。）
- **诊断**：`grep 'global step' 日志` 显示 "Epoch 4, global step 1480" ——
  而按 "0.95 步/秒" 算，那时最多跑 300 步。**两个数对不上时就该回头查自己的算法**。
- **解法**：用
  `tail -1 metrics.csv | awk -F, '{print $3}'` 前后各取一次 `step` 列相减。
- **影响文件**：无（诊断方法）
- **教训**：
  - ⚠️ **日志/指标的"行数"通常不等于"步数"。** 先看列名、看采样间隔，再算速率。
  - ⚠️ **框架自己报的吞吐（`it/s`）比自己从日志推算的可信。** 我手上有 Lightning 的
    `11.39it/s`，却用一个自己发明的、样本间隔错误的代理指标推翻了它。
  - **两个独立来源的数字对不上时，先验算自己的，而不是先怀疑系统。**
    1480 步 vs "最多 300 步" 这种矛盾非常刺眼，当时应该立刻停下来查。

---

### ENV-17 ⚠️ 长任务挂在"工具的后台作业"上会随会话一起被杀 —— 白等 4 小时

- **时间**：2026-09-15
- **现象**：本地 v2 渲染驱动起了 2 小时只出了 6 个场景，之后**再没动过**。
  查下来：**驱动进程不存在**，最后一次写日志是 **02:55**，而当时已经是 **07:15** ——
  **任务早死了 4 个多小时，我一直在"等"一个不存在的进程。**
  更糟的是它**不留任何痕迹**：没有报错、没有崩溃记录、日志还是正常收尾的样子。
- **原因**：驱动是用**工具的后台作业**（`run_in_background`）启动的。作业生命周期
  绑在工具会话上，会话状态一变（上下文压缩、作业表重置）就被连带杀掉，**而且不会通知**。
- **解法**（两条一起用）：
  1. **脱离会话启动**：`Start-Process powershell.exe -WindowStyle Hidden -PassThru`
     拿独立 pid，才真正和工具会话解耦。
  2. **断点续跑**：驱动启动时扫描输出目录**跳过已完成的单元**，并把带时间戳的状态
     追加写到 `driver_state.txt` —— 一眼就能看出"最后动过是什么时候"。
- **影响文件**：本地渲染驱动（`data/render_ws/`，不入库）
- **教训**：
  - ⚠️ **"我启动了长任务" ≠ "长任务在跑"。** 必须有**外部可观测的心跳**
    （状态文件时间戳 / pid 是否存活），不能靠"没报错"来推断。
  - ⚠️ **依赖工具或会话生命周期的后台作业，不适合跑小时级任务。**
  - ⚠️ 这是本项目**第三次**栽在"以为在跑其实没跑"上：
    `PS-07`（解析期失败，一行没执行）、`ENV-12`（`cmd | tail` 把真实退出码吃掉）、
    以及这次。**共同点都是：我没有去验证"它真的在动"。**

---

### ALGO-07 ⚠️ 用错评价口径：拿 6D 位姿的 ADD 去判一个 2D 角点任务

- **时间**：2026-09-15
- **现象**：v1 训练完，第一次评测打开了 `solve_pose`，报出

  ```
  add = 157.84 mm   （物体直径 81 mm，BOP 判据是 ADD < 0.1×直径 = 8.1 mm）
  rot_err_deg = 28.06°   trans_err_mm = 156.55 mm   acc_5cm5deg = 0.540
  ```

  据此我下了结论「6D 位姿完全失败、差了 20 倍」。
- **诊断**：**本项目的交付物是 2D 的 8 个角点，不是 6D 位姿**，而且不需要相机内参
  （psd 确认，见 `TRAINING.md` §5.1）。`ADD` / `rot_err` / `trans_err` 都要跑 PnP，
  属于**另一个任务**的指标。

  换成角点口径后，同一个 checkpoint 的真实表现是：

  ```
  中位误差 3.34 px（256 裁剪）   均值 16.04 px   差 4.8 倍
  PCK@0.05 80.8%   PCK@0.15 90.1%   框 IoU 中位 0.961
  失败率(>0.1 对角线) 12.8%
  ```

  **不是"完全失败"，是"87% 很准 + 13% 完全崩"的重尾分布。**
- **原因**：把参考实现（BoxDreamer/HCCEPose 走 6D 位姿）的指标口径，
  套到了我们这条「只出 2D 角点」的路线上。而且 70 mm 小物体的位姿对 2D 误差
  极其敏感 —— 5 mm 的 2D 误差就能让 ADD 远超判据，**这个指标对我们是失真的**。
- **解法**：新增 `scripts/eval_corners.py`，用角点任务的指标：
  归一化误差的**中位数 + p90/p95**、`PCK@t`、**失败率**、**框 IoU**、逐角点拆解。
  `metrics.solve_pose` 保持 `false`。
- **影响文件**：`scripts/eval_corners.py`（新增）、`configs/model/metrics/default.yaml`
- **教训**：
  - ⚠️ **评测口径必须跟着交付物走，不能跟着参考实现走。**
    照抄框架时最容易犯的错：把上游的**必需项**和**评价标准**一起搬过来，
    而我们的交付物其实更窄。
  - ⚠️ **均值会掩盖重尾。** 中位 3.34 / 均值 16.04 —— 只看均值会以为"普遍偏 16 px"，
    实际是"绝大多数 3 px，少数飞到画面外"。**这类分布必须看中位数和分位数。**
  - ⚠️ **看失败模式要看分位数和最差样本，不要看均值。** 均值降一点可能只是
    尾巴短了一点；真正关心的是"崩掉的比例有没有降"。

---

### ENV-15 ⚠️ 本机 Windows 并行渲染打爆"提交内存"（commit limit），3/4 分片崩溃

- **时间**：2026-09-15
- **现象**：本机开 4 个 Blender 分片渲染，约 40 分钟后 3 个崩掉：

  ```
  Malloc returns null: len=3836072 in new_bhead, total 8693725308
  Error: Failed to read blend file '': Missing DNA block
  Error   : EXCEPTION_ACCESS_VIOLATION
  ```

- **诊断**：物理内存还有 14.9 GB 空闲，但 **committed bytes = 45.9 GB / limit 46.5 GB**。
  Windows 的提交上限 = 物理内存 + 页面文件（本机页面文件上限只有 15 GB）。

  **关键：每个 Blender 进程的内存会随场景数持续增长** —— 实测从 3.6 GB 涨到 6.0 GB
  （5 个场景），崩溃那个到了 8.7 GB。4 个并发 × 涨到 8+ GB = 32 GB，
  再加上 QQ / Edge / DingTalk / 微信 / wps 等约 15 GB，直接越过提交上限。
- **解法**（两条一起用）：
  1. **每个场景用一个全新的 Blender 进程**（`BP_NUM_SCENES=1`，外面套调度循环），
     内存每场景归零。这样进程内不再累积到 8 GB，稳定在 4~5 GB。
  2. **并发数降到 2**（本机 31.7 GB 内存的前提下）。
     调度器里再加一道看护：`committed > 34 GB` 就暂停派发新任务。
- **注意**：`Get-Process` 的 `WorkingSet64` 看不出真相，要看 **`PagedMemorySize64`（提交）**，
  以及全局 `\Memory\Committed Bytes` / `\Memory\Commit Limit` 两个计数器。
- **影响文件**：本地渲染调度（`data/render_ws/`，不入库）
- **教训**：
  - ⚠️ **"物理内存还空着"不代表不会 OOM。** Windows 上真正卡住进程的是**提交上限**，
    而它受**页面文件大小**约束，不只是物理内存。
  - ⚠️ **长跑进程要怀疑内存泄漏。** 单场景测试好好的（3.6 GB），跑 13 个场景就完蛋。
    **测试规模必须覆盖真实规模**，否则测的是"能不能跑"而不是"能不能跑完"。
  - ✅ **"每个单元一个全新进程"是廉价且有效的隔离手段** —— 泄漏、状态污染、
    崩溃丢进度，一次全解决。代价只是每单元多几十秒启动。

---

### ENV-16 ⚠️ AutoDL 无卡模式的资源：1 核（配额 0.5）/ 2 GB / 无 GPU

- **时间**：2026-09-15
- **触发**：考虑把渲染放到无卡模式实例上省 GPU 机时
- **实测**：

  ```
  nproc        = 1
  cpu.max      = 50000 100000      -> 实际只有 0.5 核！
  memory.max   = 2147483648        -> 2 GB
  GPU          = 无
  ```

- **结论：渲染和训练都不能放在无卡模式。**
  - **2 GB 内存连网格都载不进来**：单个 PLY 30 MB / 32 万面 × 12 实例要 ~4 GB
    （已在 `ENV-12` 被 cgroup OOM 杀过）
  - 0.5 核 + 无 GPU：1040 帧按之前实测速率要 70+ 小时
- **无卡模式适合做什么**：装环境、改代码、跑自检用例、小文件整理。
  **不适合**：渲染、训练、任何要载入大模型/大网格的事。
- **教训**：**"无卡模式便宜"是有代价的，先看清资源再决定**。
  同类坑：不要以为 `nproc` 就是可用核数 —— 要看 `cpu.max`。

---

### PS-07 ⚠️ `.ps1` 文件里的中文会让脚本在**解析阶段**就失败

- **时间**：2026-09-15
- **现象**：写了一个带中文提示的 PowerShell 脚本，运行时直接报一堆语法错：

  ```
  The string is missing the terminator: ".
  Missing closing '}' in statement block or type definition.
  The assignment expression is not valid.
  ```

  报错行号指向中文字符串的**中间**。
- **诊断**：`write` 工具写的是 UTF-8（无 BOM），而本机 **PowerShell 5.1 按 GBK 读 `.ps1`**。
  中文字节被错误解码后，引号被吃掉/错位，于是整个文件语法崩掉。
  **注意这是解析期失败 —— 脚本一行都没执行**（所以那次渲染一帧都没出）。
- **解法**：**`.ps1` 一律写成纯 ASCII**（注释和提示都用英文）。
  需要中文时用 `-Encoding UTF8` 显式写 BOM，或者干脆改用 `pwsh`/Python。
- **影响文件**：所有本地 `.ps1`
- **教训**：
  - ⚠️ **"脚本报语法错"要先看编码，不是先改代码。** 报错位置在中文字符串中间
    是很明显的信号。
  - ⚠️ **解析失败 = 什么都没做。** 我当时以为"渲染已经在跑"，实际一帧都没出
    —— 浪费了一轮排查。**下结论前先确认脚本真的执行了。**

---

### PS-06 ⚠️ PowerShell 把子进程的 stderr 当异常，长任务报"退出码 1"但其实是 0

- **时间**：2026-09-14
- **触发命令**：`python run.py --config-name=train.yaml ... | Select-Object -Last 40`
- **现象**：训练明明跑完、日志最后一行是 `[INFO] All done. Exiting.`、没有任何 traceback，
  但工具报 `[exit code: 1]`。
- **诊断**：把输出重定向到文件（不经管道、不经 `Select-*`）后取 `$LASTEXITCODE`：

  ```powershell
  & python run.py ... *> $log
  Write-Host "真实退出码 = $LASTEXITCODE"      # -> 0
  ```

  日志里确实有 `NativeCommandError`，但它在 `CategoryInfo` 那一行——
  是 **PowerShell 对 stderr 的包装**，不是程序抛的异常。
  Lightning 的 `Seed set to 42`、git 的进度条都会走 stderr。
- **原因**：PowerShell 的原生命令输出处理：子进程只要往 stderr 写东西，
  在 `2>&1` 或管道场景下就会被包装成 `RemoteException`，
  且**管道的退出码会变成下游 cmdlet 的**（和 `cmd | tail` 是同一类坑，见 `ENV-12`）。
- **解法**：判长任务只看**产物 + 日志内容**；要拿真实退出码就**重定向到文件**再读 `$LASTEXITCODE`。
- **影响文件**：所有 PowerShell 调用点
- **教训**：这是本项目**第二次**踩同一类坑（第一次是 bash 的 `cmd | tail`，记在 `ENV-12`）。
  **"退出码"在管道里是不可信的**，两个 shell 都会骗你。

---

### ENV-14 Hydra struct 模式下，往已有配置组里加**新键**必须写 `+`

- **时间**：2026-09-14
- **触发命令**：
  `python run.py --config-name=train.yaml trainer.limit_train_batches=2`
- **现象**：
  ```
  omegaconf.errors.ConfigAttributeError: Key 'limit_train_batches' is not in struct
  full_key: trainer.limit_train_batches
  hydra.errors.ConfigCompositionException: Could not override 'trainer.limit_train_batches'.
  To append to your config use +trainer.limit_train_batches=2
  ```
- **原因**：`configs/trainer/default.yaml` 里**没有** `limit_train_batches` 这个键；
  Hydra 默认 struct 模式不允许凭空加键。
- **解法**：加 `+` 前缀 —— `+trainer.limit_train_batches=2`。
  **覆盖已存在的键**（如 `trainer.max_epochs=1`）不用 `+`。
- **影响文件**：所有命令行 override
- **教训**：报错信息最后一行 Hydra 已经把答案写出来了（`use +trainer.xxx=2`）。
  **Hydra 的报错值得读到最后一行。**

---

### CODE-06 ⚠️ 验证脚本自己的检查公式写错，报出一个假问题

- **时间**：2026-09-14
- **现象**：`scripts/verify_dataloader.py` 报
  `✗ #4165 热图峰值偏离角点 4.2px (>stride 4.0)`，看起来像数据有问题。
- **诊断**：去读 GT 热图的**实际实现**（`src/models/utils/data_processing.py`）：
  `centers = corner_2d * ratio`，热图在**整数网格**上取值，所以峰值格 = `round(centers)`。
  而我的检查用的是 `peak_cell * stride + (stride-1)/2` —— **凭空多加了 1.5 px**。
  最大真误差应是 `√2 × stride/2 ≈ 2.83 px`，加上这 1.5 px 正好 4.3 px，和报出的 4.2 吻合。
- **原因**：**验证脚本自己引入了偏移**，不是数据的问题。
- **解法**：直接比**整数格子索引**：
  `expected = round(corner_2d * ratio)`，`peak = argmax`，要求 `|peak - expected| ≤ 1 格`。
  改完误差 **0 格**，全部通过。
- **影响文件**：`scripts/verify_dataloader.py`
- **教训**：
  - ⚠️ **写验证脚本时，检查公式必须来自被测代码的实现，不能凭直觉。**
    "图像坐标 → 热图坐标" 差半个格子的偏移极难靠肉眼发现，却会让整张检查表失去意义。
  - **报错时先怀疑检查器。** 尤其是"差一点点就过"的那种失败（4.2 vs 阈值 4.0）——
    真 bug 通常差得远，差一点点的往往是自己的容差/公式有问题。

---

### CODE-05 ⚠️ numpy 广播写错一个 `[:, None]`，撑出两个 512³ 数组把进程打成 OOM

- **时间**：2026-09-13
- **触发命令**：`python /root/autodl-tmp/bp_ws/make_cc0textures.py /root/autodl-tmp/cc0textures-512`
- **现象**：`bash: line 47: 1477 Killed "$PY" .../make_cc0textures.py`（**exit 137**）。
  前面 6 个材质都正常，第 7 个（`Marble012`）直接被杀。
- **诊断**：逐步打印峰值 RSS——前 6 个材质峰值只有 **72 MiB**，第 7 个还没打印就死了。
  于是范围缩到 `_marble()` 一个函数。
- **原因**：

  ```python
  # ❌ 错：[:, None] 加在了整个和上
  vein = np.abs(np.sin((n * 7.0 + np.linspace(0, 4, h))[:, None] * 2.2))
  ```

  `n` 是 `(512,512)`，`np.linspace(0,4,h)` 是 `(512,)`，两者相加**沿最后一维广播**仍是
  `(512,512)`；再 `[:, None]` 就变成 **`(512,1,512)`**——之后的 `sin()` 和 `np.abs()`
  各产生一个 **1 GiB 的 float64 数组**。2 GB 的 cgroup 直接爆。

- **解法**：把行斜坡**先**变成列向量再参与运算，并加形状断言：

  ```python
  rows = np.linspace(0.0, 4.0, h)[:, None]          # (h, 1)
  vein = np.abs(np.sin(n * 7.0 + rows * 2.2))       # (h, w)
  ```

  同时给每个材质的输出加断言：

  ```python
  for name, arr, want in (("color", color, (size, size, 3)), ...):
      if arr.shape != want:
          raise ValueError(f"{asset}: {name} has shape {arr.shape}, expected {want}")
  ```

- **影响文件**：`data/render_ws/make_cc0textures.py`
- **教训**：**数组形状错误不该由 OOM killer 来报。**
  在"造图/造数据"这类循环里，**每个产物加一句 shape 断言**成本几乎为零，
  却能把一次 137 变成一行清晰的 `ValueError`。
  另外 `[:, None]` 要**紧跟在需要变列向量的那个数组**后面，不能图省事写在括号外。

### ALGO-08 ⚠️ focal 的**正样本数为 0**：centernet 热图没做峰值归一化

- **时间**：2026-09-17
- **现象**：损失消融的 E2 变体（`heatmap_style=centernet` + `heatmap_loss=focal`）
  训练指标**卡在 `corner_err_px = 123.6` 且 25 个 epoch 不动**，
  `pck@0.05 ≈ 0.01` —— 基本等于没学到。
- **诊断**：`focal_loss` 的正样本判据是 `pos_mask = (gt >= 1.0)`，而
  `make_heatmap_target` 的 centernet 分支**直接用了 `gaussian_2d` 的输出，没做峰值归一化**。
  角点中心是**浮点数**（如 37.3），最近格子到中心的距离不为 0，于是峰值只有 0.98x：

  ```
  boxdreamer   峰值 1.000000   值>=1.0 的格子数 = 2
  centernet    峰值 0.983113   值>=1.0 的格子数 = 0     <- 正样本 0 个！
  ```

  **正样本数为 0 时 focal 只剩负样本项，网络被训成"哪里都没有角点"。**
- **为什么 boxdreamer 风格没这个问题**：它峰值归一化到 1 之后再映射到 `[-1,1]`，
  所以 `(gt >= 1.0)` 总能匹配。**BoxDreamer 的 `make_bbox_features` 末尾同样做了
  峰值归一化 —— 我们照抄时漏了这一步**，而默认配置一直用 boxdreamer 风格，
  所以这个 bug 从最初实现 centernet 起就潜伏着，直到切到 focal 才暴露。
- **解法**：centernet 分支加 `g = g / g.max().clamp(min=1e-6)`。
  修复后峰值 1.000000、正样本 4 个；**E2b 从 123.6 → 23.81 px**。
- **影响文件**：`src/models/utils/data_processing.py`
- **教训**：
  - ⚠️ **"照抄"要抄完整。** 上游在某个不起眼的位置做了一步归一化，漏掉它不会报错，
    只会让损失悄悄退化成"只有负样本"。
  - ⚠️ **损失里凡是出现 `>= 1.0`、`== 0` 这类硬阈值判据，都要去验证它到底能匹配到几个元素。**
    一行 `print(((gt>=1.0)).sum())` 就能省下几小时。
  - ⚠️ **换损失/换表征必须先跑一个 30 秒的单元检查**（GT 的取值区间、峰值、正样本数），
    而不是直接开 100 epoch 的训练。这次是训练跑完了才发现。

---

### ALGO-09 ⚠️ 「无解样本」：裁剪图里**没有目标物体**，却要求模型预测它的角点

- **时间**：2026-09-17
- **现象**：v2（带料箱造遮挡）训练完全不收敛，v1 的模型在 v2 上也崩 18 倍。
  逐场景扫描发现 52 个场景的 median 误差从 **7.77 px 一直散到 120.99 px**。
- **诊断**：把所有被证伪的假设（标签错位、几何不一致、姿态分布、亮度、
  `BP_NUM_WORKER`、裁剪框、提取算子）逐一排除后，**直接看裁剪图**才找到：

  `data/results/stage3_train_v1/v2_hard_occlusion.jpg`（按可见度排序的 5 个样本）：

  | `visib_fract` | 可见像素 | 模型误差 | **裁剪图里有物体吗** |
  |---|---|---|---|
  | 1.00 | 32,473 | **2.8 px** | 完整 |
  | 0.50 | 16,247 | 32.9 px | 露一半 |
  | 0.36 | 12,310 | 39.1 px | 露一部分 |
  | **0.24** | 10,335 | **228.1 px** | **几乎没有** |
  | **0.11** | 4,191 | **241.9 px** | **完全没有，纯空白** |

  **第 4、5 行的裁剪图是空白的米色平面，没有任何物体，却要求模型输出 8 个具体坐标。**
  这不是"难样本"，是**自相矛盾的样本**：图像里零证据，标签给了唯一答案。
  模型只能学出"从箱壁纹理到这些坐标"的**伪映射**。

  **注意第 4 行有 10,335 个可见像素（相当多）照样崩** —— 问题不是像素不够，
  而是**那些像素不在裁剪图里**。
- **根因**：`bbox_visib`（可见部分的框）退化。真正可见的部分是画面一角的小斑，
  裁剪围着那块小斑做，自然取不到物体。而过滤阈值 `min_visib_fract=0.10`
  **恰好把 `visib=0.11` 的样本放了进来**。v2 里这类样本占 **18.1%**。
- **解法**（尚未实施，见 `docs/RESULTS.md` §8）：
  1. `visib_fract` 下限 0.10 → **0.25~0.3**
  2. 裁剪**以 `bbox_obj` 为中心**（已加开关 `datamodule.crop_use_bbox_obj`）
  3. 排除 `bbox_obj` 部分出画的实例（**截断 ≠ 遮挡**：遮挡的位姿仍可外推，
     截断的角点坐标在图像外**根本不存在**）
  4. **加一道裁剪图检查**：目标掩码在裁剪图中心区域必须有足够像素 ——
     这是**直接验证"目标在图里"**，比看 `visib_fract` 可靠
- **影响文件**：`src/datasets/bop_pbr.py`、`scripts/qa_scene.py`
- **教训**：
  - ⚠️ **统计量排除法有极限，最后还是要看图。**
    我在 bbox 统计、亮度分布、姿态分布、标签自洽性上花了大量时间，全部通过；
    **一张 `imshow` 就定位了。** "数据自洽性检查通过"不等于"数据是对的" ——
    自洽性检查用的是同一套（可能错的）前提。
  - ⚠️ **mean 和 median 要一起看，而且是看它们的关系。**
    `mean ≈ median ≈ 120` ⇒ 均匀全错；`mean = 4×median` ⇒ 重尾。
    两种形态指向完全不同的原因。
  - ⚠️ **"难"和"无解"必须区分。** amodal 标注本身没错（被遮挡的角点就该预测），
    但**图像里必须存在能够定住位姿的证据**。`visib_fract` 只是代理量，
    真正的判据是"裁剪图里有没有把目标框住"。

---

### DATA-20 ⚠️ 裁剪用 `bbox_visib`（可见部分）却配 **amodal** 标签 —— 语义不一致

- **时间**：2026-09-17
- **现象**：在排查 v2 时发现，同一份代码里两个选择互相矛盾。
- **诊断**：

  ```
  裁剪用   bbox_visib  —— "看得见的部分"的框
  标签是   3D 盒 8 角点投影 —— amodal，包含被挡住的部分
  ```

  物体被严重遮挡时 `bbox_visib` 会退化：框偏移、变小、甚至**跑到画面外**
  （实测 `bbox_visib = [-512, 0, 78, 39]`）。裁出来是无关背景，
  标签却还是这个物体的角点。
- **为什么一直没暴露**：v1 的数据里 **95.3% 的实例 `bbox_visib == bbox_obj`**
  （几乎没有遮挡），所以两种选择等价。v2 加料箱造遮挡后立刻引爆。
- **解法**：加 `datamodule.crop_use_bbox_obj`（默认 `false` 保持 v1 历史行为，
  新训练建议 `true`）。**但实测打开它对 v2 帮助很小（122.26 → 120.67）** ——
  因为过滤后只有 **2~4%** 的样本框是脏的。**它修的是语义正确性，不是 v2 那个量级的问题。**
- **影响文件**：`src/datasets/bop_pbr.py`、`src/datamodules/corner_pose_datamodule.py`、
  `configs/datamodule/bop.yaml`
- **教训**：
  - ⚠️ **裁剪框的语义必须和标签的语义一致。** 标签是 amodal 的，框就该是 amodal 的。
  - ⚠️ **"在旧数据上看不出差别"不代表两种写法等价** —— 可能只是旧数据的分布
    碰巧落在两者重合的那一段。**造新数据时，先想想它会把这个假设推到哪个区间。**

---

### ALGO-10 ⚠️ 实验设计三连错：种子混淆 / epoch 未对齐 / 看错指标

- **时间**：2026-09-17
- **现象**：一轮里连犯三次，每次都白花了 GPU 时间，而且**前两次都产出了错误结论**。

  **① A/B/C 单变量消融被「随机种子」混淆**

  想隔离"料箱 / 相机仰角"，三个变体**共用 `seed=424242`**：

  | 变体 | mean | PCK@0.05 |
  |---|---|---|
  | A：无料箱 + 仰角5°（最接近 v1） | 56.16 | 47.9% |
  | B：料箱 + 仰角5° | 50.44 | 53.4% |
  | C：无料箱 + 仰角22° | 49.49 | 52.1% |

  据此得出"料箱和仰角都不是原因"。**后来用完全相同的设置、只换 seed（777001）
  重渲一次，结果是好的（mean 6.79 / PCK@0.05 92.5%）。**
  **⇒ 那个结论无效，已撤回。**

  **② 跨对比时 epoch 数没对齐**

  E0/E1/E2 全训 **100 epoch**，v1 训 **257 epoch**。
  "E0 比 v1 差"**不能**确定是增强的锅，也可能只是没训够。
  这两个变量（增强、训练时长）**从未分开测过** ——
  "增强有害"这个说法到现在**还没被证明**。

  **③ fit 测试看错了指标**

  想验"数据是否自相矛盾"，却用了 600 个样本（从零训 200 epoch 本来就不该记住），
  还设了 `val_ratio=0.1`、读的是 **`val/corner_err_px`（验证集）**。
  结果 **fit_bin 65~78 px / fit_nobin 44~48 px —— 两个都"失败"，什么也没证明。**

- **共同根因**：
  **先跑再想。没有先问"这个实验的输出会是什么形状、它能不能区分我的两个假设"。**
- **解法（写进流程）**：
  1. **消融实验必须多 seed** —— 单 seed 的单变量消融在渲染这类高方差任务上不可信。
     至少 3 个 seed，看的是**分布**不是单点。
  2. **跨对比必须对齐所有其它变量**，尤其是训练时长。想隔离 A 就先把 B 固定住。
  3. **"能不能拟合"要看训练误差，而且要用极小样本**（32 个、`val_ratio=0`）
     **+ 必须有已知可用的对照组**（对照组也失败 ⇒ 测试本身错了，别信结论）。
     `train/loss_fine` = `SmoothL1(soft_argmax(heatmap), corner_2d)`，
     单位是图像像素，就是训练集上的角点误差。
- **影响文件**：`docs/RESULTS.md` §5、§6
- **教训**：
  - ⚠️ **跑之前先写下"如果假设成立，输出会是什么样；如果不成立，又会是什么样"。
    两种情况分不开的实验，不要跑。**
  - ⚠️ **有随机性的流程（渲染、采样）做消融，必须多 seed。**
    "单变量"只是指被测变量只有一个，不包括把随机性也压成单点。
  - ⚠️ **对照组不是可选项。** 一个没有对照的"失败"结果什么都证明不了 ——
    这次 fit 测试就是活例。

---

### PS-08 ⚠️ `pgrep -f <pattern>` 匹配到**自己的命令行**，把脚本自己的 shell 杀掉

- **时间**：2026-09-16 ~ 09-17（**一轮里连踩三次**，是本次最高频的问题）
- **现象（三次都不同，同一个根因）**：

  1. 脚本里 `pgrep -f 'loss_ablation.sh' | xargs kill -9` → **脚本自己的 shell 一起死**，
     `[remote] exit=-1`，后面的步骤全部没执行。
  2. `pgrep -f 'exp_name=train_v4'` 报"还有 2 个进程"，其实是**它自己的命令行**
     含有这个字符串（假阳性），导致我误判任务还在跑。
  3. `run_now.sh` 里 `pgrep -f 'exp_name=fitc_'` → **匹配到调用它的那个 wrapper shell**
     （wrapper 的命令行里含脚本正文，正文里就有 `exp_name=fitc_`）→ 把自己杀了。

- **诊断**：`pgrep -f` 匹配的是**完整命令行**。而
  - `/bin/bash -c "... pgrep -f 'xxx' ..."` 的**命令行本身就含 `xxx`**；
  - 用 `remote.py run --file script.sh` 时，脚本正文被塞进 shell 的 `-c` 参数里
    → **脚本正文里出现的任何字符串都会出现在自己的命令行里**。

  所以只要被搜的字符串出现在**脚本正文**里，就会匹配到自己。

- **解法（按可靠性排序）**：
  1. **pidfile（最稳）**：启动时 `echo $! > x.pid`，之后只认 pidfile 里的 pid。
  2. **用 `/proc/<pid>/cmdline` 逐个核对**，并**显式排除 `$$` 和 `$PPID`**：
     ```bash
     ME=$$; for p in $(pgrep -f PAT); do [ "$p" = "$ME" ] && continue; ... done
     ```
  3. **`pgrep -x <进程名>`**（精确匹配进程名，不匹配命令行）—— 但拿不到脚本名。
  4. **数个数用 `pgrep -f PAT | wc -l`，不要用 `pgrep -cf PAT`** —— 见下面那条。

- **⚠️ 附带发现：`pgrep -c` 在"0 匹配"时会同时打印 `0` 并以退出码 1 结束。**
  于是 `N=$(pgrep -fc PAT || echo 0)` 得到的是**两行的 `0\n0`**，
  字符串比较 `[ "$N" = "0" ]` **永远为假** —— 等待循环空转满上限（400×45s = 5 小时）。
  **这是本轮第二次因为"等待条件写错"白等**（第一次是 `ENV-17` 的会话被杀）。
  正确写法：`N=$(pgrep -f PAT 2>/dev/null | wc -l)`。

- **影响文件**：`%TEMP%\dsh_x1.sh`、`dsh_iso.sh`、`dsh_fit_correct.sh`、`dsh_e4.sh`、
  `dsh_run_now.sh`，以及多个驱动脚本
- **教训**：
  - ⚠️ **"还有 N 个进程在跑"这类判断，来源必须唯一且明确**（pidfile 或精确进程名）。
    模糊的字符串匹配会同时骗过你两次：**让你以为任务还在跑，又让你把自己的 shell 杀掉。**
  - ⚠️ **诊断/编排脚本要用只读方式写。** 别在排查脚本里 `kill` ——
    排查脚本自己出错时，损失会从"没查到"升级成"把环境搞坏"。
  - ⚠️ **凡是"等待某个条件成立"的循环，必须先单独验证那个条件判断是对的。**
    一次 `pgrep` 的行为误判，代价是 5 小时空转 —— 比写一个单元检查贵得多。

---

### ENV-18 ⚠️ `git pull` 返回 503 静默失败，训练**照常启动**跑的是旧代码

- **时间**：2026-09-17
- **现象**：在实例上一键脚本里是 `git fetch → 启动训练`。`git fetch` 返回
  **`error: 503`**（GitHub 暂时不可用），脚本没有中断，**训练正常启动了**。
  但仓库还停在 `24d511b`，**没有新的增强模块** —— 等于白跑 3 小时的旧配置。
- **诊断**：脚本用的是 `git fetch ... ; git reset --hard origin/main`，
  但**没有检查 `git fetch` 的退出码**，也没有校验"新文件是否存在"。
  `git fetch` 失败后 `origin/main` 仍是旧引用，`reset` 就把旧代码当成"最新"了。
- **解法**：
  1. `git fetch` **重试若干次**，失败则回退到 codeload tarball
  2. **启动训练前校验仓库版本和关键文件**：
     `git log --oneline -1` + `test -f src/datasets/utils/aug_boxdreamer.py` +
     `grep 新配置键 configs/...`
  3. **实测**：第二次重试就成功了（503 是暂时的）
- **影响文件**：所有"拉代码 + 启动训练"的实例脚本
- **教训**：
  - ⚠️ **一键脚本里的网络步骤必须检查退出码**，"没报错"不等于"成功了"。
  - ⚠️ **启动长任务前必须校验"我要的代码真的在"**，而不只是"命令返回 0"。
    一个 `git log --oneline -1` + 关键文件存在性检查，成本几秒，
    能避免几小时的白跑。
  - ⚠️ **`data/` 是 gitignore 的，所以渲染脚本不在仓库里。**
    在实例上跑渲染要确认脚本路径（`/root/autodl-tmp/bp_ws/`），
    别用仓库路径 —— 报错是 `Blender ... Python file ... could not be opened`。

---

### DATA-21 · PLY 用非标准 UV 属性名，Blender 静默丢掉 UV

**现象**：把 BOP 的 `obj_000001.ply` 导进 Blender 渲染，出来是**一片纯白**，
没有任何纹理。查材质：`原材质数 = 0  UV 层 = []`。

**原因**：我们的 PLY 是 VCGLIB 生成的，UV 属性名是**非标准的**
```
comment TextureFile obj_000001.png
property float texture_u          <-- 不是标准的 s / t / u / v
property float texture_v
```
**Blender 的 PLY 导入器认不出，直接忽略，不报错。**
（trimesh 能正确解析：UV 覆盖 [0,1]，只有 2.5% 的顶点是零）

**修法**：先用 trimesh 转成 OBJ/GLB 再导入（`data/render_ws/ply_to_glb.py`）。

**教训**：**"渲染出来没有纹理"要先查 UV 层存不存在**，不要先怀疑贴图路径。
导入器丢掉属性是静默的。

---

### DATA-22 · Blender 的 OBJ/GLTF 导入器默认 Y-up，会换掉机身坐标轴

**现象**：转成 GLB 再导入后，bbox 从 `Y[-22.43,22.43] Z[-16.55,16.55]`
变成 `Y[-16.55,16.55] Z[-22.43,22.43]` —— **机身的 Y(高 44.87) 和 Z(深 33.09) 被换掉了**。
OBJ 也一样。

**原因**：OBJ/GLTF 约定是 Y-up，Blender 是 Z-up，导入器自动做转换。

**修法**：显式指定 `bpy.ops.wm.obj_import(filepath=..., forward_axis="Y", up_axis="Z")`。

**教训**：**导入后第一件事是核对 bbox**。轴被换掉之后，
"在 +Z 面贴平面"会贴到错误的面上，而且**看起来像是位置算错了**，极难反查。

---

### CODE-08 · `matrix_parent_inverse` 抵消父级缩放，平面大了 1000 倍

**现象**：给物体挂屏幕平面（父级带 `mm2m` 的 0.001 缩放），
平面的世界尺寸报 **60200 × 36900 mm**，期望 60.2 × 36.9 —— **正好 1000 倍**。

**原因**：父级时写了
```python
pl.parent = parent
pl.matrix_parent_inverse = parent_mw.inverted()     # <-- 错
```
Blender 的 `child.matrix_world = parent.matrix_world @ matrix_parent_inverse @ matrix_basis`。
把 `matrix_parent_inverse` 设成父级矩阵的逆，等于**把父级变换整个抵消**，
平面就退化成"用局部单位直接当世界单位"。

**修法**：**只设 `parent`，不要碰 `matrix_parent_inverse`**（保持单位阵）。
这样局部坐标(mm) 会被父级正确地缩放成米。

**教训**：Blender 里"保持世界变换"和"在父级局部空间里定位"是**两种相反的需求**，
UI 的 Set Parent 做的是前者，程序里赋 `.parent` 做的是后者。**别把 UI 的行为带进代码。**

---

### CODE-09 · BlenderProc 跑在 Blender 内嵌 Python 里，**没有 torch**

**现象**：渲染脚本里 `from src.datasets.utils.screen_content import ...`
直接 `ModuleNotFoundError: No module named 'torch'`。

**原因**：这条 import 会先执行 `src/datasets/__init__.py`，
而它 import 了 `bop_pbr.py`，后者需要 torch。
**BlenderProc 用的是 Blender 自带的 Python，没有 torch。**

**修法**：**按文件路径加载**，绕开整条包导入链：
```python
spec = importlib.util.spec_from_file_location("screen_content", "<abs path>.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
```

**教训**：**渲染脚本里任何 `src.*` 包导入都是雷**。
渲染进程和训练进程是两套 Python 环境，**只有纯 numpy/cv2 的模块能共享**。

### DATA-23 · `get_bound_box()` 刚导入时返回全 0 —— 尺寸归一化静默失败

**现象**：加了 `0/6` 个 Objaverse 物体，日志里只有一行汇总，
**没有任何失败原因**，查了半天。

**原因**：刚 `import` 完的 GLTF 物体，BlenderProc 的 `obj.get_bound_box()`
返回**全 0**（depsgraph 还没评估）。而我把 `cur <= 1e-9` 当成"尺寸非法"跳过 ——
于是 6 个全被静默丢掉。

**修法**：
- 改从 **mesh 顶点直接算世界包围盒**（大网格子采样到 ~4000 点，避免慢）
- **每个失败分支都打印原因** —— 这次的教训就是"静默失败最贵"

**教训**：**「全 0 的包围盒」是 Blender 里非常常见的坑。**
凡是 import 之后立刻读几何量的地方，要么先 `bpy.context.view_layer.update()`，
要么干脆从顶点自己算。**别信刚导入对象的缓存几何量。**

---

### CODE-10 · `join_objects_many_list()` 返回悬空引用

**现象**：处理 8 个物体全部失败：
```
AttributeError: 'NoneType' object has no attribute 'vertices'
```

**原因**：`bproc.object.join_objects_many_list(objs)` 返回的包装对象可能是
**悬空引用** —— join 会**删掉**被合并的 object，返回值的 `.blender_obj.data`
变成 `None`，后面读顶点就崩。
（而 GLTF 导入常常产生**一个父 EMPTY + 多个 mesh 子对象**的结构，
正好会走到 join 这条路径。）

**修法**：
- 改用 **raw bpy**：`select_all(DESELECT)` → 逐个 `select_set(True)` →
  `active = bos[0]` → `bpy.ops.object.join()`。**active 的包装对象仍然有效**
- 加 `_is_mesh()` 防御检查（`.blender_obj` 存在**且** `.data` 非 None）
- join 失败时退路是"保住顶点最多的子网格"（会丢部件，但不崩）
- 每个失败分支打印原因

**教训**：**BlenderProc 里凡是"会删除对象"的 API（join / merge / delete），
返回值的有效性都必须验证**，不能假设它指向的还是活的 object。

### PS-09 ⚠️ 中文 `grep` 模式在**命令替换**里静默返回 0（今天踩了 4 次）

**症状**：`$(grep -c '位置校验通过' "$L")` 返回 `0`，但同一行直接 `grep '\[check\]'`
明明能打印出 `[check] 场景 0 位置校验通过`。于是我把「14/14 全通过」误读成
「0/14，一个都没过」，差点去查一个不存在的问题。

**根因**：`grep -c` 在 `$(...)` 里、配合脚本文件的编码/locale，中文模式匹配失败。
失败是**静默的** —— `grep -c` 匹配不到就规规矩矩输出 `0`，exit code 也是 1 而非崩溃，
所以外层 `set +e` 的脚本一路往下跑，把 `0` 当成真实计数。

**正确做法**：**匹配英文/ASCII 关键词，不要匹配中文。**
```bash
# ❌ 会静默返回 0
N=$(grep -c '位置校验通过' "$L")

# ✅ 匹配日志里的英文标记
N=$(grep -c '\[check\]' "$L")
N=$(grep -c 'Objaverse' "$L")     # 别用 '模式 cover'
```

**代价**：今天 4 次误判（3 次把"全通过"读成"0 通过"，1 次把 props 读成"0 行"，
实际 props 每场景 3/3 全部加载成功）。**每次都要多花一轮去核实一个假问题。**

**推广**：**任何"计数型"检查都不要匹配中文。** 中文只用于给人看的输出，
不用于机器判断。这与 `PS-08`（`pgrep -f` 自匹配）是同一类问题：
**检查工具本身出错时是静默的，比被检查的对象出错更危险。**

---

### ENV-19 ⚠️ 杀掉 driver 的 `bash` **不会**杀掉它的 `blender` 子进程 → 两个渲染抢 GPU

**症状**：要停掉正在跑的渲染，`kill` 掉 driver 脚本的 bash 之后以为停了，
重新启动一个 → `nvidia-smi` 显示 `blender=4`（BlenderProc 的 worker 子进程），
**两个渲染在抢同一张 4090**，两边都变慢，日志混在一起。

**根因**：`bash render.sh` 的子进程 `blender` 不受 `kill <bash_pid>` 影响 ——
它会被 `init` 收养继续跑。

**正确做法**：**先杀叶子，再杀父**。
```bash
# 1. 先杀所有 blender（含 BlenderProc worker）
for pid in $(pgrep -x blender); do kill -9 "$pid"; done
# 2. 再杀 driver 的 bash（按 cmdline 精确匹配，见 PS-08 的教训）
for pid in $(pgrep -x bash); do
  C=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$C" in *bp_ws/myjob*) kill -9 "$pid";; esac
done
sleep 4
# 3. 必须核实真的清零了才重启
[ "$(pgrep -c -x blender)" = "0" ] || { echo "没杀干净，中止"; exit 1; }
```

**⚠️ 而且这个匹配逻辑必须放在【独立脚本文件】里**，不能写在 SSH wrapper 里 ——
否则 wrapper 的 cmdline 含 `bp_ws/myjob`，`pgrep -x bash` 会匹配到它自己
（`PS-08` 的同一个坑）。放在 `killall.sh` 里就安全，因为它的 cmdline 是
`bash killall.sh`，不含目标模式。

---

### ENV-20 ⚠️ 环境变量设在 `remote.py put` **之后** → 上传静默失败

**症状**：一个 PowerShell 块里先 `git commit`、再 `remote.py put`、
最后才 `$env:AUTODL_HOST='...'`。结果：
```
[remote] exit=127
缺少环境变量 AUTODL_HOST。先设置：...
sed: can't read /root/autodl-tmp/bp_ws/geo6.sh: No such file or directory
```
**上传根本没发生**，但后面的 `remote.py run` 因为环境变量此时已设好而**成功连上**，
跑了一个**不存在的脚本**。

**根因**：PowerShell 是顺序执行的 —— `put` 那行运行时 `$env:AUTODL_HOST` 还是空的。
而 `remote.py` 缺变量时 `exit 127`，被 `set +e` / 后台作业吞掉，看不出问题。

**正确做法**：**一个 SSH 段落里，环境变量永远放在最前面。**
更稳的做法是把凭据读取放进 `remote.py` 自己（读 `.env`），
这样调用方就不需要"记得先设变量"。

**同类**：`ENV-18`（`git fetch` 503 静默失败后训练照常启动跑旧代码）。
**共同的教训：一个"本该失败却没失败"的步骤，比一个明确报错的步骤危险得多。**
所以每一步之后都要**验证产物真的存在**，而不是假设它成功。

---

### ENV-21 ⚠️ 无卡模式想跑 BoxDreamer：权重能下，但**推理一定被 SIGKILL**

**时间**：2026-09-23（用户问"无卡模式能不能用 BoxDreamer 预测目标视频"）

**结论：不能。** 但**准备工作全部可以做完**，而且这一步的排查过程本身有四个可复用的发现。

#### 一、能做的（已全部完成并校验）

| 项 | 结果 |
|---|---|
| BoxDreamer 代码 + 子模块 | ✅ `three/dust3r` + `three/dust3r/croco` + `three/GroundingDINO` 全部 init |
| DUSt3R 权重 | ✅ 2,285,005,731 B，`zipfile.testzip()` 通过（1005 条目） |
| BoxDreamer 权重 | ✅ `yyh929/BoxDreamer` 的 `BoxDreamer-vitb.safetensor`（354.6 MB / 177 张量）+ `-reproduce` 变体 |
| GroundingDINO | ✅ `groundingdino_swint_ogc.pth` 0.69 GB + `grounding-dino-tiny`（HF 11 文件） |
| 目标视频 | ✅ 290 MB 上传完成 |

**⚠️ 校验大权重不能用 `torch.load`** —— DUSt3R 2.29 GB 在这个 2 GB 上限下必然 OOM。
用**只读元数据**的办法：`.pth` 是 zip，`zipfile.ZipFile(p).testzip()` 就能验完整性；
`.safetensor` 用 `safetensors.safe_open` 只读 header。

#### 二、🔴 推理为什么不行（逐步定位）

`python -u` 每步 flush，结果：

```
STEP boot                             RSS=  9 MB
STEP imported torch                   RSS=364 MB
STEP imported transformers            RSS=555 MB
STEP model loaded                     RSS=609 MB
STEP frame read ok=True (2464,3248,3) RSS=714 MB   <- 读 3248x2464 的帧没问题
STEP   W=64 processor done (0.4s)     RSS=682 MB
                                      <- 死在这里：model(**inputs)
真实退出码 = 137（SIGKILL）
```

**在输入只有 64×48 的情况下第一次前向就被 SIGKILL**，而且 **det_w 取 256 / 384 / 512
全都一样死** ⇒ **不是图像太大**，是这个栈在 2 GB 上限下跑不了前向。

**`oom_kill=0` 但 `memory.max` 命中 66,962 次** ⇒ 杀进程的是 **AutoDL 平台层的监管**，
不是内核 cgroup OOM killer（后者会累加 `oom_kill`）。
另外 `memory.current` 里**含页缓存**（实测 `file` 项占 385 MB，正是 HF 权重的缓存）
⇒ 真正留给进程的余量**远小于 2 GB**。

**⇒ 与 `ENV-16`（1 核配额 0.5 / 2 GB / 无 GPU）一致：无卡模式只能做纯 CPU 的准备与
纯 numpy/json 的分析，任何 ViT 级模型的前向都不行。**

#### 三、⚠️ `from_pretrained` 必须配 `HF_HUB_OFFLINE=1`，否则表现为"卡死"

第一次跑时日志刷满：
```
'[Errno 99] Cannot assign requested address' thrown while requesting HEAD
  https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/main/processor_config.json
Retrying in 8s [Retry 5/5].
```
`from_pretrained` 会去 HF 做**检查更新**的 HEAD 请求，在 turbo 代理下报 `Errno 99`，
然后**每个文件重试 5 次 × 8 秒**，几分钟里一个输出都没有 —— 看着像卡死。

**权重已经全部缓存好时，就该强制离线：**
```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 一个网络请求都不发
```
**效果**：载入从"卡几分钟"变成 **4 秒**。**离线模式不要再 source network_turbo**（徒增拖慢）。

#### 四、⚠️ 我的仪表本身也是坑：块缓冲 + SIGKILL = 日志丢失

探测脚本原先用普通 `print`，重定向到文件时是**块缓冲**；进程被 SIGKILL 时
**缓冲区直接丢掉**，于是"最后打印的那一行"根本不是真正的死点 ——
我因此一度以为它死在读帧。**必须 `python -u` 或每行 `flush=True`。**

**教训（与 `ENV-12` 同族）**：
- ⚠️ **"卡住"和"被静默杀掉"在日志上可以长得一模一样**，必须拿**真实退出码**
  （`cmd | tail` 拿到的是 `tail` 的退出码，见 `ENV-12`）+ **逐步骤 flush**。
- ⚠️ **内存类故障要先看 `memory.current` 里页缓存占了多少** ——
  限额是"进程 + 页缓存"共享的，只看进程 RSS 会低估。
- ⚠️ **`oom_kill=0` 不等于"没被杀"**：平台层监管杀进程不会累加 cgroup 计数器。

**影响文件**：`/root/autodl-tmp/bd/`（代码 + 权重 + 视频 + 探测脚本）；
新增仓库脚本 `scripts/detect_gdino_frames.py`（原 `detect_grounding_dino.py`
把路径硬编码成 Windows 路径，Linux 上根本不能用）。

---

### ENV-22 ⚠️ 循环 SFTP 搬大文件：远端中途关机 → 18/36 全失败，而且**没有校验就分不清"已完成"和"半截"**

- **时间**：2026-09-23（把实例上的东西备份到本地）
- **现象**：36 个 checkpoint（每个约 140 MB，共 5.3 GB）用
  `foreach { remote.py get }` 循环搬。**前 18 个成功，后 18 个连续全部失败**，
  而且循环自己还报 `exit code 0`（失败被 `if (Test-Path)` 的 else 分支吞了，
  只有 `Write-Output` 一行提示）。
- **诊断**：`NoValidConnectionsError: Unable to connect to port 14401`
  —— **远端关机了**（不是文件问题，也不是限流）。前 18 个是关机前抢下来的。
- **暴露的三个问题**（这才是要记的）：
  1. **没有尺寸校验**。循环版用 `Test-Path` 判断"是否已下载"，
     而**传了一半的文件也满足 `Test-Path`** —— 下次会跳过它，永久坏掉。
     本项目已经栽过一次：截断的 ckpt 让 `torch.load` 报
     `PytorchStreamReader failed reading zip archive`。
  2. **不记住失败项**，得人肉从日志里扒出还缺哪些 18 个文件。
  3. **一个文件一次 SSH 握手**，慢，且容易触发服务端限流。
- **解法（本次采用）**：新写 `scripts/fetch_remote_files.py`，三个要点：
  - 先**一次 SSH** 用 `stat` 把全部远端文件大小取回，再逐个传
  - 每个文件传完**比对大小**，不符就删掉重试（`--retries`，默认 3）
  - 已存在**且大小正确**的才跳过 ⇒ **这才是真正的断点续传**
  - 结束打印缺哪些并以退出码 1 结束，便于接自动化
  - 另有 `--dry-run`：只比对大小、报缺口，不下载
- **教训**：
  - ⚠️ **"文件存在"不等于"文件完整"。** 跨机传大文件必须比大小 ——
    `Test-Path` / `os.path.isfile` 判断不了这个。
  - ⚠️ **批量长任务要把"缺口"当成一等输出**：结束时明确列出缺哪些 + 非零退出码。
    否则失败会淹没在滚动日志里（本次 18 个失败，脚本自己还报成功）。
  - ⚠️ **远端随时会没。** 长搬运要能随时中断、随时续，且续的时候不能依赖
    上一次的日志。清单文件（`CKPT_DOWNLOAD_LIST.txt`）比日志可靠。
  - ✅ 与之配套：`INSTANCE_MANIFEST.txt` 把**"远端到底有什么"**固化成一份可离线
    查阅的清单（含权重下载 URL），所以即使远端没了，也清楚缺什么、怎么补。

---

### DATA-24 ⚠️ 干扰物摆在物理**之前** → `sample_poses` 重试风暴，整体慢 3.6 倍

**症状**：50 场景的渲染预计要 15 小时（**18 分钟/场景**）。但量了 rgb 文件的
写入时间戳发现：
```
04:48:31 / :32 / :33     <- 20 帧在 3 秒内全写完
05:05:58 / :59 / 06:00
```
**⇒ 渲染本身只要 3 秒，时间全花在每场景的 SETUP。**

**根因**：日志里 `It took 830 tries to place obj_000001.087` +
大量 `Detected that ... is completely inside ...`。
**Objaverse 干扰物是静态的，但摆在了 `sample_poses()` 之前**，
于是采样器必须躲开这些静态物体，最多重试 853 次 —— 每次重试都要拿
646K 顶点的相机网格做相交检查。

**修法**：把干扰物的摆放**挪到 `simulate_physics_and_fix_final_poses()` 之后**。
干扰物是静态的（`enable_rigidbody(False)`），本来就不需要参与物理，
挪到物理之后（相机位姿已定）**完全等价**，而且：
- `sample_poses` 看不到它们 → 不再重试
- 物理看不到它们 → 更快

**效果**（2 场景 A/B 实测，验证过才全量重跑）：
```
改前:  900 ~ 1380 秒/场景（15~23 分钟）
改后:   433 秒/场景（7.2 分钟）      sample_poses 重试: 2 / 2 / 7 / 42 / 56
```
**⇒ 快 2~3 倍。50 场景从 15 小时降到 6 小时。**

**教训**：**"渲染慢"不等于"渲染慢"。** 先量每个阶段的时间戳，
再决定优化哪里。我一开始以为要降分辨率/降采样，
实际上真正的问题是**一个摆放顺序**。

---

### DATA-25 ⚠️ 相交检查是「防爆炸」和「造遮挡」之间的零和 —— 四组尝试全部失败

**背景**：`v1` 训练集里 95.6% 的实例完全可见（中位 `visib_fract` = 1.000），
所以模型在真实视频的遮挡样本上崩（严重遮挡失败率 **45%**）。想加遮挡。

**四组尝试，全部失败**：
```
配置                    实例数  中位visib  >=0.1   部分遮挡  严重<0.3
tight (spawn_half .06)    72    1.000    100.0%     0.0%     0.0%   <- 收紧反而更不重叠！
mid   (spawn_half .09)    72    1.000    100.0%     4.2%     0.0%
stack (步距 .060)         24    1.000    100.0%     0.0%     0.0%
stack (步距 .075)         72    1.000     95.8%     1.4%     4.2%
```

**根因**：`sample_poses()` **强制物体互不相交**。
- **有相交检查** → 物体永不重叠 → 遮挡恒为 0（所以"收紧撒点范围"只会让它们
  **更紧凑地不重叠**，反而降低遮挡概率）
- **没相交检查** → 深度穿透 → 物理爆炸（`GEO2` 那次：物体被抛到 z = −18.8 m，
  8.5 小时渲染 + 4 小时训练全废）

**⇒ 这是个零和：相交检查是"防爆炸"和"造遮挡"之间的硬矛盾。**

**出路（`GEO6` ✅）**：**让干扰物静态地压在目标上。**
静态物体的重叠**不产生任何力** —— 所以既能遮挡又绝不会爆炸。
给 `add_objaverse_props` 加 `target_objs` / `cover_ratio`：
随机挑一个目标，把干扰物静态摆在它上方偏一点。

```
配置              实例数  中位visib  可用    部分遮挡  完全<0.1  严重<0.3
v1（对照）         5000    1.000     99.4%     3.4%     0.6%     1.1%
cover 0.5            72    1.000     97.2%     8.3%     2.8%     6.9%   <- 采用
cover 0.0            72    1.000     91.7%     5.6%     8.3%     8.3%   <- 太狠
```
`cover_ratio=0.5`：严重遮挡 **6.3 倍于 v1**，可用率仍 **97.2%**。

**教训**：**当一个约束（相交检查）同时服务两个目标时，先问"它到底在为谁服务"。**
这里它在防爆炸，而爆炸的根源是"动态物体 + 深穿透"。
把物体改成**静态**，就同时解开了两个约束 —— 不需要在零和里找平衡点。

---

### CODE-11 · 完全遮挡的实例 `bbox` 会退化（宽或高 ≤ 0），画图直接崩

**症状**：带 GT 可视化时 `PIL.ImageDraw.rectangle` 报
`ValueError: x1 must be greater than or equal to x0`。

**根因**：`visib_fract = 0.00` 的实例（完全被挡），BOP 写出的
`bbox_obj` 宽或高是 **0 或负数**。`ImageDraw.rectangle` 要求 `x1 >= x0`。

**修法**：画之前加保护。
```python
bo = e["bbox_obj"]
if bo[2] > 1 and bo[3] > 1:          # 宽高都得 > 1 才画
    d.rectangle([bo[0], bo[1], bo[0]+bo[2], bo[1]+bo[3]], ...)
```

**注意**：这不只是画图问题 —— **它同时说明了为什么必须过滤完全遮挡的样本**
（`DATA-19` / `min_visib_fract=0.10`）。退化 bbox 本身就是"这个实例没有
可用监督信号"的信号。

---

### ALGO-11 ⚠️ 用 `uniq` 后的 mtime 差算 ETA，把自己的进度算错了

**症状**：估算渲染剩余时间时，脚本输出
```
续渲已跑 25 分钟，完成 4 个场景 -> 388 秒/场景
预计完成: 09:31
```
而实际逐帧时间戳显示是 **7 分钟/场景**，两者差一倍多。

**两个错**：
1. `S=$(stat -c %Y append_driver.log)` 取的是**日志的 mtime**，
   而日志一直在刷新 —— 所以 "已跑 25 分钟" 根本不是已经过时间
2. `ls --time-style=+%s | uniq` 之后相邻差，混进了**旧渲染遗留帧**的时间戳，
   算出 `+63 分 14 秒` 这种不可能的间隔

**正确做法**：**用最可靠的那个量：帧数差 ÷ 每场景帧数。**
```
300 帧 @ 07:20  →  380 帧 @ 07:48
= 4 个场景 / 28 分钟 = 7 分钟/场景
```
帧数是**单调、可核实、不受 mtime 语义影响**的量。

**教训**：**估算进度时优先用"累计产物的数量"，而不是文件时间戳。**
mtime 会被读写、touch、追加日志污染；产物计数不会。
（和 `CODE-07`「把 metrics.csv 行数当步数，耗时估错 10 倍」是同一类错。）

---

### DATA-26 · BOP 的 `scene_gt.json` 里 `cam_t_m2c` 单位是**毫米**

**症状**：我看到某物体的 `cam_t_m2c` 里 z ≈ 439，得出"物体被抛到 439 米外"
的错误结论，并据此写了一整段错误分析。

**根因**：**BOP 的 `cam_t_m2c` 是毫米**，而 BlenderProc 的 `o.get_location()`
是**米**。两个 API 混用时单位不一致。

**修法**：
- 读 BOP 的 GT → 当成**毫米**（除以 1000 才是米）
- 读 `o.get_location()` → 当成**米**
- 写"位置合理性"校验门时，**必须先确认这个数是哪个 API 来的**

**⚠️ 单位错会导致"看起来完全合理"的错误结论。** 439 mm = 43.9 cm 是正常距离，
"439 米"才荒谬。**如果结论荒谬，先怀疑单位，再怀疑逻辑。**

---

### TEST-01 · 计时验证要放在全量重跑**之前**

**症状**：我给渲染脚本打了一个补丁（`DATA-24`：把干扰物挪到物理之后），
补丁有明确的理论依据，本可以直接启动 50 场景的全量渲染。
我先跑了 **2 场景的计时验证**（866 秒），确认 433 秒/场景、校验 2/2、
`sample_poses` 重试降到 2~56 次，**然后**才全量重跑。

**为什么这么做**：这一轮之前，我在 `GEO2` 上已经吃过一次
「带着坏数据一路烧到训练结束」的亏。**补丁正确 ≠ 补丁生效**，
尤其是跨越 Blender / BlenderProc / 物理引擎三层的改动。

**模板**：
```bash
# 全量之前，先用 2 个样本验证
BP_NUM_SCENES=2 <跑一遍>
# 检查：耗时 / 门禁 / 关键日志行 / 产物数量
# 全部符合预期，再启动全量
```

**代价对比**：计时验证 15 分钟；如果直接全量、跑到一半发现没生效，
损失的是数小时 + 已渲帧全部作废。

**与 `ALGO-10`（实验设计三连错）同类：先做小样本、先对齐基线、
先验证补丁生效，再投入大算力。**


---

### TEST-02 ⚠️⚠️ 用一个**从未验证过能显示进度**的信号判断"训练卡死"，白查一整轮

**这是本轮最严重的一次误判，代价是数小时。** 记在最前面：

> **"没有看到进度"不是证据，除非你先证明过"有进度时这个信号会变"。**

**症状**：v1 + GEO7 合并训练（多路径 `dataset_root=[v1, GEO7]`）时，
`cmb.log` 的大小**在 430 秒内一个字节都没涨**（4354 B 恒定），
`grep -c 'Epoch\|it/s\|loss_step'` **全是 0**。
于是判定"一个 batch 都没完成，训练卡死"，并开始了一整轮排查。

**真相**：训练**一直正常**。同一个进程的 `outputs/<exp>/logs/version_0/metrics.csv`
里，step 已经在涨、loss 在降。最后实测 **6.89 步/s**，300 epoch 只需 2.8 小时。
我杀掉的那些"卡死"的进程，每一个都在正常训练。

**那个假信号是怎么来的（三个原因叠加，全是"日志不输出"而非"训练不前进"）**：

1. ⚠️ **Linux 非 TTY 下 TQDM 进度条不输出中间行**，只在**进度条关闭时**打最终行。
   所以 `nohup ... > log 2>&1` 的日志里，`Epoch 0/0 ━━━ 40/40` 这种行
   **只在 epoch 结束时**才出现 —— 一个 epoch 235 步约 34 秒内，日志当然不动。
2. **Lightning 的控制台指标（`train/loss_step` / `val/...`）也是 epoch 末才打**。
   中间过程只写 logger（CSV / TensorBoard），不写 stdout。
3. **`CSVLogger` 有 flush 延迟**，`grep loss_step` 长时间只能看到表头那 1 行。

**我当时其实已经握有反证，却没读懂**：冒烟实验 `W8`（`limit_train_batches=40`）
的日志**整段只有最后一行** `Epoch 0/0 ━━━ 40/40 • 0:00:06 • 9.45it/s`。
这已经说明"进度条只在末尾输出"，但我把它读成了"日志会输出进度"。

**为什么排查越走越偏**：我取到的每一个栈都**看起来很可疑**，其实全是正常位置：

| 取到的栈 | 我当时的解读 | 实际含义 |
|---|---|---|
| `optimizer.step → clip_grad_norm_` | "卡在梯度裁剪" | 正常的一次优化步 |
| `DataLoader._try_get_data → queue.get → wait` | "worker 不产数据" | 正常地等下一个 batch |
| `logger_connector.on_batch_end → to_item` | "卡在指标转换" | 正常的每步记录指标 |

**主进程大部分时间本来就该待在 `queue.get`**（数据 6.9 步/s 产得比 GPU 吃得快），
**worker 大部分时间本来就该 100% CPU**（`__getitem__` 是 CPU 活儿）。
"栈看起来像在等"不是卡死，**要看它是否永远停在同一个位置且产出为零** ——
而"产出"恰恰是我用错信号去量的那个东西。

**真正可用的进度信号（本轮验证过）**：

```bash
# ✅ 每步都落盘，与 TTY / 进度条无关；两次采样算出真实步/s
CSV=$(ls -t outputs/<exp>/logs/version_*/metrics.csv | head -1)
awk -F, 'NR>1 && $3!="" {s=$3} END{print s}' $CSV   # 间隔 90s 采两次
# ✅ TensorBoard 事件文件大小增长
ls -l /root/tf-logs/<exp>/version_0/events.out.tfevents*
# ✅ 卡住时取栈（容器禁 ptrace，py-spy 用不了，信号可用）
DSH_FAULTHANDLER=1 <启动>   # 然后 kill -USR1 <pid>，只对主进程发避免输出交错
```

**要注意的是**：`kill -USR1` 输出必须**只对主进程发**。同时给 8 个 worker 发，
9 个进程往同一个 fd 写，输出会完全交错成一团乱码（本轮踩了）。

**代价**：这一轮里我反复"杀掉正常训练 → 重启 → 再判卡死"，
期间 `kill -9` 掉至少 6 个正在正常训练的进程（每个都是在跑第 N 个 epoch）。

**教训**：

- ⚠️⚠️ **任何"没有进展"的判断，先在一个已知正常的短跑上验证这个信号会动。**
  本轮的 `W8` 短跑就是现成的基准，我却没拿它校准信号。
- ⚠️ **日志文件不增长 ≠ 进程不前进。** 先确认日志**是否本来就该增长**
  （stdout 是否被 TTY 检测关掉、指标是否只在 epoch 末打）。
- ⚠️ **排查脚本必须只读。** 我一边排查一边 `kill -9`，
  把"查出真相"变成了"销毁证据"：真正该做的是**先只观察一段时间，什么都别杀**。
- ⚠️ **栈"看起来可疑"不构成证据。** 要区分"停在某个位置"和"永远停在同一个位置"，
  而且必须先确认**进程有没有在产出** —— 而产出的度量方式本身要可靠。
- 与 `ALGO-11`（用 mtime 差算 ETA 算错）同类：**度量工具错了，结论一定错。**

**附带发现（真实且值得修）**：`__getitem__` 每次都要重新读+解析
`scene_gt.json`(1.4 MB) / `scene_gt_info.json`(0.8 MB) / `scene_camera.json`，
没有缓存。实测给 `_load_scene_json` 加缓存后 dataloader 吞吐
**110 → 297 样本/s（2.7×）**。当前训练不是数据瓶颈（6.9 步/s 只需 110 样本/s），
但样本量再涨或换更大模型时会成为瓶颈。

**新增诊断脚本**（都是这一轮的产物，留作工具）：
`scripts/diag_dataload.py`（逐样本计时 + 超时自动打栈）、
`scripts/diag_throughput.py`（真实多进程吞吐 + 子进程 RSS/CPU 趋势）、
`scripts/diag_dm.py`（真实 datamodule + 可开关 `Subset` / CUDA-fork）、
`scripts/merge_bop_roots.py`（多根合并成单根，符号链接不复制数据）。


---

### DATA-27 ⚠️ 渲染场景里**每一个物体都是同一款 DJI**，一个异类物体都没有

- **时间**：2026-09-21（用户提出"单实例应该一帧一个 DJI + 多个其它物体"后查实的）
- **现象**：`gen_pbr_data_demo.py` 里
  ```python
  models_ids = np.array([int(k) for k in models_info])   # = [1]
  idx_l = np.random.choice(models_ids, size=num_objs, replace=True)
  ```
  `models_info.json` 里**只有 `obj 1`**，所以 `num_objs` 个实例**全都是同一款 DJI**。
  v1 每帧 **10 个**、GEO7 每帧 **6 个**，`scene_gt.json` 的 `obj_ids` 恒为 `[1]`。
  逐帧确认：`loaded 6 object instances [1, 1, 1, 1, 1, 1]`。
- **量化**（`scripts/analyze_instance_ambiguity.py`，纯 numpy，无卡模式 0.5 核/2 GB 可跑）：

  | 数据集 | 帧 | 目标实例 | 1.4× 裁剪框内含其他同类 | 平均/最多 |
  |---|---|---|---|---|
  | v1 | 500 | 4969 | **96.20%** | 3.2 / 9 |
  | GEO7 | 700 | 3958 | **77.59%** | 1.3 / 5 |

  含 ≥2 个：v1 80.2% / GEO7 40.7%。含 ≥3 个：v1 62.1% / GEO7 14.3%。
  （比文档里旧记的 88.3% **更高**。）

- **⚠️ 但「这是标签歧义」的推论是错的。** 我本来以为"图里全是 DJI → 模型不知道预测哪台"，
  于是量了"选**离裁剪中心最近**的那台"这个位置捷径的正确率：

  | 框偏移（相对裁剪边长） | v1 捷径答错 | GEO7 捷径答错 |
  |---|---|---|
  | 0（GT 框） | 0.08% | 0.08% |
  | ±5% | 0.29% | 0.11% |
  | ±10% | 1.13% | 0.37% |
  | ±17.5% | 3.58% | 1.17% |
  | ±25% | 8.35% | 3.29% |

  **捷径即便在框偏移 ±25% 时仍有 92~97% 正确。** 所以
  **"哪台是目标"并不歧义**，减少同类实例**不会**自动修好真实域精度。
  （真实域瓶颈是外观：见 `docs/PROJECT_SUMMARY.md` §3.3 的合成/真实排名反转。）

- **那为什么还要改？** 三条**与歧义无关**的理由：
  1. **模型可以走捷径而不学外观** —— 96% 的裁剪里有同类邻居，
     模型完全可以只学"中间那个"，于是换域时没有任何判别性表征可依靠
  2. **裁剪里从来没有"非 DJI 物体"** —— 模型的"背景"外观统计 == "DJI 外观统计"
  3. **遮挡永远是同类自遮挡**（相同形状互压），学不到被任意形状遮挡

- **解法**：渲染脚本**本来就支持**这些开关，所以是纯配置改动（`scripts/render_single_instance.sh`）：
  ```
  BP_NUM_OBJS=1          # 每帧 1 个 DJI（GT 唯一），原来是 6~10
  BP_OBJAVERSE_PROPS=10  # 异类道具给足遮挡与形状多样性，原来是 3
  BP_PROP_COVER=0.6      # 静态压盖造遮挡（GEO6 验证过的机制）
  BP_SCREEN_CONTENT=1    # 屏幕内容（已经是 1）
  ```
  **两阶段**：主体 45 场景 × 20 帧（每帧 1 个 DJI）+ 追加 5 场景 × 20 帧（每帧 3 个 DJI）
  —— 因为真实视频的裁剪图里**一定**会有别的 DJI，全去掉会造成反向的分布偏移。

- **⚠️ 一个差点误判的坑**：`find /root/autodl-tmp -name '*.glb'` 只找到 9 个，
  差点得出结论"道具 GLB 丢了、重渲缺素材"。实际它们在
  **`/root/.objaverse/hf-objaverse-v1/glbs/`**（1923/1923 全部存在）——
  **在系统盘 `/root/.objaverse`，不在 autodl-tmp 数据盘上**。
  `load_keep_list()` 用 `os.path.isfile()` 过滤，路径对了就没事。
  ⚠️ 反过来说：**它在系统盘，所以「保存镜像」能覆盖它，而数据盘上的东西不能。**

- **影响文件**：`data/render_ws/gen_pbr_data_demo.py`（不在仓库，在实例 `bp_ws/`）、
  `scripts/render_single_instance.sh`（新增，在仓库）、
  `scripts/analyze_instance_ambiguity.py`（新增，在仓库）
- **教训**：
  - ⚠️ **"数据设计有病"要先量化再动手。** 我一开始准备直接认同"每帧放一堆同类
    = 歧义"，量化后发现**歧义不成立**（捷径 99.92% 正确）。如果不查，
    会花几小时 GPU 去修一个**不存在的问题**，还以为是"修好了"。
  - ⚠️ **量化要问对问题。** "有多少别的实例混进来了"（96%）是**现象**；
    "这个现象会不会让标签变歧义"（不会）才是**判据**。两个问题的答案相反。
  - ⚠️ **`find` 的范围会骗人。** 只搜数据盘得出的结论是"素材丢了"，
    实际素材在系统盘。**跨盘搜要用全盘或明确列出盘。**
  - ✅ 这条属于「用户的直觉指对了方向，但机制判断需要修正」——
    **方向和机制都要对，否则会在对的方向上做错的事。**


## 待解决问题

| 编号 | 问题 | 阻塞什么 | 状态 |
|---|---|---|---|
| `DATA-01` | 头戴相机内参 K 未知（视频被 H.264 重编码，元数据已丢） | ~~PnP 无法求解、重投影误差无法计算~~ | 🟢 **对项目目标不构成阻塞**（2026-09-14 修正）。psd 明确：**本项目不需要内参**。核对代码确认：网络本体（`CornerPoseModel.forward`）只有 `image → encoder → decoder → heatmap`，损失只吃 `pred_heatmap`/GT `heatmap`/`corner_2d`，**都不消费 K**；`crop_and_resize` 的裁剪只看 bbox。K 只出现在可选的 `predict(solve_pose=True) → cv2.solvePnP` 与其位姿指标里，已把 `configs/model/metrics/default.yaml` 的 `solve_pose` 默认改为 `false`。**对照**：BoxDreamer 把 K 当网络输入（构造 camera rays），所以它必须标定 |
| `DATA-02` | 还没有 DJI Action 4 的 3D 模型（BOP 格式 `models_info.json`） | 整个训练管线没有输入 | ✅ **v2 已产出**：4 视图生成，三轴 99%/102%/101%（官方尺寸），319668 面 → `data/dji_action4_hybrid/models/`。旧 2 视图版留在 `data/dji_action4_2view/` 对照 |
| `DATA-13` | 多视图生成时**形状和纹理的最优输入不同**：¾ 侧视能让形状变准、却会把纹理投糊 | 外观质量 | ✅ 已定位；解法=形状用 4 视图、纹理用 front/back 两张 |
| `DATA-03` | BlenderProc 官方流程要求 Ubuntu + EGL；且 `bpy` pip 模块在 AutoDL 上装不了 | 合成数据渲染 | ✅ **已跑通**：Blender 3.6.0 官方发行包 + HCCEPose 的 blenderproc 2.5.0，无卡模式也出了图 → [docs/RENDER_SETUP.md](RENDER_SETUP.md) |
| `ENV-12` | 无卡模式 cgroup 内存上限只有 **2 GB**，110 MB 的 PLY 把 Blender OOM 掉 | 渲染环境 | ✅ 已定位；用 `mesh_to_bop.py --max-faces` 简化到 13.4 万面，峰值降到 851 MiB |
| `DATA-08` | 渲染出**黑色裂纹**（真因：`preserveboundary=False` 把 38705 个 UV 岛拼的网格**撕开了**，多出 1383 条非流形边） | 外观正确性 | ✅ 已定位并修复：改回 `preserveboundary=True`；用纯灰材质渲染（`BP_FLAT_MATERIAL=1`）一锤定音 |
| `DATA-12` | 纹理图集 **38.9%（新网格 25.84%）是黑色噪点**，缩小采样会拉偏颜色 | 外观质量 | 🟡 真实但次要；已用 `scripts/repair_texture_atlas.py` 做 padding。根治需重跑纹理生成（要 GPU）或用顶点色 |
| `DATA-09` | 保纹理简化 `preserveboundary=True` 卡在 40.3% 且无法再降 | 网格规模 | ✅ 已接受 40.3%（358830 面）；用 `--ply-precision 5` 把 PLY 压到 32.89 MB 抵消内存 |
| `DATA-10` | ~~生成的 mesh 朝向是任意的~~ | BOP 8 角点约定 | ✅ **虚警，已撤销**。实测物体坐标系本来就是规范的、和包围盒对齐的：`+X` = 长轴 69.94 mm、`+Y` = 44.87 mm（上，顶部有录制键）、`+Z` = 33.09 mm（**镜头方向**）。渲染里"躺着"只是刚体掉落的结果，PBR 数据本来就是随机姿态。惯性主轴与坐标轴差 6~8° 是镜头偏心造成的，正常。**约定已写进 [DATA.md](DATA.md)** |
| `DATA-14` | 训练数据 `depth/` 不可用：深度被量化成整数米（uint8） | RGB-D 路线 | 🟡 **暂不修**（训练不用深度）。根因在 HCCEPose 版 BlenderProc 的 EXR 读写链路，定位记录见 `DATA-14` |
| `DATA-16` | 训练集**遮挡严重不足**：95% 实例完全可见，`mask` 与 `mask_visib` 几乎相同 | HCCEPose 的遮挡鲁棒能力 | 🟡 已定位（物理把物体摊平了）；三条修法见 `DATA-16`，本次未修 |
| `DATA-17` | 裁完之后的**有效分辨率偏低**：`bbox_visib` 只有 95~156 px，网络输入 224 要上采样 1.5~2.4× | 精度上限 | 🟡 对比 BoxDreamer（一图一物、512²）我们是下采样。可收紧相机 radius 或提高渲染分辨率 |
| `DATA-15` | 单模型数据集下原脚本一半场景只放 1 个物体 | 训练数据量 | ✅ 已修：先不重复取模型再平铺到 `num_objs` 个实例 |
| `DATA-11` | ~~无卡模式下只跑了预览参数~~ | 正式训练数据 | ✅ 已在 GPU 上用正式参数跑完（1024×768 / 50 采样 / `BP_WRITE_BOP=1`） |
| `DATA-11` | 无卡模式下只跑了预览参数（480×360 / 32 采样、`BP_WRITE_BOP=0`） | 正式训练数据 | ⏳ 待有 GPU 时用正式配置跑（1024×768 / 50 采样 / `BP_WRITE_BOP=1`），并验证 `scene_gt.json` + `mask_visib` |
| `ALGO-05` | 推理时用 `topk`（BoxDreamer 官方）还是 `soft_argmax`（实测更准） | 最终指标 | 🟢 **已结案**：实测两者高度一致（v1: 4.11 / 4.05 px；v2: 225 / 224 px），**差距可忽略，不必为此改动**。真正的问题不是"两个算子不一致"，而是**没有任何一项损失在优化推理时读的那个量**（峰在哪、多尖），见 `docs/RESULTS.md` §4.3 |
| `DATA-18` | v2 数据（1040 帧 / 12480 实例）**不可用**：18.1% 的实例是"裁剪图里没有目标"的无解样本 | 遮挡鲁棒性 | 🟡 **已定位，待重渲**。修法见 `docs/RESULTS.md` §8：`visib_fract` 下限提到 0.25~0.3、裁剪改用 `bbox_obj`、排除出画实例、加裁剪图检查。**注意 v2 的逐场景 median 从 7.77 散到 120.99，坏场景是成片的** —— 重渲时必须逐场景质检（`scripts/qa_scene.py`） |
| `DATA-19` | 料箱随机堆叠**不可控**：18% 实例被挡死，部分遮挡只从 1.9% 提到 4.3% | 遮挡样本的"度" | 🟡 **建议改用"训练时合成遮挡"**：拿完全可见样本在线遮掉一块，比例自己定、答案唯一确定。**但强度必须小**（10~15%），见 `docs/RESULTS.md` §3 —— 照抄 BoxDreamer 的 0~40% 会把模型压垮 |
| `DATA-27` | 渲染场景里**每个物体都是同一款 DJI**（`models_info` 只有 `obj 1`）：v1 每帧 10 个、GEO7 每帧 6 个，异类物体一个都没有 | 单实例任务是否良定义 / 模型能否学到外观判别 | ✅ **已量化并给出解法**：裁剪污染 v1 **96.2%** / GEO7 77.6%；但**"哪台是目标"不歧义**（位置捷径 99.92% 正确，框偏移 ±25% 仍 92~97%）。⇒ 改的理由是**去捷径 + 加外观多样性**，不是修歧义。解法=纯配置改动（`BP_NUM_OBJS=1` + `BP_OBJAVERSE_PROPS=10`），脚本 `scripts/render_single_instance.sh` 已就绪，**待有 GPU 时渲** |
| `ALGO-11` | 训练时增强的**强度**与"单图设定"不匹配 | 能否用增强补遮挡 | ⏳ **E4 实验在跑**（无增强 + 同样 100 epoch）。判据：E4 ≈ 4.12 ⇒ 差的是训练时长；E4 ≈ 17.91 ⇒ 差的是增强 |
| `ALGO-12` | 损失函数**不是杠杆**（E0/E1/E2b 三组打平） | 精度上限 | 🟢 **已结案**：去掉占 99.55% 权重的 fine 项无实质变化，换 focal 也没更好。**下一步该打数据和增强，不要再调损失** |
| `ALGO-06` | fine loss 用 soft-argmax 近似（BoxDreamer 用单独回归头） | 精度上限 | 🟡 若 fine 项收益不明显，再考虑加回归头 |
| `ENV-04` | 全局 torch 是 CPU 版 | 训练速度 | 🟡 训练前换 CUDA 版 |
| `DATA-04` | 渲染产物的 RGB 是 `.jpg`（`color_file_format="JPEG"`） | 数据加载 | ✅ 已兼容 `.png`/`.jpg`/`.jpeg` 及大小写变体 |
| `DATA-05` | `camera.json` 不存在时脚本会填 LINEMOD 默认内参（640×480） | 渲染数据的尺度分布 | ✅ 已手写 `data/render_ws/camera.json`（1024×768，fx=fy=800，depth_scale 0.1）并上传到远端数据集目录 |
| `DATA-06` | 渲染脚本相机采样半径 0.3~1.2 m，均值偏大（s/d≈0.094 vs 真实≈0.15） | 透视强度 sim-to-real | ✅ 适配脚本默认收紧到 `BP_RADIUS_MIN/MAX = 0.35/0.6`（注意不能再小：脚本自带 0.3 m 的 obstacle-in-view 阈值） |
| `GEN-01` | 生成的 mesh **厚度偏大约 16%**（只用了正面+背面两张图；三轴包围盒 95%/102%/**116%**，对角线已对齐官方 89.44 mm） | 包围盒比例 | 🟡 补一张侧面图可改善 |
| `GEN-02` | 纹理生成 | 外观 sim-to-real | ✅ 已跑通（1314 s，`textured.glb` 26.7 MB，2048² 图集） |
| `K` | AutoDL 实例**未保存镜像**前，环境不可丢 | 全部云端工作 | 🔴 跑完记得「保存镜像」；本次环境在 `/root/autodl-tmp`（关机保留，释放即丢） |

---

## 快速排查清单

遇到问题先按这个顺序扫一遍：

1. **命令真的失败了吗？**
   看**产物**（文件在不在、`git log` 正常吗），别只看 exit code。→ `PS-01`
2. **是显示问题还是代码问题？**
   中文乱码先验 `decode('utf-8')` + `print(sys.stdout.encoding)`。→ `ENV-03`
3. **报告里说的下载/网络错误，真的是网络问题吗？**
   用别的方式下同一个文件比哈希。→ `ENV-02`
4. **路径问题？**
   临时脚本要设 `PYTHONPATH`（`sys.path[0]` 是脚本目录不是 cwd）。→ `PS-03`
5. **配置解析失败？**
   先看 `${hydra:...}` 的键在当前运行模式下有没有值；pytest 里要手动装 `HydraConfig`。→ `CFG-02`、`CFG-03`
6. **导入/实例化报"找不到 target"？**
   查模块名和类名是不是同名了。→ `CODE-01`
7. **显存随轮次缓慢增长？**
   查有没有把带 `grad_fn` 的张量累积到 epoch 末尾。→ `CODE-02`
8. **模型不收敛但没报错？**
   查标签里有没有大量**并列值**（会让 topk/argmax 行为随机）。→ `ALGO-02`
9. **要提交敏感/大文件？**
   **先 gitignore 再 git add**，推上去就晚了。→ `GIT-01`、`GIT-02`
10. **"我以为提交上去了"？**
   用 `git ls-files <路径>` **核实**。`.gitignore` 会静默吞文件，
   `git status` 也看不到被忽略的东西。→ `GIT-03`
11. **要往 `.gitignore` 里加目录名？**
   **加前导 `/`**。无锚点模式会匹配任意深度。→ `GIT-03`
12. **`pip` 说"找不到这个包"（`from versions: none`）？**
   先 `curl -sI` 一下那个源，看是不是 **403 / Cloudflare**。pip 对"被封"和"真没有"报同一句话。
   另外国内镜像站多半**只镜像发行版目录，不镜像 pip 源**。→ `ENV-09`
13. **`pip` 报 `ResolutionImpossible`？**
   这条命令**一个包都没装上**（pip 先整体解析再安装）。把钉死依赖的老包拆出来用
   `--no-deps` 单独装。→ `ENV-10`
14. **要在无头 Linux 上渲染/离屏取图？**
   先确认 EGL 能起来（`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 在不在、
   `libEGL.so.1` 装没装），再确认 `OMP_NUM_THREADS` 是合法整数。→ `ENV-11`
15. **把一个 CLI 工具当库 `import` 用？**
   先找出它"只有 CLI 才会初始化"的全局状态（路径、临时目录、设备）。
   这类状态不报错，只会让路径悄悄退化成相对路径。→ `DATA-07`
16. **长任务"成功"了但没有产物？**
   `cmd | tail` 之后 `echo $?` 拿到的是 **`tail`** 的退出码。**重定向到文件**再看。
   容器里先查 `/sys/fs/cgroup/memory.max`，别信 `free`。→ `ENV-12`
17. **进程被 `Killed`（137）但没有任何 traceback？**
   先按内存查：cgroup 限额、`memory.events` 的 `max` 计数、以及**产物本身有多大**
   （这一例是 110 MB 的文本 PLY 在 BlenderProc 里被复制成三份）。→ `ENV-12`
18. **numpy 报 OOM / 进程莫名被杀？**
   查**广播**。多写一个 `[:, None]` 就能把 `(512,512)` 变成 `(512,1,512)`，
   再套一层 `sin` + `abs` 就是两个 GB 级数组。**给每个产物加 shape 断言**。→ `CODE-05`
19. **渲染出来不对（黑斑/裂纹/糊）？**
   按 **几何 → 纹理 → 着色/采样率** 三段拆，而且**要真的把某一类整个拿掉**去验证
   （例如断开纹理连线、换成纯灰材质）。在同一类里换三种假设都不对时，
   **下一件事是跳出这一类**，不是换第四种。
   先用**点云 splat**（按 UV 取色后正交投影，纯 numpy，秒出）排除纹理这一类。→ `DATA-08`
20. **要对网格做简化/降采样？**
   先 `ms.get_topological_measures()` 看**连通分量、边界边、非流形边**。
   本例网格是 **38705 个 UV 岛**拼的，任何"允许移动边界"的简化都会把它撕碎。
   **降得更多 ≠ 更好。** → `DATA-08`、`DATA-09`
21. **文本格式的模型/数据文件太大？**
   先想想**读它的程序会不会把整个文件读成字符串**（BlenderProc 就会，还连做两次 `.replace()`）。
   少写几位有效数字就能省一半内存。→ `ENV-12`
22. **用 `grep -c` / `wc -l` 之类做"数量检查"？**
   **只匹配英文/ASCII 关键词，别匹配中文。** 中文模式在命令替换里会静默返回 0，
   把"全部通过"读成"一个都没过"。**任何用于机器判断的标记，日志里都要是 ASCII。** → `PS-09`
23. **要停掉一个「bash 包着 blender/python」的长任务？**
   **先杀叶子再杀父**（`pkill -x blender` 之后再杀 driver 的 bash），
   而且**杀完必须核实清零**再重启，否则两个任务抢同一张卡。→ `ENV-19`
24. **一个脚本块里既有 `git`/上传/远端命令，又有 `export`/`$env:`？**
   **环境变量永远放最前面。** 顺序错了，命令会以 `exit 127` 失败但被后台作业吞掉，
   后面的步骤却因为变量已设好而"成功"连上 —— 跑一个不存在的脚本。→ `ENV-20`
25. **远端任务"慢了"，想降分辨率/降采样？**
   **先量时间戳。** 本例 20 帧渲染只要 3 秒，18 分钟全花在每场景 SETUP ——
   根因是**一个物体摆放顺序**。优化错地方等于白降质量。→ `DATA-24`
26. **想给训练集"加遮挡"，但物体总是不重叠？**
   先看 `sample_poses` 之类的**相交检查**。它是"防爆炸"和"造遮挡"之间的零和。
   出路是**把遮挡物改成静态**（静态重叠不产生力），不是收紧撒点范围。→ `DATA-25`
27. **要估一个长任务的剩余时间？**
   用**累计产物的数量 ÷ 单位产量**，别用文件 mtime。
   mtime 会被日志刷新、touch、追加写入污染。→ `ALGO-11`
28. **打了个有理论依据的补丁，想直接全量重跑？**
   **先用 2 个样本做计时验证**：耗时 / 门禁 / 关键日志行 / 产物数量。
   补丁正确 ≠ 补丁生效，尤其是跨 Blender / 物理引擎 / 框架三层的改动。→ `TEST-01`
29. **读写位姿/坐标时单位对不上？**
   **BOP 的 `cam_t_m2c` 是毫米，BlenderProc 的 `get_location()` 是米。**
   **如果算出来的结论荒谬（"物体在 439 米外"），先怀疑单位，再怀疑逻辑。** → `DATA-26`
30. **怀疑"训练卡死了"？**
   **先证明你用的进度信号在正常训练时会动。** 日志不增长 ≠ 没在训练：
   非 TTY 下 TQDM 只在进度条关闭时输一行，Lightning 控制台指标也只在 epoch 末打。
   改看 `outputs/<exp>/logs/version_*/metrics.csv` 的 `step` 列（间隔 90s 采两次算步/s）。
   **排查期间不要 `kill`** —— 先把"正在正常训练"这个可能排除掉。→ `TEST-02`

