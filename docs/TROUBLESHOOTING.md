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
| — | 渲染合成数据 | — | ⛔ 阻塞：见 [待解决](#待解决问题) |

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

---

## 待解决问题

| 编号 | 问题 | 阻塞什么 | 状态 |
|---|---|---|---|
| `DATA-01` | **头戴相机内参 K 未知**（视频被 H.264 重编码，元数据已丢） | PnP 无法求解、重投影误差无法计算 | 🔴 需向 psd 索要，或自拍棋盘格标定 |
| `DATA-02` | 还没有 DJI Action 4 的 3D 模型（BOP 格式 `models_info.json`） | 整个训练管线没有输入 | 🔴 待阶段① |
| `DATA-03` | BlenderProc 官方流程要求 Ubuntu + EGL，Windows 上大概率跑不通 | 合成数据渲染 | 🔴 需 Linux / 服务器，或换渲染方案 |
| `ALGO-05` | 推理时用 `topk`（BoxDreamer 官方）还是 `soft_argmax`（实测更准） | 最终指标 | 🟡 待有真实训练模型后用验证集实测决定，见 `configs/model/heatmap.yaml` 注释 |
| `ALGO-06` | fine loss 用 soft-argmax 近似（BoxDreamer 用单独回归头） | 精度上限 | 🟡 若 fine 项收益不明显，再考虑加回归头 |
| `ENV-04` | 全局 torch 是 CPU 版 | 训练速度 | 🟡 训练前换 CUDA 版 |
| `DATA-04` | 渲染产物的 RGB 是 `.jpg`（`color_file_format="JPEG"`） | 数据加载 | ✅ 已兼容 `.png`/`.jpg`/`.jpeg` 及大小写变体 |
| `DATA-05` | `camera.json` 不存在时脚本会填 LINEMOD 默认内参（640×480） | 渲染数据的尺度分布 | 🟡 必须在 `data/dji_action4/camera.json` 手写，见 `docs/DATA.md` §5.2 |
| `DATA-06` | 渲染脚本相机采样半径 0.3~1.2 m，均值偏大（s/d≈0.094 vs 真实≈0.15） | 透视强度 sim-to-real | 🟡 建议收紧到 0.35~0.6 m，见 `docs/DATA.md` §5.3 |
| `GEN-01` | 生成的 mesh **厚度偏大 23%**（只有正面+背面两张图） | 包围盒比例 | 🟡 补一张侧面图可改善；`mesh_to_bop.py` 会缩放到官方对角线 |
| `GEN-02` | 纹理生成尚未跑通（需 GPU 模式） | 外观 sim-to-real | ⏳ 脚本已备好：`finish_all.sh` |
| `K` | AutoDL 实例**未保存镜像**前，环境不可丢 | 全部云端工作 | 🔴 跑完记得「保存镜像」 |

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

