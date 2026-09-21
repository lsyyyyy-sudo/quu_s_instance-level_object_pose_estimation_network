#!/bin/bash
# ============================================================================
#  单实例版数据集渲染：**每帧 1 个 DJI（唯一 GT）+ 多个异类道具**
# ============================================================================
#
# 为什么要改（2026-09-21 的决定）
# ------------------------------
# 原来的渲染是 `idx_l = np.random.choice(models_ids, size=num_objs, replace=True)`，
# 而 `models_info.json` 里只有 `obj 1` —— **每一帧的每一个物体都是同一款 DJI**，
# 一个异类物体都没有。v1 每帧 10 个、GEO7 每帧 6 个，全是 DJI。
#
# 实测后果（scripts/analyze_instance_ambiguity.py，无卡模式跑的）：
#   · 1.4× GT 裁剪框里含其他同类实例的比例：v1 **96.2%**、GEO7 77.6%
#     （平均 3.2 / 1.3 个，最多 9 / 5 个）
#   · ⚠️ 但"选离裁剪中心最近的那台"这个**位置捷径**在 GT 框上 99.92% 正确，
#     即便框偏移 ±25% 仍有 92~97% 正确
#
# ⇒ **所以这不是"标签歧义"问题**（目标是可判定的），减少同类实例**不会**
#   自动修好真实域精度。改它的真正理由是另外三条：
#   1. **模型可以走捷径**：96% 的裁剪里都有同类邻居，模型完全可以只学
#      "中间那个"而**不学外观**，于是换域时没有任何鲁棒性
#   2. **裁剪里从来没有"非 DJI 物体"** → 模型的"背景"统计 == "DJI 外观"
#   3. **遮挡永远是同类自遮挡**（相同形状互压），学不到被任意形状遮挡
#
# ⇒ 同时**必须**配合"渲染对齐真实外观"（屏幕点亮 + 目标贴标签），
#   那才是真实域瓶颈（见 docs/PROJECT_SUMMARY.md §3.3 的排名反转）。
#
# ⚠️ 不要走极端：真实视频的裁剪图里**一定**会有别的 DJI。所以本脚本是
#   两个阶段：主体 1 个 DJI（消除捷径）+ 少量 3 个 DJI（保留真实情况）。
#
# 用法（**在有 GPU 的实例上**；无卡模式渲不动）
# ------------------------------------------
#   bash render_single_instance.sh smoke    # 2 场景冒烟（TEST-01：先验证再全量）
#   bash render_single_instance.sh main     # 45 场景 x 20 帧 = 900 帧，每帧 1 个 DJI
#   bash render_single_instance.sh mixed    # 追加 5 场景 = 100 帧，每帧 3 个 DJI
#   bash render_single_instance.sh all      # main + mixed
#
# 前置：BP_OBJAVERSE_KEEP 指向的 1923 个 GLB 必须在
#       `/root/.objaverse/hf-objaverse-v1/glbs/`（⚠️ 在系统盘 `/root/.objaverse`，
#       **不在** autodl-tmp 数据盘上；关机保留、释放会随系统盘一起没）
# ============================================================================
set +e

BL=/root/autodl-tmp/blender-3.6.0-linux-x64/blender
BP=/root/autodl-tmp/bp_src
SCRIPT=/root/autodl-tmp/bp_ws/gen_pbr_data_demo.py
PY=/root/autodl-tmp/envs/train/bin/python
DSET=/root/autodl-tmp/bop/v4/SINGLE/dji_action4_hybrid
V1_MODELS=/root/autodl-tmp/bop/v1/dji_action4_hybrid/models
V1_CAM=/root/autodl-tmp/bop/v1/dji_action4_hybrid/camera.json

MODE=${1:-smoke}

# ---- 数据集目录：软链 models/ 与 camera.json，避免复制 30 MB 的 PLY ----
if [ ! -d "$DSET/models" ]; then
  mkdir -p "$DSET"
  ln -sfn "$V1_MODELS" "$DSET/models"
  cp -f "$V1_CAM" "$DSET/camera.json"
  echo "  建好 $DSET（models 软链到 v1）"
fi

# ---- 公共环境 ----
export PYTHONNOUSERSITE=1 PYTHONPATH="$BP" OMP_NUM_THREADS=8
export BP_CC0TEXTURES=/root/autodl-tmp/cc0textures-512
export BP_OBJAVERSE_KEEP=/root/autodl-tmp/ov/keep_frozen.json
export BP_RES=1024x768 BP_SAMPLES=50 BP_NUM_WORKER=4
export BP_BIN=0                      # 不加料箱（料箱方案造出大量无解样本，见 DATA-19）
export BP_SCREEN_CONTENT=1           # 屏幕内容（真实目标是拍摄状态，屏幕亮）
export BP_SPAWN=random BP_SPAWN_HALF=0.13
export BP_RADIUS_MIN=0.30 BP_RADIUS_MAX=0.52 BP_ELEV_MIN=18

# ---- 关键改动：DJI 数量 ----
#   1  -> 目标唯一，消除"选中间那个"的位置捷径
#   10 -> 异类道具给足遮挡与形状多样性
export BP_NUM_OBJS=1
export BP_OBJAVERSE_PROPS=10
export BP_PROP_COVER=0.6             # 静态压盖比例（GEO6 验证过这个机制能造出遮挡）
export BP_PROP_SIZE_MIN=0.025 BP_PROP_SIZE_MAX=0.10

_clean() { rm -rf "$DSET/train_pbr"; }

_run() {
  local TAG=$1 NS=$2 NOBJ=$3 NPROP=$4 SEED=$5 FRAMES=$6
  local LOG=/root/autodl-tmp/SINGLE_$TAG.render.log
  export BP_NUM_SCENES=$NS BP_FRAMES=$FRAMES BP_NUM_OBJS=$NOBJ
  export BP_OBJAVERSE_PROPS=$NPROP BP_SEED=$SEED
  export BP_SCREEN_DIR=/root/autodl-tmp/bp_render/SINGLE_$TAG/content
  export BP_PREVIEW_DIR=/root/autodl-tmp/bp_render/SINGLE_$TAG/preview
  mkdir -p "$BP_SCREEN_DIR" "$BP_PREVIEW_DIR"

  echo "  [$TAG] 场景=$NS 帧/场景=$FRAMES DJI/帧=$NOBJ 道具/帧=$NPROP seed=$SEED"
  echo "  [$TAG] START $(date '+%H:%M:%S')"
  cd "$DSET" || return 1
  local T0=$(date +%s)
  "$BL" --background --factory-startup --python "$SCRIPT" > "$LOG" 2>&1
  local RC=$? T1=$(date +%s)
  echo "  [$TAG] END exit=$RC 用时 $(( (T1-T0)/60 )) 分 $(( (T1-T0)%60 )) 秒"
  echo "  [$TAG] 位置校验通过 $(grep -c '位置校验通过' "$LOG")/$NS"
  grep -E '\[check\] !!' "$LOG" | head -3 | sed 's/^/      /'
  echo "  [$TAG] 道具: $(grep -oE '加了 [0-9]+/[0-9]+ 个 Objaverse 物体' "$LOG" | tail -1)"
  echo "  [$TAG] 错误: $(grep -ciE 'Traceback|SystemExit' "$LOG")"
  grep -iE 'Traceback|SystemExit' "$LOG" | head -4 | sed 's/^/      /'
}

# ---- 渲染后体检：这一批数据到底长什么样 ----
_report() {
  echo
  echo "═══ 体检 ═══"
  $PY - "$DSET" <<'PYEOF'
import json, os, sys, numpy as np
root = sys.argv[1]
sd = os.path.join(root, "train_pbr")
if not os.path.isdir(sd):
    print("  （没有 train_pbr）"); raise SystemExit
nf = 0; per = []; v = []; ids = set()
for scene in sorted(os.listdir(sd)):
    gt = os.path.join(sd, scene, "scene_gt.json")
    if not os.path.isfile(gt): continue
    d = json.load(open(gt, encoding="utf-8"))
    nf += len(d)
    for k, anns in d.items():
        per.append(len(anns)); ids.update(int(a["obj_id"]) for a in anns)
    gi = os.path.join(sd, scene, "scene_gt_info.json")
    if os.path.isfile(gi):
        for k, lst in json.load(open(gi, encoding="utf-8")).items():
            v += [e["visib_fract"] for e in lst]
per = np.array(per); v = np.array(v)
print(f"  帧 {nf}   实例 {len(per)}   每帧实例 中位 {np.median(per):.0f} / 最多 {per.max()}   obj_ids={sorted(ids)}")
if len(v):
    print(f"  visib 中位 {np.median(v):.3f}  完全可见(>=0.95) {100*np.mean(v>=0.95):.1f}%  "
          f"部分遮挡(0.1~0.95) {100*np.mean((v>0.1)&(v<0.95)):.1f}%  "
          f"严重(<0.3) {100*np.mean(v<0.3):.1f}%  完全(<0.1) {100*np.mean(v<0.1):.1f}%")
    print(f"  ⚠️ 判据：每帧实例中位 = 1 ⇒ 目标唯一 ✅")
    print(f"           部分遮挡占比应显著高于 v1 的 3.4% / GEO7 的 9.0%")
print(f"  体积 {os.popen(f'du -sh {root}').read().split()[0]}")
PYEOF

  echo "  --- 用真实加载器确认保留率（会打印多根/可见性过滤行）---"
  cd /root/autodl-tmp/quu_s
  $PY -c "
import sys; sys.path.insert(0,'/root/autodl-tmp/quu_s')
from src.datasets.bop_pbr import BOPPBRDataset
ds=BOPPBRDataset(dataset_root='$DSET',split='train_pbr',obj_ids=[1],
                 image_size=256,heatmap_size=64,augment=False,min_visib_fract=0.10)
print(f'  过滤后保留 {len(ds)} 个样本')
" 2>&1 | grep -E '过滤|保留' | tail -2

  echo "  --- ⭐ 关键体检：位置捷径还有多可靠（应为 100%）---"
  cd /root/autodl-tmp/quu_s
  $PY scripts/analyze_instance_ambiguity.py --root "$DSET" --names SINGLE 2>&1 \
    | grep -E '含 ≥1|答对|排名|捷径' | sed 's/^/  /'

  df -h /root/autodl-tmp | tail -1 | sed 's/^/  /'
}

case "$MODE" in
  smoke)
    echo "═══ 冒烟：2 场景 x 4 帧（TEST-01：先验证再全量）═══"
    _clean
    _run smoke 2 1 10 770001 4
    _report
    echo
    echo "  ⚠️ 冒烟通过后再跑 main。判据："
    echo "     · 位置校验通过 2/2"
    echo "     · 每帧实例中位 = 1"
    echo "     · 部分遮挡占比 > 5%"
    echo "     · 位置捷径 答对 100%（这是本次改动的核心指标）"
    ;;
  main)
    echo "═══ 主体：45 场景 x 20 帧 = 900 帧，每帧 1 个 DJI + 10 个道具 ═══"
    _clean
    _run main 45 1 10 990001 20
    _report
    ;;
  mixed)
    echo "═══ 追加：5 场景 x 20 帧 = 100 帧，每帧 3 个 DJI（保留真实的多实例情形）═══"
    echo "  ⚠️ 刻意【不删】train_pbr —— BOP writer 会往同一个 000000 里追加帧"
    _run mixed 5 3 10 990777 20
    _report
    ;;
  all)
    echo "═══ main + mixed ═══"
    _clean
    _run main 45 1 10 990001 20
    _run mixed 5 3 10 990777 20
    _report
    ;;
  *)
    echo "用法: $0 {smoke|main|mixed|all}"; exit 2;;
esac

echo
echo "═══ DONE $(date '+%H:%M:%S') ═══"
