# 汇报 PPT（极简版）

- **成品**：[`minimal.pptx`](minimal.pptx) —— 10 页，由 `scripts/make_ppt_minimal.py` 生成
- **预览**：`preview/slide01.png` ~ `slide10.png`（用本机 PowerPoint 导出，与 PPT 实际渲染一致）

## 结构：照 `docs/HUMAN_LOG.md` 末尾「交付」的三条主线

| 页 | 内容 |
|:---:|---|
| 1 | 封面：题目 + 一句话结论 |
| 2 | 任务与交付物：交付的是 **2D 8 角点**，不需要相机内参 |
| 3 | 五步流水线（含阶段①产出的三维模型） |
| 4 | **数据集主线（1/2）**：第一版 —— 多个 DJI 随机摆放；埋下的隐患 |
| 5 | **数据集主线（2/2）**：第二版 —— 屏幕贴平面 + Objaverse 异类实例 |
| 6 | **模型主线**：架构 + 迭代历程表（v1 → DINOv2 → GEO7 → 多实例 → 增强） |
| 7 | **效果（1/2）**：自建数据集上可用（4.12 px / PCK@0.05 96.9%） |
| 8 | **效果（2/2）**：目标视频上还不可用；**排名反转** |
| 9 | 已知缺陷与根因（6 条，都定位到具体机制） |
| 10 | 教训与下一步 |

## 重新生成 / 改内容

```bash
python scripts/make_ppt_minimal.py                 # 默认输出 docs/ppt/minimal.pptx
python scripts/make_ppt_minimal.py --out x.pptx
```

图文内容全在脚本里（`build()` 函数），改文字直接改脚本；图取自
[`docs/figures/`](../figures/README.md)。要重出预览页：

```powershell
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open("<绝对路径>\minimal.pptx", $true, $false, $false)
$pres.Export("<绝对路径>\preview", "PNG", 1600, 900); $pres.Close(); $ppt.Quit()
```

> ⚠️ 两个已经踩过的坑：
> 1. **中文必须同时设 `a:ea`**。python-pptx 的 `run.font.name` 只影响西文，
>    东亚文字走 `<a:ea typeface="...">`，不设会退化成默认字体。
> 2. **图片框要匹配长宽比**。竖排多帧的长条图（如 `v1_fullframe_gt_projection.jpg` 是 900×2114）
>    塞进横向框里会被压成细柱 —— 脚本里先用 PIL 裁出单帧再用。
