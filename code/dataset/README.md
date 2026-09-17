# KFB 病理切片 → patch 数据集 预处理

把江丰生物 KFB 全切片转成病理深度学习可直接用的 patch 数据集（组织检测 → 切块 → 清单）。

```
make_labels.py  →  labels.csv      标签表（可选，用于按患者划分）
preprocess_wsi.py → data/          组织检测 + 切块 + manifest.csv
make_montage.py →  联系表          抽样目检 patch 质量
```

## 快速开始

脚本在 `code/dataset/` 下，**以下命令都在项目根目录 `SCLC-LCNEC/` 执行**。

```bash
pip install -r code/dataset/requirements.txt

# 1) 生成标签表（按病理号从病理系统 Excel 补患者号）
python code/dataset/make_labels.py --src SCLC --label sclc --xls "简单查询-肺小.xls" --out labels.csv

# 2) 先确认组织分割是否合理（不写文件，26 秒扫完全部切片）
python code/dataset/preprocess_wsi.py --src SCLC --out data --dry-run

# 3) 正式切图
python code/dataset/preprocess_wsi.py --src SCLC --out data --labels labels.csv --workers 8

# 4) 抽样目检
python code/dataset/make_montage.py --manifest data/manifest.csv --sort tissue_frac --n 36
```

## 输出结构

```
data/
  sclc/<slide_id>/<slide_id>_<x>_<y>.png    patch（256×256 @ 20×，命名含 level0 坐标）
  manifests/<slide_id>.csv                   单张切片清单（--resume 依据）
  manifest.csv                               全部切片合并清单 ← 训练时读这个
  preprocess_meta.json                       本次运行参数 + 逐张统计（复现用）
  qc/<slide_id>_thumb.jpg                    缩略图 + 组织轮廓（先看这个确认分割）
  qc/<slide_id>_mask.png                     二值组织掩膜
  qc/montage*.jpg                            patch 抽样联系表
```

## manifest.csv 列

| 列 | 说明 |
|---|---|
| `slide_id` / `label` / `patient_id` | 切片号 / 类别 / 患者号（后者用于按患者划分，防止同一患者多张切片泄漏） |
| `patch_path` | 相对 `data/` 的路径 |
| `x0,y0,x1,y1` | level0 像素坐标，可回到原片定位 |
| `level` / `mpp` / `patch_size` | 切图金字塔层 / 物理分辨率 µm/px / 边长 |
| `tissue_frac` | 该块组织覆盖率（0~1） |
| `blur` | 拉普拉斯方差，越低越模糊 |

**切完后想换筛选标准，不必重新切图**——`tissue_frac` 和 `blur` 都在清单里，直接筛：

```python
import pandas as pd
d = pd.read_csv("data/manifest.csv")
d = d[(d.tissue_frac >= 0.6) & (d.blur >= 60)]   # 更严的组织阈值 + 去掉失焦块
d.to_csv("data/manifest_filtered.csv", index=False)
```

本数据集统计：13764 块，约 **7.7% 的块 blur<60（失焦）**，`blur` 分布双峰
（5% 分位 20，10% 分位 289），所以阈值取 **50~250 之间**效果稳定；
直接用 `--min-blur 60` 重切也可以。

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--patch-size` | 256 | patch 边长（像素） |
| `--stride` | =patch_size | 滑窗步长，小于 patch_size 即重叠切块 |
| `--tissue-thr` | 0.5 | 组织覆盖率下限。肺组织含大量肺泡空腔，0.5 是合适值；调高会更严 |
| `--target-mpp` | 0（level0） | >0 时按该 µm/px 选金字塔层（如 `1.0` ≈ 10×） |
| `--mask-mpp` | 2.0 | 组织检测的分辨率，越粗越快 |
| `--max-hole-um` | 半块边长 | **可填补空洞的边长上限**，见下方"已知陷阱" |
| `--min-area-um2` | 10000 | 组织连通域面积下限，滤掉碎屑 |
| `--min-blur` | 0 | 拉普拉斯方差下限，0 = 不丢弃（blur 仍记录） |
| `--workers` | 1 | 并行进程数（按切片并行） |
| `--resume` | — | 已有 `manifests/<slide_id>.csv` 的切片跳过，支持断点续跑 |
| `--layout flat` | by_slide | 改成 `<out>/<label>/*.png` 扁平结构（配 `torchvision.ImageFolder`） |

## 已知陷阱（踩过的坑）

1. **填洞必须限面积。** 组织在切片上常围成一圈，圈内是裂隙/囊腔/大片空白。
   若不加限制地把所有封闭区域填成组织，这些纯空白块会以 `tissue_frac≈1.0` 被保留。
   实测某块真实组织占比 21.7%，无限填洞后变成 94%。现在只填小于半个 patch 的洞。

2. **`read_region` 要按条带读。** 随机读 256×256 小块只有 5 Mpx/s，
   整行条带读能到 42 Mpx/s（快 8 倍）。本脚本按行读，且只读组织所在列区间。

3. **改参数重跑会留僵尸文件。** 同一张切片重跑时会先清掉 `{slide_id}_*.png`，
   否则旧的 patch 会和新 manifest 对不上。

4. **`kfbslide` 非线程安全**，并行要用多进程（`--workers`），不要用多线程。

5. **患者级划分。** `patient_id` 已写进 manifest。划分 train/val/test 时务必
   按 `patient_id` 分组切分，同一患者的多张切片不能跨集合——否则评估指标会虚高。

## 其它格式

`preprocess_wsi.py` 按扩展名自动选读取器：`.kfb` 走 `kfbslide`，
其余（`.svs/.tif/.ndpi/.mrxs/.scn`）走 `openslide`，失败则回退 `tiffslide`。
`--src` 指向的目录下若混有多种格式会一起处理；`.downloading` 等未下完的文件会自动跳过。

## 多类别

当前只有 `SCLC/` 一个文件夹。加入 `LCNEC/` 后：

```bash
# 方式一：靠文件夹名当标签（不加 --labels）
python code/dataset/preprocess_wsi.py --src . --out data --workers 8
# 输出 data/sclc/... 和 data/lcnc/...

# 方式二：标签表里写清楚（推荐，能带 patient_id）
python code/dataset/make_labels.py --src . --xls "简单查询-肺小.xls" --out labels.csv
```
