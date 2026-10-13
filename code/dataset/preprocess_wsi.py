#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCLC / LCNEC 病理全切片 (KFB) → patch 数据集  预处理脚本
=========================================================
把江丰生物 KFB 全切片转换成病理深度学习可直接使用的 patch 数据集：
    组织区域检测 → 滑窗切块 → 丢弃背景块 → 存盘 + 生成 manifest.csv

标准做法（对齐 CLAM / GTBIS / 常规 WSI 分类流程）：
  1. 低倍缩略图 + HSV 饱和度通道 OTSU 阈值分割组织区域
     （玻片背景为高亮度低饱和，HE 染色的组织为紫红色高饱和）
  2. 形态学闭/开运算去噪 → 剔除小连通域 → 填补组织内部空洞
  3. 在目标倍率上固定步长滑窗，按"patch 内组织覆盖率 ≥ 阈值"筛选（默认 0.5）
  4. patch 命名 <slide_id>_<x>_<y>.png，x/y 为 level0 像素坐标（左上角）
  5. manifest.csv 记录每块坐标 / 组织占比 / 模糊度，便于二次筛选与训练索引

性能要点
--------
  * kfbslide 的 read_region 按"整行条带"读比随机小块快 6 倍以上；
    本脚本按行读取，且只读组织所在列区间，纯背景区间直接跳过
  * 多进程按切片并行（kfbslide 非线程安全，所以用进程不用线程）
  * --resume 断点续跑：已有 manifests/<slide_id>.csv 的切片直接跳过

用法
----
  # 1) 先看组织分割对不对（不写图，只统计）
  python preprocess_wsi.py --src SCLC --out data --dry-run

  # 2) 试跑一张
  python preprocess_wsi.py --src SCLC --out data --slides R2301108

  # 3) 正式跑（先生成 labels.csv，见 make_labels.py）
  python preprocess_wsi.py --src SCLC --out data --labels labels.csv --workers 6

输出
----
  data/
    sclc/<slide_id>/<slide_id>_<x>_<y>.png    patch 图片
    manifests/<slide_id>.csv                  单张切片清单（--resume 依据）
    manifest.csv                              全部切片合并清单
    preprocess_meta.json                      本次运行的参数与逐张统计
    qc/<slide_id>_thumb.jpg                   缩略图 + 组织轮廓（QC 用）
    qc/<slide_id>_mask.png                    二值组织掩膜（QC 用）

依赖
----
  pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
WSI_EXTS = (".kfb", ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn")
DEFAULT_FALLBACK_MPP = 0.5        # 探测不到 MPP 时的兜底（20× 常见值）
MAX_MASK_PIXELS = 120_000_000     # 掩膜层像素上限，超过就退到更粗的层，防爆内存
MANIFEST_COLS = [
    "slide_id", "label", "patient_id", "patch_path",
    "x0", "y0", "x1", "y1", "level", "mpp", "patch_size",
    "tissue_frac", "blur",
]


# --------------------------------------------------------------------------
# 1. 读取层
# --------------------------------------------------------------------------
def open_slide(path: str):
    """打开 WSI，返回 OpenSlide 兼容对象（.kfb 走 kfbslide）。"""
    ext = Path(path).suffix.lower()
    if ext == ".kfb":
        try:
            import kfbslide
        except ImportError as e:
            raise RuntimeError(
                "缺少 kfbslide，请先执行 pip install kfbslide（KFB 专用读取器）"
            ) from e
        return kfbslide.OpenSlide(path)

    try:
        import openslide
        return openslide.OpenSlide(path)
    except Exception:
        import tiffslide
        return tiffslide.open_slide(path)


def get_mpp(slide) -> float | None:
    """读 level0 物理分辨率 µm/px；读不到返回 None。"""
    props = slide.properties
    for key in ("openslide.mpp-x", "kfbio.mpp"):
        v = props.get(key)
        if v:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    power = str(props.get("openslide.objective-power", ""))
    for p, mpp in (("40", 0.25), ("20", 0.5), ("10", 1.0)):
        if p in power:
            return mpp
    return None


def pick_level_by_mpp(slide, target_mpp: float) -> int:
    """选 downsample 与目标 µm/px 最接近的金字塔层。"""
    ds = slide.level_downsamples
    return min(range(len(ds)), key=lambda i: abs(ds[i] - target_mpp))


def read_level_image(slide, level: int, chunk_rows: int = 1024) -> np.ndarray:
    """分块读入整层图像（RGB, uint8）。分块避免单次超大 read_region 失败。"""
    lw, lh = slide.level_dimensions[level]
    ds = float(slide.level_downsamples[level])
    out = np.zeros((lh, lw, 3), dtype=np.uint8)
    for y in range(0, lh, chunk_rows):
        hh = min(chunk_rows, lh - y)
        arr = np.asarray(
            slide.read_region((0, int(round(y * ds))), level, (lw, hh)).convert("RGB")
        )
        hh_eff = min(hh, arr.shape[0], lh - y)
        ww_eff = min(lw, arr.shape[1])
        out[y : y + hh_eff, :ww_eff] = arr[:hh_eff, :ww_eff]
    return out


# --------------------------------------------------------------------------
# 2. 组织区域检测
# --------------------------------------------------------------------------
def tissue_mask(rgb: np.ndarray, mask_mpp: float, max_hole_um: float = 60.0,
                min_area_um2: float = 10_000.0,
                sat_thresh: int | None = None) -> tuple[np.ndarray, int]:
    """
    HSV 饱和度通道 OTSU 分割组织，返回 (bool 掩膜, 实际使用的饱和度阈值)。

    纯饱和度阈值在"几乎全白"的切片上会被噪声带偏，所以再叠加"灰度不能太白"
    的约束；随后形态学清理、剔除小连通域、填补**小**空洞。

    空洞必须限制面积：组织在切片上常常围成一圈，圈内是裂隙 / 囊腔 / 大片空白。
    若不加限制地把所有封闭区域都填成组织，这些纯空白块会被当成 100% 组织保留下来
    （实测某块真实组织占比 19%，无限填洞后变成 94%）。这里只填小于
    max_hole_um 见方的洞，保证留下来的块里真有组织。

    Args:
        mask_mpp:    当前掩膜层每个像素的物理边长 µm
        max_hole_um: 可填补空洞的等效边长上限 µm（0 = 关闭填洞）
        min_area_um2: 连通域面积下限 µm²，更小的当碎屑剔除
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if sat_thresh is None:
        t, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sat_thresh = int(t)

    mask = ((sat > sat_thresh) & (gray < 220)).astype(np.uint8)

    # 闭运算补内部小孔 → 开运算去孤立噪点
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # 剔除小连通域（碎屑、笔迹点、灰尘）
    min_area_px = max(64, int(min_area_um2 / (mask_mpp ** 2)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        keep = np.zeros(n, dtype=np.uint8)
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_area_px).astype(np.uint8)
        mask = keep[lab]

    # 填补小空洞：先 flood fill 出所有封闭区域，再只保留面积达标的小洞
    if max_hole_um > 0:
        h, w = mask.shape
        inv = cv2.bitwise_not(mask * 255)
        ff = inv.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        # 从四角灌水；若四角本身都是组织则无法区分内外，直接跳过填洞更安全
        if any(inv[y, x] for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
            for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
                if inv[seed[1], seed[0]]:
                    cv2.floodFill(ff, ff_mask, seed, 0)
            holes = (ff > 0).astype(np.uint8)
            max_hole_px = max(1, int((max_hole_um / mask_mpp) ** 2))
            nh, labh, statsh, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
            if nh > 1:
                small = np.zeros(nh, dtype=np.uint8)
                small[1:] = (statsh[1:, cv2.CC_STAT_AREA] <= max_hole_px).astype(np.uint8)
                mask = mask | small[labh]

    return mask.astype(bool), sat_thresh


def tile_fractions(mask: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    把组织掩膜聚合成 patch 网格上的组织占比矩阵 (ny, nx)。

    cv2.INTER_AREA 缩放等价于对每个目标像素做源区域均值，
    因此 resize 到 (nx, ny) 直接得到每个 patch 的覆盖率，不需要积分图。
    """
    return cv2.resize(mask.astype(np.float32), (nx, ny), interpolation=cv2.INTER_AREA)


def find_runs(flags: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """布尔数组的连续 True 段；间隔 ≤ gap 的段合并，减少读取次数。"""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= gap + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    return runs


# --------------------------------------------------------------------------
# 3. 单张切片处理（worker 函数，必须模块级以便多进程 pickle）
# --------------------------------------------------------------------------
def process_slide(task: dict) -> dict:
    path = task["path"]
    slide_id = task["slide_id"]
    label = task["label"]
    patient_id = task.get("patient_id", "")
    out_root = Path(task["out_root"])
    ps = task["patch_size"]
    step = task["stride"]
    tissue_thr = task["tissue_thr"]
    gap = task["gap"]
    dry_run = task["dry_run"]

    t0 = time.time()
    res = {"slide_id": slide_id, "label": label, "rows": [], "n": 0, "error": None,
           "tissue_ratio": 0.0, "sat_thresh": 0, "mpp": None, "level": 0,
           "n_tile": 0, "n_pass": 0, "seconds": 0.0}

    slide = None
    try:
        slide = open_slide(path)

        # --- 3.1 MPP ---
        mpp0 = task.get("mpp_override") or get_mpp(slide)
        if not mpp0:
            mpp0 = DEFAULT_FALLBACK_MPP
        res["mpp"] = float(mpp0)

        # --- 3.2 切图层 ---
        target_mpp = task["target_mpp"]
        cut_level = pick_level_by_mpp(slide, target_mpp / mpp0) if target_mpp > 0 else 0
        ds = float(slide.level_downsamples[cut_level])
        mpp_cut = mpp0 * ds
        lw, lh = slide.level_dimensions[cut_level]
        res["level"] = cut_level

        # --- 3.3 掩膜层（比切图层更粗，省内存） ---
        mask_level = max(pick_level_by_mpp(slide, task["mask_mpp"] / mpp0), cut_level)
        while mask_level < slide.level_count - 1:
            mw, mh = slide.level_dimensions[mask_level]
            if mw * mh <= MAX_MASK_PIXELS:
                break
            mask_level += 1
        mask_ds = float(slide.level_downsamples[mask_level])
        mask_mpp = mpp0 * mask_ds

        # 可填洞上限：默认取半个 patch 的边长，保证填完洞的块里仍有真组织
        max_hole_um = task["max_hole_um"]
        if max_hole_um <= 0:
            max_hole_um = 0.5 * ps * mpp_cut

        mask, sat_thresh = tissue_mask(read_level_image(slide, mask_level),
                                       mask_mpp=mask_mpp, max_hole_um=max_hole_um,
                                       min_area_um2=task["min_area_um2"])
        res["tissue_ratio"] = float(mask.mean())
        res["sat_thresh"] = sat_thresh

        # --- 3.4 patch 网格 ---
        # 第 cx 列 patch 在切图层覆盖 [cx*step, cx*step + ps)
        nx = (lw - ps) // step + 1
        ny = (lh - ps) // step + 1
        if nx <= 0 or ny <= 0:
            raise RuntimeError(f"切片过小 ({lw}x{lh})，放不下一个 {ps}x{ps} patch")

        frac = tile_fractions(mask, nx, ny)
        res["n_tile"] = int(nx * ny)
        res["n_pass"] = int((frac >= tissue_thr).sum())

        if not dry_run and task["save_qc"]:
            _save_qc(out_root, slide_id, slide, mask, mpp_cut, ps)

        if dry_run:
            res["seconds"] = round(time.time() - t0, 1)
            return res

        # --- 3.5 输出目录 ---
        if task["layout"] == "flat":
            patch_dir = out_root / label
        else:
            patch_dir = out_root / label / slide_id
        patch_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = out_root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # 重跑同一张切片（换了阈值/尺寸）时先清掉上次的产物，
        # 否则残留的旧 patch 会和新的 manifest 对不上
        stale = list(patch_dir.glob(f"{Path(path).stem}_*.png"))
        for p in stale:
            p.unlink()
        if stale:
            print(f"    [{slide_id}] 清理旧 patch {len(stale)} 个", flush=True)

        # --- 3.6 逐行条带切块 ---
        # 预筛阈值放宽一点，避免网格量化误差把刚好达标的块漏掉
        cand_thr = max(0.02, tissue_thr * 0.8)
        gap_tiles = max(1, int(round(gap / step)))
        min_blur = task["min_blur"]
        png_c = [cv2.IMWRITE_PNG_COMPRESSION, task["png_compress"]]
        stem = Path(path).stem
        rows: list[list] = []
        n_patch = 0
        max_patches = task["max_patches"]
        stop = False

        for ty in range(ny):
            if stop:
                break
            y_l0 = int(round(ty * step * ds))
            for cx0, cx1 in find_runs(frac[ty] >= cand_thr, gap_tiles):
                # 一次读入该列区间、整条 patch 高（快 6 倍）
                w_read = (cx1 - cx0) * step + ps
                x_l0 = int(round(cx0 * step * ds))
                strip = np.asarray(
                    slide.read_region((x_l0, y_l0), cut_level, (w_read, ps)).convert("RGB")
                )[:, :, :3]
                sw, sh = strip.shape[1], strip.shape[0]

                for cx in range(cx0, cx1 + 1):
                    f = float(frac[ty, cx])
                    if f < tissue_thr:
                        continue
                    ox = (cx - cx0) * step
                    if ox + ps > sw or ps > sh:
                        continue  # 边界不足整块，丢弃
                    tile = strip[:, ox : ox + ps]

                    # 模糊度始终计算并写入 manifest（单块约 0.5ms），
                    # 这样事后可以用任意阈值筛选，不必重新切图
                    bl = float(cv2.Laplacian(
                        cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
                    if min_blur > 0 and bl < min_blur:
                        continue

                    x0 = int(round(cx * step * ds))
                    fpath = patch_dir / f"{stem}_{x0}_{y_l0}.png"
                    ok, buf = cv2.imencode(
                        ".png", np.ascontiguousarray(tile[:, :, ::-1]), png_c)
                    if not ok:
                        continue
                    fpath.write_bytes(buf.tobytes())

                    rows.append([
                        slide_id, label, patient_id,
                        str(fpath.relative_to(out_root)).replace("\\", "/"),
                        x0, y_l0, x0 + int(round(ps * ds)), y_l0 + int(round(ps * ds)),
                        cut_level, f"{mpp_cut:.4f}", ps, f"{f:.4f}", f"{bl:.1f}",
                    ])
                    n_patch += 1
                    if max_patches and n_patch >= max_patches:
                        stop = True
                        break
                    if n_patch % 500 == 0:
                        print(f"    [{slide_id}] {n_patch} 块 ({time.time()-t0:.0f}s)",
                              flush=True)
                if stop:
                    break

        # 单张切片的清单落盘 —— --resume 就靠它判断该切片是否已完成
        with open(manifest_dir / f"{slide_id}.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(MANIFEST_COLS)
            w.writerows(rows)
        res["rows"] = rows
        res["n"] = n_patch

    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        res["seconds"] = round(time.time() - t0, 1)
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass
    return res


def _save_qc(out_root: Path, slide_id: str, slide, mask, mpp_cut, ps) -> None:
    """QC 图：缩略图叠组织轮廓 + 比例尺，以及二值掩膜。"""
    qc = out_root / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    try:
        thumb = slide.get_thumbnail((1400, 1400)).convert("RGB")
        ta = np.asarray(thumb).copy()
        scale = ta.shape[1] / slide.dimensions[0]      # 缩略图相对 level0 的缩放
        m = cv2.resize(mask.astype(np.uint8), (ta.shape[1], ta.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ta, cnts, -1, (255, 0, 0), 2)

        # 比例尺：取一个 patch 的物理边长
        bar_um = ps * mpp_cut
        bar_px = max(8, int(round(bar_um * scale)))
        if bar_um >= 1000:
            txt = f"{bar_um/1000:.2f} mm"
        else:
            txt = f"{bar_um:.0f} um"
        cv2.line(ta, (14, 24), (14 + bar_px, 24), (0, 0, 255), 5)
        cv2.putText(ta, f"1 patch = {txt}", (14, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(ta[:, :, ::-1]),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            (qc / f"{slide_id}_thumb.jpg").write_bytes(buf.tobytes())
        ok, buf = cv2.imencode(".png", (m * 255))
        if ok:
            (qc / f"{slide_id}_mask.png").write_bytes(buf.tobytes())
    except Exception as e:
        print(f"    [warn] QC 图生成失败 {slide_id}: {e}", flush=True)


# --------------------------------------------------------------------------
# 4. 标签表 / 数据发现
# --------------------------------------------------------------------------
def load_labels(path: Path) -> dict[str, dict]:
    """读 labels.csv → {slide_id: {'label':..., 'patient_id':...}}。"""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = (row.get("slide_id") or "").strip()
            if sid:
                out[sid] = {"label": (row.get("label") or "").strip(),
                            "patient_id": (row.get("patient_id") or "").strip()}
    return out


def discover_slides(src: Path, labels: dict[str, dict] | None) -> list[dict]:
    """
    发现待处理切片。

    无 --labels：src 下每个子文件夹名即标签（ImageFolder 风格）；只有 SCLC/ 就是单类 sclc。
    有 --labels：标签以表为准，扫到的切片按文件名 stem 匹配（大小写不敏感）。
    """
    if src.is_file():
        files, folder_label = [src], src.parent.name
    else:
        files = [p for p in sorted(src.rglob("*"))
                 if p.suffix.lower() in WSI_EXTS and not p.name.endswith(".downloading")]
        folder_label = src.name

    tasks: list[dict] = []
    for p in files:
        stem = p.stem
        if labels is None:
            tasks.append({"path": str(p), "slide_id": stem,
                          "label": folder_label.lower(), "patient_id": ""})
            continue

        hit = labels.get(stem) or labels.get(p.name)
        if hit is None:
            low = stem.lower()
            hit = next((v for k, v in labels.items()
                        if k.lower() == low or Path(k).stem.lower() == low), None)
        if hit is None or not hit["label"]:
            print(f"  [skip] {p.name}: labels.csv 中无匹配 slide_id 或 label 为空")
            continue
        tasks.append({"path": str(p), "slide_id": stem,
                      "label": hit["label"], "patient_id": hit["patient_id"]})
    return tasks


# --------------------------------------------------------------------------
# 5. 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="KFB/WSI → patch 数据集预处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--src", default="SCLC",
                    help="输入切片目录（子文件夹名=标签）或单个切片文件")
    ap.add_argument("--out", default="data", help="输出根目录")
    ap.add_argument("--labels", default=None,
                    help="labels.csv（列：slide_id,label,patient_id[,diagnosis]）")
    ap.add_argument("--slides", default=None,
                    help="只处理指定切片，逗号分隔（调试用）")

    g = ap.add_argument_group("切块参数")
    g.add_argument("--patch-size", type=int, default=256)
    g.add_argument("--stride", type=int, default=0, help="0 = 等于 patch_size（不重叠）")
    g.add_argument("--tissue-thr", type=float, default=0.5,
                   help="patch 内组织占比下限，低于则丢弃")
    g.add_argument("--target-mpp", type=float, default=0.0,
                   help=">0 时按该 µm/px 选层切图；0 = level0 原生最高倍率")
    g.add_argument("--mask-mpp", type=float, default=2.0,
                   help="组织检测等效分辨率 µm/px（越粗越快，越细越准）")
    g.add_argument("--max-hole-um", type=float, default=0.0,
                   help="可填补空洞的等效边长上限 µm；0 = 自动取半个 patch 边长")
    g.add_argument("--min-area-um2", type=float, default=10_000.0,
                   help="组织连通域面积下限 µm²，更小的当碎屑剔除")
    g.add_argument("--mpp-override", type=float, default=0.0,
                   help="强制指定 level0 的 MPP（切片元数据不准时使用）")
    g.add_argument("--gap", type=int, default=256,
                   help="同一行内两段组织间隔小于该像素数时合并读取")
    g.add_argument("--png-compress", type=int, default=3, choices=range(0, 10),
                   help="PNG 压缩级别（无损，仅影响体积与速度）")
    g.add_argument("--min-blur", type=float, default=0.0,
                   help="拉普拉斯方差下限，低于则丢弃（0=不丢弃，但 blur 值仍会写进 "
                        "manifest；本数据集约 8%% 的块失焦，推荐 60）")
    g.add_argument("--max-patches", type=int, default=0,
                   help="每张切片最多输出块数（0=不限，试跑用）")

    o = ap.add_argument_group("输出 / 运行")
    o.add_argument("--layout", choices=["by_slide", "flat"], default="by_slide",
                   help="by_slide: <out>/<label>/<slide_id>/*.png")
    o.add_argument("--workers", type=int, default=1, help="并行进程数（按切片并行）")
    o.add_argument("--resume", action="store_true",
                   help="已有 manifests/<slide_id>.csv 的切片直接跳过")
    o.add_argument("--qc", dest="qc", action="store_true", default=True,
                   help="生成 QC 缩略图/掩膜（默认开启）")
    o.add_argument("--no-qc", dest="qc", action="store_false")
    o.add_argument("--dry-run", action="store_true",
                   help="只做组织检测并统计可切块数，不写任何文件")

    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    src = Path(args.src)
    if not src.exists():
        print(f"[error] 输入不存在: {src}", file=sys.stderr)
        return 1

    labels = None
    if args.labels:
        lp = Path(args.labels)
        if not lp.exists():
            print(f"[error] 找不到标签表: {lp}", file=sys.stderr)
            return 1
        labels = load_labels(lp)
        print(f"标签表: {lp}（{len(labels)} 条）")

    tasks = discover_slides(src, labels)
    if args.slides:
        want = {s.strip() for s in args.slides.split(",")}
        tasks = [t for t in tasks if t["slide_id"] in want]
    if not tasks:
        print(f"[error] {src} 下没有可处理的切片", file=sys.stderr)
        return 1

    if args.resume and not args.dry_run:
        mdir = out_root / "manifests"
        done = {p.stem for p in mdir.glob("*.csv")} if mdir.exists() else set()
        before = len(tasks)
        tasks = [t for t in tasks if t["slide_id"] not in done]
        print(f"resume: 跳过已完成 {before - len(tasks)} 张，剩余 {len(tasks)} 张")
        if not tasks:
            print("没有需要处理的切片。")
            return 0

    stride = args.stride if args.stride > 0 else args.patch_size
    common = dict(out_root=str(out_root), patch_size=args.patch_size, stride=stride,
                  tissue_thr=args.tissue_thr, mask_mpp=args.mask_mpp,
                  max_hole_um=args.max_hole_um, min_area_um2=args.min_area_um2,
                  target_mpp=args.target_mpp, png_compress=args.png_compress,
                  min_blur=args.min_blur, gap=args.gap, layout=args.layout,
                  save_qc=args.qc, max_patches=args.max_patches, dry_run=args.dry_run)
    if args.mpp_override > 0:
        common["mpp_override"] = args.mpp_override
    for t in tasks:
        t.update(common)

    print(f"\n待处理 {len(tasks)} 张切片 | patch={args.patch_size}px "
          f"组织阈值={args.tissue_thr} | 输出 -> {out_root}\n")

    t0 = time.time()
    results: list[dict] = []
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_slide, t): t for t in tasks}
            for i, fu in enumerate(as_completed(futs), 1):
                r = fu.result()
                results.append(r)
                _print_result(i, len(tasks), r, args.dry_run)
    else:
        for i, t in enumerate(tasks, 1):
            r = process_slide(t)
            results.append(r)
            _print_result(i, len(tasks), r, args.dry_run)

    results.sort(key=lambda r: r["slide_id"])

    # ---------- dry-run 报告 ----------
    if args.dry_run:
        print("\n" + "=" * 76)
        print(f"{'slide_id':<26}{'组织占比':>10}{'可切块':>10}{'网格总数':>10}{'通过率':>9}")
        print("-" * 76)
        tp = tt = 0
        for r in results:
            if r["error"]:
                print(f"{r['slide_id']:<26}ERROR {r['error']}")
                continue
            tp += r["n_pass"]
            tt += r["n_tile"]
            print(f"{r['slide_id']:<26}{r['tissue_ratio']*100:>9.1f}%{r['n_pass']:>10}"
                  f"{r['n_tile']:>10}{(r['n_pass']/r['n_tile']*100 if r['n_tile'] else 0):>8.1f}%")
        print("-" * 76)
        print(f"合计 {tp} / {tt} 块可通过组织筛选"
              f"（{tp/tt*100 if tt else 0:.1f}%），耗时 {time.time()-t0:.0f}s")
        print(f"QC 图未生成（dry-run 不写文件），确认统计合理后去掉 --dry-run 正式跑")
        return 0

    # ---------- 合并 manifest ----------
    all_rows: list[list] = []
    n_err = 0
    for r in results:
        if r["error"]:
            n_err += 1
        else:
            all_rows.extend(r["rows"])
    all_rows.sort(key=lambda r: (r[0], int(r[4]), int(r[5])))

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLS)
        w.writerows(all_rows)

    meta = {
        "src": str(src), "labels": str(args.labels) if args.labels else None,
        "patch_size": args.patch_size, "stride": stride,
        "tissue_thr": args.tissue_thr,
        "target_mpp": args.target_mpp if args.target_mpp > 0 else "level0",
        "mask_mpp": args.mask_mpp, "min_blur": args.min_blur,
        "max_hole_um": args.max_hole_um or "auto(0.5*patch)",
        "min_area_um2": args.min_area_um2,
        "n_slides": len(tasks), "n_slides_failed": n_err,
        "n_patches": len(all_rows), "elapsed_s": round(time.time() - t0, 1),
        "slides": [{"slide_id": r["slide_id"], "label": r["label"], "mpp": r["mpp"],
                    "level": r["level"], "n_patches": r["n"],
                    "n_tile_grid": r["n_tile"],
                    "tissue_ratio": round(r["tissue_ratio"], 4),
                    "sat_thresh": r["sat_thresh"], "seconds": r["seconds"],
                    "error": r["error"]} for r in results],
    }
    (out_root / "preprocess_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 收尾报告 ----------
    print("\n" + "=" * 76)
    print(f"{'slide_id':<24}{'label':<8}{'块数':>8}{'组织占比':>10}{'耗时':>9}  状态")
    print("-" * 76)
    for r in results:
        print(f"{r['slide_id']:<24}{r['label']:<8}{r['n']:>8}"
              f"{r['tissue_ratio']*100:>9.1f}%{r['seconds']:>8.0f}s  "
              f"{'ERROR ' + r['error'] if r['error'] else 'ok'}")
    print("-" * 76)
    print(f"总计 {len(all_rows)} 个 patch，{len(tasks)-n_err}/{len(tasks)} 张切片成功，"
          f"耗时 {time.time()-t0:.0f}s")
    by_label: dict[str, int] = {}
    for r in all_rows:
        by_label[r[1]] = by_label.get(r[1], 0) + 1
    print("按类别: " + ", ".join(f"{k}={v}" for k, v in sorted(by_label.items())))
    print(f"\n清单     -> {manifest}")
    print(f"元信息   -> {out_root / 'preprocess_meta.json'}")
    if args.qc:
        print(f"QC 图    -> {out_root / 'qc'}  "
              f"（建议先看 <slide_id>_thumb.jpg 确认组织轮廓是否贴合）")
    return 1 if n_err else 0


def _print_result(i: int, n: int, r: dict, dry_run: bool) -> None:
    if r["error"]:
        print(f"[{i}/{n}] {r['slide_id']}: ERROR {r['error']}", flush=True)
    elif dry_run:
        print(f"[{i}/{n}] {r['slide_id']}: 组织 {r['tissue_ratio']*100:.1f}%, "
              f"可切 {r['n_pass']}/{r['n_tile']} 块 ({r['seconds']:.0f}s)", flush=True)
    else:
        print(f"[{i}/{n}] {r['slide_id']}: {r['n']} 块 "
              f"(组织 {r['tissue_ratio']*100:.1f}%, {r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
