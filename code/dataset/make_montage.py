#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 manifest.csv 抽样拼一张联系表（contact sheet），用于目检 patch 质量。
=========================================================================
切完图之后第一件事就是抽几十块看一眼：是不是真组织、有没有大片空白、
有没有全是血/坏死/笔迹的块。这个脚本把抽样结果拼成一张大图，一眼看完。

用法
----
  python make_montage.py --manifest data/manifest.csv --out data/qc/montage.jpg
  python make_montage.py --manifest data/manifest.csv --slide R2301108 --n 36
  python make_montage.py --manifest data/manifest.csv --sort tissue_frac --reverse
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="patch 抽样联系表")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default=None, help="默认 <manifest 同级的 qc>/montage.jpg")
    ap.add_argument("--slide", default=None, help="只抽某张切片")
    ap.add_argument("--label", default=None, help="只抽某个类别")
    ap.add_argument("--n", type=int, default=36, help="抽样块数")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--cell", type=int, default=160, help="每格显示边长(px)")
    ap.add_argument("--sort", default="random",
                    choices=["random", "tissue_frac", "blur", "x0"],
                    help="抽样前排序；tissue_frac 可用来抽查最'边缘'的块")
    ap.add_argument("--reverse", action="store_true", help="配合 --sort 反序")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.exists():
        print(f"[error] 找不到 {mpath}")
        return 1
    root = mpath.parent

    with open(mpath, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if (not args.slide or r["slide_id"] == args.slide)
                and (not args.label or r["label"] == args.label)]
    if not rows:
        print("[error] 没有匹配的记录")
        return 1

    if args.sort == "random":
        random.Random(args.seed).shuffle(rows)
    else:
        rows.sort(key=lambda r: float(r[args.sort]), reverse=args.reverse)

    rows = rows[: args.n * 4]  # 多取一些，跳过读不出来的
    cell = args.cell
    cols = args.cols
    picked: list[tuple[np.ndarray, dict]] = []
    for r in rows:
        if len(picked) >= args.n:
            break
        p = root / r["patch_path"]
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        # 从大 patch 中心裁一块显示，避免缩小后看不清细胞
        h, w = img.shape[:2]
        c = min(h, w)
        img = img[(h - c) // 2 : (h + c) // 2, (w - c) // 2 : (w + c) // 2]
        picked.append((cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA), r))

    if not picked:
        print("[error] 一块都读不出来，检查 manifest 里的 patch_path 是否相对 manifest 目录")
        return 1

    nrow = (len(picked) + cols - 1) // cols
    pad, label_h = 4, 16
    sheet = np.full((nrow * (cell + label_h + pad) + pad,
                     cols * (cell + pad) + pad, 3), 245, np.uint8)
    for i, (img, r) in enumerate(picked):
        ry, rx = divmod(i, cols)
        y = pad + ry * (cell + label_h + pad)
        x = pad + rx * (cell + pad)
        sheet[y : y + cell, x : x + cell] = img
        txt = f"{r['slide_id'][:12]} f={float(r['tissue_frac']):.2f}"
        cv2.putText(sheet, txt, (x + 2, y + cell + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1, cv2.LINE_AA)

    out = Path(args.out) if args.out else root / "qc" / "montage.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"联系表 -> {out}  ({len(picked)} 块, 排序={args.sort}"
          f"{' 反序' if args.reverse else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
