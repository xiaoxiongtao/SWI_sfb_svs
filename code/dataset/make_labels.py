#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 labels.csv 标签表 —— 给 preprocess_wsi.py --labels 用
==========================================================
扫切片目录里的文件名，和病理系统导出的 Excel 按「病理号」对接，
产出标准三列（+1 列参考诊断）：

    slide_id,label,patient_id,diagnosis

  slide_id   = 切片文件名（不含扩展名），必须和 kfb 文件名一致
  label      = 类别，默认取 --label；若目录是「一个子文件夹一个类别」则自动用文件夹名
  patient_id = 来自 Excel 的患者Id —— 用于后续按患者划分 train/val/test，防止同一患者
               的多张切片泄漏到不同集合（病理研究里这是最常见的评估虚高来源）
  diagnosis  = 病理诊断原文，仅供人工核对 label 用，不参与训练

隐私提示
--------
病理系统导出的 Excel 里通常含姓名、住院号等直接身份信息。本脚本只提取
病理号 / 患者Id / 诊断结果三列，不写出任何直接身份字段；生成的 labels.csv
仍属受控临床数据，别提交到公开仓库。

用法
----
  # 单类别，从 Excel 补患者号
  python make_labels.py --src SCLC --label sclc --xls ../简单查询-肺小.xls --out labels.csv

  # 多类别（SCLC/ 与 LCNEC/ 各一个文件夹），类别自动取文件夹名
  python make_labels.py --src . --xls ../简单查询-肺小.xls --out labels.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

WSI_EXTS = (".kfb", ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn")

# Excel 导出表里可能出现的列名（不同医院系统叫法不同）
CASE_COL_CANDIDATES = ("病理号", "病理号/条码号", "样本编号", "蜡块号", "case_id", "slide_id")
PATIENT_COL_CANDIDATES = ("患者Id", "患者ID", "患者id", "病人ID", "病案号", "patient_id")
DIAG_COL_CANDIDATES = ("诊断结果", "病理诊断", "诊断", "diagnosis")

ID_LIKE = ("R",)  # 常见病理号前缀，仅用于提示，不参与过滤


def pick_col(df, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # 大小写/空格不敏感兜底
    norm = {str(c).strip().lower().replace(" ", ""): c for c in df.columns}
    for c in candidates:
        hit = norm.get(c.strip().lower().replace(" ", ""))
        if hit:
            return hit
    return None


def load_case_table(xls: Path) -> dict[str, dict]:
    """读 Excel → {病理号: {patient_id, diagnosis}}。"""
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("需要 pandas：pip install pandas xlrd openpyxl")

    try:
        df = pd.read_excel(xls, dtype=str)
    except Exception as e:
        raise SystemExit(f"读取 {xls} 失败：{e}\n"
                         f".xls 需要 xlrd，.xlsx 需要 openpyxl：pip install xlrd openpyxl")

    case_col = pick_col(df, list(CASE_COL_CANDIDATES))
    if case_col is None:
        raise SystemExit(f"{xls} 里找不到病理号列，现有列：{list(df.columns)[:20]}")
    pat_col = pick_col(df, list(PATIENT_COL_CANDIDATES))
    diag_col = pick_col(df, list(DIAG_COL_CANDIDATES))
    print(f"Excel: {xls.name}  行数={len(df)}")
    print(f"  病理号列 = {case_col}")
    print(f"  患者Id列 = {pat_col or '(未找到)'}")
    print(f"  诊断列   = {diag_col or '(未找到)'}")

    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = str(row[case_col]).strip()
        if not key or key.lower() == "nan":
            continue
        diag = ""
        if diag_col:
            diag = str(row[diag_col]).replace("\n", " ").replace("\r", " ").strip()
            diag = " ".join(diag.split())
        pat = ""
        if pat_col:
            v = str(row[pat_col]).strip()
            pat = "" if v.lower() == "nan" else v
            if pat.endswith(".0"):          # Excel 把整数 ID 读成 2861545.0
                pat = pat[:-2]
        out[key.upper()] = {"patient_id": pat, "diagnosis": diag}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 labels.csv")
    ap.add_argument("--src", default="SCLC", help="切片目录（一个子文件夹一个类别）或单个文件")
    ap.add_argument("--out", default="labels.csv")
    ap.add_argument("--label", default=None,
                    help="统一类别名；不给则用子文件夹名（src 直接是切片目录时必填）")
    ap.add_argument("--xls", default=None, help="病理系统导出的 Excel，按病理号补 patient_id/diagnosis")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[error] 输入不存在: {src}", file=sys.stderr)
        return 1

    # 收集 (文件, 类别)
    items: list[tuple[Path, str]] = []
    if src.is_file():
        items = [(src, args.label or src.parent.name.lower())]
    else:
        subdirs = [d for d in sorted(src.iterdir()) if d.is_dir()]
        direct = [p for p in sorted(src.iterdir())
                  if p.suffix.lower() in WSI_EXTS and not p.name.endswith(".downloading")]
        for p in direct:
            items.append((p, args.label or src.name.lower()))
        for d in subdirs:
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in WSI_EXTS and not p.name.endswith(".downloading"):
                    items.append((p, args.label or d.name.lower()))

    if not items:
        print(f"[error] {src} 下没找到切片", file=sys.stderr)
        return 1
    if any(not lab for _, lab in items):
        print("[error] 有切片无法确定类别，请用 --label 指定", file=sys.stderr)
        return 1

    cases = load_case_table(Path(args.xls)) if args.xls else {}

    rows, unmatched = [], []
    for p, lab in items:
        sid = p.stem
        info = cases.get(sid.upper()) if cases else None
        if cases and info is None:
            # 大小写/扩展名不一致时再试一次
            info = next((v for k, v in cases.items() if k == sid.upper()), None)
            unmatched.append(sid)
        rows.append({
            "slide_id": sid,
            "label": lab,
            "patient_id": (info or {}).get("patient_id", ""),
            "diagnosis": (info or {}).get("diagnosis", ""),
        })

    rows.sort(key=lambda r: (r["label"], r["slide_id"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["slide_id", "label", "patient_id", "diagnosis"])
        w.writeheader()
        w.writerows(rows)

    n_by_label: dict[str, int] = {}
    for r in rows:
        n_by_label[r["label"]] = n_by_label.get(r["label"], 0) + 1
    n_pat = len({r["patient_id"] for r in rows if r["patient_id"]})
    n_with_pat = sum(1 for r in rows if r["patient_id"])

    print(f"\n写出 {out}")
    print(f"  切片 {len(rows)} 张，类别 " +
          ", ".join(f"{k}={v}" for k, v in sorted(n_by_label.items())))
    if cases:
        print(f"  匹配到患者号 {n_with_pat}/{len(rows)} 张，涉及 {n_pat} 个患者")
        if unmatched:
            print(f"  [注意] 以下 {len(unmatched)} 张在 Excel 里没有对应病理号，"
                  f"patient_id 留空（这些切片将无法参与按患者的划分）：")
            for s in unmatched[:20]:
                print(f"      {s}")
            if len(unmatched) > 20:
                print(f"      ... 另外 {len(unmatched) - 20} 张")
    print("\n下一步（在项目根目录执行）：")
    print(f"  python code/dataset/preprocess_wsi.py --src {args.src} --out data "
          f"--labels {out} --workers 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
