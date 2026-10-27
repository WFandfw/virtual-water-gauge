# -*- coding: utf-8 -*-
"""水槽数据集核查：结构完整性 + 标注对账 + 像素误差统计。

用法：
    python eval_kit/audit_dataset.py --root "C:/Users/a1595/Desktop/AIC/水槽数据集" --outdir audit_out

产出：
    audit_out/dataset_audit.md      核查报告（结构、缺口、异常）
    audit_out/pixel_metrics.md      像素误差指标（可直接进 5.3）
    audit_out/records_from_dataset.csv  按 records_template 的字段整理出的记录（真实数据）
"""

import argparse
import hashlib
import os
import re
import statistics

import pandas as pd

SLOPE_SETS = {
    "混凝土数据": "混凝土",
    "浆砌石数据": "浆砌石",
    "砂石数据": "砂石",
}
NEED_FILES = ["original.jpg", "mask_orig.jpg", "result.jpg"]


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def px_stats(diff_px):
    """diff_px: 带符号像素差 list（模型-人工）。"""
    d = [abs(x) for x in diff_px if x is not None]
    if not d:
        return None
    s = pd.Series(d)
    return {
        "n": len(d),
        "MAE_px": s.mean(),
        "RMSE_px": (pd.Series(diff_px) ** 2).mean() ** 0.5,
        "P95_px": s.quantile(0.95),
        "Max_px": s.max(),
        "Bias_px": statistics.mean(diff_px),
        "<=5px": (s <= 5).mean() * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", default="audit_out")
    args = ap.parse_args()

    R = os.path.abspath(args.root)
    os.makedirs(args.outdir, exist_ok=True)

    md = ["# 水槽数据集核查报告", "", f"根目录：`{R}`", ""]
    px_md = ["# 像素误差指标（由标注表直接计算）", ""]
    records = []
    issues = []

    # ---------------- 一、三类边坡数据 ----------------
    md += ["## 一、三类边坡数据（逐样本文件夹 + 人工标注）", ""]
    md += ["| 数据集 | 文件夹数 | 标注行数 | 图片数 | 缺失文件 | 编号对不上的 | 图高分布 |",
           "|---|---|---|---|---|---|---|"]
    px_md += ["## 一、三类边坡（模型坐标 vs 人工坐标）", "",
              "| 数据集 | 样本数 | MAE (px) | RMSE (px) | P95 (px) | 最大 (px) | Bias (px) | ≤5px 占比 |",
              "|---|---|---|---|---|---|---|---|"]

    all_diff = []
    for dname, label in SLOPE_SETS.items():
        d = os.path.join(R, dname)
        if not os.path.isdir(d):
            issues.append(f"缺目录：{d}")
            md.append(f"| {label} | — | — | — | 目录不存在 | — | — |")
            continue

        folders = sorted([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))])
        xlsx = os.path.join(d, "水位线Y坐标.xlsx")
        ann = pd.read_excel(xlsx) if os.path.isfile(xlsx) else pd.DataFrame()
        ann["序号"] = pd.to_numeric(ann.get("序号"), errors="coerce")

        n_img, miss = 0, []
        heights = {}
        for f in folders:
            for fn in NEED_FILES:
                p = os.path.join(d, f, fn)
                if os.path.isfile(p):
                    n_img += 1
                else:
                    miss.append(f"{f}/{fn}")
            op = os.path.join(d, f, "original.jpg")
            if os.path.isfile(op):
                h = image_hw(op)[0]
                heights[h] = heights.get(h, 0) + 1

        fset = {int(x) for x in folders if x.isdigit()}
        aset = {int(x) for x in ann["序号"].dropna()}
        only_f = sorted(fset - aset)
        only_a = sorted(aset - fset)
        if only_f:
            issues.append(f"{label}：有文件夹但无标注 → {only_f[:10]}（共 {len(only_f)}）")
        if only_a:
            issues.append(f"{label}：有标注但无文件夹 → {only_a[:10]}（共 {len(only_a)}）")
        if miss:
            issues.append(f"{label}：缺文件 {len(miss)} 个 → {miss[:6]}")

        hs = "、".join(f"{k}px×{v}" for k, v in sorted(heights.items()))
        md.append(f"| {label} | {len(folders)} | {len(ann)} | {n_img} | {len(miss)} | "
                  f"{len(only_f)}+{len(only_a)} | {hs} |")

        if not ann.empty and {"模型坐标", "人工坐标"}.issubset(ann.columns):
            ann["模型坐标"] = pd.to_numeric(ann["模型坐标"], errors="coerce")
            ann["人工坐标"] = pd.to_numeric(ann["人工坐标"], errors="coerce")
            ok = ann.dropna(subset=["模型坐标", "人工坐标"])
            diff = (ok["模型坐标"] - ok["人工坐标"]).astype(float).tolist()
            all_diff += diff
            st = px_stats(diff)
            if st:
                px_md.append(f"| {label} | {st['n']} | {st['MAE_px']:.2f} | {st['RMSE_px']:.2f} | "
                             f"{st['P95_px']:.1f} | {st['Max_px']:.0f} | {st['Bias_px']:+.2f} | "
                             f"{st['<=5px']:.1f}% |")
            for _, r in ok.iterrows():
                sid = int(r["序号"])
                folder = f"{sid:04d}"
                records.append({
                    "sample_id": f"{label}-{sid:04d}",
                    "site_type": label,
                    "site_name": f"{label}数据集",
                    "condition": "正常",
                    "water_level_true_m": "",
                    "true_source": "人工标注水位线像素坐标",
                    "true_readings": f"{r['人工坐标']:.0f}",
                    "image_file": f"{dname}/{folder}/original.jpg",
                    "csv_file": "",
                    "mode": "单帧",
                    "frame_files": "",
                    "waterline_gt_px": f"{r['人工坐标']:.0f}",
                    "waterline_px": f"{r['模型坐标']:.0f}",
                    "water_level_pred_m": "",
                    "abs_err_mm": "",
                    "note": "标注表 waterlineY坐标.xlsx 自带模型坐标",
                })

    # ---------------- 二、十五里河数据 ----------------
    md += ["", "## 二、十五里河数据集（原图/分割图 + 理论值对比）", ""]
    xl = os.path.join(R, "十五里河数据集", "挑选图片信息.xlsx")
    if os.path.isfile(xl):
        ann2 = pd.read_excel(xl)
        d2 = os.path.join(R, "十五里河数据集")
        n_orig = len([f for f in os.listdir(os.path.join(d2, "原图"))]) if os.path.isdir(os.path.join(d2, "原图")) else 0
        n_seg = len([f for f in os.listdir(os.path.join(d2, "分割图"))]) if os.path.isdir(os.path.join(d2, "分割图")) else 0
        md += [f"- 标注行数：**{len(ann2)}**，原图 **{n_orig}**，分割图 **{n_seg}**"]
        md += [f"- 列：{'、'.join(map(str, ann2.columns))}"]

        c_meas, c_theo = "测量水位线", "理论值"
        if {c_meas, c_theo}.issubset(ann2.columns):
            a = ann2.dropna(subset=[c_meas, c_theo])
            diff2 = (pd.to_numeric(a[c_meas], errors="coerce")
                     - pd.to_numeric(a[c_theo], errors="coerce")).dropna().tolist()
            st = px_stats(diff2)
            if st:
                md += ["", f"- 像素误差：MAE **{st['MAE_px']:.2f} px**，RMSE {st['RMSE_px']:.2f} px，"
                           f"P95 {st['P95_px']:.1f} px，最大 {st['Max_px']:.0f} px，"
                           f"Bias {st['Bias_px']:+.2f} px，≤5px 占比 {st['<=5px']:.1f}%"]
                px_md += ["", "## 二、十五里河（测量水位线 vs 理论值）", "",
                          f"n={st['n']}，MAE {st['MAE_px']:.2f} px，RMSE {st['RMSE_px']:.2f} px，"
                          f"P95 {st['P95_px']:.1f} px，最大 {st['Max_px']:.0f} px，Bias {st['Bias_px']:+.2f} px，"
                          f"≤5px 占比 {st['<=5px']:.1f}%"]
            all_diff += diff2
            for _, r in a.iterrows():
                sid = int(r["编号"])
                records.append({
                    "sample_id": f"十五里河-{sid:04d}",
                    "site_type": "十五里河",
                    "site_name": "十五里河",
                    "condition": "正常",
                    "water_level_true_m": "",
                    "true_source": "理论值（像素）",
                    "true_readings": f"{r[c_theo]:.0f}",
                    "image_file": f"十五里河数据集/原图/{sid:04d}.jpg",
                    "csv_file": "",
                    "mode": "单帧",
                    "frame_files": "",
                    "waterline_gt_px": f"{r[c_theo]:.0f}",
                    "waterline_px": f"{r[c_meas]:.0f}",
                    "water_level_pred_m": "",
                    "abs_err_mm": "",
                    "note": "挑选图片信息.xlsx 自带测量值与理论值",
                })
        # 原图与分割图配对检查
        if n_orig != n_seg:
            issues.append(f"十五里河：原图 {n_orig} 张 ≠ 分割图 {n_seg} 张")
    else:
        issues.append(f"缺标注表：{xl}")

    # ---------------- 三、合计 ----------------
    st_all = px_stats(all_diff)
    if st_all:
        md += ["", "## 三、合计（全部有标注的样本）", "",
               f"- 样本总数：**{st_all['n']}**",
               f"- 像素误差：MAE **{st_all['MAE_px']:.2f} px**，RMSE **{st_all['RMSE_px']:.2f} px**，"
               f"P95 **{st_all['P95_px']:.1f} px**，最大 **{st_all['Max_px']:.0f} px**，"
               f"Bias {st_all['Bias_px']:+.2f} px",
               f"- 误差 ≤5 px 的样本占比：**{st_all['<=5px']:.1f}%**"]
        px_md += ["", "## 三、合计", "",
                  f"n={st_all['n']}，MAE **{st_all['MAE_px']:.2f} px**，RMSE **{st_all['RMSE_px']:.2f} px**，"
                  f"P95 {st_all['P95_px']:.1f} px，最大 {st_all['Max_px']:.0f} px，"
                  f"≤5px 占比 **{st_all['<=5px']:.1f}%**"]

    # ---------------- 四、问题清单 ----------------
    md += ["", "## 四、发现的问题", ""]
    md += [f"{i}. {s}" for i, s in enumerate(issues, 1)] or ["未发现结构性问题。"]

    md += ["", "## 五、说明", "",
           "- 三类边坡数据每个样本文件夹含 `original.jpg`（原图）、`mask_orig.jpg`（掩码）、`result.jpg`（结果图），"
           "标注表里的「模型坐标/人工坐标」即系统输出与人工标注的水位线像素 y，可直接算像素误差。",
           "- 标注表里的「误差」列是空的，本脚本按 `模型坐标 − 人工坐标` 现算，未采用该列。",
           "- 本数据只含**像素坐标**，不含真实水位（米），因此只能支撑「像素误差 <5 px」这条指标，"
           "米制 MAE/RMSE 仍需配 CSV 标定表换算后才能得出。"]

    with open(os.path.join(args.outdir, "dataset_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.outdir, "pixel_metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(px_md))

    rec = pd.DataFrame(records)
    rec.to_csv(os.path.join(args.outdir, "records_from_dataset.csv"),
               index=False, encoding="utf-8-sig")

    print("\n".join(md))
    print("\n" + "\n".join(px_md))
    print(f"\n[audit] 记录表：{os.path.join(args.outdir, 'records_from_dataset.csv')}（{len(rec)} 行）")
    return 0


def image_hw(path):
    """不依赖 PIL 也能读 JPEG 尺寸：解析 SOF 段。"""
    try:
        with open(path, "rb") as f:
            data = f.read()
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                return h, w
            seg = int.from_bytes(data[i + 2:i + 4], "big")
            i += 2 + seg
    except Exception:
        pass
    return (-1, -1)


if __name__ == "__main__":
    raise SystemExit(main())
