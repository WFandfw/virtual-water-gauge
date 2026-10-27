# -*- coding: utf-8 -*-
"""从测试记录表计算第五章 5.3 的全部指标（离线、无需 GPU）。

用法：
    python eval_kit/compute_metrics.py --records results.csv --outdir eval_out

输入：records_template.csv 填好的记录（至少要有 water_level_true_m 与 water_level_pred_m）
输出：
    eval_out/metrics_summary.md     可直接粘进报告的表格（表9 / 表10 / 表11）
    eval_out/metrics_by_group.csv   分场景汇总明细
    eval_out/fig5_error_distribution.png  图5（误差散点 + 直方图 + 分组箱线）
    eval_out/check_report.md        记录自检报告（缺列、组数、真值一致性）

设计原则：所有数字都由本脚本从原始记录算出，报告里不手抄。
"""

import argparse
import os
import statistics
import sys

import pandas as pd

# ---------------------------------------------------------------- 字段约定
TRUE_COL = "water_level_true_m"     # 基准真值（米）
PRED_COL = "water_level_pred_m"     # 系统读数（米）
PX_COL = "waterline_px"             # 系统输出的水位线像素 y
GT_PX_COL = "waterline_gt_px"       # 人工标注的水位线像素 y（可选，用于像素误差）
GROUP_COLS = ["condition", "site_type"]


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 丢掉模板里的示例行
    if "sample_id" in df.columns:
        demo = df["sample_id"].astype(str).str.startswith("【示例")
        if demo.any():
            print(f"[metrics] 跳过 {int(demo.sum())} 行示例数据")
            df = df[~demo].copy()
    for c in (TRUE_COL, PRED_COL, PX_COL, GT_PX_COL):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def check(df):
    """生成自检报告：缺值、组数、真值一致性。"""
    lines = ["# 记录自检报告", ""]
    n = len(df)
    lines.append(f"- 记录条数：**{n}**" + ("（与报告的 285 组一致）" if n == 285 else "（⚠️ 与报告中写的 285 组不一致，二者必须统一）"))

    miss = df[df[PRED_COL].isna()]
    lines.append(f"- 缺系统读数：**{len(miss)}** 条" + (f"（{', '.join(map(str, miss['sample_id'].head(8)))}…）" if len(miss) else ""))
    if TRUE_COL in df.columns:
        lines.append(f"- 缺真值：**{int(df[TRUE_COL].isna().sum())}** 条")

    # 真值一致性：true_readings 的 9 次读数中位数应等于 water_level_true_m（容差 1mm）
    if "true_readings" in df.columns:
        bad = []
        for _, r in df.iterrows():
            s = str(r.get("true_readings", "") or "")
            vals = []
            for tok in s.replace("；", "|").replace(";", "|").split("|"):
                tok = tok.strip()
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if len(vals) >= 3 and pd.notna(r[TRUE_COL]):
                med = statistics.median(vals)
                if abs(med - r[TRUE_COL]) > 0.001:
                    bad.append((r["sample_id"], med, r[TRUE_COL]))
        lines.append(f"- 真值中位数与 `{TRUE_COL}` 不一致：**{len(bad)}** 条")
        for sid, med, tv in bad[:8]:
            lines.append(f"    - {sid}：读数中位数 {med:.3f} ≠ 记录值 {tv:.3f}")

    # 分组覆盖
    for col in GROUP_COLS:
        if col in df.columns:
            vc = df[col].fillna("未标注").value_counts()
            lines.append(f"- `{col}` 分布：" + "、".join(f"{k} {v}" for k, v in vc.items()))

    if "mode" in df.columns:
        vc = df["mode"].fillna("未标注").value_counts()
        lines.append("- `mode` 分布：" + "、".join(f"{k} {v}" for k, v in vc.items()))

    lines.append("")
    return "\n".join(lines)


def err_stats(g):
    """一组记录的误差统计。返回 dict。"""
    e = (g[PRED_COL] - g[TRUE_COL]).dropna() * 1000.0     # 转 mm
    if e.empty:
        return None
    ae = e.abs()
    return {
        "n": int(e.size),
        "MAE": ae.mean(),
        "RMSE": (e.pow(2).mean()) ** 0.5,
        "P95": ae.quantile(0.95),
        "Max": ae.max(),
        "Bias": e.mean(),
        "Std": e.std(ddof=1) if e.size > 1 else float("nan"),
    }


def fmt(d, key):
    return "—" if d is None or pd.isna(d[key]) else f"{d[key]:.1f}"


def table(rows, header=None):
    """把 dict 列表渲染成 Markdown 表。"""
    if not rows:
        return "_无可用数据_"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(header or cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def build_tables(df, outdir):
    md = []
    valid = df.dropna(subset=[TRUE_COL, PRED_COL])
    overall = err_stats(valid)

    # ---------- 表9-1 像素级精度（不依赖米制，只有像素坐标也能算） ----------
    if PX_COL in df.columns and GT_PX_COL in df.columns:
        d = df.dropna(subset=[PX_COL, GT_PX_COL])
        if len(d):
            md.append("### 表9-1 像素级精度（模型水位线 vs 人工标注，由脚本自动生成）\n")
            prow = []
            for name, sub in [("全场景合计", d)] + [(n, g) for n, g in d.groupby("site_type")] if "site_type" in d.columns else [("全场景合计", d)]:
                diff = sub[PX_COL] - sub[GT_PX_COL]
                ae = diff.abs()
                prow.append({
                    "数据集/场景": name, "样本数": len(sub),
                    "MAE (px)": f"{ae.mean():.2f}",
                    "RMSE (px)": f"{(diff.pow(2).mean()) ** 0.5:.2f}",
                    "P95 (px)": f"{ae.quantile(0.95):.1f}",
                    "最大 (px)": f"{ae.max():.0f}",
                    "Bias (px)": f"{diff.mean():+.2f}",
                    "≤5px 占比": f"{(ae <= 5).mean() * 100:.1f}%",
                })
            md.append(table(prow))
            if "condition" in d.columns and d["condition"].nunique() > 1:
                crows = []
                for name, sub in d.groupby("condition"):
                    diff = sub[PX_COL] - sub[GT_PX_COL]
                    ae = diff.abs()
                    crows.append({"工况": name, "样本数": len(sub), "MAE (px)": f"{ae.mean():.2f}",
                                  "P95 (px)": f"{ae.quantile(0.95):.1f}",
                                  "≤5px 占比": f"{(ae <= 5).mean() * 100:.1f}%"})
                md.append("\n" + table(crows))
            md.append("")

    # ---------- 表9 测量精度（米制） ----------
    md.append("### 表9 分场景测量精度（米制，由脚本自动生成，勿手改）\n")
    if not valid.empty:
        rows = []
        if overall:
            rows.append({"工况": "全场景合计", "组数": overall["n"], "MAE (mm)": fmt(overall, "MAE"),
                         "RMSE (mm)": fmt(overall, "RMSE"), "P95 (mm)": fmt(overall, "P95"),
                         "最大误差 (mm)": fmt(overall, "Max"), "Bias (mm)": fmt(overall, "Bias")})
        if "condition" in valid.columns:
            order = [c for c in ["正常", "遮挡物", "水流波纹", "耀斑"] if c in set(valid["condition"].dropna())]
            for name in order + [c for c in valid["condition"].dropna().unique() if c not in order]:
                g = err_stats(valid[valid["condition"] == name])
                if g:
                    rows.append({"工况": name, "组数": g["n"], "MAE (mm)": fmt(g, "MAE"),
                                 "RMSE (mm)": fmt(g, "RMSE"), "P95 (mm)": fmt(g, "P95"),
                                 "最大误差 (mm)": fmt(g, "Max"), "Bias (mm)": fmt(g, "Bias")})
        md.append(table(rows))
        if "condition" in valid.columns:
            bad = valid[valid["condition"].isin(["遮挡物", "水流波纹", "耀斑"])]
            gb = err_stats(bad)
            if gb:
                md.append(f"\n恶劣工况（遮挡/波纹/耀斑合并）n={gb['n']}，MAE {fmt(gb,'MAE')} mm，"
                          f"P95 {fmt(gb,'P95')} mm，最大 {fmt(gb,'Max')} mm。")
    else:
        md.append("_本记录表没有米制真值（`water_level_true_m` / `water_level_pred_m` 为空），"
                  "米制指标无法计算——这是正常的：像素级验证与米制验证是两层，见 5.3 表述。_")

    # ---------- 表10 稳定性 ----------
    md.append("\n### 表10 稳定性与重复性（由脚本自动生成）\n")
    rows10 = []
    if "site_name" in valid.columns:
        groups = valid.groupby(["site_name", TRUE_COL])
        sig = []
        for _, g in groups:
            if len(g) >= 3:
                sig.append(g[PRED_COL].std(ddof=1) * 1000.0)
        if sig:
            rows10.append({"指标": "重复性 σ（同点同水位 ≥3 次）",
                           "实测值": f"{statistics.mean(sig):.1f} mm",
                           "样本组数": len(sig)})
            rows10.append({"指标": "重复性 σ（最差组）",
                           "实测值": f"{max(sig):.1f} mm", "样本组数": len(sig)})
    if PX_COL in valid.columns and GT_PX_COL in valid.columns:
        d = (valid[PX_COL] - valid[GT_PX_COL]).abs().dropna()
        if len(d):
            rows10.append({"指标": "像素误差 均值", "实测值": f"{d.mean():.2f} px", "样本组数": len(d)})
            rows10.append({"指标": "像素误差 最大", "实测值": f"{d.max():.0f} px", "样本组数": len(d)})
    if "mode" in valid.columns:
        for mode, g in valid.groupby("mode"):
            s = err_stats(g)
            if s:
                rows10.append({"指标": f"模式「{mode}」MAE", "实测值": f"{s['MAE']:.1f} mm",
                               "样本组数": s["n"]})
    md.append(table(rows10) if rows10 else "_需米制数据且同一水位重复 ≥3 次才能算重复性 σ；本表暂缺。_")

    # ---------- 分组明细 ----------
    detail = []
    for col in GROUP_COLS:
        if col in valid.columns:
            for name, g in valid.groupby(col):
                s = err_stats(g)
                if s:
                    detail.append({"分组维度": col, "取值": name, "组数": s["n"],
                                   "MAE": round(s["MAE"], 2), "RMSE": round(s["RMSE"], 2),
                                   "P95": round(s["P95"], 2), "Max": round(s["Max"], 2)})
    pd.DataFrame(detail).to_csv(os.path.join(outdir, "metrics_by_group.csv"),
                                index=False, encoding="utf-8-sig")

    return "\n".join(md), overall, valid


def build_figure(valid, outdir):
    """图5：误差分布三联图。缺 matplotlib 时跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[metrics] 未生成图5（matplotlib 缺失: {e}）")
        return None

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    d = valid.dropna(subset=[TRUE_COL, PRED_COL]).copy()
    d["err_mm"] = (d[PRED_COL] - d[TRUE_COL]) * 1000.0

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=200)

    ax[0].axhline(0, color="#888", lw=0.8)
    ax[0].scatter(d[TRUE_COL], d["err_mm"], s=18, alpha=0.7, color="#c0392b", edgecolors="none")
    ax[0].set_xlabel("真值水位 (m)")
    ax[0].set_ylabel("绝对误差 (mm)")
    ax[0].set_title("(a) 误差—水位散点")

    ax[1].hist(d["err_mm"], bins=20, color="#2c7fb8", edgecolor="white")
    ax[1].axvline(0, color="#888", lw=0.8)
    ax[1].set_xlabel("误差 (mm)")
    ax[1].set_ylabel("样本数")
    ax[1].set_title("(b) 误差分布")

    if "condition" in d.columns and d["condition"].notna().any():
        cats = [c for c in ["正常", "遮挡物", "水流波纹", "耀斑"] if c in set(d["condition"])]
        data = [d.loc[d["condition"] == c, "err_mm"].values for c in cats]
        if cats:
            try:
                ax[2].boxplot(data, tick_labels=cats, showmeans=True)   # matplotlib >= 3.9
            except TypeError:
                ax[2].boxplot(data, labels=cats, showmeans=True)        # 旧版本
            ax[2].axhline(0, color="#888", lw=0.8)
            ax[2].set_ylabel("误差 (mm)")
            ax[2].set_title("(c) 分工况误差箱线")

    fig.tight_layout()
    p = os.path.join(outdir, "fig5_error_distribution.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] 已生成 {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="记录表 csv（UTF-8，可带 BOM）")
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.records)
    if df.empty:
        print("[metrics] 记录表为空（只剩示例行？）——请先填入实测数据")
        return 1

    chk = check(df)
    with open(os.path.join(args.outdir, "check_report.md"), "w", encoding="utf-8") as f:
        f.write(chk)

    body, overall, valid = build_tables(df, args.outdir)
    head = "# 第五章 5.3 指标汇总（脚本生成）\n\n" \
           f"数据源：`{os.path.basename(args.records)}`，有效样本 {len(valid)} 组。\n\n"
    with open(os.path.join(args.outdir, "metrics_summary.md"), "w", encoding="utf-8") as f:
        f.write(head + body + "\n")

    fig = build_figure(valid, args.outdir)

    print("\n===== 自检 =====")
    print(chk)
    print("===== 指标 =====")
    print(body)
    if fig:
        print(f"\n图5 已输出：{fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
