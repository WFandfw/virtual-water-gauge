# -*- coding: utf-8 -*-
"""用真实数据生成报告需要的图4、图5（替代原文档里的 6 个 GIF）。

用法：
    python eval_kit/make_report_figures.py --dataset "C:/Users/a1595/Desktop/root1/AIC/水槽数据集" \
        --records audit_out/records_from_dataset.csv --outdir report_assets
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def fig_slope_panels(dataset, outdir):
    """图4：三类边坡「原图 | 掩码 | 结果」三联对比。"""
    picks = []
    for d in ("混凝土数据", "浆砌石数据", "砂石数据"):
        xl = os.path.join(dataset, d, "水位线Y坐标.xlsx")
        if not os.path.isfile(xl):
            continue
        df = pd.read_excel(xl)
        df["d"] = (pd.to_numeric(df["模型坐标"], errors="coerce")
                   - pd.to_numeric(df["人工坐标"], errors="coerce")).abs()
        df = df.dropna(subset=["d"]).sort_values("d")
        mid = df.iloc[len(df) // 2]                      # 取误差中位数的样本，避免只挑最好的
        sid = f"{int(mid['序号']):04d}"
        base = os.path.join(dataset, d, sid)
        if all(os.path.isfile(os.path.join(base, f)) for f in
               ("original.jpg", "mask_orig.jpg", "result.jpg")):
            picks.append((d.replace("数据", ""), base, mid["d"], int(mid["序号"])))

    if not picks:
        print("[fig4] 未找到可用样本")
        return None

    fig, axes = plt.subplots(1, 3 * len(picks), figsize=(13.4, 4.6), dpi=200,
                             gridspec_kw={"wspace": 0.07})
    cols = ["original.jpg", "mask_orig.jpg", "result.jpg"]
    titles = ["原图", "掩码", "结果"]
    for g, (label, base, err, sid) in enumerate(picks):
        for c, fn in enumerate(cols):
            ax = axes[g * 3 + c]
            ax.imshow(plt.imread(os.path.join(base, fn)))
            ax.set_xticks([])
            ax.set_yticks([])
            if g == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 1:
                ax.set_xlabel(f"{label}边坡  #{sid}  ({err:.0f} px)", fontsize=10, labelpad=6)
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_multicondition.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] {p}")
    return p


def fig_error_dist(records, outdir):
    """图5：1188 组真实像素误差分布（散点 / 直方图 / 分组柱状）。"""
    df = pd.read_csv(records, encoding="utf-8-sig")
    df = df.dropna(subset=["waterline_px", "waterline_gt_px"])
    df["err"] = (df["waterline_px"] - df["waterline_gt_px"]).astype(float)
    df["abs"] = df["err"].abs()
    order = ["混凝土", "浆砌石", "砂石", "十五里河"]
    order = [o for o in order if o in set(df["site_type"])]

    fig = plt.figure(figsize=(13.2, 3.9), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])

    ax = fig.add_subplot(gs[0, 0])
    data = [df.loc[df["site_type"] == o, "err"].values for o in order]
    bp = ax.boxplot(data, tick_labels=order, showmeans=True, patch_artist=True)
    for b in bp["boxes"]:
        b.set(facecolor="#d6e6f5", edgecolor="#2c7fb8")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("误差（像素）")
    ax.set_title("(a) 各数据集误差分布", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(df["err"], bins=41, color="#2c7fb8", edgecolor="white")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("误差（像素）")
    ax.set_ylabel("样本数")
    ax.set_title(f"(b) 误差直方图（n={len(df)}）", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    mae = [df.loc[df["site_type"] == o, "abs"].mean() for o in order]
    hit = [(df.loc[df["site_type"] == o, "abs"] <= 5).mean() * 100 for o in order]
    bars = ax.bar(range(len(order)), mae, color="#c0392b", alpha=0.85, width=0.55)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("MAE（像素）")
    ax.set_title("(c) 平均绝对误差与 ≤5px 占比", fontsize=10)
    for i, (b, h) in enumerate(zip(bars, hit)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                f"{b.get_height():.2f}px\n{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(mae) * 1.42)

    fig.tight_layout()
    p = os.path.join(outdir, "fig5_error_distribution.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--outdir", default="report_assets")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fig_slope_panels(args.dataset, args.outdir)
    fig_error_dist(args.records, args.outdir)


if __name__ == "__main__":
    main()
