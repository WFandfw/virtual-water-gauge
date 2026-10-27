# -*- coding: utf-8 -*-
"""在服务器上批量跑分割，把「系统读数」自动写回记录表（第五章 5.3 的数据来源）。

必须在跑得动 SAM3 的机器上运行：
    python eval_kit/batch_eval.py --records records_filled.csv --out results.csv \
        --base /root/sam --image-root /root/sam/eval_images --csv-dir /root/sam/virtual

要点：
* 系统读数由本脚本跑出来，不要手抄——这是「可核验」的前提：别人用同样图片+同样权重能复现同一数字。
* 每行跑完立即落盘，中断后可加 --resume 续跑。
* 不修改原记录文件，输出写到 --out。

记录表至少要有的列：sample_id, image_file, csv_file, mode
可选列：frame_files（多帧，用 ; 分隔）, water_level_true_m
"""

import argparse
import csv
import os
import sys
import time
import traceback

OUT_FIELDS_EXTRA = ["waterline_px", "water_level_pred_m", "abs_err_mm", "infer_ms", "eval_error"]


def read_rows(path):
    """读记录表，丢掉模板里的示例行。"""
    rows, fields = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        for r in rd:
            if str(r.get("sample_id", "")).startswith("【示例"):
                continue
            rows.append(r)
    return rows, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--base", default="/root/sam", help="代码根目录（含 sam3first.py）")
    ap.add_argument("--image-root", default="", help="图片根目录，image_file 相对此目录")
    ap.add_argument("--csv-dir", default="", help="标定 CSV 所在目录；留空则按记录里的相对路径")
    ap.add_argument("--checkpoint", default="", help="默认 <base>/sam3.pt")
    ap.add_argument("--prompt", default="black water")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="跳过 results 里已成功的 sample_id")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    sys.path.insert(0, base)
    ckpt = args.checkpoint or os.path.join(base, "sam3.pt")

    from PIL import Image
    import sam3first
    import ruler

    # ---- 模型构建：直连 sam3，失败再回退 sam3video（与 main_video 修复后的写法一致）
    try:
        from sam3.model_builder import build_sam3_image_model
        src = "sam3.model_builder"
    except Exception:
        import sam3video
        build_sam3_image_model = sam3video.build_sam3_image_model
        src = "sam3video"
    print(f"[batch] 模型构建函数来源: {src}")
    print(f"[batch] 正在加载 SAM3（{ckpt}）...")
    model = build_sam3_image_model(checkpoint_path=ckpt)
    print("[batch] 模型就绪")

    try:
        from main_video import select_waterline   # 多帧筛选复用生产代码
    except Exception as e:
        print(f"[batch] 多帧筛选不可用（{e}），将只跑单帧")
        select_waterline = None

    rows, fields = read_rows(args.records)
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if args.resume and os.path.isfile(args.out):
        with open(args.out, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("eval_error") in ("", None) and r.get("waterline_px"):
                    done.add(r["sample_id"])
        print(f"[batch] --resume：已完成 {len(done)} 条，跳过")

    out_fields = list(fields) + [c for c in OUT_FIELDS_EXTRA if c not in fields]
    new_file = not os.path.isfile(args.out) or not args.resume
    fo = open(args.out, "w" if new_file else "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(fo, fieldnames=out_fields)
    if new_file:
        w.writeheader()

    ok = fail = 0
    for i, r in enumerate(rows, 1):
        sid = r.get("sample_id", f"row{i}")
        if sid in done:
            continue
        t0 = time.time()
        rec = {k: r.get(k, "") for k in fields}
        rec.update({k: "" for k in OUT_FIELDS_EXTRA})
        try:
            img = r.get("image_file", "")
            img_path = img if os.path.isabs(img) else os.path.join(args.image_root, img)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"图片不存在: {img_path}")

            csv_rel = (r.get("csv_file") or "").strip()
            if csv_rel and args.csv_dir and not os.path.isabs(csv_rel):
                csv_path = os.path.join(args.csv_dir, csv_rel)
            elif csv_rel and not os.path.isabs(csv_rel):
                csv_path = os.path.join(base, csv_rel)
            else:
                csv_path = csv_rel or ""      # 空串 = 无边坡，不调用 SSD、不换算高程

            frames = (r.get("frame_files") or "").strip()
            fps = [p.strip() for p in frames.replace("；", ";").split(";") if p.strip()] or None

            if fps and select_waterline:
                cands = []
                for fp in fps:
                    p = fp if os.path.isabs(fp) else os.path.join(args.image_root, fp)
                    im = Image.open(p)
                    ret = sam3first.process_single_image_video(
                        im, model, args.prompt, ruler_csv=csv_path or None)
                    if ret[0] and ret[1] is not None:
                        cands.append({"final_y": ret[1], "score": ret[3]})
                    im.close()
                best = select_waterline(cands, is_continuous=False)
                py = best["final_y"] if best else None
            else:
                im = Image.open(img_path)
                ret = sam3first.process_single_image_video(
                    im, model, args.prompt, ruler_csv=csv_path or None)
                py = ret[1] if ret[0] else None
                im.close()

            rec["waterline_px"] = "" if py is None else int(py)
            rec["infer_ms"] = int((time.time() - t0) * 1000)

            if csv_path and py is not None:
                hh = Image.open(img_path).height
                H = ruler.pixel_to_real_waterline(py, csv_path, image_height=hh,
                                                  original_image_height=sam3first.BASE_H)
                rec["water_level_pred_m"] = "" if H is None else f"{H:.4f}"
                tv = (r.get("water_level_true_m") or "").strip()
                if tv and H is not None:
                    rec["abs_err_mm"] = f"{abs(H - float(tv)) * 1000:.1f}"
            ok += 1
        except Exception as e:
            rec["eval_error"] = f"{type(e).__name__}: {e}"
            fail += 1
            print(f"  ✗ {sid} {rec['eval_error']}")
            if os.environ.get("DEBUG"):
                traceback.print_exc()

        w.writerow(rec)
        fo.flush()
        if i % 10 == 0 or i == len(rows):
            print(f"[batch] {i}/{len(rows)}  成功 {ok}  失败 {fail}")

    fo.close()
    print(f"\n[batch] 完成：成功 {ok}，失败 {fail} → {args.out}")
    print("[batch] 下一步：python eval_kit/compute_metrics.py --records " + args.out)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
