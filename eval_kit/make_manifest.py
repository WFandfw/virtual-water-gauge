# -*- coding: utf-8 -*-
"""生成图片/权重清单与 SHA256，作为 5.3 数据的「锚点」。

用途：
1. 报告里写「285 组原始记录」时，评审可凭 manifest 校验图片未被替换；
2. 记录表中的 image_file 是否齐全、是否重复，一次性查出来。

用法：
    python eval_kit/make_manifest.py --dir eval_images --out manifest.csv
    python eval_kit/make_manifest.py --dir eval_images --records results.csv   # 顺带做记录↔图片对账
"""

import argparse
import csv
import hashlib
import os


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def walk(root, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower().endswith(exts):
                p = os.path.join(dp, fn)
                yield p, os.path.relpath(p, root).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="图片目录")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--records", default="", help="（可选）记录表，用于对账")
    args = ap.parse_args()

    rows = []
    for p, rel in walk(args.dir):
        rows.append({"file": rel, "size_bytes": os.path.getsize(p), "sha256": sha256(p)})

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "size_bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)

    total = sum(r["size_bytes"] for r in rows)
    print(f"[manifest] {len(rows)} 个文件，合计 {total/1024/1024:.1f} MB → {args.out}")
    dup = {}
    for r in rows:
        dup.setdefault(r["sha256"], []).append(r["file"])
    same = {k: v for k, v in dup.items() if len(v) > 1}
    if same:
        print(f"[manifest] ⚠️ 有 {len(same)} 组完全相同的图片（可能是重复计数，检查样本独立性）：")
        for v in list(same.values())[:5]:
            print("    ", " == ".join(v[:3]))

    if args.records and os.path.isfile(args.records):
        have = {r["file"] for r in rows}
        with open(args.records, encoding="utf-8-sig", newline="") as f:
            recs = [r for r in csv.DictReader(f)
                    if not str(r.get("sample_id", "")).startswith("【示例")]
        miss = []
        for r in recs:
            for key in ("image_file", "frame_files"):
                for p in str(r.get(key, "") or "").replace("；", ";").split(";"):
                    p = p.strip()
                    if p and p not in have and os.path.normpath(p) not in have:
                        miss.append((r.get("sample_id"), p))
        print(f"[manifest] 记录 {len(recs)} 条，引用图片缺失 {len(miss)} 个")
        for sid, p in miss[:10]:
            print(f"    - {sid}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
