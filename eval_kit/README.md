# 5.3 可核验数据工具集

这套脚本的作用只有一个：**让报告 5.3 里的每一个数字都能被追溯和复算**，而不是手抄进 Word。

评审视角的"可核验"= 给他三样东西：

1. **原始记录**：每一行 = 一次测量（图片 → 真值 → 系统读数 → 误差）
2. **能重算的脚本**：同一份记录跑一遍，得到的 MAE/RMSE/P95 与报告完全一致
3. **不可篡改的锚点**：图片 SHA256 清单

---

## 一、四个文件

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `audit_dataset.py` | 任意机器（需 pandas/openpyxl） | 核查水槽数据集结构完整性，并从标注表直接算出**像素误差**指标 |
| `records_template.csv` | — | 记录表模板，25 列，含 2 行示例（脚本会自动跳过带"【示例"的行） |
| `batch_eval.py` | 服务器（有 GPU + SAM3 权重） | 批量跑分割，把「系统读数」自动写进 `results.csv` |
| `compute_metrics.py` | 任意机器（只要 pandas） | 从记录算出表9/表10/表11 + 图5 + 自检报告 |
| `make_manifest.py` | 任意机器 | 生成图片 SHA256 清单，并对账记录表引用的图片是否齐全 |

## 二、执行顺序

```
① 现场/离线采集          原始图片 + 人工水尺读数（9 次/组）
        ↓  手工填 records_template.csv（或从 Excel 另存为 CSV）
② 服务器：batch_eval.py   跑出 water_level_pred_m / waterline_px / infer_ms
        ↓
③ 本地：compute_metrics.py 得到 metrics_summary.md（表9/表10）+ fig5_error_distribution.png
        ↓
④ 本地：make_manifest.py   得到 manifest.csv（图片指纹）
        ↓
⑤ 把 ②③④ 的产物连同报告一起开放（见《上传与开放方案.md》）
```

具体命令：

```bash
# ② 服务器上（权重、标定 CSV 都在 /root/sam 下）
python eval_kit/batch_eval.py \
    --records records_filled.csv --out results.csv \
    --base /root/sam --image-root /root/sam/eval_images --csv-dir /root/sam/virtual
# 中断后可以续跑
python eval_kit/batch_eval.py --records records_filled.csv --out results.csv \
    --base /root/sam --image-root /root/sam/eval_images --resume

# ③ 任意机器
python eval_kit/compute_metrics.py --records results.csv --outdir eval_out

# ④ 任选
python eval_kit/make_manifest.py --dir /root/sam/eval_images --out manifest.csv --records results.csv
```

## 三、字段说明（照这个填，不要随意改名）

| 列名 | 必填 | 说明 |
|---|---|---|
| `sample_id` | ✅ | 唯一编号，建议 `S001` 起 |
| `site_type` | ✅ | 边坡类型：混凝土 / 浆砌石 / 砂石 |
| `site_name` | ✅ | 测点名，如 `野外-2`。**重复性 σ 靠它分组，必须填准** |
| `condition` | ✅ | 工况：正常 / 遮挡物 / 水流波纹 / 耀斑（**用词必须统一**，否则分组表会多出几行） |
| `water_level_true_m` | ✅ | 真值，米，3 位小数 |
| `true_source` | ✅ | 真值来源，如 `实体水尺人工读数` / `试验渠设定值` |
| `true_readings` | 建议 | 9 次读数，用 `\|` 分隔。脚本会校验其中位数是否等于 `water_level_true_m`（容差 1 mm），用来抓录入错误 |
| `image_file` | ✅ | 图片相对路径 |
| `csv_file` | ✅ | 标定 CSV 路径；**选"无边坡"时留空**（脚本据此跳过 SSD 与高程换算） |
| `mode` | ✅ | `单帧` 或 `5帧`，用于消融对比 |
| `frame_files` | 条件 | `mode=5帧` 时填 5 个文件，`;` 分隔 |
| `waterline_gt_px` | 建议 | 人工标注的水位线像素 y，用于算"像素误差 <5px"这条指标 |
| `note` | 否 | 备注 |

以下三列由 `batch_eval.py` 自动写回，不要手填：`waterline_px`、`water_level_pred_m`、`infer_ms`。

## 四、三条会直接影响结果的记录规则

1. **同一水位重复 5 次，记 5 行**（`sample_id` 不同、`water_level_true_m` 相同、图片是连续帧）。
   只有 ≥3 行同组，脚本才能算重复性 σ；这也是 285 组 ÷ 5 = 57 个水位的原因。
2. **真值 = 9 次读数（3 人 × 3 次）的中位数**，填进 `water_level_true_m`。不要把 9 次读数取平均后四舍五入到 cm，那会引入系统性偏差。
3. **工况用词全表统一**。`遮挡物`/`遮挡`/`有遮挡` 会被当成三个不同工况，分组表直接崩。

## 五、输出长什么样

`eval_out/metrics_summary.md` 里直接是能粘进 Word 的表格：

```
| 工况 | 组数 | MAE (mm) | RMSE (mm) | P95 (mm) | 最大误差 (mm) | Bias (mm) |
| 全场景合计 | 285 | 5.9 | 8.7 | ... | ... | ... |
| 正常 | ... |
| 遮挡物 | ... |
```

`eval_out/check_report.md` 是自检报告，会明确告诉你：条数是否等于 285、有没有缺读数、真值中位数是否对得上、各工况/边坡的分布是否均衡。**先把这份自检清干净，再往报告里抄数字。**

## 六、注意事项

- `batch_eval.py` 直接复用生产代码（`sam3first.process_single_image_video`、`main_video.select_waterline`、`ruler.pixel_to_real_waterline`），所以测的就是线上跑的那套逻辑，不存在"另写一套评测代码"的偏差。
- `compute_metrics.py` 不依赖 GPU、不依赖 torch，只要有 pandas 就能跑，方便评审复算。
- 想给评审一个"零 GPU 也能复现"的路径，就把 `results.csv`（含 pred）一起开放——评审只需重跑 `compute_metrics.py` 就能核对全部指标。
- `demo_synthetic_records.csv` 是**合成数据**，只用来验证脚本能跑通，**绝不能写进报告**。真数据到位后删掉它。
