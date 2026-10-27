# 发布清单（RELEASE MANIFEST）

生成时间：2026-10-27 21:53（按仓库实际内容重新统计）
总计：**1929 个文件，82.12 MB**

仓库地址：https://github.com/WFandfw/virtual-water-gauge

| 内容 | 文件数 | 体积 (MB) | 说明 |
|---|---|---|---|
| MobileNet-SSD/ | 511 | 55.86 | 自训检测模型与推理框架：API/ 推理包、path/ 自训权重、VOCdevkit/ 训练集 |
| 水槽数据集/ | 1195 | 22.14 | 验证数据集：1188 张原图（已转 JPG）+ 4 份人工标注表 |
| model_lib/ | 181 | 3.39 | SAM3 依赖库（Meta 官方代码 180 个 + LICENSE 副本） |
| cloud/ | 15 | 0.26 | 云端服务（15 个 Python 文件）：任务调度、SAM3 分割、水位解算、水泵联动、实时推流 |
| audit_out/ | 3 | 0.22 | 指标复算产物（由 eval_kit/audit_dataset.py 生成，可直接核对报告第五章数字） |
| eval_kit/ | 8 | 0.10 | 评测脚本（8 个）：数据核查、指标复算、指纹校验、记录模板 |
| web/ | 3 | 0.09 | 网页客户端（HTML/JS/CSS）：实时画面、分割结果、水位折线、参数配置 |
| (根)/ | 6 | 0.03 | README.md、LICENSE（Apache-2.0）、THIRD_PARTY_LICENSES.md、开源及第三方资源使用清单.md、RELEASE_MANIFEST.md、.gitignore |
| edge_openmv/ | 1 | 0.01 | OpenMV 边缘端固件 OPENMV.py：图像采集、WiFi 联网、25 fps TCP 上传 |
| licenses/ | 2 | 0.01 | 第三方许可证原文：LICENSE_SAM3.txt（SAM License）、LICENSE_MobilenetSSD.txt（MIT） |
| docs/ | 1 | 0.01 | 文档：标定与提示词配置说明 |
| weights/ | 1 | 0.00 | 权重获取说明（权重本体不入库，见 README） |
| calibration/ | 2 | 0.00 | 边坡标定查找表：3.csv（分段线性）、4.csv（线性，1—6.5 m） |

## 体检结果

- ✅ 最大单文件 25.8 MB（`MobileNet-SSD/path/001.pth`），无超大单文件
- ✅ 总体积 82.12 MB，可直接普通 git 提交（未使用 LFS）
- ⚠️ 发现疑似信息：[('RELEASE_MANIFEST.md', '指导教师'), ('RELEASE_MANIFEST.md', '学校名称')]
- ✅ 自训权重 001.pth / 004.pth 以普通文件提交，clone 即可直接获得

## 许可边界

- 根目录 `LICENSE`：**Apache License 2.0**，适用于团队自主代码（cloud/、web/、edge_openmv/、calibration/、eval_kit/、docs/ 等）。
- `model_lib/sam3/` 为 Meta 官方代码，适用 **SAM License**，许可原文见 `licenses/LICENSE_SAM3.txt` 与 `model_lib/sam3/LICENSE`。
- `MobileNet-SSD/` 框架部分适用 **MIT**（`licenses/LICENSE_MobilenetSSD.txt`）；自训权重 001.pth / 004.pth 为团队自研。
- 第三方资源权利归属不因本作品开源与参赛行为而改变。

## 未包含（需另置）

- `sam3.pt`（约 3.4 GB）：受 SAM License 约束，**不随仓库分发**，由使用者从官方渠道获取后自行放置。
- 训练中间快照：**42 个 .pth，约 1.1 GB**（`MobileNet-SSD/pytorch-ssd-master/models/`，本地留存），仅保留最终权重，快照不公开。
- 演示视频与答辩材料：通过大赛报名系统提交，不进仓库。

## 体积前 5

- 25.8 MB　`MobileNet-SSD/path/001.pth`
- 25.8 MB　`MobileNet-SSD/path/004.pth`
- 1.3 MB　`model_lib/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- 0.5 MB　`MobileNet-SSD/pytorch-ssd-master/readme_ssd_example.jpg`
- 0.2 MB　`audit_out/records_from_dataset.csv`
