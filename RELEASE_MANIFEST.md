# 发布清单（自动生成）

生成时间：2026-10-27 16:35
总计：**82.0 MB**

| 内容 | 文件数 | 体积 (MB) | 说明 |
|---|---|---|---|
| cloud/（云端服务） | 15 | 0.26 |  |
| web/（前端） | 3 | 0.09 |  |
| edge_openmv/（边缘端） | 1 | 0.01 |  |
| model_lib/sam3（依赖库） | 180 | 3.38 | Meta 官方代码，随 LICENSE 再分发 |
| model_lib/sam3（依赖库） | 180 | 3.38 | Meta 官方代码，随 LICENSE 再分发 |
| README / 许可证 | 6 | 0.02 |  |
| calibration/（标定表） | 2 | 0.00 |  |
| MobileNet-SSD/API | 45 | 0.21 | vision 推理包 |
| MobileNet-SSD/path/*.pth | 2 | 51.66 | 自训权重，建议 LFS/Release |
| MobileNet-SSD/pytorch-ssd-master | 259 | 3.16 | 已排除 models/ 训练快照 |
| MobileNet-SSD/tabel.txt | 1 | 0.00 | BACKGROUND / water / bank |
| MobileNet-SSD/VOCdevkit | 204 | 0.82 | 团队自采自标训练集 |
| 水槽数据集（轻量版） | 1195 | 22.31 | 已转 JPG |
| eval_kit/（评测脚本） | 8 | 0.10 |  |

## 体检结果

- ✅ 无超大单文件
- ✅ 总体积 82.0 MB，可直接普通 git 提交
- ✅ 未见已删模块残留、未见学校/教师信息

## 未包含（需另置）

- `sam3.pt`（约 3.4 GB）：受 SAM License 约束，**不再分发**，由使用者从官方渠道获取后自行放置
- `MobileNet-SSD/pytorch-ssd-master/models/`（43 个训练快照，约 1.06 GB）：仅保留最终权重，快照不公开
- 演示视频与答辩材料：走大赛系统提交，不进仓库