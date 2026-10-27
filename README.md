# 虚拟水尺水位视觉监测系统

基于端云协同的水位视觉监测系统：OpenMV 摄像头采集图像，云端用 SAM3 语义分割提取水面，结合虚拟水尺把像素坐标换算成真实水位高程，通过网页实时展示，并联动继电器自动控制抽/放水泵。

用普通工业摄像头 + 云端算力替代实体水尺与专用硬件，在不规则边坡（折坡、曲面）场景下实现低成本、可规模化部署的水位自动监测。

## 目录结构

```
AIC/
├── cloud/                      云端（Linux 服务器）：视觉计算 + 控制服务
│   ├── control_first.py        HTTP 服务入口：/api/* 接口、任务调度、水泵联动
│   ├── main_video.py           常驻 worker：模型只加载一次，单帧/视频任务分发
│   ├── sam3first.py            SAM3 单帧分割：框提示、掩码后处理、水位线提取
│   ├── sam3video.py            SAM3 视频传播：关键帧筛选、tracker 传播、渲染
│   ├── model.py                MobileNet-SSD 水体检测（自训权重，生成框提示）
│   ├── ruler.py                虚拟水尺：CSV 标定查找表、像素→高程换算
│   ├── time_tracker.py         关键帧均值聚类筛选
│   ├── TCP.py                  图像接收：长度头解析 → 畸变校正 → ROI 裁剪
│   ├── live_stream.py          WebSocket 实时推流（/ws/live）
│   ├── ewelink_control.py      易微联 OAuth2.0 继电器控制（凭据走环境变量）
│   ├── save_images.py          MQTT 旁路收图脚本
│   ├── state_tracker.py        状态追踪辅助
│   ├── sort.py                 文件排序工具
│   ├── main.py                 离线批量处理入口（可选，非在线链路必需）
│   ├── main_ui.py              PyQt5 桌面客户端（可选，本地调试用）
│   ├── OPENMV_exposure_calib.py OpenMV 曝光标定脚本（暗/亮帧交替采集）
│   └── .env.example            易微联凭据模板
├── edge_openmv/
│   └── OPENMV.py               边缘端：摄像头初始化、WiFi 联网、25fps JPEG 采集并 TCP 上传
├── web/                        前端页面
│   ├── index.html              实时画面 / 分割结果 / 历史折线 / 虚拟水尺
│   ├── app.js                  WebSocket、API 轮询、折线图、自动控制
│   └── style.css               样式
├── model_lib/sam3/             SAM3 模型库（依赖层，Meta 官方代码）
├── MobileNet-SSD/              MobileNet-SSD 水体检测（自训）
│   ├── API/                    vision 推理包（BoundingBox / Predictor 等）
│   ├── path/                   自训权重 001.pth / 004.pth（按标定站号命名）
│   ├── pytorch-ssd-master/     pytorch-ssd 框架（MIT，训练快照 models/ 不随仓库发布）
│   ├── VOCdevkit/              训练集与标注（99 张，VOC 格式）
│   └── tabel.txt               类别表：BACKGROUND / water / bank
├── calibration/                边坡标定查找表
│   ├── 3.csv                   分段线性标定（垂直坡 / 斜坡 / 平地）
│   └── 4.csv                   线性标定（1—6.5 m）
├── 水槽数据集/                  验证数据集（1188 组带人工标注，见该目录下 README.md）
├── eval_kit/                   评测脚本（批量评测、指标计算、数据集核查）
└── weights/                    SAM3 权重说明（sam3.pt 约 3.4 GB，需自行获取，不随仓库分发）
```

## 运行架构（端云协同）

```
OpenMV(RT1062) ──TCP 25fps──▶ 云服务器（TCP.py 接收 + 畸变校正 + 裁剪）
                                    │
                                    ▼ SAM3 语义分割 + MobileNet-SSD 框提示 + 虚拟水尺换算
                              control_first.py（HTTP / WebSocket）
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              网页实时画面     水位数值/折线图    易微联继电器（水泵联动）
```

## 快速复现（无需摄像头）

评审或无硬件环境下最短路径：一台能跑 SAM3 的服务器 + 一张现场水面照片。

1. 启动服务：`python cloud/control_first.py`
2. 浏览器打开 `http://<服务器IP>:6008`
3. 在页面上选择站点（无边坡，或某个已标定的边坡）→ 上传本地图片
4. 系统返回分割掩码、水位线、叠加虚拟水尺的换算水位

整条链路不依赖 OpenMV、不依赖继电器，用于验证分割与高程换算是否正确。

## 部署要点

1. 边缘端：`edge_openmv/OPENMV.py` 烧录进 OpenMV RT1062，修改 `WIFI_SSID` / `WIFI_PASS` 与 `TCP_HOST`。
2. 云端：安装依赖（见下），把 `model_lib/sam3` 与 `MobileNet-SSD` 放到可导入路径；
   SAM3 权重 `sam3.pt` 从官方 `huggingface.co/facebook/sam3` 申请获取后放置到 `cloud/` 同级目录。
3. 易微联凭据通过环境变量注入（不硬编码），复制 `.env.example` 为 `.env` 并填入真实值：
   ```bash
   cp cloud/.env.example cloud/.env
   export $(grep -v '^#' cloud/.env | xargs)
   ```
4. 服务入口：`python cloud/control_first.py`，默认监听 6008（HTTP/WebSocket）+ TCP 6006 收图。
5. 继电器首次授权：`python cloud/ewelink_control.py auth`。

依赖安装：
```bash
pip install torch opencv-python numpy pandas Pillow matplotlib scipy openpyxl flask paho-mqtt requests
pip install huggingface_hub iopath einops timm triton==2.3.0 decord pycocotools ftfy regex
```

## 主要功能

- **分割方式**：视频分割 / 单帧分割，单次与连续检测均按所选方式执行。
- **框提示策略**：选择已标定边坡时，用该边坡对应的 MobileNet-SSD 权重给出水体框/岸框提示，再交给 SAM3 精分；
  选择「无边坡」时跳过 MobileNet-SSD，直接用语义文本提示分割（不叠加水尺、不换算高程）。
- **自动控制**：按「持续 / 单次」模式联动水泵；单次模式达到目标水位并稳定后自动停止并关闭水泵。
- **异常帧过滤**：相邻帧水位突变时暂不改水泵，连续两帧保持才确认；异常帧不写入折线。
- **三路开关**：1 抽水泵、2 放水泵、3 摄像头电源，非分割状态可点击切换。

## 评测

`eval_kit/` 提供离线可复算的评测脚本，指标与报告 5.3 一致：

```bash
# 核查数据集结构，并从标注表直接算出像素误差
python eval_kit/audit_dataset.py --root "AIC/水槽数据集" --outdir audit_out
# 计算 MAE / RMSE / P95 / 重复性，输出表格与误差分布图
python eval_kit/compute_metrics.py --records audit_out/records_from_dataset.csv --outdir eval_out
```

## 数据集

`水槽数据集/` 含四套子集、共 1188 组带人工标注的样本（三类边坡 393 组 + 十五里河 795 组），
字段说明、采集方式、许可与使用边界见 `水槽数据集/README.md`。

## 开源与第三方资源及许可证

本作品基于以下开源资源构建，来源、版本、许可证与关键义务见：

- 仓库根目录 `LICENSE`：为**团队自主代码**（cloud/、web/、edge_openmv/、calibration/、eval_kit/ 等）所采用，协议为 **Apache License 2.0**。
- 第三方资源的许可证原文集中存放于 `licenses/`，并在各自所在目录保留副本：
  - `licenses/LICENSE_SAM3.txt`（SAM License，亦见 `model_lib/sam3/LICENSE`）
  - `licenses/LICENSE_MobilenetSSD.txt`（MIT）
- 汇总与署名见 `THIRD_PARTY_LICENSES.md`，逐条资源与自研边界见 `开源及第三方资源使用清单.md`。
- 第三方资源的权利归属不因本作品的开源与参赛行为而改变。

SAM3 权重（`sam3.pt`，约 3.4 GB）受 SAM License 约束，**不在本仓库分发**，需使用者自行从官方渠道获取。
