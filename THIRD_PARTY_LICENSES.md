# 第三方许可证汇总（Third-Party Licenses）

本文件汇总作品所使用的第三方开源软件、模型、框架、组件与服务的许可证及署名信息，作为《开源及第三方资源使用清单》的补充。完整清单见《开源及第三方资源使用清单.md》。

> **许可边界说明**：仓库根目录 `LICENSE` 为**团队自主代码**所采用的 **Apache License 2.0**；
> 第三方资源的许可证文本集中存放于 `licenses/`（如 `LICENSE_SAM3.txt`、`LICENSE_MobilenetSSD.txt`），
> 并在各自所在目录（如 `model_lib/sam3/LICENSE`）保留副本。第三方资源的权利归属不因参赛或开源行为而改变。

---

## 一、模型

### 1. SAM 3（Segment Anything Model 3）

- **来源**：Meta（facebookresearch/sam3，huggingface.co/facebook/sam3）
- **许可证**：SAM License（完整文本见 `licenses/LICENSE_SAM3.txt`，`model_lib/sam3/LICENSE` 另有副本）
- **署名**：本作品使用了 Meta 的 SAM 3（Segment Anything Model 3）模型进行图像/视频语义分割。
- **关键义务**：分发或再分发 SAM Materials 及其衍生作品须附带本许可副本；发表相关研究须注明使用了 SAM Materials；遵守出口管制。

### 2. MobileNet-SSD 检测权重

- **来源**：团队自训练（自标注水体数据，类别 BACKGROUND / water）
- **许可证**：不适用（自研）

---

## 二、代码框架与算法

### 1. pytorch-ssd（MobileNet-SSD 推理框架）

- **来源**：Hao Gao，2019（https://github.com/qfgaohao/pytorch-ssd）
- **许可证**：MIT License
- **版权声明**：Copyright (c) 2019 Hao Gao

```
MIT License
Copyright (c) 2019 Hao Gao
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 2. SSD（Single Shot MultiBox Detector）模型结构

- **来源**：Liu, W., et al. "SSD: Single Shot MultiBox Detector." ECCV 2016（论文）及 Caffe 开源实现
- **引用**：Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. ECCV 2016.

---

## 三、Python 依赖库许可证

| 库 | 许可证 |
|---|---|
| torch (PyTorch) | BSD-3-Clause |
| opencv-python | Apache-2.0 |
| numpy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| Pillow (PIL) | HPND / MIT-CMU |
| matplotlib | PSF-derived |
| scipy | BSD-3-Clause |
| openpyxl | MIT |
| flask | BSD-3-Clause |
| paho-mqtt | EPL-2.0 / EDL-1.0 |
| huggingface_hub | Apache-2.0 |
| iopath | MIT |
| einops | MIT |
| timm | Apache-2.0 |
| triton | MIT |
| decord | Apache-2.0 |
| pycocotools | BSD-style (FreeBSD) |
| ftfy | MIT |
| regex | Apache-2.0 |

> 上述库均以 `import` 方式调用，未修改其源码；均为宽松许可证，无 GPL 等传染性义务。详细许可证文本见各库官方仓库。

---

## 四、前端资源

- **hls.js**：Apache-2.0（https://github.com/video-dev/hls.js）

## 五、硬件与固件

- **OpenMV**：MIT License（https://github.com/openmv/openmv）

## 六、第三方服务

- **易微联 eWeLink 开放平台 API**：酷宅科技 CoolKit 服务条款 + OAuth 2.0 授权。凭据通过环境变量注入，不随代码公开。
