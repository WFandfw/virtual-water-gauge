#!/usr/bin/env python3
# model.py
"""
MobileNet-SSD 水体检测库（粗分割框提示）。

云端目录结构（API 放代码/标签，path 放权重）：
  /root/sam/MobileNet-SSD/API/    除权重外所有必需文件：vision 包、voc-model-labels.txt
  /root/sam/MobileNet-SSD/path/   各 CSV 对应的 .pth 权重文件（如 001.pth、004.pth）

labels 三分类：BACKGROUND / water / bank。water=水体框（正框提示），bank=岸框（负框提示）。

按 CSV 选择权重：
  - CSV 名为数字：001.csv -> 1 -> path/001.pth（也认 1.pth / 01.pth / 0001.pth）
  - CSV 名为中文等非数字：读 /root/sam/MobileNet-SSD/model_map.json 映射
  - 无 CSV：默认编号 1
  - 找不到对应权重或 labels：跳过 SSD 提示，只告警一次，不影响主分割
"""

import os
import sys
import json
import threading

# ==================== 配置 ====================
SSD_BASE = "/root/sam/MobileNet-SSD"
API_DIR = os.path.join(SSD_BASE, "API")
WEIGHTS_DIR = os.path.join(SSD_BASE, "path")
MODEL_MAP_PATH = os.path.join(SSD_BASE, "model_map.json")
DEFAULT_MODEL_NUM = 1

CONF_THRESHOLD = 0.3

_lock = threading.Lock()
_cache = {}                 # num -> (predictor, class_names)
_warned = set()             # 已告警过的 CSV 编号（避免每帧刷屏）
_vision_loaded = False
_create_ssd = None
_create_predictor = None
_no_slope_noticed = False     # 「无边坡」提示只打一次，避免每帧刷屏


def _load_vision(code_root):
    """把包含 vision 包的目录加入 sys.path，并延迟导入。"""
    global _vision_loaded, _create_ssd, _create_predictor
    if _vision_loaded:
        return
    if code_root not in sys.path:
        sys.path.append(code_root)
    from vision.ssd.mobilenetv1_ssd import (
        create_mobilenetv1_ssd,
        create_mobilenetv1_ssd_predictor,
    )
    _create_ssd = create_mobilenetv1_ssd
    _create_predictor = create_mobilenetv1_ssd_predictor
    _vision_loaded = True


def _code_root():
    """找到包含 vision 包的目录（优先 API 目录）。"""
    candidates = [
        API_DIR,
        os.path.join(SSD_BASE, str(DEFAULT_MODEL_NUM)),
        os.path.join(SSD_BASE, "pytorch-ssd-master"),
    ]
    for d in candidates:
        if os.path.isfile(os.path.join(d, "vision", "ssd", "mobilenetv1_ssd.py")):
            return d
    return API_DIR


def _labels_path():
    """在 API 目录里找 labels 文件。"""
    for base in (API_DIR, os.path.join(API_DIR, "models")):
        for name in ("voc-model-labels.txt", "labels.txt", "model-labels.txt", "tabel.txt"):
            p = os.path.join(base, name)
            if os.path.isfile(p):
                return p
    return None


def _load_model_map():
    if not os.path.isfile(MODEL_MAP_PATH):
        return {}
    try:
        with open(MODEL_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[model] 读取 {MODEL_MAP_PATH} 失败: {e}")
        return {}


def _num_from_csv(csv_path):
    """从 CSV 路径解析模型编号。"""
    if not csv_path:
        return DEFAULT_MODEL_NUM
    base = os.path.basename(str(csv_path))
    stem = os.path.splitext(base)[0].strip()
    if stem.isdigit():
        return int(stem)

    mapping = _load_model_map()
    for key in (stem, base):
        if key in mapping:
            try:
                return int(mapping[key])
            except (TypeError, ValueError):
                pass
    if stem not in _warned:
        _warned.add(stem)
        print(f"[model] CSV '{base}' 没有模型编号映射，使用默认 {DEFAULT_MODEL_NUM}"
              f"（可配置 {MODEL_MAP_PATH}）")
    return DEFAULT_MODEL_NUM


def _weight_path(num):
    """在 path 目录里按编号找权重文件。"""
    names = [f"{num}.pth", f"{num:02d}.pth", f"{num:03d}.pth", f"{num:04d}.pth"]
    for n in names:
        p = os.path.join(WEIGHTS_DIR, n)
        if os.path.isfile(p):
            return p
    return None


def _load(num):
    """加载并缓存指定编号的模型。"""
    if num in _cache:
        return _cache[num]

    pth = _weight_path(num)
    labels = _labels_path()
    if not pth or not labels:
        raise FileNotFoundError(f"权重或 labels 缺失：num={num}, pth={pth}, labels={labels}")

    _load_vision(_code_root())
    class_names = [n.strip() for n in open(labels, encoding="utf-8").readlines() if n.strip()]
    net = _create_ssd(len(class_names), is_test=True)
    net.load(pth)
    net.eval()
    net.to("cpu")
    # SSD 内部把 priors 放到了 cuda，.to('cpu') 不会移动它，需手动移回 CPU
    if hasattr(net, "priors"):
        net.priors = net.priors.to("cpu")
    predictor = _create_predictor(net, candidate_size=200, device="cpu")

    print(f"[model] 已加载 {pth}（类别: {class_names}）")
    _cache[num] = (predictor, class_names)
    return _cache[num]


def load_model(csv_path=None):
    """兼容旧调用：加载并返回该 CSV 对应的 predictor。"""
    num = _num_from_csv(csv_path)
    with _lock:
        return _load(num)[0]


def detect_water_box(img_rgb, csv_path=None):
    """用与 CSV 对应的 MobileNet-SSD 检测，返回 (water_box, bank_box, water_conf)。

    water_box / bank_box 为 [xmin, ymin, xmax, ymax] 像素坐标，未检出为 None。
    水体框作正框提示，岸框作负框提示。
    img_rgb 为 RGB 图像（与 sam3first 传入一致，不在此处做通道转换）。
    csv_path 为空字符串表示用户端选择了「无边坡」：无边坡不做 SSD 框提示，
    直接返回 (None, None, None)（不加载权重、不占用显存）。"""
    if csv_path is not None and not str(csv_path).strip():
        global _no_slope_noticed
        if not _no_slope_noticed:
            _no_slope_noticed = True
            print("[model] 无边坡：不使用 MobileNet-SSD 水体框，跳过检测（走纯语义提示分割）")
        return None, None, None
    num = _num_from_csv(csv_path)
    try:
        with _lock:
            predictor, class_names = _load(num)

        boxes, labels, probs = predictor.predict(img_rgb, 10, CONF_THRESHOLD)
        if boxes is None or boxes.size(0) == 0 or len(probs) == 0:
            return None, None, None

        water_idx = class_names.index("water") if "water" in class_names else 1
        bank_idx = class_names.index("bank") if "bank" in class_names else -1

        water_box = None
        bank_box = None
        water_conf = -1.0
        bank_conf = -1.0
        for i in range(boxes.size(0)):
            cls = int(labels[i])
            prob = float(probs[i])
            box = [float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3])]
            if cls == water_idx and prob > water_conf:
                water_conf = prob
                water_box = box
            elif bank_idx > 0 and cls == bank_idx and prob > bank_conf:
                bank_conf = prob
                bank_box = box
        if water_box is not None:
            print(f"[model] 水体框(正框) xmin={water_box[0]:.0f} ymin={water_box[1]:.0f} xmax={water_box[2]:.0f} ymax={water_box[3]:.0f}")
        if bank_box is not None:
            print(f"[model] 岸框(负框) xmin={bank_box[0]:.0f} ymin={bank_box[1]:.0f} xmax={bank_box[2]:.0f} ymax={bank_box[3]:.0f}  下边缘 ymax={bank_box[3]:.0f}")
        return water_box, bank_box, water_conf
    except Exception as e:
        if num not in _warned:
            _warned.add(num)
            print(f"[model] 水体检测不可用（csv={csv_path}, num={num}）: {e}")
        return None, None, None
