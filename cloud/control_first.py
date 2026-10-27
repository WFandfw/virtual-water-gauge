#!/usr/bin/env python3
# control_first.py
"""
SAM3 检测控制服务（纯标准库，无第三方依赖）

提供 HTTP 接口供 web 前端调用：
    GET  /api/status            -> {"task_running": bool}
    POST /api/detect            body {"timestamp": ms, "mode": "single"|"continuous"}
    GET  /api/result            -> {"ready": false} 或
                                   {"ready": true, "result_video_url": "/output_video/0001.mp4", ...}

同时托管静态资源：web 前端页面（/）与分割结果视频（/output_video/*.mp4，支持 Range 断点续传）。

流程：前端点击“单次检测” -> POST /api/detect -> 后台线程执行一次
      “合成视频 -> SAM3 分割 -> 输出到 output_video” -> /api/result 返回视频地址 -> 前端播放。
"""

import os
import sys
import re
import json
import time
import atexit
import shutil
import signal
import subprocess
import threading
import mimetypes
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

# 确保可导入同目录下的 main_video 模块，以及项目根目录下的 sam3 包
# （main_video 用 spawn 起子进程时会重新 import，需把根目录一并加入 sys.path/PYTHONPATH）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for p in (SCRIPT_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.append(p)
os.environ["PYTHONPATH"] = ROOT_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

import main_video
import live_stream

# 导入虚拟水尺库
import ruler

# 导入易微联继电器控制库
import ewelink_control

# ==================== 服务配置 ====================
# AutoDL 端口映射（服务只需在容器内监听明文 HTTP）：
#   容器 6008(http) -> 公网 https://uu1109182-b505-9f8649e7.westc.seetacloud.com:8443
#   容器 6006(tcp)  -> 公网 connect.westc.seetacloud.com:47509（TCP.py 图片上传占用）
# 也可用环境变量 CONTROL_PORT 覆盖监听端口。
HOST = os.environ.get("CONTROL_HOST", "0.0.0.0")
PORT = int(os.environ.get("CONTROL_PORT", "6008"))

CHECKPOINT_PATH = "/root/sam/sam3.pt"
IMAGE_INPUT_DIR = "/root/sam/input"
VIDEO_INPUT_DIR = "/root/sam/input_video"
OUTPUT_VIDEO_DIR = "/root/sam/output_video"
SAVE_TARGET_DIR = "/root/sam/save"
ALL_OUTPUT_DIR = "/root/sam/all"
ALL_ONLY_DIR = "/root/sam/all_only"
WEB_DIR = "/root/sam/web"

# 等待输入图片凑齐一帧视频的最长时间（秒）
WAIT_TIMEOUT = int(os.environ.get("CONTROL_WAIT_TIMEOUT", "600"))

# 虚拟水尺查找表目录与默认路径
VIRTUAL_DIR = "/root/sam/virtual"
VIRTUAL_RULER_CSV = "/root/sam/virtual/1.csv"

# 边坡配置文件：{边坡名: csv文件名}，持久化到 sites.json
SITES_CONFIG = "/root/sam/virtual/sites.json"

# 设定水位线配置文件（自动控制的目标水位）
AUTO_SETPOINT_CONFIG = "/root/sam/virtual/setpoint.json"

# 分割方式配置文件（"video" 视频分割 | "frame" 单帧分割）
SEG_MODE_CONFIG = "/root/sam/virtual/seg_mode.json"

# 本地测试配置：网页设置后由浏览器直接读取本地文件夹并上传到云端处理（纯前端，无需本地客户端）。
#   method:     "image" | "video"      测试方式
#   five_frame: bool                   图片测试是否启用 5 帧筛选（视频方式忽略）
#   path:       str                    兼容旧字段，现已不使用（浏览器用文件夹选择器）
#   enabled:    bool                   是否启用（仅点亮顶部按钮，作为标记）
LOCAL_TEST_CONFIG = "/root/sam/virtual/local_test.json"
LOCAL_VIDEO_INBOX = "/root/sam/local_video"     # 网页上传的本地测试视频临时落盘目录
DEFAULT_LOCAL_TEST = {"method": "image", "five_frame": True, "path": "", "enabled": False}

# 水位比较的显示精度（四舍五入到的小数位数，与前端 avgElevation.toFixed 显示一致）
DISPLAY_PRECISION = 2


def load_sites():
    """读取边坡配置 {name: csv_filename}，不存在则从 virtual 目录现有 csv 初始化。"""
    if os.path.isfile(SITES_CONFIG):
        try:
            with open(SITES_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"[control_first] 读取边坡配置失败: {e}")
    sites = {}
    if os.path.isdir(VIRTUAL_DIR):
        csv_files = [f for f in os.listdir(VIRTUAL_DIR) if re.fullmatch(r'\d+\.csv', f)]
        csv_files.sort(key=lambda f: int(f[:-4]))
        for i, f in enumerate(csv_files, 1):
            sites[f"边坡{i}"] = f
    save_sites(sites)
    return sites


def save_sites(sites):
    """保存边坡配置到 sites.json。"""
    try:
        os.makedirs(VIRTUAL_DIR, exist_ok=True)
        with open(SITES_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(sites, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[control_first] 保存边坡配置失败: {e}")


def get_next_csv_filename():
    """返回最小空缺编号文件名（从 1 开始补空位），如存在 001、002、004 则返回 003.csv。"""
    os.makedirs(VIRTUAL_DIR, exist_ok=True)
    nums = set()
    for f in os.listdir(VIRTUAL_DIR):
        m = re.fullmatch(r'(\d+)\.csv', f)
        if m:
            nums.add(int(m.group(1)))
    n = 1
    while n in nums:
        n += 1
    return f"{n:03d}.csv"


def get_ruler_csv_for_site(site):
    """根据站点返回对应的虚拟水尺查找表路径；'无边坡' 返回空字符串（不叠加水尺、不换算高程）。"""
    if site == "无边坡":
        return ""
    sites = load_sites()
    csv_name = sites.get(site)
    if csv_name:
        return os.path.join(VIRTUAL_DIR, csv_name)
    return VIRTUAL_RULER_CSV


# 输出文件设置配置文件（是否启用分割完成后自动下载到本地）
OUTPUT_SETTINGS_CONFIG = "/root/sam/virtual/settings.json"


def load_output_settings():
    """读取输出文件设置 {download_enabled, download_path}。"""
    default = {"download_enabled": False, "download_path": ""}
    if os.path.isfile(OUTPUT_SETTINGS_CONFIG):
        try:
            with open(OUTPUT_SETTINGS_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {
                    "download_enabled": bool(data.get("download_enabled", False)),
                    "download_path": data.get("download_path", ""),
                }
        except Exception as e:
            print(f"[control_first] 读取输出设置失败: {e}")
    return default


def save_output_settings(settings):
    """保存输出文件设置。"""
    try:
        os.makedirs(os.path.dirname(OUTPUT_SETTINGS_CONFIG), exist_ok=True)
        with open(OUTPUT_SETTINGS_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[control_first] 保存输出设置失败: {e}")


# 连续检测设置配置文件（分割间隔秒数 + 分割次数；0 表示不限/默认一直连续）
CONTINUOUS_SETTINGS_CONFIG = "/root/sam/virtual/continuous_settings.json"


def load_continuous_settings():
    """读取连续检测设置 {interval_seconds, count}。"""
    default = {"interval_seconds": 0, "count": 0}
    if os.path.isfile(CONTINUOUS_SETTINGS_CONFIG):
        try:
            with open(CONTINUOUS_SETTINGS_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {
                    "interval_seconds": int(data.get("interval_seconds", 0) or 0),
                    "count": int(data.get("count", 0) or 0),
                }
        except Exception as e:
            print(f"[control_first] 读取连续检测设置失败: {e}")
    return default


def save_continuous_settings(settings):
    """保存连续检测设置。"""
    try:
        os.makedirs(os.path.dirname(CONTINUOUS_SETTINGS_CONFIG), exist_ok=True)
        with open(CONTINUOUS_SETTINGS_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[control_first] 保存连续检测设置失败: {e}")

# 将 main_video 内的路径常量统一指向本服务配置，保证 segment_single_video 输出到同一目录
main_video.CHECKPOINT_PATH = CHECKPOINT_PATH
main_video.IMAGE_INPUT_DIR = IMAGE_INPUT_DIR
main_video.VIDEO_INPUT_DIR = VIDEO_INPUT_DIR
main_video.OUTPUT_VIDEO_DIR = OUTPUT_VIDEO_DIR
main_video.SAVE_TARGET_DIR = SAVE_TARGET_DIR

TARGET_FRAMES = main_video.TARGET_FRAMES
FPS = main_video.FPS


# ==================== 检测任务执行 ====================

def transcode_h264(src_path):
    """把分割结果视频转成浏览器可播放的 H.264(yuv420p)，失败则保留原文件。"""
    if not os.path.isfile(src_path):
        return False

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        tmp = src_path + ".tmp.mp4"
        cmd = [ffmpeg, "-y", "-i", src_path,
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-an", tmp]
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[control_first] ffmpeg 转码异常: {e}")
            r = None
        if r is not None and r.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, src_path)
            print(f"[control_first] 已转码为 H.264(浏览器可播放): {src_path}")
            return True
        if os.path.exists(tmp):
            os.remove(tmp)

    # 回退：用 OpenCV 重编码为 avc1
    try:
        cap = cv2.VideoCapture(src_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or float(FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp = src_path + ".tmp.mp4"
        out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
        if out.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            out.release()
        cap.release()
        if os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, src_path)
            print(f"[control_first] 已用 OpenCV 转码为 avc1: {src_path}")
            return True
    except Exception as e:
        print(f"[control_first] 转码失败，保留原 mp4v（浏览器可能无法播放）: {e}")
    return False


def clear_input_directory():
    """清空 input 目录及其下所有子文件夹（含 processed），保证每次检测从干净状态开始。"""
    try:
        if os.path.exists(IMAGE_INPUT_DIR):
            shutil.rmtree(IMAGE_INPUT_DIR)
            print(f"[control_first] 已清空输入目录及其子文件夹: {IMAGE_INPUT_DIR}")
        os.makedirs(IMAGE_INPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"[control_first] 清空输入目录 {IMAGE_INPUT_DIR} 失败: {e}")


# ==================== 水泵自动控制 ====================

# 泵状态缓存：outlet -> bool（True=开 False=关），避免重复下发相同指令
_pump_state = {}
_pump_lock = threading.Lock()

# 设定水位线（None 表示未设定）
_setpoint = None

# 自动控制模式状态：single 单次(默认) | continuous 持续
_auto_mode_state = "single"

# 单帧多选筛选：历史水位线（像素 y，时间顺序），用于连续分割时的时序预测
_waterline_history = []
HISTORY_MAX_LEN = 20

# 手动框提示：单帧分割进行中，人工在 5 张图上逐个点选真实水位点，
# 框提示上沿 = 该点上方 5px，用于替代自动动态框。
_manual_lock = threading.Lock()
_manual_eligible = False        # 是否有单帧分割正在进行（决定按钮可否点击）
_manual_requested = False       # 用户已点击“手动框提示”，等待下一次分割暂停
_manual_pending = None          # 等待人工点选时: {"names":[...], "urls":[...], "points":[None,...]}
_manual_event = threading.Event()


def load_seg_config():
    """读取分割方式配置 {mode, frame5}，默认 {"mode": "video", "frame5": True}。"""
    cfg = {"mode": "video", "frame5": True}
    if os.path.isfile(SEG_MODE_CONFIG):
        try:
            with open(SEG_MODE_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if data.get("mode") in ("video", "frame"):
                    cfg["mode"] = data["mode"]
                if "frame5" in data:
                    cfg["frame5"] = bool(data["frame5"])
        except Exception as e:
            print(f"[control_first] 读取分割方式失败: {e}")
    return cfg


def load_seg_mode():
    """读取分割方式，默认 video。"""
    return load_seg_config()["mode"]


def save_seg_mode(mode, frame5=True):
    """保存分割方式及图片分割是否 5 选 1。"""
    try:
        os.makedirs(os.path.dirname(SEG_MODE_CONFIG), exist_ok=True)
        with open(SEG_MODE_CONFIG, 'w', encoding='utf-8') as f:
            json.dump({"mode": mode, "frame5": bool(frame5)}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[control_first] 保存分割方式失败: {e}")


def frame_batch():
    """图片分割每批张数：5 选 1 时为 5；关闭时每张图片立即分割（batch=1）。"""
    return 5 if load_seg_config()["frame5"] else 1


def load_local_test():
    """读取本地测试配置，缺省用 DEFAULT_LOCAL_TEST。"""
    cfg = dict(DEFAULT_LOCAL_TEST)
    if os.path.isfile(LOCAL_TEST_CONFIG):
        try:
            with open(LOCAL_TEST_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if data.get("method") in ("image", "video"):
                    cfg["method"] = data["method"]
                cfg["five_frame"] = bool(data.get("five_frame", cfg["five_frame"]))
                cfg["path"] = str(data.get("path", "") or "")
                cfg["enabled"] = bool(data.get("enabled", False))
        except Exception as e:
            print(f"[control_first] 读取本地测试配置失败: {e}")
    return cfg


def save_local_test(cfg):
    """保存本地测试配置。"""
    try:
        os.makedirs(os.path.dirname(LOCAL_TEST_CONFIG), exist_ok=True)
        with open(LOCAL_TEST_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[control_first] 保存本地测试配置失败: {e}")


def load_setpoint():
    """启动时读取设定水位线。"""
    global _setpoint
    if os.path.isfile(AUTO_SETPOINT_CONFIG):
        try:
            with open(AUTO_SETPOINT_CONFIG, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("setpoint") is not None:
                _setpoint = float(data["setpoint"])
        except Exception as e:
            print(f"[control_first] 读取设定水位线失败: {e}")
    return _setpoint


def save_setpoint(value):
    """保存设定水位线到文件。"""
    global _setpoint
    _setpoint = value
    try:
        os.makedirs(os.path.dirname(AUTO_SETPOINT_CONFIG), exist_ok=True)
        with open(AUTO_SETPOINT_CONFIG, 'w', encoding='utf-8') as f:
            json.dump({"setpoint": value}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[control_first] 保存设定水位线失败: {e}")


def set_pump(outlet, on):
    """控制某路继电器；状态不变则不重复下发指令。"""
    with _pump_lock:
        if _pump_state.get(outlet) == on:
            return
        try:
            at = ewelink_control.get_access_token()
            ewelink_control.set_switch(at, outlet, on)
            _pump_state[outlet] = on
            print(f"[control_first] 继电器 outlet={outlet} -> {'开' if on else '关'}")
        except Exception as e:
            print(f"[control_first] 控制继电器 outlet={outlet} 失败: {e}")


def get_pump_state(outlet):
    """查询某路继电器状态（以内存缓存为准）。"""
    with _pump_lock:
        return bool(_pump_state.get(outlet, False))


def toggle_pump(outlet):
    """切换某路继电器状态，返回切换后的状态。"""
    cur = get_pump_state(outlet)
    set_pump(outlet, not cur)
    return bool(_pump_state.get(outlet, False))


def init_relay_state():
    """服务启动时：开开关三(outlet 2)，关开关一(outlet 0)、二(outlet 1)。"""
    try:
        ewelink_control.get_access_token()
    except Exception as e:
        print(f"[control_first] 继电器初始化跳过（可能尚未授权）: {e}")
        return
    for outlet, on in ((0, False), (1, False), (2, True)):
        set_pump(outlet, on)
    print("[control_first] 继电器初始化完成：开三、关一、关二")


def _display_round(v):
    """四舍五入到显示精度（DISPLAY_PRECISION 位小数）。"""
    return round(float(v), DISPLAY_PRECISION)


def apply_auto_control(current):
    """根据设定水位线与当前水位的关系自动控制水泵。
       当前<设定：开一关二；当前>设定：开二关一；相等(四舍五入后)：关一关二(停止)。"""
    global _setpoint
    if _setpoint is None or current is None:
        return
    c = _display_round(current)
    s = _display_round(_setpoint)
    if c == s:
        set_pump(0, False)
        set_pump(1, False)
        print(f"[control_first] 水位 {c} == 设定 {s}，水泵停止（关一关二）")
    elif c < s:
        set_pump(0, True)
        set_pump(1, False)
        print(f"[control_first] 水位 {c} < 设定 {s}，开一关二")
    else:
        set_pump(0, False)
        set_pump(1, True)
        print(f"[control_first] 水位 {c} > 设定 {s}，开二关一")


def run_one_detection(site=None, stop_event=None):
    """执行一次检测：等待图片 -> 合成视频 -> SAM3 分割 -> 叠加虚拟水尺 -> 输出视频。返回 (out_path, video_name, avg_elevation, ruler_min, ruler_max)。"""
    ruler_csv = get_ruler_csv_for_site(site)

    # 每次检测前优先清空 input 及其子文件夹，避免历史残留图片混入本批
    clear_input_directory()

    main_video.ensure_dir(VIDEO_INPUT_DIR)
    main_video.ensure_dir(OUTPUT_VIDEO_DIR)
    main_video.ensure_dir(SAVE_TARGET_DIR)
    processed_dir = os.path.join(IMAGE_INPUT_DIR, "processed")
    main_video.ensure_dir(processed_dir)

    deadline = time.time() + WAIT_TIMEOUT
    image_files = main_video.get_image_files(IMAGE_INPUT_DIR)
    while len(image_files) < TARGET_FRAMES:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("检测已被用户中断")
        if time.time() > deadline:
            raise RuntimeError(f"等待图片超时：当前 {len(image_files)}/{TARGET_FRAMES}")
        time.sleep(1)
        image_files = main_video.get_image_files(IMAGE_INPUT_DIR)

    selected = image_files[:TARGET_FRAMES]
    first_img = cv2.imread(selected[0])
    if first_img is None:
        main_video.move_to_processed(selected, processed_dir)
        raise RuntimeError(f"无法读取第一张图片: {selected[0]}")
    h, w = first_img.shape[:2]

    video_name = main_video.get_next_video_name(VIDEO_INPUT_DIR)
    video_path = os.path.join(VIDEO_INPUT_DIR, video_name)

    # 创建输出子文件夹 /root/sam/all/{video_name}，并复制 125 张 ROI 序列
    all_output_dir = os.path.join(ALL_OUTPUT_DIR, os.path.splitext(video_name)[0])
    os.makedirs(all_output_dir, exist_ok=True)
    roi_dir = os.path.join(all_output_dir, "roi")
    os.makedirs(roi_dir, exist_ok=True)
    for p in selected:
        try:
            shutil.copy(p, os.path.join(roi_dir, os.path.basename(p)))
        except Exception as e:
            print(f"  ⚠️ 复制 ROI 图片失败 {p}: {e}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {video_path}")
    for p in selected:
        img = cv2.imread(p)
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()

    print(f"[control_first] 即将分割，站点={site}，ruler_csv = {ruler_csv!r}")
    success, avg_elevation = main_video.segment_single_video(video_path, CHECKPOINT_PATH, SAVE_TARGET_DIR, ruler_csv, image_paths=selected, all_output_dir=all_output_dir)
    main_video.move_to_processed(selected, processed_dir)
    if not success:
        print(f"[control_first] ⚠️ 视频 {video_name} 分割失败，仍输出原视频结果")

    out_path = os.path.join(OUTPUT_VIDEO_DIR, video_name)

    # ----- 叠加虚拟水尺 -----
    if os.path.isfile(ruler_csv):
        tmp_gauge_path = out_path + ".gauge.mp4"
        try:
            # 调用 ruler 库对视频叠加水尺
            ruler.ruler(
                input_path=out_path,
                output_path=tmp_gauge_path,
                csv_path=ruler_csv,
                step_main=0.05,                # 主刻度间隔 0.05m
                original_image_height=240      # 虚拟水尺标定基准高度(v_star 按 240px 标定)，帧高不同会自动缩放
            )
            os.replace(tmp_gauge_path, out_path)
            print(f"[control_first] 已叠加虚拟水尺: {out_path}")
        except Exception as e:
            print(f"[control_first] 叠加虚拟水尺失败，保留原视频: {e}")
            if os.path.exists(tmp_gauge_path):
                os.remove(tmp_gauge_path)
    else:
        print(f"[control_first] 未找到虚拟水尺查找表 {ruler_csv}，跳过叠加")

    # 转码为 H.264 确保浏览器兼容（ruler 已尝试输出 avc1，但再次调用可保证兼容性）
    transcode_h264(out_path)
    # 需求 6：复制结果视频到子文件夹
    try:
        shutil.copy(out_path, os.path.join(all_output_dir, video_name))
    except Exception as e:
        print(f"  ⚠️ 复制结果视频失败: {e}")
    print(f"[control_first] 检测完成，输出视频: {out_path}，平均水位高程: {avg_elevation}")
    ruler_min, ruler_max = ruler.get_ruler_range(ruler_csv)
    result_url = "/output_video/" + video_name
    return result_url, video_name, avg_elevation, ruler_min, ruler_max, True


# 单帧结果合成视频时重复的帧数（约 1 秒，便于浏览器循环播放）
FRAME_LOOP = 25


def get_next_all_only_name():
    """返回 all_only 目录下下一个可用的四位数文件夹名（0001 起递增）。"""
    os.makedirs(ALL_ONLY_DIR, exist_ok=True)
    max_num = 0
    for f in os.listdir(ALL_ONLY_DIR):
        if os.path.isdir(os.path.join(ALL_ONLY_DIR, f)) and f.isdigit():
            max_num = max(max_num, int(f))
    return f"{max_num + 1:04d}"


def manual_request_pending():
    """是否已请求手动框提示（供 HTTP 端点读取）。"""
    with _manual_lock:
        return bool(_manual_requested)


def set_manual_requested():
    """用户点击“手动框提示”：仅在单帧分割进行中可用。
    同时中断正在推理的当前组，使其尽快改为人工点选。"""
    global _manual_requested
    with _manual_lock:
        if not _manual_eligible:
            return False
        _manual_requested = True
    main_video.request_manual_abort()
    return True


def manual_status():
    """返回手动框提示状态，供前端轮询。"""
    with _manual_lock:
        pending = _manual_pending is not None
        return {
            "eligible": bool(_manual_eligible) and not pending,
            "pending": pending,
            "names": list(_manual_pending["names"]) if pending else [],
            "images": list(_manual_pending["urls"]) if pending else [],
            "points": list(_manual_pending["points"]) if pending else [],
        }


def submit_manual_points(points):
    """前端提交人工点选的水位像素 y，唤醒暂停中的分割。"""
    with _manual_lock:
        if _manual_pending is None:
            return False
        n = len(_manual_pending["names"])
        vals = []
        for i in range(n):
            v = points[i] if i < len(points) else None
            try:
                vals.append(float(v) if v is not None else None)
            except (TypeError, ValueError):
                vals.append(None)
        _manual_pending["points"] = vals
    _manual_event.set()
    return True


def cancel_manual_wait():
    """取消手动点选等待并唤醒阻塞的分割线程（供 /api/stop 使用）。"""
    global _manual_pending
    with _manual_lock:
        _manual_pending = None
        _manual_event.set()


def _wait_for_manual_points(img_paths, stop_event):
    """把 img_paths 暴露给前端，阻塞等待人工逐个点选。返回像素 y 列表，或被中断时返回 None。"""
    global _manual_pending, _manual_requested
    names = [os.path.basename(p) for p in img_paths]
    urls = ["/input/" + n for n in names]
    _manual_event.clear()
    with _manual_lock:
        _manual_requested = False
        _manual_pending = {"names": names, "urls": urls, "points": [None] * len(names)}
    print(f"[control_first] 手动框提示：暂停分割，等待人工点选 {len(names)} 张图片的水位点...")
    while True:
        if _manual_event.wait(timeout=0.5):
            break
        if stop_event is not None and stop_event.is_set():
            with _manual_lock:
                _manual_pending = None
            print("[control_first] 手动框提示被中断")
            return None
    with _manual_lock:
        pts = list(_manual_pending["points"]) if _manual_pending else None
        _manual_pending = None
    print(f"[control_first] 手动框提示：收到水位点 {pts}，继续分割")
    return pts


def _prompt_manual_crop_edges(img_paths, stop_event):
    """暂停并等待人工点选，返回每张图的框提示上边缘（y-5）。被中断时抛异常。"""
    pts = _wait_for_manual_points(img_paths, stop_event)
    if pts is None:
        raise RuntimeError("检测已被用户中断")
    edges = []
    for p in pts:
        if p is None:
            edges.append(None)
        else:
            edges.append(max(0, int(round(p)) - 5))
    return edges


def run_one_frame_detection(site=None, stop_event=None, is_continuous=False, batch=5):
    """单帧分割入口：标记“手动框提示”可用区间，结束后复位。"""
    global _manual_eligible
    _manual_eligible = True
    try:
        return _run_one_frame_detection(site, stop_event, is_continuous, batch)
    finally:
        _manual_eligible = False


def _run_one_frame_detection(site=None, stop_event=None, is_continuous=False, batch=5):
    """执行一次单帧分割：收集 batch 张连续图 -> 批量分割筛选 -> 结果存 all_only/{四位数}/（原图+分割图）。
    batch=5 表示启用 5 帧筛选；batch=1 表示不筛选（仅取一张）。
    返回 (result_url, folder_name, avg_elevation, ruler_min, ruler_max, send)。"""
    global _waterline_history
    ruler_csv = get_ruler_csv_for_site(site)
    _mode_tag = "（无边坡：应跳过 MobileNet-SSD）" if ruler_csv == "" else "（边坡：应走 MobileNet-SSD 框提示）"
    print(f"[control_first] 单帧分割 站点={site!r}，ruler_csv={ruler_csv!r}{_mode_tag}")

    try:
        batch = int(batch)
    except (TypeError, ValueError):
        batch = 5
    if batch < 1:
        batch = 1

    main_video.ensure_dir(VIDEO_INPUT_DIR)
    main_video.ensure_dir(OUTPUT_VIDEO_DIR)
    main_video.ensure_dir(SAVE_TARGET_DIR)
    processed_dir = os.path.join(IMAGE_INPUT_DIR, "processed")
    main_video.ensure_dir(processed_dir)

    # 收集 batch 张完整可读的图片（从最新的往次新收集，兼容相机流式到达与本地测试一次性批量写入）
    deadline = time.time() + WAIT_TIMEOUT
    img_paths = []
    first_name = None
    while len(img_paths) < batch:
        image_files = main_video.get_image_files(IMAGE_INPUT_DIR)
        if image_files:
            for candidate in reversed(image_files):
                if len(img_paths) >= batch:
                    break
                base = os.path.basename(candidate)
                if base in [os.path.basename(p) for p in img_paths]:
                    continue
                tmp = cv2.imread(candidate)
                if tmp is not None:
                    img_paths.append(candidate)
                    if first_name is None:
                        first_name = base
        if len(img_paths) >= batch:
            break
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("检测已被用户中断")
        if time.time() > deadline:
            if img_paths:
                break  # 超时但已有部分可用图片
            raise RuntimeError("等待图片超时")
        time.sleep(0.15)

    img_paths.sort(key=lambda x: main_video.natural_sort_key(os.path.basename(x)))

    # 手动框提示：若用户已点击，暂停分割并在 5 张图上等待人工点选水位点；
    # 框提示上沿 = 该点上方 5px（即 crop_edge = y - 5）。
    manual_crop_edges = None
    if manual_request_pending():
        manual_crop_edges = _prompt_manual_crop_edges(img_paths, stop_event)

    # 四位数递增文件夹：all_only/{folder_name}/
    folder_name = get_next_all_only_name()
    out_dir = os.path.join(ALL_ONLY_DIR, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # 保存第一张原图 + 分割结果图
    original_path = os.path.join(out_dir, "original.jpg")
    result_path = os.path.join(out_dir, "result.jpg")
    first_frame = cv2.imread(img_paths[0])
    if first_frame is not None:
        try:
            cv2.imwrite(original_path, first_frame)
        except Exception as e:
            print(f"  ⚠️ 保存原图失败: {e}")

    success, avg_elevation, selected_y, send, aborted = main_video.segment_single_frame_batch(
        img_paths, result_path, CHECKPOINT_PATH, SAVE_TARGET_DIR, ruler_csv, out_dir,
        history=_waterline_history, is_continuous=is_continuous,
        manual_crop_edges=manual_crop_edges, stop_event=stop_event)

    if aborted:
        # 当前这组被“手动框提示”中断：保留输入图，人工点选后用人工框重算同一组
        print("[control_first] 手动框提示：当前组已中断，改为人工点选后重算同一组")
        manual_crop_edges = _prompt_manual_crop_edges(img_paths, stop_event)
        success, avg_elevation, selected_y, send, _ = main_video.segment_single_frame_batch(
            img_paths, result_path, CHECKPOINT_PATH, SAVE_TARGET_DIR, ruler_csv, out_dir,
            history=_waterline_history, is_continuous=is_continuous,
            manual_crop_edges=manual_crop_edges, stop_event=stop_event)

    # 维护历史水位线（像素），限制长度
    if success and selected_y is not None:
        _waterline_history.append(selected_y)
        if len(_waterline_history) > HISTORY_MAX_LEN:
            _waterline_history = _waterline_history[-HISTORY_MAX_LEN:]

    # 分割完成后清空 input 文件夹（含 processed），避免图片积压
    clear_input_directory()

    ruler_min, ruler_max = ruler.get_ruler_range(ruler_csv)
    result_url = f"/all_only/{folder_name}/result.jpg"
    print(f"[control_first] 单帧分割完成（{len(img_paths)} 张筛选），结果: {out_dir}，水位: {avg_elevation}")
    return result_url, folder_name, avg_elevation, ruler_min, ruler_max, send


def run_one_local_video(site=None, stop_event=None, mp4_path=None):
    """本地测试-视频：直接把本地 mp4 交给 SAM3 视频分割 -> 叠加虚拟水尺 -> 输出。
    返回 (result_url, video_name, avg_elevation, ruler_min, ruler_max)。"""
    ruler_csv = get_ruler_csv_for_site(site)

    main_video.ensure_dir(VIDEO_INPUT_DIR)
    main_video.ensure_dir(OUTPUT_VIDEO_DIR)
    main_video.ensure_dir(SAVE_TARGET_DIR)

    if not mp4_path or not os.path.isfile(mp4_path):
        raise RuntimeError(f"视频文件不存在: {mp4_path}")

    v_name = os.path.basename(mp4_path)
    all_output_dir = os.path.join(ALL_OUTPUT_DIR, os.path.splitext(v_name)[0])
    os.makedirs(all_output_dir, exist_ok=True)

    print(f"[control_first] 本地测试视频分割: {v_name}（ruler_csv={ruler_csv!r}）")
    success, avg_elevation = main_video.segment_single_video(
        mp4_path, CHECKPOINT_PATH, SAVE_TARGET_DIR, ruler_csv,
        image_paths=None, all_output_dir=all_output_dir)
    if not success:
        print(f"[control_first] ⚠️ 视频 {v_name} 分割失败，仍输出原视频结果")

    out_path = os.path.join(OUTPUT_VIDEO_DIR, v_name)

    # ----- 叠加虚拟水尺 -----
    if os.path.isfile(ruler_csv) and os.path.isfile(out_path):
        tmp_gauge_path = out_path + ".gauge.mp4"
        try:
            ruler.ruler(
                input_path=out_path,
                output_path=tmp_gauge_path,
                csv_path=ruler_csv,
                step_main=0.05,
                original_image_height=240
            )
            os.replace(tmp_gauge_path, out_path)
            print(f"[control_first] 已叠加虚拟水尺: {out_path}")
        except Exception as e:
            print(f"[control_first] 叠加虚拟水尺失败，保留原视频: {e}")
            if os.path.exists(tmp_gauge_path):
                os.remove(tmp_gauge_path)
    else:
        print(f"[control_first] 未找到虚拟水尺查找表 {ruler_csv} 或输出视频，跳过叠加")

    transcode_h264(out_path)
    try:
        shutil.copy(out_path, os.path.join(all_output_dir, v_name))
    except Exception as e:
        print(f"  ⚠️ 复制结果视频失败: {e}")

    ruler_min, ruler_max = ruler.get_ruler_range(ruler_csv)
    result_url = "/output_video/" + v_name
    print(f"[control_first] 本地测试视频完成，输出: {out_path}，平均水位高程: {avg_elevation}")
    return result_url, v_name, avg_elevation, ruler_min, ruler_max


class TaskManager:
    """后台检测任务状态管理（线程安全）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._mode = "idle"
        self._result = None
        self._start_time_ms = None
        self._stop = threading.Event()
        self._thread = None
        self._site = None
        self._extra = {}

    def status(self):
        with self._lock:
            return {"task_running": self._running}

    def start(self, mode, timestamp_ms=None, site=None, extra=None):
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._mode = mode
            self._site = site
            self._extra = extra or {}
            self._result = None
            self._start_time_ms = timestamp_ms or int(time.time() * 1000)
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(mode, site), daemon=True)
            self._thread.start()
            return True

    def stop(self):
        """请求立即终止当前检测任务（单次与连续均适用），并关闭两个水泵（开关一、二）。"""
        with self._lock:
            self._stop.set()
        main_video.request_stop()
        # 兜底：即使部署的 request_stop 还是旧实现，也强制结束常驻 worker，
        # 否则它会在子进程里把当前这组/整段视频继续跑完。
        try:
            p = getattr(main_video, "_worker_process", None)
            if p is not None and p.is_alive():
                print("[control_first] 兜底：强制终止常驻 worker")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)
                main_video._worker_process = None
        except Exception as e:
            print(f"[control_first] 兜底终止 worker 失败: {e}")
        # 中断时暂停两个水泵：关闭开关一(outlet 0)、开关二(outlet 1)
        set_pump(0, False)
        set_pump(1, False)
        print("[control_first] 任务中断，已关闭开关一、二（暂停两个水泵）")

    def _run(self, mode, site=None):
        try:
            if mode == "continuous":
                settings = load_continuous_settings()
                interval = settings.get("interval_seconds", 0)
                count = settings.get("count", 0)
                done = 0
                while not self._stop.is_set():
                    if count > 0 and done >= count:
                        print(f"[control_first] 连续检测已完成 {count} 次，自动停止")
                        break

                    start_time = time.time()
                    try:
                        result_url, video_name, avg_elevation, ruler_min, ruler_max, send = self._run_once(site, self._stop)
                        if send:
                            self._publish(result_url, avg_elevation, ruler_min, ruler_max)
                    except Exception as e:
                        if self._stop.is_set():
                            break
                        print(f"[control_first] 连续检测单次失败: {e}")
                        time.sleep(0.5)
                        continue

                    done += 1
                    print(f"[control_first] 连续检测第 {done} 次完成")

                    if count > 0 and done >= count:
                        break

                    # 间隔控制：从本次分割开始时间算，不足则等待到间隔
                    if interval > 0:
                        elapsed = time.time() - start_time
                        if elapsed < interval:
                            wait = interval - elapsed
                            print(f"[control_first] 等待 {wait:.1f} 秒后开始下次分割")
                            while wait > 0 and not self._stop.is_set():
                                s = min(wait, 1.0)
                                time.sleep(s)
                                wait -= s
            elif mode == "continuous_frame":
                frame_count = 0
                stable_count = 0
                while not self._stop.is_set():
                    try:
                        result_url, folder_name, avg_elevation, ruler_min, ruler_max, _send = run_one_frame_detection(site, self._stop, is_continuous=True, batch=frame_batch())
                        frame_count += 1

                        self._publish(result_url, avg_elevation, ruler_min, ruler_max)
                        if avg_elevation is not None:
                            apply_auto_control(avg_elevation)

                        # 单次模式：达到目标水位后，连续 3 帧稳定在目标则自动中断并关闭水泵
                        if _auto_mode_state == "single" and _setpoint is not None and avg_elevation is not None:
                            if _display_round(avg_elevation) == _display_round(_setpoint):
                                stable_count += 1
                                print(f"[control_first] 单次模式：达标稳定 {stable_count}/3")
                            else:
                                stable_count = 0
                            if stable_count >= 3:
                                print("[control_first] 单次模式：连续 3 帧稳定达标，自动停止并关闭开关一、二")
                                set_pump(0, False)
                                set_pump(1, False)
                                self._stop.set()
                                break
                    except Exception as e:
                        if self._stop.is_set():
                            break
                        print(f"[control_first] 单帧连续分割单次失败: {e}")
                        time.sleep(0.5)
                        continue
            elif mode == "local_frame":
                # 本地测试-图片：处理当前 input 中已有的 batch 张图（batch=5 启用 5 帧筛选，否则 1 张）
                batch = int(self._extra.get("batch", 5) or 5)
                result_url, folder_name, avg_elevation, ruler_min, ruler_max, send = \
                    run_one_frame_detection(site, self._stop, is_continuous=False, batch=batch)
                self._publish(result_url, avg_elevation, ruler_min, ruler_max)
            elif mode == "local_video":
                # 本地测试-视频：直接把上传的 mp4 交给 SAM3 视频分割
                mp4_path = self._extra.get("file")
                result_url, video_name, avg_elevation, ruler_min, ruler_max = \
                    run_one_local_video(site, self._stop, mp4_path)
                self._publish(result_url, avg_elevation, ruler_min, ruler_max)
            else:
                # 单次检测
                result_url, video_name, avg_elevation, ruler_min, ruler_max, _send = self._run_once(site, self._stop)
                self._publish(result_url, avg_elevation, ruler_min, ruler_max)
        except Exception as e:
            print(f"[control_first] 检测任务异常: {e}")
            with self._lock:
                self._result = {"ready": True, "error": str(e)}
        finally:
            with self._lock:
                self._running = False
                self._mode = "idle"

    def _run_once(self, site=None, stop_event=None):
        """按当前分割方式执行一次检测；视频分割走 run_one_detection，单帧分割走 run_one_frame_detection。"""
        if load_seg_mode() == "frame":
            return run_one_frame_detection(site, stop_event, batch=frame_batch())
        return run_one_detection(site, stop_event)

    def _publish(self, result_url, avg_elevation=None, ruler_min=None, ruler_max=None):
        with self._lock:
            self._result = {
                "ready": True,
                "sample_start_time": self._start_time_ms,
                "avg_elevation": avg_elevation,
                "ruler_min": ruler_min,
                "ruler_max": ruler_max,
                "result_video_url": result_url,
                "switches": [
                    {"outlet": 0, "state": get_pump_state(0)},
                    {"outlet": 1, "state": get_pump_state(1)},
                    {"outlet": 2, "state": get_pump_state(2)},
                ],
            }

    def result(self):
        with self._lock:
            return self._result if self._result is not None else {"ready": False}


TASK = TaskManager()
LIVE = live_stream.LiveStreamManager(IMAGE_INPUT_DIR)


# ==================== HTTP 处理器 ====================

# 高频轮询接口日志节流：同一接口每 60 秒最多打印一次，避免刷屏
_LOG_THROTTLE_PATHS = {"/api/relay", "/api/result", "/api/status"}
_LOG_THROTTLE_SEC = 60
_LOG_THROTTLE = {}


class Handler(BaseHTTPRequestHandler):
    server_version = "control_first/1.0"

    def log_message(self, fmt, *args):
        # 只对高频轮询的 GET 请求做节流，其余（POST、错误等）照常打印
        # 注意：log_request 与 log_error 调用本函数的签名不同，
        # 请求行统一从 self.requestline 取，避免把状态码整数当成请求行。
        requestline = getattr(self, "requestline", "") or ""
        parts = requestline.split()
        if len(parts) >= 2 and parts[0] == "GET":
            path = parts[1].split("?")[0]
            if path in _LOG_THROTTLE_PATHS:
                now = time.time()
                last = _LOG_THROTTLE.get(path, 0.0)
                if now - last < _LOG_THROTTLE_SEC:
                    return
                _LOG_THROTTLE[path] = now
        try:
            msg = fmt % args if args else fmt
        except Exception:
            msg = fmt
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), msg))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/ws/live":
            live_stream.handle_websocket(self, LIVE)
            return

        if path == "/api/status":
            self._send_json(TASK.status())
        elif path == "/api/result":
            self._send_json(TASK.result())
        elif path == "/api/sites":
            sites = load_sites()
            site_list = [{"name": n, "csv": c} for n, c in sites.items()]
            self._send_json({"sites": site_list})
        elif path == "/api/settings":
            self._send_json(load_output_settings())
        elif path == "/api/continuous_settings":
            self._send_json(load_continuous_settings())
        elif path == "/api/seg_mode":
            cfg = load_seg_config()
            self._send_json({"mode": cfg["mode"], "frame5": cfg["frame5"]})
        elif path == "/api/setpoint":
            self._send_json({"setpoint": _setpoint, "auto_mode": _auto_mode_state})
        elif path == "/api/auto_mode":
            self._send_json({"auto_mode": _auto_mode_state})
        elif path == "/api/local_test":
            self._send_json(load_local_test())
        elif path == "/api/manual_status":
            self._send_json(manual_status())
        elif path == "/api/relay":
            self._send_json({"switches": [
                {"outlet": 0, "state": get_pump_state(0)},
                {"outlet": 1, "state": get_pump_state(1)},
                {"outlet": 2, "state": get_pump_state(2)},
            ]})
        elif path == "/api/download_all":
            self._handle_download_all(qs.get('video', [''])[0])
        elif path == "/api/download_all_only":
            self._handle_download_all_only(qs.get('folder', [''])[0])
        elif path.startswith("/api/"):
            self._send_json({"error": "not found"}, 404)
        else:
            self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        site = qs.get('site', [None])[0]

        if path == "/api/stop":
            # 若正在等待手动点选，先唤醒，避免分割线程一直阻塞
            cancel_manual_wait()
            print("[control_first] /api/stop 收到中断请求")
            TASK.stop()
            self._send_json({"ok": True, "stopped": True})
            return

        if path == "/api/manual_request":
            if set_manual_requested():
                self._send_json({"ok": True})
            else:
                self._send_json({"ok": False, "error": "当前没有单帧分割在进行"}, 409)
            return

        if path == "/api/manual_points":
            data = self._read_json_body()
            ok = submit_manual_points(data.get("points") or [])
            self._send_json({"ok": ok})
            return

        if path == "/api/sites/upload":
            self._handle_site_upload()
            return

        if path == "/api/sites/rename":
            self._handle_site_rename()
            return

        if path == "/api/sites/delete":
            self._handle_site_delete()
            return

        if path == "/api/settings":
            data = self._read_json_body()
            settings = {
                "download_enabled": bool(data.get("download_enabled", False)),
                "download_path": data.get("download_path", ""),
            }
            save_output_settings(settings)
            self._send_json({"ok": True, "settings": settings})
            return

        if path == "/api/continuous_settings":
            data = self._read_json_body()
            settings = {
                "interval_seconds": int(data.get("interval_seconds", 0) or 0),
                "count": int(data.get("count", 0) or 0),
            }
            save_continuous_settings(settings)
            self._send_json({"ok": True, "settings": settings})
            return

        if path == "/api/seg_mode":
            data = self._read_json_body()
            mode = data.get("mode")
            if mode not in ("video", "frame"):
                self._send_json({"error": "mode 非法"}, 400)
                return
            frame5 = bool(data.get("frame5", True))
            save_seg_mode(mode, frame5)
            self._send_json({"ok": True, "mode": mode, "frame5": frame5})
            return

        if path == "/api/auto_mode":
            data = self._read_json_body()
            auto_mode = data.get("auto_mode")
            if auto_mode not in ("continuous", "single"):
                self._send_json({"error": "auto_mode 非法"}, 400)
                return
            global _auto_mode_state
            _auto_mode_state = auto_mode
            self._send_json({"ok": True, "auto_mode": auto_mode})
            return

        if path == "/api/relay":
            data = self._read_json_body()
            outlet = int(data.get("outlet", 1))
            action = data.get("action", "toggle")
            if action == "on":
                set_pump(outlet, True)
            elif action == "off":
                set_pump(outlet, False)
            else:
                toggle_pump(outlet)
            self._send_json({"ok": True, "outlet": outlet, "state": get_pump_state(outlet)})
            return

        if path == "/api/setpoint":
            data = self._read_json_body()
            try:
                value = float(data.get("setpoint"))
            except (TypeError, ValueError):
                self._send_json({"error": "setpoint 非法"}, 400)
                return
            save_setpoint(value)
            # 设定后立刻开始单帧连续分割（若未在运行），按当前 auto_mode 状态执行
            if not TASK.status()["task_running"]:
                TASK.start("continuous_frame", None, site)
            self._send_json({"ok": True, "setpoint": value, "auto_mode": _auto_mode_state})
            return

        if path == "/api/local_test":
            data = self._read_json_body()
            method = data.get("method", "image")
            if method not in ("image", "video"):
                method = "image"
            cfg = {
                "method": method,
                "five_frame": bool(data.get("five_frame", True)),
                "path": str(data.get("path", "") or ""),
                "enabled": bool(data.get("enabled", False)),
            }
            save_local_test(cfg)
            self._send_json({"ok": True, "config": cfg})
            return

        if path == "/api/local_upload":
            self._handle_local_upload()
            return

        if path == "/api/local_clear":
            # 清空本地测试收件箱：input 图片目录 + 本地视频临时目录，避免上轮残留影响本轮
            clear_input_directory()
            try:
                if os.path.isdir(LOCAL_VIDEO_INBOX):
                    for fn in os.listdir(LOCAL_VIDEO_INBOX):
                        fp = os.path.join(LOCAL_VIDEO_INBOX, fn)
                        if os.path.isfile(fp):
                            os.remove(fp)
            except Exception as e:
                print(f"[control_first] 清理本地视频收件箱失败: {e}")
            self._send_json({"ok": True})
            return

        if path == "/api/local_run":
            data = self._read_json_body()
            kind = str(data.get("kind", "image")).lower()
            if kind == "video":
                name = os.path.basename(str(data.get("name", "") or ""))
                mp4_path = os.path.join(LOCAL_VIDEO_INBOX, name)
                if not name or not os.path.isfile(mp4_path):
                    self._send_json({"error": f"视频不存在: {name}"}, 404)
                    return
                if TASK.start("local_video", None, site, extra={"file": mp4_path}):
                    self._send_json({"ok": True, "kind": "video", "name": name})
                else:
                    self._send_json({"ok": False, "error": "检测任务正在运行中"}, 409)
            else:
                try:
                    batch = int(data.get("batch", 5) or 5)
                except (TypeError, ValueError):
                    batch = 5
                if batch <= 1:
                    batch = 1
                else:
                    batch = 5
                if TASK.start("local_frame", None, site, extra={"batch": batch}):
                    self._send_json({"ok": True, "kind": "image", "batch": batch})
                else:
                    self._send_json({"ok": False, "error": "检测任务正在运行中"}, 409)
            return

        if path != "/api/detect":
            self._send_json({"error": "not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            data = {}

        mode = data.get("mode", "single")
        if mode not in ("single", "continuous", "continuous_frame"):
            mode = "single"
        timestamp_ms = data.get("timestamp")

        if TASK.start(mode, timestamp_ms, site):
            self._send_json({"ok": True, "mode": mode, "site": site})
        else:
            self._send_json({"ok": False, "error": "检测任务正在运行中"}, 409)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return {}

    def _handle_site_upload(self):
        data = self._read_json_body()
        content_b64 = data.get("content", "")
        try:
            content = base64.b64decode(content_b64)
        except Exception:
            self._send_json({"ok": False, "error": "文件内容解析失败"}, 400)
            return
        if not content:
            self._send_json({"ok": False, "error": "文件内容为空"}, 400)
            return
        csv_name = get_next_csv_filename()
        csv_path = os.path.join(VIRTUAL_DIR, csv_name)
        try:
            with open(csv_path, 'wb') as f:
                f.write(content)
        except Exception as e:
            self._send_json({"ok": False, "error": f"写入文件失败: {e}"}, 500)
            return
        sites = load_sites()
        num = int(csv_name[:-4])  # csv 编号数字
        new_name = f"边坡{num}"
        # 若该边坡名已被占用（重命名等情况），向后找第一个不冲突的
        while new_name in sites:
            num += 1
            new_name = f"边坡{num}"
        sites[new_name] = csv_name
        save_sites(sites)
        self._send_json({"ok": True, "name": new_name, "csv": csv_name})

    def _handle_local_upload(self):
        """本地客户端上传文件：?kind=image|video&name=xxx，body 为文件原始字节。
        图片落到 input 目录，视频落到 LOCAL_VIDEO_INBOX 目录。"""
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        kind = str(qs.get('kind', ['image'])[0] or 'image').lower()
        name = os.path.basename(str(qs.get('name', [''])[0] or ''))

        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            self._send_json({"error": "空文件"}, 400)
            return
        raw = self.rfile.read(length)
        if not raw:
            self._send_json({"error": "空文件"}, 400)
            return

        if kind == 'video':
            os.makedirs(LOCAL_VIDEO_INBOX, exist_ok=True)
            if not name:
                name = f"local_{int(time.time() * 1000)}.mp4"
            if not name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                name = name + '.mp4'
            with open(os.path.join(LOCAL_VIDEO_INBOX, name), 'wb') as f:
                f.write(raw)
            print(f"[control_first] 本地上传视频: {name} ({len(raw)} bytes)")
            self._send_json({"ok": True, "kind": "video", "name": name, "bytes": len(raw)})
            return

        os.makedirs(IMAGE_INPUT_DIR, exist_ok=True)
        if not name:
            name = f"{int(time.time() * 1000)}.jpg"
        if not name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            name = name + '.jpg'
        with open(os.path.join(IMAGE_INPUT_DIR, name), 'wb') as f:
            f.write(raw)
        print(f"[control_first] 本地上传图片: {name} ({len(raw)} bytes)")
        self._send_json({"ok": True, "kind": "image", "name": name, "bytes": len(raw)})

    def _handle_site_rename(self):
        data = self._read_json_body()
        old_name = data.get("old_name", "")
        new_name = data.get("new_name", "").strip()
        if not old_name or not new_name:
            self._send_json({"ok": False, "error": "名称不能为空"}, 400)
            return
        sites = load_sites()
        if old_name not in sites:
            self._send_json({"ok": False, "error": "边坡不存在"}, 404)
            return
        if new_name in sites and new_name != old_name:
            self._send_json({"ok": False, "error": "名称已存在"}, 400)
            return
        sites[new_name] = sites.pop(old_name)
        save_sites(sites)
        self._send_json({"ok": True, "name": new_name})

    def _handle_site_delete(self):
        data = self._read_json_body()
        name = data.get("name", "")
        if not name:
            self._send_json({"ok": False, "error": "名称不能为空"}, 400)
            return
        sites = load_sites()
        if name not in sites:
            self._send_json({"ok": False, "error": "边坡不存在"}, 404)
            return
        csv_name = sites.pop(name)
        csv_path = os.path.join(VIRTUAL_DIR, csv_name)
        try:
            if os.path.isfile(csv_path):
                os.remove(csv_path)
        except Exception as e:
            print(f"[control_first] 删除 csv 失败: {e}")
        save_sites(sites)
        self._send_json({"ok": True, "name": name})

    def _handle_download_all(self, video):
        """把 all/{video} 目录打包成 zip 下载。"""
        import zipfile
        import io
        if not video:
            self._send_json({"error": "缺少 video 参数"}, 400)
            return
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', video):
            self._send_json({"error": "非法参数"}, 400)
            return
        src_dir = os.path.join(ALL_OUTPUT_DIR, video)
        if not os.path.isdir(src_dir):
            self._send_json({"error": "目录不存在"}, 404)
            return
        buf = io.BytesIO()
        try:
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(src_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        arcname = os.path.relpath(fp, src_dir)
                        zf.write(fp, arcname)
        except Exception as e:
            print(f"[control_first] 打包失败: {e}")
            self._send_json({"error": f"打包失败: {e}"}, 500)
            return
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{video}.zip"')
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    # ---------- 静态资源托管（含 Range 支持，供 <video> 播放/拖动） ----------

    def _resolve_static(self, path):
        if path in ("/", "/index.html"):
            return os.path.join(WEB_DIR, "index.html")
        if path.startswith("/input/"):
            # /input/{file}：手动框提示时展示待标注的原图
            return os.path.join(IMAGE_INPUT_DIR, os.path.basename(path))
        if path.startswith("/output_video/"):
            return os.path.join(OUTPUT_VIDEO_DIR, os.path.basename(path))
        if path.startswith("/all_only/"):
            # /all_only/{folder}/{file}
            rel = path[len("/all_only/"):]
            return os.path.join(ALL_ONLY_DIR, rel)
        if path.startswith("/save/"):
            return os.path.join(SAVE_TARGET_DIR, os.path.basename(path))
        name = os.path.basename(path)
        if name in ("app.js", "style.css"):
            return os.path.join(WEB_DIR, name)
        return None

    def _serve_static(self, path):
        filepath = self._resolve_static(path)
        if not filepath or not os.path.isfile(filepath):
            self.send_error(404)
            return

        size = os.path.getsize(filepath)
        ctype = mimetypes.guess_type(filepath)[0] or "application/octet-stream"

        range_header = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", (range_header or "").strip())
        if m:
            start_s, end_s = m.group(1), m.group(2)
            if start_s == "":
                length = int(end_s or 0)
                start = max(0, size - length)
                end = size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(filepath, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)


# ==================== 入口 ====================

def clean_startup():
    """启动时清空输入图片目录（含 processed 等子目录）"""
    for dir_path in (IMAGE_INPUT_DIR,):
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"[control_first] 已清空目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"[control_first] 清空目录 {dir_path} 失败: {e}")


def main():
    # 手动停止程序（Ctrl+C / kill）时兜底清理常驻 worker，防止其残留占用显存
    atexit.register(main_video.shutdown)

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except Exception:
        pass

    clean_startup()
    main_video.ensure_dir(LOCAL_VIDEO_INBOX)
    LIVE.start()

    # 读取设定水位线 + 初始化继电器（开三、关一二）
    load_setpoint()
    init_relay_state()

    # 服务启动时立即预加载常驻 worker（后台异步加载模型，不阻塞服务）
    print("[control_first] 预加载 SAM3 模型（常驻 worker）...")
    main_video._init_worker(CHECKPOINT_PATH, SAVE_TARGET_DIR)

    print("=" * 60)
    print("control_first 检测控制服务")
    print(f"  监听地址: http://{HOST}:{PORT}")
    print(f"  前端页面: http://{HOST}:{PORT}/")
    print(f"  输出目录: {OUTPUT_VIDEO_DIR}")
    print(f"  单次检测 = {TARGET_FRAMES} 张图片合成视频后 SAM3 分割")
    print(f"  虚拟水尺查找表: {VIRTUAL_RULER_CSV}")
    print("=" * 60)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[control_first] 用户中断，服务退出")
    finally:
        LIVE.stop()
        main_video.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()