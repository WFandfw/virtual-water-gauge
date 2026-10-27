import os
import sys
import time
import shutil
import re
import cv2
import numpy as np
import multiprocessing as mp
import threading
import signal
from pathlib import Path
from PIL import Image

# 确保可以导入同目录下的 sam3video 模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ================== 路径配置 ==================
IMAGE_INPUT_DIR = "/root/sam/input"
VIDEO_INPUT_DIR = "/root/sam/input_video"
OUTPUT_VIDEO_DIR = "/root/sam/output_video"
SAVE_TARGET_DIR = "/root/sam/save"
CHECKPOINT_PATH = "/root/sam/sam3.pt"

FPS = 25
TARGET_FRAMES = 25   # 1 秒 × 25fps = 25 帧
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# ================== 工具函数 ==================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def get_image_files(folder):
    files = []
    if not os.path.isdir(folder):
        return files
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTENSIONS:
            files.append(os.path.join(folder, f))
    files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    return files

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def move_to_processed(file_list, processed_dir):
    ensure_dir(processed_dir)
    for f in file_list:
        try:
            shutil.move(f, os.path.join(processed_dir, os.path.basename(f)))
        except Exception as e:
            print(f"  ⚠️ 移动文件 {f} 失败: {e}")

def get_next_video_name(output_dir):
    ensure_dir(output_dir)
    max_num = 0
    for f in os.listdir(output_dir):
        if f.lower().endswith('.mp4'):
            name = os.path.splitext(f)[0]
            if name.isdigit():
                max_num = max(max_num, int(name))
    return f"{max_num + 1:04d}.mp4"

def clean_startup_dirs():
    """启动时清空 input 和 input_video 目录（包括 processed 子目录）"""
    for dir_path in [IMAGE_INPUT_DIR, VIDEO_INPUT_DIR]:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"🧹 已清空目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"⚠️ 清空目录 {dir_path} 失败: {e}")

# ================== 常驻视频分割 worker（模型只加载一次） ==================

_worker_lock = threading.Lock()
_worker_process = None
_task_queue = None
_result_queue = None
# 记录最近一次 worker 的启动参数，用于中断后后台预加载
_worker_args = None
# 服务整体退出标记：置位后禁止中断后的后台预加载，避免退出瞬间又拉起 worker
_shutting_down = False
_interrupt_event = threading.Event()
# 手动框提示：请求中断当前“单帧批量”等待，改为人工点选后重算同一组
_manual_abort = threading.Event()
# 单帧批量任务序号：用于丢弃被中断组的残留结果，避免串号
_task_token = 0
_token_lock = threading.Lock()


def _next_token():
    global _task_token
    with _token_lock:
        _task_token += 1
        return _task_token


def _init_worker(checkpoint_path, save_target_dir):
    """初始化常驻 worker 进程（只启动一次，模型只加载一次）。"""
    global _worker_process, _task_queue, _result_queue, _worker_args
    with _worker_lock:
        if _shutting_down:
            return
        if _worker_process is not None and _worker_process.is_alive():
            return
        _worker_args = (checkpoint_path, save_target_dir)
        ctx = mp.get_context("spawn")
        _task_queue = ctx.Queue()
        _result_queue = ctx.Queue()
        _worker_process = ctx.Process(
            target=_worker_loop,
            args=(checkpoint_path, save_target_dir, _task_queue, _result_queue)
        )
        _worker_process.start()
        print("  [main_video] 常驻 worker 已启动（模型只加载一次）")


def _preload_after_stop(delay=2.0):
    """中断杀掉 worker 后，在后台重新拉起一个并预加载模型。

    SIGKILL 后 CUDA 上下文释放有延迟，所以先等 delay 秒；若这期间用户已经
    点了下一次分割（worker 已被 _init_worker 拉起），这里直接跳过。
    """
    def _run():
        try:
            time.sleep(delay)
            if _shutting_down:
                return
            with _worker_lock:
                if _worker_process is not None and _worker_process.is_alive():
                    return
                if _shutting_down:
                    return
                args = _worker_args
            if not args or _shutting_down:
                return
            print("  [main_video] 中断后后台预加载模型（下次分割可直接使用）...")
            _init_worker(*args)
        except Exception as e:
            print(f"  [main_video] 后台预加载失败: {e}")

    threading.Thread(target=_run, daemon=True, name="worker-preload").start()


def request_stop():
    """请求立即终止当前正在执行的分割任务。

    注意：worker 是 spawn 出来的独立进程，父进程里的 threading.Event 它看不到，
    仅 set(_interrupt_event) 只能让“父进程停止等待”，worker 仍会把当前推理跑完。
    这里直接 terminate worker，才能真正停下云端分割；下一次任务会自动重新加载模型。
    """
    global _worker_process
    _interrupt_event.set()
    killed = False
    with _worker_lock:
        p = _worker_process
        if p is not None and p.is_alive():
            print("  [main_video] 收到中断请求，正在终止 worker 进程...")
            try:
                # worker 在 _worker_loop 里调用了 os.setsid()，杀掉整个进程组，
                # 连它内部再 spawn 出来的子进程（如视频分割）一起清掉。
                pgid = os.getpgid(p.pid)
                if pgid != os.getpgid(0):
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    p.terminate()
            except Exception:
                try:
                    p.terminate()
                except Exception:
                    pass
            p.join(timeout=5)
            if p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass
                p.join(timeout=2)
            killed = True
            print("  [main_video] worker 已终止（后台将预加载模型）")
        else:
            print("  [main_video] 收到中断请求，当前没有运行中的 worker")
        _worker_process = None

    if killed:
        _preload_after_stop()


def shutdown():
    """服务整体退出时终止常驻 worker 及其子进程，释放显存。

    仅用于手动停止程序（Ctrl+C / SIGTERM）的退出路径，不影响网页端 /api/stop。
    """
    global _shutting_down, _worker_process, _worker_args
    _shutting_down = True
    _worker_args = None
    _interrupt_event.set()
    with _worker_lock:
        p = _worker_process
        _worker_process = None
    if p is None or not p.is_alive():
        return
    print("  [main_video] 服务退出，终止常驻 worker 及子进程...")
    try:
        pgid = os.getpgid(p.pid)
        if pgid != os.getpgid(0):
            os.killpg(pgid, signal.SIGKILL)
        else:
            p.terminate()
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass
    p.join(timeout=5)
    if p.is_alive():
        try:
            p.kill()
        except Exception:
            pass
        p.join(timeout=2)
    print("  [main_video] worker 已终止")


def request_manual_abort():
    """请求中断当前“单帧批量”等待（用于手动框提示：改为人工点选后重算同一组）。"""
    _manual_abort.set()


def _drain_result_queue():
    """清空结果队列中残留的历史结果，避免下次分割拿到串号结果。"""
    global _result_queue
    if _result_queue is None:
        return
    try:
        while True:
            _result_queue.get_nowait()
    except Exception:
        pass


def predict_waterline(history, poly_degree=3):
    """用历史水位线（像素 y，时间顺序）做多项式拟合，预测下一帧水位线像素。"""
    n = len(history)
    if n == 0:
        return None
    if n == 1:
        return float(history[0])
    xs = np.arange(n, dtype=float)
    ys = np.array(history, dtype=float)
    degree = min(poly_degree, n - 1)
    coeffs = np.polyfit(xs, ys, degree)
    return float(np.polyval(coeffs, n))


def cluster_select(candidates, band=3):
    """聚类筛选：均值 ±band 分三区，保留帧数最多的区，迭代剔除，最后选离均值最近的。"""
    active = list(candidates)
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    while True:
        ys = [c['final_y'] for c in active]
        mean_y = float(np.mean(ys))
        left = [c for c in active if c['final_y'] < mean_y - band]
        center = [c for c in active if mean_y - band <= c['final_y'] <= mean_y + band]
        right = [c for c in active if c['final_y'] > mean_y + band]
        counts = [len(left), len(center), len(right)]
        max_cnt = max(counts)
        if len(center) == max_cnt:
            keep = center
        elif len(right) == max_cnt:
            keep = right
        else:
            keep = left
        if len(keep) == len(active):
            break
        active = keep
        if len(active) <= 1:
            break
    ys = [c['final_y'] for c in active]
    mean_y = float(np.mean(ys))
    return min(active, key=lambda c: abs(c['final_y'] - mean_y))


def select_waterline(candidates, history=None, is_continuous=False):
    """
    从多张候选分割结果中筛选最终水位线。
    candidates: list of dict，每项含 final_y(像素)、score(置信度) 等。
    返回选中的候选 dict。
    """
    if not candidates:
        return None

    # 步骤1：置信度筛选（>0.8 的图里选置信度最高的作为基准，偏离 ±20px 舍去）
    high_conf = [c for c in candidates if c.get('score') is not None and c['score'] > 0.8]
    if high_conf:
        base = max(high_conf, key=lambda c: c['score'])['final_y']
        kept = [c for c in candidates if abs(c['final_y'] - base) <= 20]
    else:
        kept = candidates
    if not kept:
        kept = candidates

    ys = [c['final_y'] for c in kept]
    y_min, y_max = min(ys), max(ys)

    # 步骤2：最高最低差距 ≤5px，直接选水位最高（y 最小）
    if y_max - y_min <= 5:
        return min(kept, key=lambda c: c['final_y'])

    # 步骤3：连续分割且历史 >3 组，用多项式拟合预测，选最近的
    if is_continuous and history and len(history) > 3:
        pred = predict_waterline(history)
        if pred is not None:
            return min(kept, key=lambda c: abs(c['final_y'] - pred))

    # 步骤4：聚类（均值 ±3px），选离均值最近的
    return cluster_select(kept, band=3)


def _blend_mask(overlay, mask, color, alpha=0.5):
    """把 bool 掩码以半透明颜色叠加到 BGR 图上（原地修改）。"""
    if mask is None:
        return
    mask_bool = np.squeeze(mask).astype(bool)
    h, w = overlay.shape[:2]
    if mask_bool.ndim != 2 or mask_bool.shape[:2] != (h, w):
        return
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c]
        )


def _draw_dashed_line(img, y, color=(255, 0, 0), dash_len=10, gap_len=6, thickness=2):
    """在 BGR 图上画一条水平虚线。"""
    h, w = img.shape[:2]
    y = max(0, min(int(y), h - 1))
    x = 0
    while x < w:
        x_end = min(x + dash_len, w)
        cv2.line(img, (x, y), (x_end, y), color, thickness)
        x = x_end + gap_len


def _process_single_frame(image_path, out_image_path, ruler_csv, all_output_dir, image_model):
    """在常驻 worker 进程内处理单张图片：SAM3 单帧分割 -> 水位线 -> 叠加掩码/水位线/水尺 -> 保存结果图。
    返回 (success, real_waterline, send)。send=True 表示应发送给前端，False 表示 auto 突变不发。"""
    import sam3first
    import ruler

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  [单帧] 无法读取图片: {image_path}")
        return False, None, False

    h, w = frame.shape[:2]
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    success, final_y, final_mask, final_score, floating_mask, crop_edge, mask_b = \
        sam3first.process_single_image_video(pil_frame, image_model, prompt="black water", ruler_csv=ruler_csv)

    if not success or final_mask is None:
        print("  [单帧] 水位线识别失败")
        return False, None, False

    # 三路水位线对比结果已在 process_single_image_video 内确定，作为最终蓝色掩码，不再二次 optimize
    print(f"  [单帧] 水位线 Y = {final_y}")

    # 单帧分割总是发送前端（auto 模板已移除）
    send = True

    # 完整红掩码：水位线以下全部填充红色
    complete_mask = np.zeros((h, w), dtype=bool)
    complete_mask[int(final_y):, :] = True

    # 输出结果图（完整红掩码 + 蓝虚线水位线）
    overlay = frame.copy()
    _blend_mask(overlay, complete_mask, (0, 0, 255))            # 红色
    _draw_dashed_line(overlay, int(final_y), color=(255, 0, 0))  # 蓝虚线

    # 原本红掩码图（分割原始掩码：B 红 + A 蓝 + 蓝虚线）
    orig_overlay = frame.copy()
    _blend_mask(orig_overlay, mask_b, (0, 0, 255))              # 原本 B 红色
    _blend_mask(orig_overlay, final_mask, (255, 0, 0))          # 原本 A 蓝色
    _draw_dashed_line(orig_overlay, int(final_y), color=(255, 0, 0))

    # 换算真实水位
    real_waterline = None
    if ruler_csv and os.path.isfile(ruler_csv):
        try:
            real_waterline = ruler.pixel_to_real_waterline(
                final_y, ruler_csv, image_height=h, original_image_height=240)
        except Exception as e:
            print(f"  [单帧] 换算真实水位失败: {e}")

    # 叠加虚拟水尺（两张都叠，便于对比）
    if ruler_csv and os.path.isfile(ruler_csv):
        try:
            overlay = ruler._overlay_on_frame(overlay, ruler_csv, original_image_height=240)
            orig_overlay = ruler._overlay_on_frame(orig_overlay, ruler_csv, original_image_height=240)
        except Exception as e:
            print(f"  [单帧] 叠加水尺失败: {e}")

    try:
        os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
        cv2.imwrite(out_image_path, overlay)
        cv2.imwrite(os.path.join(os.path.dirname(out_image_path), "mask_orig.jpg"), orig_overlay)
    except Exception as e:
        print(f"  [单帧] 保存结果图失败: {e}")

    return True, real_waterline, send


def _process_frame_batch(image_paths, out_image_path, ruler_csv, all_output_dir, image_model, history=None, is_continuous=False, manual_crop_edges=None):
    """处理多张连续图，筛选出最终水位线。返回 (success, real_waterline, selected_final_y, send)。
    manual_crop_edges: 与 image_paths 等长的人手框上边缘列表（None 表示该帧走自动动态框）。"""
    import sam3first
    import ruler

    candidates = []
    for idx, image_path in enumerate(image_paths):
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        manual_edge = None
        if manual_crop_edges and idx < len(manual_crop_edges):
            manual_edge = manual_crop_edges[idx]
        success, final_y, final_mask, final_score, floating_mask, crop_edge, mask_b = \
            sam3first.process_single_image_video(pil_frame, image_model, prompt="black water", manual_crop_edge=manual_edge, ruler_csv=ruler_csv)
        if not success or final_mask is None:
            continue
        # 三路水位线对比结果已在 process_single_image_video 内确定，作为最终蓝色掩码，不再二次 optimize
        candidates.append({'final_y': final_y, 'score': final_score, 'mask': final_mask, 'mask_b': mask_b, 'frame': frame})

    if not candidates:
        print("  [单帧批量] 全部识别失败")
        return False, None, None, False

    best = select_waterline(candidates, history=history, is_continuous=is_continuous)
    if best is None:
        return False, None, None, False

    final_y = best['final_y']
    frame = best['frame']
    final_mask = best['mask']
    mask_b = best.get('mask_b')
    h, w = frame.shape[:2]
    print(f"  [单帧批量] {len(candidates)} 张有效，筛选得水位线 Y = {final_y}")

    send = True

    # 完整红掩码：水位线以下全部填充红色
    complete_mask = np.zeros((h, w), dtype=bool)
    complete_mask[int(final_y):, :] = True

    # 输出结果图（完整红掩码 + 蓝虚线水位线）
    overlay = frame.copy()
    _blend_mask(overlay, complete_mask, (0, 0, 255))            # 红色
    _draw_dashed_line(overlay, int(final_y), color=(255, 0, 0))  # 蓝虚线

    # 原本红掩码图（分割原始掩码：B 红 + A 蓝 + 蓝虚线）
    orig_overlay = frame.copy()
    _blend_mask(orig_overlay, mask_b, (0, 0, 255))              # 原本 B 红色
    _blend_mask(orig_overlay, final_mask, (255, 0, 0))          # 原本 A 蓝色
    _draw_dashed_line(orig_overlay, int(final_y), color=(255, 0, 0))

    # 换算真实水位
    real_waterline = None
    if ruler_csv and os.path.isfile(ruler_csv):
        try:
            real_waterline = ruler.pixel_to_real_waterline(
                final_y, ruler_csv, image_height=h, original_image_height=240)
        except Exception as e:
            print(f"  [单帧批量] 换算真实水位失败: {e}")

    # 叠加虚拟水尺（两张都叠，便于对比）
    if ruler_csv and os.path.isfile(ruler_csv):
        try:
            overlay = ruler._overlay_on_frame(overlay, ruler_csv, original_image_height=240)
            orig_overlay = ruler._overlay_on_frame(orig_overlay, ruler_csv, original_image_height=240)
        except Exception as e:
            print(f"  [单帧批量] 叠加水尺失败: {e}")

    try:
        os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
        cv2.imwrite(out_image_path, overlay)
        cv2.imwrite(os.path.join(os.path.dirname(out_image_path), "mask_orig.jpg"), orig_overlay)
    except Exception as e:
        print(f"  [单帧批量] 保存结果图失败: {e}")

    return True, real_waterline, final_y, send


def _print_worker_env(checkpoint_path):
    """加载模型前打印环境信息：权重路径、torch/CUDA、显存、模块真实来源。

    服务器上同时存在 /root/sam 与 /root/sam/all_linux 两份代码时，靠「模块来源」
    一眼就能看出子进程实际加载的是哪一份。"""
    print(f"  [worker] cwd={os.getcwd()}")
    exists = bool(checkpoint_path) and os.path.isfile(str(checkpoint_path))
    print(f"  [worker] checkpoint={checkpoint_path!r} 存在={exists}")
    try:
        import torch
        print(f"  [worker] torch={torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  [worker] GPU={torch.cuda.get_device_name(0)} "
                  f"空闲显存 {free / 1024 ** 3:.2f}/{total / 1024 ** 3:.2f} GiB")
    except Exception as e:
        print(f"  [worker] 显存信息查询失败: {e}")
    for mod_name in ("sam3video", "sam3first"):
        try:
            mod = __import__(mod_name)
            print(f"  [worker] 模块来源 {mod_name}={getattr(mod, '__file__', '?')}")
        except Exception as e:
            print(f"  [worker] 模块来源 {mod_name} 查询失败: {e}")
    print(f"  [worker] sys.path[0:3]={sys.path[:3]}")


def _print_worker_failure(what):
    """模型加载失败时补充可判读的信息（多半是权重缺失或显存被占满）。"""
    print(f"  ❌ [worker] {what} 加载失败，完整异常如下：")
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            tip = "（显存几乎占满：很可能旧进程/上一轮模型没释放，先用 nvidia-smi 看一下）" \
                if free / 1024 ** 3 < 2 else ""
            print(f"     GPU 空闲显存 {free / 1024 ** 3:.2f}/{total / 1024 ** 3:.2f} GiB{tip}")
    except Exception:
        pass
    import traceback
    traceback.print_exc()


def _worker_loop(checkpoint_path, save_target_dir, task_queue, result_queue):
    """常驻 worker 进程：加载一次模型，循环处理视频任务，直到收到 None 停止。"""
    # 独立进程组，方便中断时连内部子进程一起杀掉
    try:
        os.setsid()
    except Exception:
        pass

    import sam3video
    import gc
    import torch

    # 模型构建函数优先直接从 sam3 包导入，不再依赖 sam3video 是否把它再导出：
    # 服务器上若残留旧版 sam3video.py，会报 module 'sam3video' has no attribute ...
    try:
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
        _builder_src = "sam3.model_builder（直连）"
    except Exception as _e:
        build_sam3_image_model = getattr(sam3video, "build_sam3_image_model", None)
        build_sam3_video_predictor = getattr(sam3video, "build_sam3_video_predictor", None)
        _builder_src = "sam3video（回退）"
        if build_sam3_image_model is None or build_sam3_video_predictor is None:
            print(f"  ❌ [worker] 找不到模型构建函数：sam3.model_builder 导入失败（{_e}），"
                  f"sam3video 也没有再导出它们。")
            print(f"     当前 sam3video 来源: {getattr(sam3video, '__file__', '?')}")
            _names = [n for n in dir(sam3video) if "build" in n.lower() or "model" in n.lower()]
            print(f"     sam3video 内与模型相关的符号: {_names[:12]}")
            raise
    print(f"  [worker] 模型构建函数来源: {_builder_src}")

    # 关闭 cudnn benchmark，减少其 workspace 显存需求，缓解传播时碎片化 OOM
    torch.backends.cudnn.benchmark = False

    print("  [worker] 正在加载 SAM3 Image 模型（常驻）...")
    _print_worker_env(checkpoint_path)
    try:
        image_model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    except Exception:
        _print_worker_failure("SAM3 Image 模型")
        raise
    print("  [worker] 正在加载 SAM3 Video 预测模型（常驻）...")
    try:
        video_predictor = build_sam3_video_predictor(checkpoint_path)
    except Exception:
        _print_worker_failure("SAM3 Video 预测模型")
        raise
    print("  [worker] 模型加载完毕，等待视频任务...")

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            # 单帧任务：("frame", image_path, out_image_path, ruler_csv, all_output_dir)
            if isinstance(task, tuple) and len(task) == 5 and task[0] == "frame":
                _, frame_image_path, frame_out_path, frame_ruler_csv, frame_all_dir = task
                try:
                    success, elev, send = _process_single_frame(
                        frame_image_path, frame_out_path, frame_ruler_csv, frame_all_dir, image_model)
                    result_queue.put((success, elev, send))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    result_queue.put((False, None, False))
                continue
            # 单帧批量任务：("frame_batch", image_paths, out_image_path, ruler_csv, all_output_dir, history, is_continuous, manual_crop_edges, token)
            if isinstance(task, tuple) and len(task) == 9 and task[0] == "frame_batch":
                _, frame_image_paths, frame_out_path, frame_ruler_csv, frame_all_dir, frame_history, frame_cont, frame_manual, frame_token = task
                try:
                    success, elev, selected_y, send = _process_frame_batch(
                        frame_image_paths, frame_out_path, frame_ruler_csv, frame_all_dir, image_model,
                        history=frame_history, is_continuous=frame_cont, manual_crop_edges=frame_manual)
                    result_queue.put((success, elev, selected_y, send, frame_token))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    result_queue.put((False, None, None, False, frame_token))
                continue
            video_path, out_path, v_name, ruler_csv, image_paths, all_output_dir = task
            try:
                success, avg_elevation = sam3video.process_single_video(
                    v_path=video_path,
                    out_path=out_path,
                    v_name=v_name,
                    checkpoint_path=checkpoint_path,
                    save_target_folder=save_target_dir,
                    n_sample_frames=7,
                    ruler_csv=ruler_csv,
                    image_model=image_model,
                    video_predictor=video_predictor,
                    image_paths=image_paths,
                    all_output_dir=all_output_dir,
                )
                result_queue.put((success, avg_elevation))
            except Exception:
                import traceback
                traceback.print_exc()
                result_queue.put((False, None))
    finally:
        print("  [worker] 退出，释放常驻模型...")
        try:
            del image_model, video_predictor
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def segment_single_video(video_path, checkpoint_path, save_target_dir, ruler_csv=None, image_paths=None, all_output_dir=None):
    global _worker_process, _task_queue, _result_queue
    print(f"  [main_video] segment_single_video 收到 ruler_csv = {ruler_csv!r}")
    v_name = os.path.basename(video_path)
    out_path = os.path.join(OUTPUT_VIDEO_DIR, v_name)
    ensure_dir(OUTPUT_VIDEO_DIR)
    ensure_dir(save_target_dir)

    print(f"  🎞️ 开始分割视频: {v_name}")

    # 新分割开始，清空中断标志，并排空残留结果，避免拿到上一次的串号结果
    _interrupt_event.clear()
    _drain_result_queue()

    # 确保常驻 worker 已启动（首次调用启动，后续复用）
    _init_worker(checkpoint_path, save_target_dir)

    # 发送任务给常驻 worker
    _task_queue.put((video_path, out_path, v_name, ruler_csv, image_paths, all_output_dir))

    # 等待结果（每 1 秒检查中断/worker 崩溃）
    success = False
    avg_elevation = None
    while True:
        if _interrupt_event.is_set():
            print("  ⛔ 分割已被用户中断")
            return False, None
        try:
            res = _result_queue.get(timeout=1)
            success, avg_elevation = res if isinstance(res, tuple) else (bool(res), None)
            break
        except Exception:
            if _interrupt_event.is_set():
                print("  ⛔ 分割已被用户中断")
                return False, None
            if _worker_process is not None and not _worker_process.is_alive():
                print("  ⚠️ 常驻 worker 异常退出，重启后重试...")
                _worker_process = None
                _init_worker(checkpoint_path, save_target_dir)
                _task_queue.put((video_path, out_path, v_name, ruler_csv, image_paths, all_output_dir))
            # worker 还活着，继续等

    return success, avg_elevation


def segment_single_frame(image_path, out_image_path, checkpoint_path, save_target_dir, ruler_csv=None, all_output_dir=None):
    """对单张图片做 SAM3 单帧分割，输出结果图到 out_image_path，返回 (success, avg_elevation, send)。"""
    global _worker_process, _task_queue, _result_queue

    _interrupt_event.clear()
    _drain_result_queue()
    _init_worker(checkpoint_path, save_target_dir)
    _task_queue.put(("frame", image_path, out_image_path, ruler_csv, all_output_dir))

    success = False
    avg_elevation = None
    send = True
    while True:
        if _interrupt_event.is_set():
            print("  ⛔ 单帧分割已被用户中断")
            return False, None, False
        try:
            res = _result_queue.get(timeout=1)
            if isinstance(res, tuple) and len(res) >= 3:
                success, avg_elevation, send = res[0], res[1], res[2]
            else:
                success = bool(res) if not isinstance(res, tuple) else bool(res[0])
                avg_elevation = res[1] if isinstance(res, tuple) and len(res) >= 2 else None
                send = True
            break
        except Exception:
            if _interrupt_event.is_set():
                print("  ⛔ 单帧分割已被用户中断")
                return False, None, False
            if _worker_process is not None and not _worker_process.is_alive():
                print("  ⚠️ 常驻 worker 异常退出，重启后重试...")
                _worker_process = None
                _init_worker(checkpoint_path, save_target_dir)
                _task_queue.put(("frame", image_path, out_image_path, ruler_csv, all_output_dir))

    return success, avg_elevation, send


def segment_single_frame_batch(image_paths, out_image_path, checkpoint_path, save_target_dir, ruler_csv=None, all_output_dir=None, history=None, is_continuous=False, manual_crop_edges=None, stop_event=None):
    """对多张连续图做 SAM3 单帧分割并筛选，返回 (success, avg_elevation, selected_final_y, send, aborted)。
    aborted=True 表示本组等待被“手动框提示”中断（结果作废，调用方应保留输入图并重算同一组）。
    stop_event: 父任务（TaskManager）的停止事件，优先级高于模块内的 _interrupt_event。"""
    global _worker_process, _task_queue, _result_queue

    def _stopped():
        return (stop_event is not None and stop_event.is_set()) or _interrupt_event.is_set()

    # 启动前已停止：直接返回，绝不能再 clear 掉停止标志或往队列里塞新任务
    if stop_event is not None and stop_event.is_set():
        print("  ⛔ 单帧批量分割启动前已被中断")
        return False, None, None, False, False

    _interrupt_event.clear()
    _manual_abort.clear()
    token = _next_token()
    _drain_result_queue()

    if _stopped():
        print("  ⛔ 单帧批量分割启动前已被中断")
        return False, None, None, False, False

    _init_worker(checkpoint_path, save_target_dir)
    if _stopped():
        print("  ⛔ 单帧批量分割启动前已被中断")
        return False, None, None, False, False

    _task_queue.put(("frame_batch", image_paths, out_image_path, ruler_csv, all_output_dir, history, is_continuous, manual_crop_edges, token))

    success = False
    avg_elevation = None
    selected_final_y = None
    send = True
    while True:
        if _manual_abort.is_set():
            print("  ⏸ 单帧批量分割等待被“手动框提示”中断，结果作废")
            return False, None, None, False, True
        if _stopped():
            print("  ⛔ 单帧批量分割已被用户中断")
            return False, None, None, False, False
        try:
            res = _result_queue.get(timeout=1)
            # 只接受本组结果（token 匹配），否则视为上一组残留并丢弃
            if isinstance(res, tuple) and len(res) >= 5 and res[4] == token:
                success, avg_elevation, selected_final_y, send = res[0], res[1], res[2], res[3]
                break
            print(f"  ♻️ 丢弃非本组分割结果（token {res[4] if isinstance(res, tuple) and len(res) >= 5 else '?'} != {token}）")
            continue
        except Exception:
            if _manual_abort.is_set():
                print("  ⏸ 单帧批量分割等待被“手动框提示”中断，结果作废")
                return False, None, None, False, True
            if _stopped():
                print("  ⛔ 单帧批量分割已被用户中断")
                return False, None, None, False, False
            if _worker_process is not None and not _worker_process.is_alive():
                print("  ⚠️ 常驻 worker 异常退出，重启后重试...")
                _worker_process = None
                _init_worker(checkpoint_path, save_target_dir)
                _task_queue.put(("frame_batch", image_paths, out_image_path, ruler_csv, all_output_dir, history, is_continuous, manual_crop_edges, token))

    return success, avg_elevation, selected_final_y, send, False


# ================== 主函数（唯一入口） ==================
def main():
    # 启动时清理目录
    clean_startup_dirs()

    ensure_dir(IMAGE_INPUT_DIR)
    ensure_dir(VIDEO_INPUT_DIR)
    ensure_dir(OUTPUT_VIDEO_DIR)
    ensure_dir(SAVE_TARGET_DIR)
    processed_dir = os.path.join(IMAGE_INPUT_DIR, "processed")
    ensure_dir(processed_dir)

    print("🚀 程序启动：自动监控图片并合成视频+分割")
    print(f"📂 图片输入目录: {IMAGE_INPUT_DIR}")
    print(f"📂 视频生成目录: {VIDEO_INPUT_DIR}")
    print(f"📂 分割输出目录: {OUTPUT_VIDEO_DIR}")
    print(f"📦 每 {TARGET_FRAMES} 张图片合成一个视频并立即分割")
    print("-" * 60)

    while True:
        image_files = get_image_files(IMAGE_INPUT_DIR)
        count = len(image_files)

        if count < TARGET_FRAMES:
            print(f"⏳ 当前图片数量: {count}/{TARGET_FRAMES}，等待更多图片...")
            time.sleep(1)
            continue

        # 连续处理，直到剩余不足125张
        while len(get_image_files(IMAGE_INPUT_DIR)) >= TARGET_FRAMES:
            image_files = get_image_files(IMAGE_INPUT_DIR)
            if len(image_files) < TARGET_FRAMES:
                break

            selected = image_files[:TARGET_FRAMES]
            print(f"\n✅ 检测到 {len(image_files)} 张图片，使用前 {TARGET_FRAMES} 张制作视频")

            first_img = cv2.imread(selected[0])
            if first_img is None:
                print(f"❌ 无法读取第一张图片: {selected[0]}，跳过该批")
                move_to_processed(selected, processed_dir)
                continue

            h, w = first_img.shape[:2]
            print(f"📐 视频尺寸: {w}×{h}")

            video_filename = get_next_video_name(VIDEO_INPUT_DIR)
            video_path = os.path.join(VIDEO_INPUT_DIR, video_filename)
            print(f"🎬 合成视频: {video_filename}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
            if not video_writer.isOpened():
                print(f"❌ 无法创建视频文件: {video_path}，跳过该批")
                time.sleep(1)
                continue

            successful_frames = 0
            for idx, img_path in enumerate(selected, 1):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  ⚠️ 读取失败，跳过: {img_path}")
                    continue
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                video_writer.write(img)
                successful_frames += 1
                if idx % 25 == 0:
                    print(f"  已写入 {idx}/{TARGET_FRAMES} 帧")

            video_writer.release()
            print(f"✅ 视频合成完成: {video_path} (成功写入 {successful_frames} 帧)")

            # 立即分割
            success, _ = segment_single_video(video_path, CHECKPOINT_PATH, SAVE_TARGET_DIR)
            if success:
                print(f"✅ 视频 {video_filename} 分割成功")
            else:
                print(f"⚠️ 视频 {video_filename} 分割失败")

            # 移动已处理的图片
            move_to_processed(selected, processed_dir)
            print(f"📦 已将 {len(selected)} 张图片移动到 processed 文件夹")
            print("-" * 60)

        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 用户中断，程序退出")