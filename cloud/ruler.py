#!/usr/bin/env python3
"""
ruler.py - 虚拟水尺叠加库

提供函数:
    ruler(input_path, output_path, csv_path, step_main=0.05, ...)
        对视频或单张图像叠加虚拟水尺，并保存输出。

依赖:
    opencv-python
    numpy
    pandas
"""

import os
import math
import numpy as np
import pandas as pd
import cv2


def _overlay_on_frame(frame, csv_path, step_main=0.05,
                      color=(0, 0, 255), text_color=(0, 255, 0),
                      axis_offset_right=3, tick_len_main=12,
                      tick_thickness=2, original_image_height=None):
    """
    在单张图像上叠加虚拟水尺（内部函数）。
    返回叠加水尺后的图像副本。

    参数:
        frame: numpy.ndarray (BGR)
        csv_path: str, lookup_table.csv 路径
        step_main: float, 主刻度间隔（米）
        color, text_color: tuple, BGR 颜色
        axis_offset_right: int, 水尺主轴距离右边缘像素数
        tick_len_main: int, 刻度线长度（像素）
        tick_thickness: int, 刻度线粗细
        original_image_height: int, 生成 lookup_table 时使用的图像高度
    """
    img = frame.copy()
    h_img, w_img = img.shape[:2]

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"查找表文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'Z' not in df.columns or 'v_star' not in df.columns:
        raise ValueError("CSV 中必须包含 'Z' 和 'v_star' 列")

    z_arr = df['Z'].values.astype(np.float64)
    vstar_arr = df['v_star'].values.astype(np.float64)

    # 如果提供了原始图像高度且与当前帧高度不同，则缩放 v_star
    if original_image_height is not None and original_image_height != h_img:
        scale = h_img / float(original_image_height)
        vstar_arr = vstar_arr * scale

    z_min = float(z_arr.min())
    z_max = float(z_arr.max())

    # 动态计算主刻度间隔：根据图片内实际显示的高程范围自动选步长，使刻度标注不超过 max_labels 个
    max_labels = 10
    if vstar_arr[-1] < vstar_arr[0]:
        vs = vstar_arr[::-1].copy()
        zs = z_arr[::-1].copy()
    else:
        vs = vstar_arr
        zs = z_arr
    z_at_top = float(np.interp(0.0, vs, zs))
    z_at_bottom = float(np.interp(float(h_img - 1), vs, zs))
    display_min = min(z_at_top, z_at_bottom)
    display_max = max(z_at_top, z_at_bottom)
    span = display_max - display_min
    if span > 0:
        nice_steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        for s in nice_steps:
            if span / s <= max_labels:
                step_main = s
                break
        else:
            step_main = span / max_labels

    # 水尺主轴位置（图像右侧）
    x_axis = w_img - 1 - axis_offset_right
    x_axis = max(0, x_axis)
    tick_len = max(4, min(tick_len_main, max(1, x_axis)))

    # 绘制主轴
    cv2.line(img, (x_axis, 0), (x_axis, h_img - 1), color, 1)

    # 生成对齐到 0.00m 的刻度序列
    def build_aligned_z_list(z_min, z_max, step, start_base=0.0):
        if step <= 0:
            raise ValueError("step 必须 > 0")
        k0 = math.ceil((z_min - start_base) / step)
        first = start_base + k0 * step
        z_list = []
        z = first
        while z <= z_max + 1e-12:
            z_list.append(round(z, 6))
            z += step
        return z_list

    z_list = build_aligned_z_list(z_min, z_max, step_main)

    # 绘制刻度线和标注
    for z in z_list:
        v = np.interp(z, z_arr, vstar_arr)
        v = int(round(v))
        if 0 <= v < h_img:
            x0 = max(0, x_axis - tick_len)
            cv2.line(img, (x0, v), (x_axis, v), color, tick_thickness)

            text_x = max(0, x0 - 42)
            text_y = min(max(v + 5, 15), h_img - 5)
            cv2.putText(img, f"{z:.2f}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    return img


def ruler(input_path, output_path, csv_path, step_main=0.05,
          color=(0, 0, 255), text_color=(0, 255, 0),
          axis_offset_right=3, tick_len_main=12,
          tick_thickness=2, original_image_height=None,
          fps=None):
    """
    对视频或图像叠加虚拟水尺并输出。

    参数:
        input_path: str
            输入文件路径（视频或图像）。
        output_path: str
            输出文件路径。
        csv_path: str
            lookup_table.csv 路径，必须包含 'Z' 和 'v_star' 列。
        step_main: float
            主刻度间隔（米）的初始/兜底值，默认 0.05。
            实际会按图片内显示的高程范围动态调整，使刻度标注不超过 10 个。
        color, text_color: tuple
            刻度线颜色和文字颜色（BGR 格式）。
        axis_offset_right: int
            水尺主轴距离右边缘像素数。
        tick_len_main: int
            刻度线长度（像素）。
        tick_thickness: int
            刻度线粗细。
        original_image_height: int, optional
            生成 lookup_table 时使用的图像高度。
            如果当前图像/视频帧高度不同，将自动缩放 v_star。
            不提供则假设高度一致。
        fps: float, optional
            输出视频帧率，仅当输入为视频时有效。
            若为 None，则保持输入视频帧率。

    返回:
        bool: 是否成功。
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 判断是视频还是图像：尝试用 VideoCapture 打开，若成功则为视频
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened():
        # ================= 视频处理 =================
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # 优先使用 mp4v 编码器（通用兼容，避免硬件编码器报错）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            # 备用 avc1（如果环境支持）
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("无法创建视频写入器")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_with_gauge = _overlay_on_frame(
                frame, csv_path,
                step_main=step_main,
                color=color,
                text_color=text_color,
                axis_offset_right=axis_offset_right,
                tick_len_main=tick_len_main,
                tick_thickness=tick_thickness,
                original_image_height=original_image_height
            )
            writer.write(frame_with_gauge)

        cap.release()
        writer.release()
        print(f"[ruler] 视频水尺叠加完成: {output_path}")
        return True

    else:
        # ================= 图像处理 =================
        # 使用 imdecode 以支持中文路径
        img_np = np.fromfile(input_path, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取输入文件（既不是视频也不是图像）: {input_path}")

        result = _overlay_on_frame(
            img, csv_path,
            step_main=step_main,
            color=color,
            text_color=text_color,
            axis_offset_right=axis_offset_right,
            tick_len_main=tick_len_main,
            tick_thickness=tick_thickness,
            original_image_height=original_image_height
        )

        # 保存图像（支持中文路径）
        ext = os.path.splitext(output_path)[1]
        if ext == "":
            ext = ".jpg"
            output_path += ext
        success, img_encode = cv2.imencode(ext, result)
        if not success:
            raise ValueError("图像编码失败")
        img_encode.tofile(output_path)
        print(f"[ruler] 图像水尺叠加完成: {output_path}")
        return True


def pixel_to_real_waterline(pixel_y, csv_path, image_height=None, original_image_height=None):
    """
    通过虚拟水尺查找表 (lookup_table.csv 的 Z / v_star 列) 将像素水位线高度
    映射为真实水位线 Z（米），线性插值。

    参数:
        pixel_y: float/int, 像素水位线高度（v_star 坐标）
        csv_path: str, lookup_table.csv 路径，需包含 'Z' 和 'v_star' 列
        image_height: int, 当前图像/视频帧高度
        original_image_height: int, 生成查找表时使用的图像高度；两者不同会自动缩放
    返回:
        float: 真实水位线（米）；pixel_y 为 None 时返回 None
    """
    print(f"[ruler] pixel_to_real_waterline 收到 pixel_y={pixel_y}, csv_path={csv_path!r}")
    if pixel_y is None:
        return None
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"查找表文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    print("已经获得csv文件")
    print(f"[ruler] CSV 列: {list(df.columns)}")
    if 'Z' not in df.columns or 'v_star' not in df.columns:
        raise ValueError("CSV 中必须包含 'Z' 和 'v_star' 列")

    z_arr = df['Z'].values.astype(np.float64)
    vstar_arr = df['v_star'].values.astype(np.float64)
    print(f"[ruler] Z = {z_arr}, v_star = {vstar_arr}")

    # 当前图像高度与生成查找表时不一致，则按比例缩放 v_star（与 _overlay_on_frame 保持一致）
    if original_image_height is not None and image_height is not None \
            and original_image_height != image_height:
        scale = image_height / float(original_image_height)
        vstar_arr = vstar_arr * scale

    # np.interp 要求 xp 单调递增；若 v_star 递减则反向
    if vstar_arr[-1] < vstar_arr[0]:
        vstar_arr = vstar_arr[::-1].copy()
        z_arr = z_arr[::-1].copy()

    result = float(np.interp(float(pixel_y), vstar_arr, z_arr))
    print(f"[ruler] 映射结果(真实水位线): {result}")
    return result


def get_ruler_range(csv_path, pixel_top=15, pixel_bottom=200):
    """
    根据图片实际能体现的水位观测像素区间，映射出虚拟水尺的有效高程范围。
    图片高 240px，只有像素区间 [pixel_top, pixel_bottom]（默认 15~200px）能体现水位变化，
    通过查找表 (Z ↔ v_star) 插值得到该区间对应的实际高程范围。
    返回 (z_low, z_high)；文件不存在或列缺失返回 (None, None)。
    """
    if not os.path.isfile(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path)
        if 'Z' not in df.columns or 'v_star' not in df.columns:
            return None, None
        z_arr = df['Z'].values.astype(np.float64)
        vstar_arr = df['v_star'].values.astype(np.float64)
        # 保证 v_star 单调递增（与 pixel_to_real_waterline 一致）
        if vstar_arr[-1] < vstar_arr[0]:
            vstar_arr = vstar_arr[::-1].copy()
            z_arr = z_arr[::-1].copy()
        z_top = float(np.interp(float(pixel_top), vstar_arr, z_arr))
        z_bottom = float(np.interp(float(pixel_bottom), vstar_arr, z_arr))
        return min(z_top, z_bottom), max(z_top, z_bottom)
    except Exception as e:
        print(f"[ruler] 读取查找表范围失败: {e}")
        return None, None


if __name__ == "__main__":
    # 简单测试示例（请根据实际路径修改）
    import sys
    if len(sys.argv) >= 4:
        ruler(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("用法: python ruler.py <输入视频/图像> <输出路径> <lookup_table.csv>")