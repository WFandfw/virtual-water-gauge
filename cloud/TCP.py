import socket
import struct
import os
import time

import cv2
import numpy as np

SAVE_DIR = '/root/sam/input'
LISTEN_PORT = 6006          # 与自定义服务端口一致

# ==================== 相机内参 / 畸变系数（800x600 SVGA 标定） ====================
CAMERA_MATRIX = np.array([
    [920.356664, 0.0, 418.646752],
    [0.0,       920.038830, 298.595686],
    [0.0,       0.0,       1.0]
], dtype=np.float64)

DIST_COEFFS = np.array([
    -0.463282,
    0.310009,
    -0.001516,
    0.001346,
    0.0
], dtype=np.float64)

# 固定分辨率
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

# ---------- 预计算畸变矫正映射表（只计算一次，提升性能） ----------
MAP1, MAP2 = cv2.initUndistortRectifyMap(
    CAMERA_MATRIX,
    DIST_COEFFS,
    None,                    # 旋转矩阵
    CAMERA_MATRIX,           # 使用原内参，保持原图尺寸，不裁剪黑边
    (IMAGE_WIDTH, IMAGE_HEIGHT),
    cv2.CV_16SC2             # 16位定点格式，remap 速度快
)

# JPEG 编码质量
JPEG_QUALITY = 70


def undistort_image_fast(img_bgr):
    """
    使用预计算映射表进行快速畸变矫正。
    输入: img_bgr, 尺寸必须为 800x600
    输出: 矫正后的 BGR 图像，同尺寸
    """
    return cv2.remap(img_bgr, MAP1, MAP2, cv2.INTER_LINEAR)


def recv_exact(conn, n):
    """精确接收 n 字节数据"""
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def recv_line(conn):
    """读取一行以换行符结尾的文本（用于曝光信息包）。"""
    data = b''
    while True:
        c = conn.recv(1)
        if not c:
            return None
        if c == b'\n':
            return data.decode('utf-8', errors='ignore')
        data += c


def get_next_number(save_dir):
    """返回下一个可用的图片序号（从 1 开始，五位数字格式）"""
    max_num = 0
    if not os.path.exists(save_dir):
        return 1
    for filename in os.listdir(save_dir):
        if filename.lower().endswith('.jpg'):
            name = filename[:-4]  # 去掉 .jpg 扩展名
            if name.isdigit():    # 纯数字文件名，兼容五位编号
                num = int(name)
                if num > max_num:
                    max_num = num
    return max_num + 1


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', LISTEN_PORT))
    server.listen(1)
    print(f'Listening on 0.0.0.0:{LISTEN_PORT}, saving to {SAVE_DIR}')
    print(f'Undistortion map precomputed for {IMAGE_WIDTH}x{IMAGE_HEIGHT}')

    while True:
        print('Waiting for OpenMV connection...')
        conn, addr = server.accept()
        print('Connected by', addr)

        # 获取当前应使用的下一个图片序号
        next_number = get_next_number(SAVE_DIR)
        print(f'Next image number: {next_number:05d}')

        frame_count = 0
        last_print = time.time()

        try:
            while True:
                # 读取 4 字节长度头
                header = recv_exact(conn, 4)
                if not header:
                    print('Client disconnected')
                    break

                # 曝光信息包：魔数 0xFFFFFFFF 后跟一行文本（非图片）
                if header == b'\xff\xff\xff\xff':
                    info = recv_line(conn)
                    if info is None:
                        print('Failed to receive exposure info')
                        break
                    print(f'曝光信息: {info}')
                    continue

                size = struct.unpack('>I', header)[0]
                # 基本校验，防止异常数据
                if size <= 0 or size > 10 * 1024 * 1024:
                    print(f'Invalid size: {size}')
                    break

                img_data = recv_exact(conn, size)
                if not img_data:
                    print('Failed to receive image data')
                    break

                # ---------- 快速处理流水线 ----------
                # 1. JPEG 解码为 BGR numpy 数组
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    print('Failed to decode image, skipping frame')
                    continue

                # 2. 尺寸固定，如果意外不是 800x600，则调整到固定尺寸
                if img.shape[1] != IMAGE_WIDTH or img.shape[0] != IMAGE_HEIGHT:
                    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

                # 3. 畸变矫正（使用预计算映射，速度很快）
                img_undistorted = undistort_image_fast(img)

                # 4. 裁剪：800 宽下对应原 320 宽的 [108:186]（按 2.5 倍缩放 → [270:465]）
                img_cropped = img_undistorted[:, 270:465]   # 宽度变为 195，高度仍为 600

                # 5. 编码为 JPEG（质量 70）
                ret, buf = cv2.imencode(
                    '.jpg',
                    img_cropped,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if not ret:
                    print('Failed to encode image, skipping frame')
                    continue

                img_data_to_save = buf.tobytes()
                # --------------------------------------

                # 按五位数字命名：00001.jpg, 00002.jpg, ...
                filename = os.path.join(SAVE_DIR, f'{next_number:05d}.jpg')
                with open(filename, 'wb') as f:
                    f.write(img_data_to_save)

                print(f'Received, undistorted and cropped: {os.path.basename(filename)}')

                next_number += 1
                frame_count += 1

                # 每 10 秒打印一次状态汇总
                now = time.time()
                if now - last_print >= 10:
                    print(f'Received {frame_count} frames, last saved: {filename}')
                    last_print = now

        except Exception as e:
            print('Error:', e)
        finally:
            conn.close()
            print('Connection closed, waiting for next client...\n')


if __name__ == '__main__':
    main()