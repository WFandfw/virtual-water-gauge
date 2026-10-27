#!/usr/bin/env python3
# live_stream.py
"""
输入目录新图片实时推流（WebSocket /ws/live，纯标准库 + cv2/numpy）

监控 input 目录：出现新图片时读取原图（非 JPEG 自动转 JPEG），
通过 WebSocket 二进制帧推送给网页前端，左侧「实时画面」实时显示。
图片本身仍保留在 input 中，供分割流程使用。
"""

import os
import re
import time
import base64
import hashlib
import struct
import threading

import cv2
import numpy as np

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def ws_accept_key(key):
    return base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()


def _send_frame(conn, payload, opcode=0x2):
    header = bytearray()
    header.append(0x80 | opcode)
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header += struct.pack(">H", n)
    else:
        header.append(127)
        header += struct.pack(">Q", n)
    conn.sendall(bytes(header) + payload)


def _read_exact(conn, n):
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _load_jpeg_bytes(path):
    """读取图片，返回浏览器可解码的 JPEG 字节。"""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except OSError:
        return None
    if not raw:
        return None
    if os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg'):
        return raw
    try:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes() if ok else None


class LiveStreamManager:
    def __init__(self, watch_dir, poll_interval=0.02):
        self.watch_dir = watch_dir
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._clients = {}
        self._sent = set()
        self._sizes = {}
        self._last_data = None
        self._stop = threading.Event()
        self._thread = None

    def _list_images(self):
        if not os.path.isdir(self.watch_dir):
            return []
        files = [f for f in os.listdir(self.watch_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
        files.sort(key=_natural_sort_key)
        return files

    def start(self):
        self._sent = set(self._list_images())
        self._sizes = {}
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                files = self._list_images()
                for f in files:
                    path = os.path.join(self.watch_dir, f)
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        continue
                    prev = self._sizes.get(f)
                    self._sizes[f] = size
                    # 文件大小连续两次轮询一致且非空，才视为写入完成，避免读到半张图
                    if f not in self._sent and prev == size and size > 0:
                        self._broadcast_file(path, f)
                        self._sent.add(f)
                current = set(files)
                self._sent &= current
                self._sizes = {k: v for k, v in self._sizes.items() if k in current}
            except Exception as e:
                print(f"[live_stream] 监控异常: {e}")
            time.sleep(self.poll_interval)

    def _broadcast_file(self, path, name):
        data = _load_jpeg_bytes(path)
        if not data:
            return
        print(f"[live_stream] 推送新图片: {name} ({len(data)} bytes)")
        self.broadcast(data)

    def _latest_image_bytes(self):
        files = self._list_images()
        if not files:
            return None
        return _load_jpeg_bytes(os.path.join(self.watch_dir, files[-1]))

    def register(self, conn):
        with self._lock:
            self._clients[conn] = threading.Lock()
            last = self._last_data
        data = self._latest_image_bytes() or last
        if data:
            try:
                with self._clients[conn]:
                    _send_frame(conn, data, 0x2)
            except Exception:
                self.unregister(conn)

    def unregister(self, conn):
        with self._lock:
            self._clients.pop(conn, None)

    def broadcast(self, data):
        with self._lock:
            self._last_data = data
            clients = list(self._clients.items())
        for conn, lock in clients:
            try:
                with lock:
                    _send_frame(conn, data, 0x2)
            except Exception:
                self.unregister(conn)


def handle_websocket(handler, manager):
    """在 HTTP handler 的 do_GET 中调用：完成 WebSocket 握手并进入读取循环。"""
    key = handler.headers.get("Sec-WebSocket-Key")
    if not key:
        handler.send_error(400)
        return

    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", ws_accept_key(key))
    handler.end_headers()

    conn = handler.connection
    manager.register(conn)
    try:
        while True:
            head = _read_exact(conn, 2)
            if head is None:
                break
            opcode = head[0] & 0x0F
            length = head[1] & 0x7F
            masked = (head[1] & 0x80) != 0
            if length == 126:
                ext = _read_exact(conn, 2)
                if ext is None:
                    break
                length = struct.unpack(">H", ext)[0]
            elif length == 127:
                ext = _read_exact(conn, 8)
                if ext is None:
                    break
                length = struct.unpack(">Q", ext)[0]
            mask_key = b''
            if masked:
                mask_key = _read_exact(conn, 4)
                if mask_key is None:
                    break
            payload = b''
            if length:
                payload = _read_exact(conn, length)
                if payload is None:
                    break
            if opcode == 0x8:  # close
                try:
                    _send_frame(conn, b'', 0x8)
                except Exception:
                    pass
                break
            if opcode == 0x9:  # ping -> pong
                try:
                    _send_frame(conn, payload, 0xA)
                except Exception:
                    break
    finally:
        manager.unregister(conn)
        try:
            conn.close()
        except Exception:
            pass
