import sensor, image, time, os, gc
import network, usocket, machine
from machine import Pin, UART

# ==================== 板载 LED 初始化 ====================
try:
    led_red   = Pin("LED_RED",   Pin.OUT, value=1)   # 高电平熄灭
    led_green = Pin("LED_GREEN", Pin.OUT, value=1)
    led_blue  = Pin("LED_BLUE",  Pin.OUT, value=1)
except:
    led_red = led_green = led_blue = None

# ==================== 串口初始化 ====================
# 根据硬件选择串口号，OpenMV Cam 常见 UART3：TX=P1，RX=P0
# 若使用其他串口，请修改参数
try:
    uart = UART(3, 115200, timeout=1000)
    print("UART3 初始化成功，波特率 115200")
except Exception as e:
    print("UART3 初始化失败:", e)
    uart = None

# ==================== 网络配置 ====================
WIFI_SSID = "WF"
WIFI_PASS = "123456789"

# 安卓热点静态 IP（DHCP 失败时兜底）
STATIC_IP = ('192.168.43.100', '255.255.255.0', '192.168.43.1', '192.168.43.1')

TCP_HOST = "connect.westc.seetacloud.com"   # AutoDL 公网域名
TCP_PORT = 47509                            # 对应容器内 6006 端口的公网端口

# ==================== 图像上传参数 ====================
UPLOAD_IMAGE_MS = 40         # 40ms -> 25fps
JPEG_QUALITY    = 70

# ==================== RTC 辅助函数 ====================
rtc = machine.RTC()
MON = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
       "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

def rtc_sync():
    print("RTC 对时...")
    try:
        addr = usocket.getaddrinfo("api.thingspeak.com", 80)[0][-1]
        s = usocket.socket()
        s.settimeout(5)
        s.connect(addr)
        s.write("HEAD / HTTP/1.1\r\nHost: api.thingspeak.com\r\nConnection: close\r\n\r\n")
        resp = s.read(512)
        s.close()
        if isinstance(resp, bytes):
            resp = resp.decode()
        for line in resp.split("\r\n"):
            if line.lower().startswith("date:"):
                p = line[5:].strip().replace(",", "").split()
                d, m, y = int(p[1]), MON[p[2]], int(p[3])
                hh, mm, ss = [int(x) for x in p[4].split(":")]
                hh = (hh + 8) % 24
                rtc.datetime((y, m, d, 1, hh, mm, ss, 0))
                print("RTC 同步成功: %04d-%02d-%02d %02d:%02d:%02d" % (y, m, d, hh, mm, ss))
                return True
    except Exception as e:
        print("RTC 同步失败:", e)
    return False

def rtc_short():
    try:
        dt = rtc.datetime()
        return "%02d:%02d:%02d" % (dt[4], dt[5], dt[6])
    except:
        return "--:--:--"

# ==================== Wi-Fi 连接 ====================
wlan = None

def is_wifi_ok():
    if wlan.isconnected():
        return True
    try:
        ip = wlan.ifconfig()[0]
        if ip and ip != "0.0.0.0":
            return True
    except:
        pass
    return False

def wifi_connect():
    global wlan
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if is_wifi_ok():
        print("WiFi 已连接:", wlan.ifconfig()[0])
        return True

    print("正在扫描 WiFi...")
    nets = wlan.scan()
    found = False
    for ssid, bssid, channel, rssi, security, hidden in nets:
        ssid_str = ssid.decode() if isinstance(ssid, bytes) else ssid
        print("  发现网络: {} ch={} rssi={}".format(ssid_str, channel, rssi))
        if ssid_str == WIFI_SSID:
            found = True

    if not found:
        print("未扫描到目标 WiFi:", WIFI_SSID)
        return False

    print("正在连接 WiFi:", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASS)
    for i in range(30):          # 15 秒
        if is_wifi_ok():
            print("WiFi 连接成功, IP:", wlan.ifconfig()[0])
            return True
        if i % 4 == 0:
            print("  等待连接... status={}".format(wlan.status()))
        time.sleep_ms(500)

    print("DHCP 连接失败，尝试静态 IP...")
    try:
        wlan.ifconfig(STATIC_IP)
        print("静态 IP 已设置:", wlan.ifconfig()[0])
        return True
    except Exception as e:
        print("静态 IP 设置失败:", e)

    print("WiFi 连接最终失败")
    return False

# ==================== TCP 连接管理 ====================
tcp_socket = None
tcp_ok = False
TCP_IP = None    # 缓存解析到的 IP，避免每次 DNS

tcp_reconnect_active = False
tcp_reconnect_attempts = 0
tcp_reconnect_last_try = 0
tcp_failed_permanent = False

TCP_RECONNECT_INTERVAL_MS = 5000
TCP_RECONNECT_MAX_ATTEMPTS = 3

def tcp_connect():
    global tcp_socket, tcp_ok, TCP_IP
    try:
        if tcp_socket is not None:
            try:
                tcp_socket.close()
            except:
                pass
        print("正在连接 TCP 服务器 {}:{} ...".format(TCP_HOST, TCP_PORT))
        if TCP_IP is None:
            addr_info = usocket.getaddrinfo(TCP_HOST, TCP_PORT, 0, usocket.SOCK_STREAM)
            if not addr_info:
                raise OSError("DNS resolve failed")
            TCP_IP = addr_info[0][-1][0]
            print("解析到 IP:", TCP_IP)
        tcp_socket = usocket.socket(usocket.AF_INET, usocket.SOCK_STREAM)
        tcp_socket.settimeout(10)
        tcp_socket.connect((TCP_IP, TCP_PORT))
        tcp_socket.settimeout(None)
        tcp_ok = True
        print("TCP 连接成功:", TCP_IP)
        return True
    except Exception as e:
        print("TCP 连接失败:", e)
        tcp_ok = False
        return False

def send_all(sock, data):
    data = bytes(data)
    while data:
        sent = sock.send(data)
        if sent <= 0:
            raise OSError("send failed")
        data = data[sent:]

def send_image(jpeg_data):
    global tcp_socket
    try:
        if hasattr(jpeg_data, 'bytearray'):
            jpeg_data = jpeg_data.bytearray()
        elif hasattr(jpeg_data, 'to_bytes'):
            jpeg_data = jpeg_data.to_bytes()

        if not isinstance(jpeg_data, (bytes, bytearray)):
            return False

        jpeg_data = bytes(jpeg_data)
        size = len(jpeg_data)
        header = bytes((
            (size >> 24) & 0xff,
            (size >> 16) & 0xff,
            (size >> 8) & 0xff,
            size & 0xff
        ))
        packet = header + jpeg_data
        send_all(tcp_socket, packet)
        return True
    except Exception as e:
        print("图片发送失败:", e)
        try:
            tcp_socket.close()
        except:
            pass
        tcp_socket = None
        tcp_ok = False
        return False

# ==================== 网络状态自愈 ====================
def check_wifi():
    global wifi_ok, wlan
    if wlan is None:
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
    if is_wifi_ok():
        if not wifi_ok:
            print("WiFi 恢复连接")
            wifi_ok = True
        return True
    if wifi_ok:
        print("WiFi 已断开，尝试重连...")
        wifi_ok = False
    return wifi_connect()

def start_tcp_reconnect():
    global tcp_reconnect_active, tcp_reconnect_attempts, tcp_reconnect_last_try, tcp_failed_permanent
    if not tcp_reconnect_active:
        print("TCP 连接断开，开始自动重连（最多3次，间隔5秒）")
    tcp_reconnect_active = True
    tcp_reconnect_attempts = 0
    tcp_reconnect_last_try = time.ticks_ms() - TCP_RECONNECT_INTERVAL_MS
    tcp_failed_permanent = False

def process_tcp_reconnect():
    global tcp_reconnect_active, tcp_reconnect_attempts, tcp_reconnect_last_try, tcp_failed_permanent, tcp_ok

    if not tcp_reconnect_active:
        return tcp_ok

    now = time.ticks_ms()
    if time.ticks_diff(now, tcp_reconnect_last_try) >= TCP_RECONNECT_INTERVAL_MS:
        tcp_reconnect_last_try = now
        tcp_reconnect_attempts += 1

        print("TCP 重连尝试 {}/{}".format(tcp_reconnect_attempts, TCP_RECONNECT_MAX_ATTEMPTS))
        if tcp_connect():
            tcp_reconnect_active = False
            tcp_failed_permanent = False
            print("TCP 重连成功")
            return True
        else:
            if tcp_reconnect_attempts >= TCP_RECONNECT_MAX_ATTEMPTS:
                print("TCP 重连失败（3次尝试均失败），红灯常亮")
                tcp_reconnect_active = False
                tcp_failed_permanent = True
                return False
    return False

def check_tcp():
    global tcp_ok, tcp_failed_permanent

    if tcp_ok and tcp_socket is not None:
        return True

    if tcp_failed_permanent:
        print("周期性 TCP 检查：重新尝试连接（之前已失败）")
        tcp_failed_permanent = False
        start_tcp_reconnect()
        return False

    if not tcp_reconnect_active:
        start_tcp_reconnect()

    return process_tcp_reconnect()

# ==================== 摄像头初始化 ====================
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)       # 320x240（原注释有误）
sensor.skip_frames(time=2000)

# ==================== 启动网络 ====================
wifi_ok = wifi_connect()
tcp_ok = False
tcp_socket = None
tcp_reconnect_active = False
tcp_reconnect_attempts = 0
tcp_reconnect_last_try = 0
tcp_failed_permanent = False

if wifi_ok:
    tcp_ok = tcp_connect()
    rtc_sync()
else:
    print("WiFi 连接失败，程序将在主循环中尝试重连")

# ==================== 主循环变量 ====================
clock = time.clock()
last_image_ticks = 0
last_wifi_check  = 0
last_tcp_check   = 0
image_count = 0
gc_counter = 0

last_led_toggle = 0
led_red_state = False
led_blue_state = False

# ==================== LED 状态更新函数 ====================
def update_leds_state():
    global last_led_toggle, led_red_state, led_blue_state
    now = time.ticks_ms()

    if time.ticks_diff(now, last_led_toggle) >= 500:
        last_led_toggle = now
        led_red_state = not led_red_state
        led_blue_state = not led_blue_state

    if not wifi_ok:
        if led_red:
            if led_red_state:
                led_red.off()
            else:
                led_red.on()
        if led_green: led_green.on()
        if led_blue:  led_blue.on()
    elif tcp_failed_permanent:
        if led_red: led_red.off()
        if led_green: led_green.on()
        if led_blue:  led_blue.on()
    elif tcp_reconnect_active:
        if led_red: led_red.on()
        if led_green: led_green.on()
        if led_blue:
            if led_blue_state:
                led_blue.off()
            else:
                led_blue.on()
    elif tcp_ok:
        if led_red: led_red.on()
        if led_green: led_green.off()
        if led_blue:  led_blue.on()
    else:
        if led_red: led_red.off()
        if led_green: led_green.on()
        if led_blue:  led_blue.on()

# ==================== 主循环 ====================
print("=== 图像传输启动（无图像叠加，25fps，串口输出图片序号） ===")
while True:
    clock.tick()
    img = sensor.snapshot()
    t_now = time.ticks_ms()

    if time.ticks_diff(t_now, last_wifi_check) >= 30000:
        wifi_ok = check_wifi()
        last_wifi_check = t_now
        if wifi_ok and not tcp_ok and not tcp_reconnect_active and not tcp_failed_permanent:
            print("WiFi 重连成功，尝试重新建立 TCP")
            tcp_ok = tcp_connect()

    if time.ticks_diff(t_now, last_tcp_check) >= 60000:
        tcp_ok = check_tcp()
        last_tcp_check = t_now

    if tcp_reconnect_active:
        tcp_ok = process_tcp_reconnect()

    if time.ticks_diff(t_now, last_image_ticks) >= UPLOAD_IMAGE_MS:
        image_count += 1
        last_image_ticks = t_now

        if tcp_ok and not tcp_reconnect_active:
            try:
                if gc_counter >= 100:
                    gc.collect()
                    gc_counter = 0
                else:
                    gc_counter += 1

                jpeg = img.compress(quality=JPEG_QUALITY)
                if jpeg is not None:
                    if send_image(jpeg):
                        # ----- 新增：串口输出成功发送的图片序号 -----
                        if uart is not None:
                            msg = "Image %05d sent\r\n" % image_count
                            uart.write(msg.encode())
                        # -----------------------------------------
                    else:
                        print("图像发送失败，准备重连 TCP")
                        tcp_ok = False
                        start_tcp_reconnect()
            except Exception as e:
                print("图像处理异常:", e)
                tcp_ok = False
                start_tcp_reconnect()

    update_leds_state()

    if image_count % 200 == 0:
        gc.collect()
