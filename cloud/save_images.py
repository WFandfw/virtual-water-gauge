import paho.mqtt.client as mqtt
import base64
import os

# ---------- 配置 ----------
MQTT_BROKER = "broker-cn.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "water/cam/image"
SAVE_DIR = "/root/sam/input"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 自动检测已有最大编号，实现接续 ----------
image_counter = 0
existing_files = [f for f in os.listdir(SAVE_DIR)
                  if f.endswith('.jpg') and f[:-4].isdigit()]
if existing_files:
    max_num = max(int(f[:-4]) for f in existing_files)
    image_counter = max_num
    print(f"检测到已有 {len(existing_files)} 张图片，最大编号: {max_num:04d}.jpg，将从 {max_num+1:04d}.jpg 开始")
else:
    print("未检测到已有图片，从 0001.jpg 开始")

# ---------- MQTT 回调（新版 paho-mqtt 签名，消除弃用警告）----------
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"✅ 已连接 MQTT Broker，返回码: {reason_code}")
    client.subscribe(MQTT_TOPIC)
    print(f"📡 已订阅主题: {MQTT_TOPIC}")

def on_message(client, userdata, msg):
    global image_counter
    try:
        jpg_data = base64.b64decode(msg.payload.decode())
        image_counter += 1
        filename = f"{image_counter:04d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(jpg_data)
        print(f"💾 已保存: {filepath}  大小: {len(jpg_data)} 字节")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

# ---------- 启动客户端（使用 VERSION2 避免弃用警告）----------
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

print(f"🔗 正在连接 {MQTT_BROKER} ...")
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()