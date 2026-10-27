#!/usr/bin/env python3
"""
易微联（eWeLink）OAuth 2.0 继电器控制模块（纯标准库，无第三方依赖）

流程：
  1. 生成授权页 URL -> 用户在浏览器登录授权 -> 回调拿 code
  2. code 换 access token（30天）+ refresh token（60天），缓存到本地
  3. 控制 3 路继电器（outlet 0/1/2），token 过期自动用 refresh token 续期

用法：
  python ewelink_control.py auth        # 生成授权页URL，登录后输入code，自动换token并缓存
  python ewelink_control.py status      # 查设备状态
  python ewelink_control.py on 0        # 开第1路
  python ewelink_control.py off 1       # 关第2路
"""

import os
import sys
import json
import time
import hmac
import hashlib
import base64
import random
import string
import urllib.request
import urllib.parse

def _load_dotenv(path):
    """若存在 .env，把 KEY=VALUE 载入环境变量（不覆盖已有的）。仅支持简单格式。"""
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print(f"[ewelink] 读取 .env 失败: {e}")


# 与本模块同目录的 .env 自动加载（也便于 CONTROL_PORT 等在 import 时生效）
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ================= 配置（凭据一律通过环境变量注入，不硬编码） =================
APP_ID = os.environ.get("EWELINK_APP_ID", "")
APP_SECRET = os.environ.get("EWELINK_APP_SECRET", "")
DEVICE_ID = os.environ.get("EWELINK_DEVICE_ID", "")
REDIRECT_URL = os.environ.get("EWELINK_REDIRECT_URL", "")

API_HOST = "https://cn-apia.coolkit.cn"
AUTH_URL = "https://c2ccdn.coolkit.cc/oauth/index.html"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ewelink_token.json")


def _require_config():
    """校验必需配置项，缺失则给出明确提示。"""
    required = {
        "EWELINK_APP_ID": APP_ID,
        "EWELINK_APP_SECRET": APP_SECRET,
        "EWELINK_DEVICE_ID": DEVICE_ID,
        "EWELINK_REDIRECT_URL": REDIRECT_URL,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            "缺少配置: " + ", ".join(missing) +
            "。请通过环境变量注入（参考 .env.example，例如 export EWELINK_APP_SECRET=xxx）"
        )


def make_sign(secret, message):
    """HMAC-SHA256 签名并 Base64 编码"""
    return base64.b64encode(
        hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()
    ).decode()


def _request(method, path, query=None, body=None, access_token=None):
    url = API_HOST + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    headers = {"X-CK-Appid": APP_ID, "Content-Type": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        if access_token:
            headers["Authorization"] = "Bearer " + access_token
        else:
            headers["Authorization"] = "Sign " + make_sign(APP_SECRET, json.dumps(body))
    elif access_token:
        headers["Authorization"] = "Bearer " + access_token

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_token():
    if os.path.isfile(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_token(token):
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(token, f, ensure_ascii=False, indent=2)


def gen_auth_url():
    """生成授权页 URL"""
    seq = str(int(time.time() * 1000))
    nonce = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    state = "stat" + nonce
    authorization = make_sign(APP_SECRET, f"{APP_ID}_{seq}")
    params = {
        "clientId": APP_ID,
        "seq": seq,
        "authorization": authorization,
        "redirectUrl": REDIRECT_URL,
        "grantType": "authorization_code",
        "state": state,
        "nonce": nonce,
        "showQRCode": "false",
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code(code):
    """用授权码 code 换 token"""
    body = {"code": code, "redirectUrl": REDIRECT_URL, "grantType": "authorization_code"}
    return _request("POST", "/v2/user/oauth/token", body=body)


def refresh_token(token):
    """用 refresh token 续期"""
    body = {"rt": token.get("refreshToken")}
    return _request("POST", "/v2/user/refresh", body=body, access_token=token.get("accessToken"))


def get_access_token():
    """获取有效 access token，必要时自动续期"""
    _require_config()
    token = _load_token()
    if not token or not token.get("accessToken"):
        raise RuntimeError("尚未授权，请先运行: python ewelink_control.py auth")
    # 判断是否过期（atExpiredTime 毫秒时间戳，提前 1 天续期）
    expired_at = token.get("atExpiredTime", 0)
    if time.time() * 1000 > expired_at - 24 * 3600 * 1000:
        print("access token 即将/已经过期，正在用 refresh token 续期...")
        resp = refresh_token(token)
        data = resp.get("data", {})
        if resp.get("error") == 0 and data.get("at"):
            token["accessToken"] = data["at"]
            token["atExpiredTime"] = time.time() * 1000 + 30 * 24 * 3600 * 1000
            if data.get("rt"):
                token["refreshToken"] = data["rt"]
            _save_token(token)
            print("续期成功")
        else:
            print("续期失败，请重新授权:", json.dumps(resp, ensure_ascii=False))
            raise RuntimeError("refresh token 失效，请重新运行 auth")
    return token["accessToken"]


def get_device_status(access_token):
    return _request("GET", "/v2/device/thing/status",
                    query={"type": 1, "id": DEVICE_ID}, access_token=access_token)


def set_switch(access_token, outlet, on):
    body = {
        "type": 1,
        "id": DEVICE_ID,
        "params": {"switches": [{"switch": "on" if on else "off", "outlet": outlet}]},
    }
    return _request("POST", "/v2/device/thing/status", body=body, access_token=access_token)


def do_auth():
    """引导授权：打印授权页URL -> 用户登录 -> 输入code -> 换token并缓存"""
    _require_config()
    url = gen_auth_url()
    print("=" * 60)
    print("请在浏览器打开下面的授权页 URL，登录易微联账号并授权：")
    print()
    print(url)
    print()
    print("授权成功后，浏览器会跳转到:")
    print(f"  {REDIRECT_URL}?code=xxxx&region=cn&state=xxxx")
    print("请把跳转后 URL 里的 code 值复制下来（xxxx 那段）。")
    print("=" * 60)
    code = input("粘贴 code: ").strip()
    if not code:
        print("未输入 code，退出")
        return
    resp = exchange_code(code)
    print("换取 token 返回：")
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    data = resp.get("data", {})
    if resp.get("error") == 0 and data.get("accessToken"):
        _save_token(data)
        print("✅ 授权成功，token 已缓存")
    else:
        print("❌ 换取 token 失败，请检查 code 是否正确（30秒内有效）")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]

    if cmd == "auth":
        do_auth()
        return

    at = get_access_token()
    if cmd == "status":
        r = get_device_status(at)
        print("设备状态：")
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif cmd in ("on", "off"):
        outlet = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        r = set_switch(at, outlet, cmd == "on")
        print("控制结果：")
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print("未知命令:", cmd)


if __name__ == "__main__":
    main()
