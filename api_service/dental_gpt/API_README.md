# DentalGPT REST API 使用文档

## 📋 目录
- [简介](#简介)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [启动服务](#启动服务)
- [API 端点说明](#api-端点说明)
- [客户端调用](#客户端调用)
- [常见问题](#常见问题)

---

## 简介

DentalGPT REST API 提供了一个基于 FastAPI 的 Web 服务，可以在局域网中调用 DentalGPT 模型进行牙科图像分析。

**主要特性:**
- ✅ 支持图像文件上传
- ✅ 支持 Base64 编码图像
- ✅ 自动 GPU 加速（如果可用）
- ✅ 支持局域网访问
- ✅ 完整的 API 文档（Swagger UI）
- ✅ 健康检查端点

---

## 环境要求

### 服务器端
- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.9+
- **GPU**: NVIDIA GPU with CUDA support (推荐)
- **显存**: ≥16GB
- **内存**: ≥32GB

### 客户端
- **Python**: 3.7+
- **网络**: 能访问服务器的局域网

---

## 安装步骤

### 1. 安装依赖

在服务器上安装所需的 Python 包:

```bash
cd /mnt/hdd/sdc/zlyu/dentalGPT
pip install -r requirements.txt
```

如果你已经有 transformers、torch 等包，可以只安装 FastAPI 相关依赖:

```bash
pip install fastapi uvicorn[standard] python-multipart
```

### 2. 检查模型路径

确保 `api_server.py` 中的模型路径正确:

```python
LOCAL_MODEL_PATH = "/mnt/hdd/sdb/zlyu/.cache/huggingface/hub/models--Eric3200--DentalGPT-7B-1026/snapshots/dce584f3ec6cf929cc37f6156cc41c8ebc8178a9"
```

---

## 启动服务

### 方法 1: 使用启动脚本（推荐）

**注意**: 启动脚本会自动激活 `zl_dentalGPT` conda 环境

```bash
cd /mnt/hdd/sdc/zlyu/dentalGPT
./start_server.sh
```

### 方法 2: 直接运行 Python

```bash
# 先激活 conda 环境
conda activate zl_dentalGPT

cd /mnt/hdd/sdc/zlyu/dentalGPT
python api_server.py
```

### 方法 3: 使用 uvicorn（高级）

```bash
# 先激活 conda 环境
conda activate zl_dentalGPT

# 基础启动
uvicorn api_server:app --host 0.0.0.0 --port 8566

# 生产环境（多进程）
uvicorn api_server:app --host 0.0.0.0 --port 8566 --workers 1

# 开发模式（自动重载）
uvicorn api_server:app --host 0.0.0.0 --port 8566 --reload
```

### 启动成功标志

当你看到以下输出时，说明服务已成功启动:

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
2024-xx-xx xx:xx:xx - __main__ - INFO - 启动 DentalGPT API 服务...
2024-xx-xx xx:xx:xx - __main__ - INFO - 正在加载 processor...
2024-xx-xx xx:xx:xx - __main__ - INFO - Processor 加载成功
2024-xx-xx xx:xx:xx - __main__ - INFO - 正在加载 DentalGPT 模型...
2024-xx-xx xx:xx:xx - __main__ - INFO - 模型加载成功
2024-xx-xx xx:xx:xx - __main__ - INFO - 服务启动完成
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 查看本机 IP

```bash
# 查看本机局域网 IP
hostname -I

# 或者
ip addr show
```

记下你的局域网 IP (例如: `192.168.1.100`)

---

## API 端点说明

### 1. 根路径 - GET `/`

返回 API 基本信息

**请求:**
```bash
curl http://192.168.1.100:8566/
```

**响应:**
```json
{
  "service": "DentalGPT API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "analyze": "/analyze (POST)",
    "analyze_base64": "/analyze/base64 (POST)"
  }
}
```

### 2. 健康检查 - GET `/health`

检查服务状态和模型加载情况

**请求:**
```bash
curl http://192.168.1.100:8566/health
```

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2024-12-11T10:30:00.123456"
}
```

### 3. 图像分析（文件上传）- POST `/analyze`

上传图像文件进行分析

**参数:**
- `image` (必需): 图像文件
- `question` (可选): 要问的问题，默认为 "请分析图像中有问题的牙齿,并给出有问题的牙齿所在位置,牙齿类型"
- `max_new_tokens` (可选): 最大生成 token 数，默认 4096

**请求示例 (curl):**
```bash
curl -X POST http://192.168.1.100:8566/analyze \
  -F "image=@/path/to/dental_image.jpg" \
  -F "question=请按照 FDI 分区格式，介绍 1 区的牙齿情况"
```

**请求示例 (Python):**
```python
import requests

files = {'image': open('dental_image.jpg', 'rb')}
data = {'question': '请分析图像中有问题的牙齿'}

response = requests.post(
    'http://192.168.1.100:8566/analyze',
    files=files,
    data=data
)
print(response.json())
```

**响应:**
```json
{
  "success": true,
  "question": "请分析图像中有问题的牙齿",
  "answer": "根据图像分析，发现以下问题...",
  "timestamp": "2024-12-11T10:30:00.123456"
}
```

### 4. 图像分析（Base64）- POST `/analyze/base64`

使用 Base64 编码的图像进行分析

**参数:**
- `image_base64` (必需): Base64 编码的图像字符串
- `question` (可选): 要问的问题
- `max_new_tokens` (可选): 最大生成 token 数，默认 4096

**请求示例 (Python):**
```python
import requests
import base64

# 读取并编码图像
with open('dental_image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

data = {
    'image_base64': image_base64,
    'question': '请分析图像中有问题的牙齿'
}

response = requests.post(
    'http://192.168.1.100:8566/analyze/base64',
    data=data
)
print(response.json())
```

### 5. API 文档 - GET `/docs`

访问自动生成的 Swagger UI 文档

在浏览器中打开:
```
http://192.168.1.100:8566/docs
```

---

## 客户端调用

### 方式 1: 使用提供的客户端示例

1. 将 `client_example.py` 复制到客户端机器
2. 修改配置:
   ```python
   SERVER_IP = "192.168.1.100"  # 改为你的服务器 IP
   ```
3. 运行示例:
   ```bash
   python client_example.py
   ```

### 方式 2: 使用 curl

```bash
# 健康检查
curl http://192.168.1.100:8566/health

# 分析图像
curl -X POST http://192.168.1.100:8566/analyze \
  -F "image=@dental_image.jpg" \
  -F "question=请分析这张牙科图像"
```

### 方式 3: 使用 Python requests

```python
import requests

# 健康检查
response = requests.get('http://192.168.1.100:8566/health')
print(response.json())

# 分析图像
files = {'image': open('dental_image.jpg', 'rb')}
data = {'question': '请分析这张牙科图像'}
response = requests.post(
    'http://192.168.1.100:8566/analyze',
    files=files,
    data=data
)
print(response.json())
```

### 方式 4: 使用其他语言

参考 `/docs` 中的 API 文档，使用任何支持 HTTP 的编程语言调用。

---

## 常见问题

### Q1: 无法连接到服务器

**检查清单:**
1. 确认服务器已启动
2. 检查 IP 地址是否正确
3. 检查防火墙设置:
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 8566/tcp
   
   # CentOS/RHEL
   sudo firewall-cmd --add-port=8566/tcp --permanent
   sudo firewall-cmd --reload
   ```
4. 确认客户端和服务器在同一局域网

### Q2: 模型加载失败

**可能原因:**
- 模型路径不正确
- GPU 显存不足
- 缺少依赖包

**解决方法:**
1. 检查 `api_server.py` 中的 `LOCAL_MODEL_PATH`
2. 使用 `nvidia-smi` 查看 GPU 状态
3. 重新安装依赖: `pip install -r requirements.txt`
4. 确保在 `zl_dentalGPT` conda 环境中运行

### Q3: 推理速度慢

**优化建议:**
1. 确保使用 GPU: 检查 `/health` 端点中的 `device` 字段
2. 减少 `max_new_tokens` 参数
3. 确保没有其他程序占用 GPU

### Q4: 如何修改端口?

修改 `api_server.py` 底部:
```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8888,  # 修改为你想要的端口
    log_level="info"
)
```

或使用命令行:
```bash
conda activate zl_dentalGPT
uvicorn api_server:app --host 0.0.0.0 --port 8888
```

### Q5: 如何在后台运行服务?

**方法 1: 使用 nohup**
```bash
conda activate zl_dentalGPT
nohup python api_server.py > server.log 2>&1 &
```

**方法 2: 使用 screen**
```bash
screen -S dentalgpt
conda activate zl_dentalGPT
python api_server.py
# 按 Ctrl+A+D 退出 screen
# 重新连接: screen -r dentalgpt
```

**方法 3: 使用 systemd (推荐生产环境)**

创建服务文件 `/etc/systemd/system/dentalgpt-api.service`:
```ini
[Unit]
Description=DentalGPT API Server
After=network.target

[Service]
Type=simple
User=zlyu
WorkingDirectory=/mnt/hdd/sdc/zlyu/dentalGPT
Environment="PATH=/home/zlyu/anaconda3/envs/zl_dentalGPT/bin:/usr/bin:/bin"
ExecStart=/home/zlyu/anaconda3/envs/zl_dentalGPT/bin/python api_server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

启动服务:
```bash
sudo systemctl daemon-reload
sudo systemctl start dentalgpt-api
sudo systemctl enable dentalgpt-api  # 开机自启
sudo systemctl status dentalgpt-api  # 查看状态
```

### Q6: 如何查看日志?

服务器端日志直接输出到终端。要保存日志:

```bash
python api_server.py > server.log 2>&1
```

或使用 tee 同时显示和保存:
```bash
python api_server.py 2>&1 | tee server.log
```

---

## 安全建议

1. **局域网使用**: 此 API 设计用于局域网环境，不建议暴露到公网
2. **添加认证**: 如需公网访问，建议添加 API Key 认证
3. **HTTPS**: 生产环境建议使用 HTTPS (需要配置 nginx 反向代理)
4. **速率限制**: 考虑添加请求速率限制

---

## 技术支持

如有问题，请检查:
1. 服务器日志
2. 客户端错误信息
3. `/health` 端点的响应

---

## 文件清单

- `api_server.py` - API 服务器主文件
- `requirements.txt` - Python 依赖列表
- `start_server.sh` - 启动脚本
- `client_example.py` - 客户端调用示例
- `API_README.md` - 本文档

---

**版本**: 1.0.0  
**更新日期**: 2024-12-11  
**作者**: DentalGPT Team
