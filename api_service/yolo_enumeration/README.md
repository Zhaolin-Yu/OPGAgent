# Tooth Enumeration Detection Tool

牙齿枚举检测工具 - 使用YOLOv9c模型识别全景X光片中的8种牙齿类型

## 模型信息

- **架构**: YOLOv9c
- **检测类别**: 8种牙齿类型（标记为1-8）
  - 1 (Central incisor - 中切牙)
  - 2 (Lateral incisor - 侧切牙)
  - 3 (Canine - 尖牙)
  - 4 (First premolar - 第一前磨牙)
  - 5 (Second premolar - 第二前磨牙)
  - 6 (First molar - 第一磨牙)
  - 7 (Second molar - 第二磨牙)
  - 8 (Third molar/Wisdom tooth - 第三磨牙/智齿)
- **输入尺寸**: 640x640
- **训练数据**: 634张图像，18,095个标注

## 安装

### 使用 uv (推荐)
```bash
# 安装 uv（如果尚未安装）
pip install uv

# 同步依赖
uv sync
```

### 使用 pip
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 本地预测（无需API服务）

```bash
# 单张图像检测
python predict.py --image path/to/xray.png

# 批量检测
python predict.py --dir path/to/images/

# 自定义参数
python predict.py --image xray.png --conf 0.3 --iou 0.5 --output results/

# 不保存结果（仅显示）
python predict.py --image xray.png --no-save
```

### 2. API服务模式

#### 启动服务
```bash
# 使用启动脚本（推荐）
bash start_api.sh

# 或直接使用uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 8001

# 使用uv运行
uv run uvicorn api_server:app --host 0.0.0.0 --port 8001
```

#### 使用Python客户端
```python
from api_client import ToothEnumerationClient

# 创建客户端
client = ToothEnumerationClient("http://localhost:8001")

# 检测图像
result = client.detect_from_file("xray.png", conf_threshold=0.25)
print(result)

# 格式化输出
formatted = client.format_result(result)
print(formatted)
```

#### 使用Agent函数
```python
from api_client import detect_tooth_enumeration, get_tooth_enumeration_summary

# 检测并获取格式化结果
result_text = detect_tooth_enumeration("xray.png")
print(result_text)

# 获取统计摘要
summary = get_tooth_enumeration_summary("xray.png")
print(f"检测到 {summary['total_teeth']} 颗牙齿")
print(f"各类型统计: {summary['tooth_count_by_type']}")
```

#### 命令行客户端
```bash
# 检测单张图像
python api_client.py xray.png

# 自定义参数
python api_client.py xray.png --url http://localhost:8001 --conf 0.3 --save
```

### 3. API端点

- `POST /detect` - 上传图像文件进行检测
- `POST /detect_base64` - 使用Base64编码图像进行检测
- `GET /health` - 健康检查
- `GET /model/info` - 获取模型信息

## 输出说明

### 检测结果格式
```json
{
  "success": true,
  "num_detections": 15,
  "detections": [
    {
      "class_id": 0,
      "class_name": "1",
      "confidence": 0.89,
      "bbox": [100, 200, 150, 250]
    }
  ]
}
```

### 统计摘要格式
```json
{
  "success": true,
  "total_teeth": 15,
  "tooth_count_by_type": {
    "1": 2,
    "2": 2,
    "3": 2,
    "6": 4
  },
  "num_detections": 15
}
```

## 部署到其他环境

1. **复制必需文件**
```bash
# 打包工具
tar -czf enumeration_tool.tar.gz enumeration_tool/

# 在目标环境解压
tar -xzf enumeration_tool.tar.gz
cd enumeration_tool
```

2. **安装依赖**
```bash
pip install -r requirements.txt
# 或使用 uv sync
```

3. **确保模型文件存在**
   - 检查 `model/best.pt` 是否存在
   - 如果不存在，从训练目录复制：`cp ../yolov9_enumeration/runs/detect/tooth_enumeration_yolov9c_*/weights/best.pt model/`

4. **启动服务**
```bash
bash start_api.sh
```

## MCP工具集成

本工具提供Model Context Protocol (MCP)工具定义，可在支持MCP的环境中使用：

- `detect_tooth_enumeration` - 检测牙齿枚举
- `get_tooth_enumeration_summary` - 获取统计摘要

工具定义参见 `mcp_tools.json`

## 性能说明

- **推理速度**: ~30ms/张 (NVIDIA GPU)
- **内存占用**: ~1GB (含模型加载)
- **准确率**: 取决于训练结果，查看训练日志获取详细指标

## 环境变量

```bash
# API服务配置
export HOST=0.0.0.0        # 默认: 0.0.0.0
export PORT=8001           # 默认: 8001
export WORKERS=1           # 默认: 1
```

## 故障排除

1. **模型文件未找到**
   - 确保 `model/best.pt` 存在
   - 从训练目录复制模型文件

2. **端口占用**
   - 修改 `PORT` 环境变量
   - 检查 8001 端口是否被占用：`lsof -i :8001`

3. **依赖问题**
   - 使用 `pip install -r requirements.txt` 重新安装
   - 或使用 `uv sync` 同步依赖

## License

MIT
