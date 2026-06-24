# Agent_v3 API 服务

本目录包含 Agent_v3 使用的所有工具 API 服务。

## 端口分配

| 服务 | 端口 | GPU | 说明 |
|------|------|-----|------|
| yolo_enumeration | 6600 | 可配置 | 牙齿编号检测 |
| tvem | 6602 | 可配置 | MaskDINO 多模型检测 |
| medsam | 6603 | 可配置 | MedSAM 医学图像分割 |
| oral_gpt | 6604-6607 | 0-3 | OralGPT-Omni VLM (4副本) |
| dental_gpt | 6608-6611 | 0-3 | DentalGPT VLM (4副本) |

## 快速启动

### 测试环境 (GPU 0-3)

```bash
# 启动所有服务（测试环境）
bash start_all_services.sh

# 停止所有服务
bash stop_all_services.sh
```

### 单独启动服务

```bash
# YOLO 牙齿编号检测
cd yolo_enumeration && bash start_api.sh

# TVEM MaskDINO
cd tvem && bash start_api.sh

# MedSAM
cd medsam && bash start_api.sh

# OralGPT (单副本)
cd oral_gpt && bash start_api.sh

# OralGPT (所有4个副本)
cd oral_gpt && bash start_all_replicas.sh

# DentalGPT (单副本)
cd dental_gpt && bash start_api.sh

# DentalGPT (所有4个副本)
cd dental_gpt && bash start_all_replicas.sh
```

### 指定端口和 GPU

所有服务支持通过环境变量配置端口和 GPU：

```bash
# 指定端口和 GPU
PORT=6700 GPU=1 bash start_api.sh

# 仅指定 GPU
GPU=2 bash start_api.sh
```

## 服务详情

### 1. YOLO 牙齿编号检测 (yolo_enumeration)
- **端口**: 6600
- **功能**: 识别口腔全景片中的牙齿编号 (1-8)
- **虚拟环境**: `/data/zhaolin/tool_env/yolo_enumeration/.venv`
- **API 文档**: http://localhost:6600/docs
- **主要端点**:
  - `POST /detect` - 检测图像
  - `GET /health` - 健康检查
  - `POST /load_model` - 加载模型
  - `POST /unload_model` - 卸载模型

### 2. TVEM MaskDINO (tvem)
- **端口**: 6602
- **功能**: 高精度牙科图像分割，支持多种检测任务
- **虚拟环境**: `/data/zhaolin/tool_env/TVEM/.venv`
- **API 文档**: http://localhost:6602/docs
- **支持的模型**:
  - `4quadrants` - 四象限分割
  - `11diseases` - 11种疾病检测
  - `bone_loss` - 骨质流失检测
  - `mandibular_maxillary` - 下颌管和上颌窦检测
- **主要端点**:
  - `POST /detect/{model}` - 使用指定模型检测
  - `GET /models` - 获取可用模型列表
  - `GET /health` - 健康检查
  - `POST /load/{model}` - 加载指定模型
  - `POST /unload/{model}` - 卸载指定模型

### 3. MedSAM (medsam)
- **端口**: 6603
- **功能**: 医学图像分割，根据 bbox 进行精确分割
- **虚拟环境**: `./medsam/.venv`
- **API 文档**: http://localhost:6603/docs
- **主要端点**:
  - `POST /segment` - 分割图像
  - `GET /health` - 健康检查

### 4. OralGPT-Omni (oral_gpt)
- **端口**: 6604-6607 (4个副本)
- **功能**: 多模态牙科诊断模型
- **虚拟环境**: `/data/zhaolin/tool_env/OralGPT-omni/.venv`
- **API 文档**: http://localhost:6604/docs
- **负载均衡**: Round-robin 策略
- **主要端点**:
  - `POST /analyze` - 分析图像
  - `GET /health` - 健康检查

### 5. DentalGPT (dental_gpt)
- **端口**: 6608-6611 (4个副本)
- **功能**: 专业牙科 VLM 分析模型
- **虚拟环境**: `/data/zhaolin/tool_env/dentalGPT/.venv`
- **API 文档**: http://localhost:6608/docs
- **负载均衡**: Round-robin 策略
- **主要端点**:
  - `POST /analyze` - 分析图像
  - `GET /health` - 健康检查
  - `POST /load_model` - 加载模型
  - `POST /unload_model` - 卸载模型

## GPU 分配建议

### 测试环境 (4 GPU)

对于在 GPU 0-3 上测试：

| GPU | 服务 | 显存估计 |
|-----|------|----------|
| 0 | yolo_enumeration + oral_gpt_0 | ~14GB |
| 1 | oral_gpt_1 | ~12GB |
| 2 | tvem + oral_gpt_2 | ~20GB |
| 3 | medsam + oral_gpt_3 | ~16GB |

或者专门用于 VLM：

| GPU | 服务 | 显存估计 |
|-----|------|----------|
| 0 | oral_gpt_0 + dental_gpt_0 | ~24GB |
| 1 | oral_gpt_1 + dental_gpt_1 | ~24GB |
| 2 | oral_gpt_2 + dental_gpt_2 | ~24GB |
| 3 | oral_gpt_3 + dental_gpt_3 | ~24GB |

### 生产环境

建议根据实际 GPU 数量和显存调整配置。

## 健康检查

```bash
# 检查所有服务状态
bash check_health.sh

# 手动检查单个服务
curl http://localhost:6600/health  # yolo_enumeration
curl http://localhost:6602/health  # tvem
curl http://localhost:6603/health  # medsam
curl http://localhost:6604/health  # oral_gpt replica 0
curl http://localhost:6608/health  # dental_gpt replica 0
```

## 日志

VLM 服务的日志存放在各自目录下的 `logs/` 文件夹中：
- `oral_gpt/logs/oral_gpt_replica_*.log`
- `dental_gpt/logs/dental_gpt_replica_*.log`

## 故障排除

1. **端口被占用**: 使用 `lsof -i:端口号` 查看占用进程
2. **GPU 内存不足**: 使用 `nvidia-smi` 检查 GPU 使用情况
3. **模型加载失败**: 检查模型文件路径和权限
4. **虚拟环境问题**: 确保虚拟环境存在且依赖已安装
