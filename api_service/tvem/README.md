# 🦷 Teeth Visual Experts Models (TVEM)

基于 MaskDINO 的牙科疾病检测与分割模型集合。

## 📋 可用模型

### MaskDINO 分割模型（4个）

| 模型 | 配置文件 | 权重文件 | 用途 |
|------|---------|---------|------|
| **11类疾病检测** | `configs/11diseases.yaml` | `weights/11diseases.pth` | 龋齿、充填、阻生、根尖病变、深龋、残根、种植体、牙冠、残冠、桥体、义齿 |
| **4象限分割** | `configs/4quadrants.yaml` | `weights/4quadrants.pth` | 右上、左上、左下、右下象限 |
| **解剖结构** | `configs/mandibular_maxillary.yaml` | `weights/mandibular_maxillary.pth` | 下颌管、上颌窦 |
| **骨质流失** | `configs/bone_loss.yaml` | `weights/bone_loss.pth` | 牙槽骨吸收检测 |

## 🚀 快速开始

### 1. 环境设置

```bash
# 安装依赖（首次使用）
bash setup_maskdino_only.sh
```

这将会：
- 创建虚拟环境 (`.venv`)
- 克隆 MaskDINO 仓库
- 安装所有必要的 Python 依赖

### 2. 运行推理

**批量推理所有模型（推荐）：**

```bash
bash infer_all_with_json.sh examples/sample.jpg
```

这将运行所有4个模型并生成：
- 每个模型的可视化图像和JSON结果
- 汇总文件 `results/summary.json`

**单个模型：**

```bash
source .venv/bin/activate

# 11类疾病检测
uv run python infer_with_json.py \
  --config configs/11diseases.yaml \
  --checkpoint weights/11diseases.pth \
  --image examples/sample.jpg \
  --output-dir ./results \
  --category-file categories/11diseases_category.json \
  --confidence 0.3
```

### 3. 查看结果

```bash
# 查看所有结果
ls -lh results/*/

# 查看汇总
cat results/summary.json
```

输出文件：
- `{model}/sample_visualization.jpg` - 可视化图像（只显示轮廓边框）
- `{model}/sample_results.json` - 详细检测结果
- `summary.json` - 所有模型的汇总结果

## 📊 输出格式

### JSON 结构示例

```json
{
  "image": "sample.jpg",
  "timestamp": "2026-01-07T18:00:00",
  "confidence_threshold": 0.3,
  "total_detections": 11,
  "category_counts": {
    "Filling": 7,
    "Crown": 2,
    "Pontic": 1,
    "Impacted": 1
  },
  "detections": [
    {
      "id": 1,
      "category_id": 2,
      "category_name": "Filling",
      "confidence": 0.825,
      "bbox": {"x": 100, "y": 200, "width": 50, "height": 60},
      "mask_area": 2057,
      "mask_shape": [976, 1976]
    }
  ]
}
```

### 汇总文件 (summary.json)

```json
{
  "11diseases": {
    "total_detections": 11,
    "category_counts": {
      "Filling": 7,
      "Crown": 2,
      "Pontic": 1,
      "Impacted": 1
    }
  },
  "4quadrants": {
    "total_detections": 4,
    "category_counts": {
      "Upper Left": 1,
      "Upper Right": 1,
      "Lower Left": 1,
      "Lower Right": 1
    }
  },
  "mandibular_maxillary": {
    "total_detections": 4,
    "category_counts": {
      "Mandibular Canal": 2,
      "Maxillary Sinus": 2
    }
  },
  "bone_loss": {
    "total_detections": 2,
    "category_counts": {
      "Bone Loss": 2
    }
  }
}
```

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 模型配置文件 | 必需 |
| `--checkpoint` | 模型权重文件 | 必需 |
| `--image` | 输入图像路径 | 必需 |
| `--output-dir` | 输出目录 | `./results` |
| `--category-file` | 类别映射文件 | 可选 |
| `--confidence` | 置信度阈值 | 0.3 |
| `--device` | 计算设备 | `cuda` |

### 置信度阈值建议

- **0.3-0.4**: 平衡模式（推荐）- 较全面的检测
- **0.5-0.6**: 严格模式 - 只保留高置信度结果
- **0.2-0.3**: 宽松模式 - 包含更多潜在检测

## 📁 文件结构

```
TVEM/
├── README.md                          # 本文件
├── setup_maskdino_only.sh            # 环境安装脚本
├── infer_with_json.py                # 单模型推理脚本（轮廓可视化）
├── infer_all_with_json.sh            # 批量推理脚本
├── create_category_files.py          # 类别文件生成器
│
├── configs/                           # 配置文件
│   ├── 11diseases.yaml
│   ├── 4quadrants.yaml
│   ├── mandibular_maxillary.yaml
│   └── bone_loss.yaml
│
├── weights/                           # 模型权重 (~10GB)
│   ├── 11diseases.pth
│   ├── 4quadrants.pth
│   ├── mandibular_maxillary.pth
│   └── bone_loss.pth
│
├── categories/                        # 类别映射
│   ├── 11diseases_category.json
│   ├── 4quadrants_category.json
│   ├── mandibular_maxillary_category.json
│   └── bone_loss_category.json
│
├── examples/                          # 样例文件
│   └── sample.jpg
│
└── MaskDINO/                         # MaskDINO 框架（自动克隆）
```

## 📝 文件命名对照表

为了简化使用，所有文件都使用简短的功能性名称：

### 配置文件 (configs/)

| 简化名称 | 原始名称 | 功能 |
|---------|---------|------|
| `11diseases.yaml` | `maskdino_SwinL_bs16_50ep_4s_dowsample1_2048_x-ray_11diseases.yaml` | 11类牙科疾病检测 |
| `4quadrants.yaml` | `maskdino_SwinL_bs16_50ep_4s_dowsample1_2048_panoramic_x-ray_4quadrants.yaml` | 4象限分割 |
| `mandibular_maxillary.yaml` | `maskdino_Swinl_bs16_50ep_4s_dowsample1_2048_panoramic_x-ray_Mandibular_Canal_Maxillary_Sinus.yaml` | 下颌管和上颌窦 |
| `bone_loss.yaml` | `maskdino_SwinL_bs16_50ep_4s_dowsample1_2048_x-ray_bone_loss_1diseases.yaml` | 骨质流失检测 |

### 模型权重 (weights/)

| 简化名称 | 原始名称 | 大小 |
|---------|---------|------|
| `11diseases.pth` | `Teeth_Visual_Experts_Maskdino_Swinl_x-ray_11diseases.pth` | ~2.5GB |
| `4quadrants.pth` | `Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_4quadrants.pth` | ~2.5GB |
| `mandibular_maxillary.pth` | `Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_Mandibular_Canal_Maxillary_Sinus.pth` | ~2.5GB |
| `bone_loss.pth` | `Teeth_Visual_Experts_Maskdino_Swinl_x-ray_bone_loss_1disease.pth` | ~2.5GB |

## ✨ 主要特性

### 1. 轮廓可视化
- ✅ 只显示mask的轮廓边框（无半透明填充）
- ✅ 显示类别名称（而非category_id）
- ✅ 自定义可视化器 `OutlineOnlyVisualizer`

### 2. JSON输出
- ✅ 详细的检测信息（边界框、mask面积、置信度）
- ✅ 类别统计
- ✅ 自动汇总所有模型结果

### 3. 置信度过滤
- ✅ 可配置的置信度阈值
- ✅ 过滤低质量检测
- ✅ 统计信息基于过滤后的结果

## 🔧 故障排除

### 1. 依赖错误

```bash
# 重新安装关键依赖
cd MaskDINO
uv pip install timm==0.6.13 numpy==1.26.4
uv pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 2. CUDA 内存不足

- 关闭其他 GPU 程序
- 减小输入图像尺寸
- 使用 CPU 模式：`--device cpu`

### 3. 配置文件路径错误

如果遇到 `_BASE_` 路径错误，检查 YAML 文件中的路径是否为 `../MaskDINO/...`：

```yaml
_BASE_: ../MaskDINO/configs/coco/instance-segmentation/Base-COCO-InstanceSegmentation.yaml
```

### 4. 类别映射文件缺失

```bash
# 重新生成类别文件
source .venv/bin/activate
python create_category_files.py
```

## 💡 使用示例

### 示例1: 批量检测

```bash
# 对单张图片运行所有模型
bash infer_all_with_json.sh patient001.jpg

# 查看汇总结果
cat results/summary.json
```

### 示例2: 单个模型 - 11类疾病检测

```bash
source .venv/bin/activate

uv run python infer_with_json.py \
  --config configs/11diseases.yaml \
  --checkpoint weights/11diseases.pth \
  --image patient_panoramic.jpg \
  --output-dir ./results/patient001 \
  --category-file categories/11diseases_category.json \
  --confidence 0.3
```

### 示例3: 象限分割（高置信度）

```bash
source .venv/bin/activate

uv run python infer_with_json.py \
  --config configs/4quadrants.yaml \
  --checkpoint weights/4quadrants.pth \
  --image patient_panoramic.jpg \
  --output-dir ./results/quadrants \
  --category-file categories/4quadrants_category.json \
  --confidence 0.5
```

## 📚 参考资料

- **原始项目**: [OralGPT](https://github.com/isbrycee/OralGPT)
- **模型仓库**: [HuggingFace - Teeth_Visual_Experts_Models](https://huggingface.co/Bryceee/Teeth_Visual_Experts_Models)
- **MaskDINO**: [IDEA-Research/MaskDINO](https://github.com/IDEA-Research/MaskDINO)
- **Detectron2**: [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

## 📄 许可

遵循原始模型的许可协议。本项目仅用于研究和教育目的。

## 🙏 致谢

- OralGPT 团队提供的预训练模型
- IDEA Research 的 MaskDINO 框架
- Facebook Research 的 Detectron2

---

**版本**: 2.0  
**最后更新**: 2026-01-07  
**主要改进**:
- ✨ 文件结构重组（configs/、weights/、categories/）
- ✨ 文件名简化（功能性命名）
- ✨ 轮廓可视化（无填充）
- ✨ 自动汇总功能
