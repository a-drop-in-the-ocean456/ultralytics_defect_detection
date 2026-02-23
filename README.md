# YOLOv8n 训练 NG100_split 数据集记录

## 1. 任务概述

本任务使用 YOLOv8n 模型训练一个单类别缺陷检测数据集 NG100_split，用于检测名为 "ng" 的缺陷类型。该数据集已经过划分，包含训练集和验证集。

## 2. 数据集信息

- **数据集位置**: `dataset/NG100_split/`
- **数据集结构**: 已划分为训练集和验证集
  - 训练集: `train/` - 89 张图像
  - 验证集: `val/` - 10 张图像
- **图像数量**: 总 99 张 (BMP 格式)
- **类别数量**: 1 类 (`ng`)
- **标注格式**: YOLO 格式 (txt 文件)
- **类别名称**: `ng` (类别 ID: 0)

## 3. 训练配置

### 3.1 模型选择
- **模型**: YOLOv8n (nano 版本，轻量化模型)
- **预训练权重**: yolov8n.pt (COCO 数据集预训练)

### 3.2 训练参数
- **任务类型**: 目标检测 (detect)
- **训练轮数**: 50 epochs
- **图像尺寸**: 640x640
- **批次大小**: 8
- **学习率**: 自动调整 (默认)
- **工作进程**: 0 (禁用多进程以避免权限问题)
- **数据集配置**: ng100.yaml
- **设备**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)

### 3.3 配置文件内容 (`ng100.yaml`)
```yaml
# NG100 Split Dataset Configuration

path: ../NG100_split  # 划分后的数据集根目录
train: ./train        # 训练集路径
val: ./val            # 验证集路径
test: ./val           # 测试集路径（使用验证集作为测试集）

# 类别信息
names:
  0: ng               # 单类别，类别名为 ng
```

## 4. 预测结果展示

以下是使用训练好的模型对验证集图像进行预测的结果：

![预测结果1](predictions/predicted_17.bmp)
![预测结果2](predictions/predicted_38.bmp)
![预测结果3](predictions/predicted_47.bmp)
![预测结果4](predictions/predicted_64.bmp)

## 5. 训练结果

### 5.1 训练输出位置
训练结果将保存在 `runs/detect/train/` 目录下，包含：
- `weights/best.pt` - 最佳验证集性能模型
- `weights/last.pt` - 最后一轮训练模型
- `results.png` - 训练过程曲线
- `confusion_matrix.png` - 混淆矩阵
- `val_batch0_pred.jpg` - 验证集预测示例

### 5.2 性能指标
训练过程中会记录以下指标：
- **mAP@0.5**: 在 IoU=0.5 时的平均精度
- **mAP@0.5:0.95**: 在 IoU 从 0.5 到 0.95 时的平均精度
- **Precision**: 精确率
- **Recall**: 召回率

## 6. 模型验证与使用

### 6.1 验证模型
训练完成后，可以使用最佳模型进行验证：
```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=ng100.yaml
```

### 6.2 推理预测
使用训练好的模型对新图像进行预测：
```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=dataset/NG100_split/val/
```

## 7. 项目结构

```
ultralytics_defect_detection/
├── dataset/
│   ├── NG100_split/
│   │   ├── train/          # 训练集
│   │   └── val/            # 验证集
│   └── OK90_paste/         # OK样本数据集
├── predictions/            # 预测结果
├── predictions1000/        # 1000张图像预测结果
├── ng100.yaml             # 数据集配置
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
└── split_dataset.py       # 数据集划分脚本
```

---

**训练完成日期**: 2026-01-30
**数据集划分**: 训练集 89 张，验证集 10 张
