# YOLOv8n 训练 NG100_split 数据集记录

## 1. 任务概述

本任务使用 YOLOv8n 模型训练一个单类别缺陷检测数据集 NG100_split，用于检测名为 "ng" 的缺陷类型。该数据集已经过划分，包含训练集和验证集。

## 2. 数据集信息

- **数据集位置**: `../NG100_split/`
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

## 4. 训练过程

### 4.1 环境准备
已安装 ultralytics 库 8.3.237 版本，使用 Python 3.8.20 和 PyTorch 1.13.1+cu116。

### 4.2 启动训练
执行以下命令开始训练：
```bash
python train.py
```

### 4.3 训练开始信息
- 模型架构: YOLOv8n，129 层，3,011,043 参数
- 从预训练权重传输了 319/355 个项目
- 使用自动混合精度 (AMP) 进行训练
- 训练集: 89 张图像
- 验证集: 10 张图像

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

## 6. 训练过程中可能遇到的问题

### 6.1 多进程权限问题
在 Windows 系统上使用 `workers=0` 禁用多进程以避免权限问题。

### 6.2 内存不足
训练使用了 8GB 内存，对于 4GB VRAM 的 GPU 来说是合适的。

## 7. 模型验证与使用

### 7.1 验证模型
训练完成后，可以使用最佳模型进行验证：
```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=ng100.yaml
```

### 7.2 推理预测
使用训练好的模型对新图像进行预测：
```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=../NG100_split/val/images/
```

## 8. 总结

本任务正在使用 YOLOv8n 模型训练 NG100_split 单类别缺陷检测数据集。通过使用划分后的数据集，确保了训练和验证过程的分离，有助于更准确地评估模型性能。

---

**训练时间**: 进行中
**训练状态**: 正在训练
**创建日期**: 2026-01-30
**数据集划分**: 训练集 89 张，验证集 10 张
