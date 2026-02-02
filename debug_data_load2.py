#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test YOLO dataset loading and image reading.
"""

import sys
import traceback
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from PIL import Image
from ultralytics.data.utils import check_det_dataset


def debug_image_reading():
    """Debug function to test image reading."""
    try:
        print("=== 调试图像读取过程 ===")
        
        # 加载数据集配置
        data_path = "ng100.yaml"
        print(f"\n1. 加载数据集配置: {data_path}")
        data = check_det_dataset(data_path)
        
        # 获取训练集图像文件
        train_path = Path(data["train"])
        print(f"\n2. 训练集路径: {train_path}")
        
        # 获取所有图像文件
        img_files = list(train_path.rglob("*.*"))
        img_files = [f for f in img_files if f.suffix[1:].lower() in {"bmp", "jpg", "jpeg", "png"}]
        img_files = sorted(img_files)
        
        print(f"找到 {len(img_files)} 张图像")
        
        # 测试读取图像
        success_count = 0
        fail_count = 0
        
        for i, img_path in enumerate(img_files[:5]):  # 只测试前 5 张图像
            print(f"\n3. 测试读取图像 {i+1}: {img_path.name}")
            
            # 尝试使用 PIL 读取
            try:
                with Image.open(img_path) as img:
                    print(f"PIL 读取成功: {img.size}, {img.mode}")
                    img_np = np.array(img)
                    print(f"转换为 numpy 数组: {img_np.shape}, {img_np.dtype}")
                    success_count += 1
            except Exception as e:
                print(f"PIL 读取失败: {e}")
                fail_count += 1
            
            # 尝试使用 OpenCV 读取
            try:
                img_cv = cv2.imread(str(img_path))
                if img_cv is not None:
                    print(f"OpenCV 读取成功: {img_cv.shape}, {img_cv.dtype}")
                else:
                    print(f"OpenCV 读取失败")
            except Exception as e:
                print(f"OpenCV 读取异常: {e}")
        
        print(f"\n=== 图像读取测试完成 ===")
        print(f"成功读取: {success_count} 张")
        print(f"失败读取: {fail_count} 张")
        
        return True
        
    except Exception as e:
        print(f"\n=== 错误发生 ===")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常内容: {e}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = debug_image_reading()
    sys.exit(0 if success else 1)
