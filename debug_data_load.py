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

from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset


def debug_data_loading():
    """Debug function to test data loading."""
    try:
        print("=== 调试数据加载过程 ===")
        
        # 加载数据集配置
        data_path = "ng100.yaml"
        print(f"\n1. 加载数据集配置: {data_path}")
        data = check_det_dataset(data_path)
        print(f"数据集配置加载成功")
        print(f"训练集路径: {data['train']}")
        print(f"验证集路径: {data['val']}")
        print(f"类别数量: {data['nc']}")
        print(f"类别名称: {data['names']}")
        
        # 创建配置
        args = get_cfg(DEFAULT_CFG)
        args.imgsz = 640
        args.batch = 8
        args.workers = 0
        args.mosaic = 0.0  # 禁用 mosaic
        args.cache = False
        
        # 测试创建训练数据集
        print("\n2. 创建训练数据集")
        train_dataset = YOLODataset(
            img_path=data["train"],
            imgsz=args.imgsz,
            cache=args.cache,
            augment=True,
            hyp=args,
            prefix="train: ",
            rect=False,
            batch_size=args.batch,
            stride=32,
            single_cls=False,
            classes=None
        )
        print(f"训练数据集创建成功，包含 {len(train_dataset)} 张图像")
        
        # 测试加载第一张图像
        print("\n3. 测试加载第一张图像")
        if len(train_dataset) > 0:
            index = 0
            item = train_dataset[index]
            print(f"图像加载成功")
            print(f"图像键值: {list(item.keys())}")
            if "img" in item:
                print(f"图像形状: {item['img'].shape}")
                print(f"图像数据类型: {item['img'].dtype}")
            if "cls" in item:
                print(f"类别数量: {len(item['cls'])}")
            if "bboxes" in item:
                print(f"边界框数量: {len(item['bboxes'])}")
        
        # 测试数据加载器
        print("\n4. 测试数据加载器")
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers
        )
        
        print(f"数据加载器创建成功")
        
        # 测试加载第一个 batch
        print("\n5. 测试加载第一个 batch")
        for batch in train_loader:
            print(f"Batch 加载成功")
            print(f"Batch 图像形状: {batch['img'].shape}")
            print(f"Batch 类别形状: {batch['cls'].shape}")
            print(f"Batch 边界框形状: {batch['bboxes'].shape}")
            break  # 只测试一个 batch
        
        print("\n=== 数据加载调试完成 ===")
        
    except Exception as e:
        print(f"\n=== 错误发生 ===")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常内容: {e}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())
        return False
    
    return True


if __name__ == "__main__":
    success = debug_data_loading()
    sys.exit(0 if success else 1)
