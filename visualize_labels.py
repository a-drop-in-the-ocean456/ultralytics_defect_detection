import os
import cv2
import numpy as np
from pathlib import Path

def visualize_yolo_labels(data_config, save_dir=None):
    """
    可视化 YOLO 数据集的标签边界框
    
    参数:
        data_config: 数据集配置文件路径 (yaml 文件)
        save_dir: 保存可视化结果的目录，None 表示只显示不保存
    """
    import yaml
    
    # 读取配置文件
    with open(data_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取数据集路径和类别信息
    dataset_path = Path(config['path'])
    class_names = config['names']
    
    # 处理训练集文件夹
    all_visualizations = []
    
    print('图片路径：', config['train'])
    for train_dir in config['train']:
        img_dir = dataset_path / train_dir
        label_dir = img_dir.with_name(f"{img_dir.name}_labels") if "labels" not in img_dir.name else img_dir
        
        # 检查标签文件夹是否存在
        if not label_dir.exists():
            label_dir = img_dir.parent / "labels" / img_dir.name
            if not label_dir.exists():
                label_dir = img_dir  # 直接在图片文件夹中查找 .txt 文件
        
        print(f"处理文件夹: {img_dir}")
        print(f"标签文件夹: {label_dir}")
        
        # 遍历图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                # 对应的标签文件
                label_file = label_dir / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    # 读取图片
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                        
                    h, w = img.shape[:2]
                    
                    # 读取标签
                    with open(label_file, 'r', encoding='utf-8') as f:
                        labels = f.readlines()
                    
                    # 绘制每个边界框
                    for label in labels:
                        label = label.strip()
                        if label:
                            try:
                                # YOLO 格式: class x_center y_center width height
                                parts = list(map(float, label.split()))
                                cls = int(parts[0])
                                x_center, y_center, box_w, box_h = parts[1:]
                                
                                # 转换为像素坐标
                                x1 = int((x_center - box_w/2) * w)
                                y1 = int((y_center - box_h/2) * h)
                                x2 = int((x_center + box_w/2) * w)
                                y2 = int((y_center + box_h/2) * h)
                                
                                # 绘制边界框
                                color = (0, 255, 0)  # 绿色
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                
                                # 绘制类别名称
                                class_name = class_names[cls]
                                cv2.putText(img, class_name, (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            except Exception as e:
                                print(f"解析标签文件 {label_file} 时出错: {e}")
                                continue
                    
                    # 保存或显示图片
                    if save_dir:
                        save_path = Path(save_dir)
                        save_path.mkdir(exist_ok=True)
                        output_file = save_path / f"{train_dir}_{img_file.name}"
                        cv2.imwrite(str(output_file), img)
                        print(f"已保存: {output_file}")
                    else:
                        # 显示图片
                        cv2.imshow(f"Visualization - {img_file.name}", img)
                        
                        # 按 ESC 键退出，按其他键继续
                        key = cv2.waitKey(0)
                        if key == 27:  # ESC
                            cv2.destroyAllWindows()
                            return
        
        if not save_dir:
            cv2.destroyAllWindows()

def main():
    # 配置参数
    data_config = "ng100.yaml"
    save_dir = "visualized_labels"  # 保存可视化结果的目录
    
    print("开始可视化 YOLO 标签边界框...")
    visualize_yolo_labels(data_config, save_dir)
    print(f"可视化完成！结果保存在 {save_dir} 目录中")

if __name__ == "__main__":
    main()