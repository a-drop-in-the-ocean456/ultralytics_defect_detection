import os
import random
import shutil

def split_dataset(input_dir, output_dir, train_ratio=0.9):
    # 创建输出目录结构
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.bmp') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            image_files.append(file)
    
    # 随机打乱文件列表
    random.shuffle(image_files)
    
    # 划分训练集和验证集
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    # 复制训练集文件
    for file in train_files:
        # 复制图像文件
        src_img = os.path.join(input_dir, file)
        dst_img = os.path.join(train_images_dir, file)
        shutil.copy(src_img, dst_img)
        
        # 复制对应的标注文件
        label_file = os.path.splitext(file)[0] + '.txt'
        src_label = os.path.join(input_dir, label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(train_labels_dir, label_file)
            shutil.copy(src_label, dst_label)
    
    # 复制验证集文件
    for file in val_files:
        # 复制图像文件
        src_img = os.path.join(input_dir, file)
        dst_img = os.path.join(val_images_dir, file)
        shutil.copy(src_img, dst_img)
        
        # 复制对应的标注文件
        label_file = os.path.splitext(file)[0] + '.txt'
        src_label = os.path.join(input_dir, label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(val_labels_dir, label_file)
            shutil.copy(src_label, dst_label)
    
    print(f"数据集划分完成！")
    print(f"训练集数量: {len(train_files)} 张图像")
    print(f"验证集数量: {len(val_files)} 张图像")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    # 数据集路径
    input_directory = '../NG100'
    output_directory = '../NG100_split'
    
    # 划分比例
    training_ratio = 0.9
    
    split_dataset(input_directory, output_directory, training_ratio)
