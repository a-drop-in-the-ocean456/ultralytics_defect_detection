from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="ng100.yaml",
        epochs=50,  # 增加训练轮数（小数据必须）
        imgsz=640,
        batch=8,
        workers=0,  # 关键：先关掉多进程
        lr0=0.0005,  # 降低学习率（小数据必做）
        freeze=10,  # 冻结前10层（backbone）只训练检测头，避免小数据把特征搞崩。
        weight_decay=0.0005,
        mosaic=0.0,  # 关闭 YOLO 默认增强（工业缺陷必须），YOLO 默认增强是为 COCO 设计的，对工业缺陷是毒药
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5
    )

if __name__ == "__main__":
    main()
