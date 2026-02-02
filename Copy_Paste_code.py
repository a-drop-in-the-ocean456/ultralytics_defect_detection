import os
import cv2
import random
import numpy as np

# ================== 配置 ==================
DEFECT_IMG_DIR = "../NG100_split/train"
DEFECT_LABEL_DIR = "../NG100_split/train"

NORMAL_IMG_DIR = "../OK90"

OUT_IMG_DIR = "../OK90_paste/"
OUT_LABEL_DIR = "../OK90_paste/"

NUM_SYNTHETIC = 300  # 生成多少张新图

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# ================== 工具函数 ==================
def load_yolo_labels(label_path, w, h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            boxes.append((int(cls), x1, y1, x2, y2))
    return boxes

def save_yolo_labels(label_path, boxes, w, h):
    with open(label_path, "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def random_transform(patch):
    h, w = patch.shape[:2]

    # 随机缩放
    scale = random.uniform(0.6, 1.4)
    patch = cv2.resize(patch, (int(w * scale), int(h * scale)))

    # 随机旋转
    angle = random.uniform(-15, 15)
    h, w = patch.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    patch = cv2.warpAffine(patch, M, (w, h), borderValue=(0, 0, 0))

    return patch

# ================== 主逻辑 ==================
def main():
    defect_imgs = os.listdir(DEFECT_IMG_DIR)
    normal_imgs = os.listdir(NORMAL_IMG_DIR)

    for i in range(NUM_SYNTHETIC):
        # 随机选一张正常图
        bg_name = random.choice(normal_imgs)
        bg_path = os.path.join(NORMAL_IMG_DIR, bg_name)
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue

        H, W = bg_img.shape[:2]
        new_boxes = []

        # 随机选一张缺陷图
        defect_name = random.choice(defect_imgs)
        defect_img_path = os.path.join(DEFECT_IMG_DIR, defect_name)
        defect_label_path = os.path.join(
            DEFECT_LABEL_DIR, defect_name.replace(".bmp", ".txt").replace(".png", ".txt")
        )

        defect_img = cv2.imread(defect_img_path)
        if defect_img is None:
            continue

        dh, dw = defect_img.shape[:2]
        defect_boxes = load_yolo_labels(defect_label_path, dw, dh)

        if len(defect_boxes) == 0:
            continue

        # 随机选一个缺陷框
        cls, x1, y1, x2, y2 = random.choice(defect_boxes)
        patch = defect_img[y1:y2, x1:x2]

        if patch.size == 0:
            continue

        # 随机变换
        patch = random_transform(patch)
        ph, pw = patch.shape[:2]

        # 随机粘贴位置
        px = random.randint(0, W - pw - 1)
        py = random.randint(0, H - ph - 1)

        # 粘贴
        mask = patch > 0
        bg_img[py:py+ph, px:px+pw][mask] = patch[mask]

        # 新 bbox
        nx1, ny1 = px, py
        nx2, ny2 = px + pw, py + ph
        new_boxes.append((cls, nx1, ny1, nx2, ny2))

        # 保存
        out_img_name = f"cp_{i}.jpg"
        out_label_name = f"cp_{i}.txt"

        cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_name), bg_img)
        save_yolo_labels(os.path.join(OUT_LABEL_DIR, out_label_name), new_boxes, W, H)

        print(f"[OK] generated {out_img_name}")

if __name__ == "__main__":
    main()
