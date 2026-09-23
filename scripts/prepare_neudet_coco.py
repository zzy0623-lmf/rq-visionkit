# -*- coding: utf-8 -*-
"""T1.7 数据准备：NEU-DET 的 YOLO 标注转 COCO JSON（YOLOX 训练格式）。

输入：
    demo/neu_det/dataset/IMAGES/       1800 张 200×200 灰度图
    demo/neu_det/dataset/yolo_labels/ 1800 个 YOLO txt + classes.txt

输出：
    demo/neu_det/dataset/coco/train.json  8:2 划分（固定 seed=42）
    demo/neu_det/dataset/coco/val.json

YOLO 格式（归一化）：class_id x_center y_center w h
COCO 格式（像素）：bbox=[x, y, w, h]（左上角 + 宽高）
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "demo" / "neu_det" / "dataset" / "IMAGES"
LABELS = ROOT / "demo" / "neu_det" / "dataset" / "yolo_labels"
OUT = ROOT / "demo" / "neu_det" / "dataset" / "coco"

# NEU-DET 所有图片均为 200×200（T0.3 实测确认）
IMG_W, IMG_H = 200, 200
SPLIT = 0.8
SEED = 42


def load_classes() -> list:
    return (LABELS / "classes.txt").read_text(encoding="utf-8").splitlines()


def parse_yolo_line(line: str) -> tuple:
    cid, xc, yc, w, h = line.split()
    cid = int(cid)
    xc, yc, w, h = float(xc), float(yc), float(w), float(h)
    # 归一化 → 像素 bbox（左上角 + 宽高）
    x = (xc - w / 2) * IMG_W
    y = (yc - h / 2) * IMG_H
    bw = w * IMG_W
    bh = h * IMG_H
    return cid, x, y, bw, bh


def build_coco(images_list: list, class_names: list) -> dict:
    images, annotations = [], []
    ann_id = 0
    for img_id, name in enumerate(images_list):
        images.append({"id": img_id, "file_name": name, "width": IMG_W, "height": IMG_H})
        label_file = LABELS / (Path(name).stem + ".txt")
        if label_file.exists():
            for line in label_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cid, x, y, bw, bh = parse_yolo_line(line)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                })
                ann_id += 1
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    return {"images": images, "annotations": annotations, "categories": categories}


def main() -> int:
    class_names = load_classes()
    print(f"类别（{len(class_names)}）: {class_names}")

    all_images = sorted(p.name for p in IMAGES.glob("*.jpg"))
    print(f"图片总数: {len(all_images)}")

    rng = random.Random(SEED)
    shuffled = all_images[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * SPLIT)
    train_imgs = sorted(shuffled[:n_train])
    val_imgs = sorted(shuffled[n_train:])

    # COCO 标准目录结构：data_dir/{train2017,val2017}/ + data_dir/annotations/*.json
    ann_dir = OUT / "annotations"
    train_img_dir = OUT / "train2017"
    val_img_dir = OUT / "val2017"
    for d in (ann_dir, train_img_dir, val_img_dir):
        d.mkdir(parents=True, exist_ok=True)

    for name in train_imgs:
        shutil.copy2(IMAGES / name, train_img_dir / name)
    for name in val_imgs:
        shutil.copy2(IMAGES / name, val_img_dir / name)

    train_coco = build_coco(train_imgs, class_names)
    val_coco = build_coco(val_imgs, class_names)
    (ann_dir / "train.json").write_text(json.dumps(train_coco), encoding="utf-8")
    (ann_dir / "val.json").write_text(json.dumps(val_coco), encoding="utf-8")

    print(f"train: {len(train_coco['images'])} 图 / {len(train_coco['annotations'])} 框")
    print(f"val:   {len(val_coco['images'])} 图 / {len(val_coco['annotations'])} 框")
    print(f"输出: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
