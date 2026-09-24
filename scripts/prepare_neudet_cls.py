# -*- coding: utf-8 -*-
"""T1.6 数据准备：NEU-DET COCO → 6 类图像分类结构。

将 T1.7 用的 NEU-DET 检测数据（COCO 格式）套用为「6 类钢材缺陷分类」任务，
产出 image_dataset_from_directory 所需的结构：

    data/dataset/{train,val,test}/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}/

规则：
    - 每张图取「框数量最多」的缺陷作为主要类别（NEU-DET 少数图含多类标注）。
    - 合并 train2017(1440) + val2017(360) = 1800 张，按类别分层 8:1:1 划分。
"""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# 路径环境变量驱动（默认值向后兼容）：
#   RQ_NEUDET_COCO_DIR      NEU-DET COCO 数据目录（默认 C:\neudet_coco）
#   RQ_SORTING_DATASET_DIR  机械臂分拣分类数据集输出目录
SRC = Path(os.environ.get("RQ_NEUDET_COCO_DIR", r"C:\neudet_coco"))
DST = Path(os.environ.get(
    "RQ_SORTING_DATASET_DIR",
    r"C:\Users\zzyly\Desktop\嵌入式AI视觉机械臂智能分拣系统V1.0_源代码\data\dataset",
))
CATS = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def load_entries(split: str):
    """读取一个 split 的 (文件路径, 主要类别id) 列表。"""
    ann = json.load(open(SRC / "annotations" / f"{split}.json", encoding="utf-8"))
    id2file = {img["id"]: img["file_name"] for img in ann["images"]}
    img2cats = defaultdict(list)
    for a in ann["annotations"]:
        img2cats[a["image_id"]].append(a["category_id"])
    img_dir = SRC / f"{split}2017"
    entries = []
    for img_id, cats in img2cats.items():
        main_cat = Counter(cats).most_common(1)[0][0]  # 框数量最多的缺陷类别
        entries.append((img_dir / id2file[img_id], main_cat))
    return entries


def split_stratified(entries, train_ratio=0.8, val_ratio=0.1):
    """按类别分层划分 train/val/test，返回 {split: [entries]}。"""
    by_cat = defaultdict(list)
    for e in entries:
        by_cat[e[1]].append(e)
    splits = {"train": [], "val": [], "test": []}
    for cat, es in by_cat.items():
        es = sorted(es, key=lambda x: x[0].name)
        n = len(es)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"] += es[:n_train]
        splits["val"] += es[n_train:n_train + n_val]
        splits["test"] += es[n_train + n_val:]
    return splits


def main():
    entries = load_entries("train") + load_entries("val")
    splits = split_stratified(entries)
    print(f"总图 {len(entries)} 张：train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    for split, es in splits.items():
        # 清空旧目录
        split_dir = DST / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for file_path, cat in es:
            cls_dir = split_dir / CATS[cat]
            cls_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, cls_dir / file_path.name)

    # 打印各类别数量
    print("\n类别分布：")
    for cat in CATS:
        counts = {s: sum(1 for _, c in splits[s] if CATS[c] == cat) for s in splits}
        print(f"  {cat:20s} train={counts['train']:3d} val={counts['val']:3d} test={counts['test']:3d}")
    print("\n完成 ->", DST)


if __name__ == "__main__":
    main()
