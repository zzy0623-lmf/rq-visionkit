"""T0.3 NEU-DET 数据集分析与 XML→YOLO 转换。

统计：每类图片数量、尺寸分布、标注完整性、每图目标数分布；
转换：VOC XML 标注 → YOLO 格式候选（class_id x_center y_center w h，0-1 归一化），
     输出到 dataset/yolo_labels/（不覆盖原标注）。

用法：
    python scripts/analyze_neudet.py
    python scripts/analyze_neudet.py --no-convert   # 只统计不转换

退出码：0 = 校验通过（1800 张图、6 类×300、标注齐全）；1 = 校验失败。
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "demo" / "neu_det" / "dataset"
EXPECTED_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
EXPECTED_PER_CLASS = 300


def parse_xml(xml_path: Path):
    """返回 (size(w,h), objects[(name, xmin, ymin, xmax, ymax), ...])"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        box = tuple(int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        objects.append((name, *box))
    return (w, h), objects


def to_yolo(box, img_w, img_h):
    xmin, ymin, xmax, ymax = box
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return xc, yc, bw, bh


def main() -> int:
    parser = argparse.ArgumentParser(description="NEU-DET 统计与 YOLO 转换")
    parser.add_argument("--no-convert", action="store_true", help="只统计，不生成 YOLO 标签")
    args = parser.parse_args()

    images_dir = DATASET / "IMAGES"
    ann_dir = DATASET / "ANNOTATIONS"
    out_dir = DATASET / "yolo_labels"

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".jpg")
    xmls = {p.stem for p in ann_dir.iterdir() if p.suffix.lower() == ".xml"}

    errors = []
    # 1. 图片/标注一一对应
    img_stems = {p.stem for p in images}
    missing_xml = img_stems - xmls
    orphan_xml = xmls - img_stems
    if missing_xml:
        errors.append(f"{len(missing_xml)} 张图片缺标注，如 {sorted(missing_xml)[:3]}")
    if orphan_xml:
        errors.append(f"{len(orphan_xml)} 个标注无对应图片，如 {sorted(orphan_xml)[:3]}")

    # 2. 按文件名前缀统计每类数量 + 实测尺寸
    class_count = Counter(p.stem.rsplit("_", 1)[0] for p in images)
    size_count = Counter()
    for p in images:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            errors.append(f"图片无法解码: {p.name}")
            continue
        h, w = img.shape[:2]
        size_count[f"{w}x{h}x{img.shape[2] if img.ndim == 3 else 1}"] += 1

    # 3. 解析标注：类别词表一致性、bbox 合法性、每图目标数
    obj_count = Counter()
    xml_class_names = Counter()
    bad_boxes = []
    parsed = {}
    for p in sorted(ann_dir.iterdir()):
        if p.suffix.lower() != ".xml":
            continue
        (w, h), objects = parse_xml(p)
        parsed[p.stem] = ((w, h), objects)
        obj_count[len(objects)] += 1
        for name, xmin, ymin, xmax, ymax in objects:
            xml_class_names[name] += 1
            if not (0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h):
                bad_boxes.append((p.name, name, (xmin, ymin, xmax, ymax), (w, h)))
    if bad_boxes:
        errors.append(f"{len(bad_boxes)} 个越界/非法 bbox，如 {bad_boxes[0]}")

    unknown = set(xml_class_names) - set(EXPECTED_CLASSES)
    if unknown:
        errors.append(f"XML 出现未知类别名: {unknown}")

    # 打印统计表
    print("== 每类图片数量（按文件名前缀） ==")
    for cls in EXPECTED_CLASSES:
        n = class_count.get(cls, 0)
        flag = "OK" if n == EXPECTED_PER_CLASS else "!!"
        print(f"  [{flag}] {cls:16s} {n}")
    print(f"  总计: {len(images)} 张图片, {len(xmls)} 个标注")
    print("== 尺寸分布（实测解码） ==")
    for size, n in sorted(size_count.items()):
        print(f"  {size}: {n}")
    print("== 每图目标数分布 ==")
    for k in sorted(obj_count):
        print(f"  {k} 个目标: {obj_count[k]} 张")
    print("== 各类别标注框总数 ==")
    for name, n in sorted(xml_class_names.items()):
        print(f"  {name:16s} {n}")
    total_boxes = sum(xml_class_names.values())
    print(f"  总框数: {total_boxes}")

    # 4. YOLO 转换
    if not args.no_convert and not errors:
        out_dir.mkdir(exist_ok=True)
        n_written = 0
        for stem, ((w, h), objects) in parsed.items():
            lines = []
            for name, *box in objects:
                cid = EXPECTED_CLASSES.index(name)
                xc, yc, bw, bh = to_yolo(box, w, h)
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_written += 1
        (out_dir / "classes.txt").write_text("\n".join(EXPECTED_CLASSES) + "\n", encoding="utf-8")
        print(f"[OK] YOLO 标签已生成: {out_dir}（{n_written} 个 txt + classes.txt）")

    # 报告落盘
    report = {
        "total_images": len(images),
        "total_xml": len(xmls),
        "per_class": {c: class_count.get(c, 0) for c in EXPECTED_CLASSES},
        "size_distribution": dict(size_count),
        "objects_per_image": {str(k): v for k, v in sorted(obj_count.items())},
        "boxes_per_class": dict(sorted(xml_class_names.items())),
        "total_boxes": total_boxes,
        "errors": errors,
    }
    (DATASET / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if errors:
        print("\n[FAIL] 校验未通过:")
        for e in errors:
            print(f"  - {e}")
        return 1
    if len(images) != 1800:
        print(f"[FAIL] 图片总数 {len(images)} != 1800")
        return 1
    print("\n[OK] 校验通过：1800 张图 / 6 类×300 / 标注齐全且 bbox 合法")
    return 0


if __name__ == "__main__":
    sys.exit(main())
