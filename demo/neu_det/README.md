# NEU-DET 钢材表面缺陷检测（主 Demo）

RQ-VisionKit 主 Demo：基于 NEU-DET 数据集，演示「采集标注 → 模型适配转换 → 低代码部署 → 端侧推理」全流程。

## 数据集概况

| 项 | 值 |
|---|---|
| 名称 | NEU-DET（东北大学热轧钢带表面缺陷数据集，检测版） |
| 规模 | 1800 张灰度图（200×200，JPEG 存储为 3 通道）+ 1800 个 VOC XML 标注 |
| 类别 | 6 类，每类 300 张 |
| 标注框 | 共 4189 个，每图 1–9 个目标 |
| 原作者 | Kechen Song, Yunhui Yan（东北大学） |

## 类别表（class_id 与 YOLO 导出一致）

| id | 类别 | 中文 | 标注框数 |
|---|---|---|---|
| 0 | crazing | 龟裂 | 689 |
| 1 | inclusion | 夹杂 | 1011 |
| 2 | patches | 斑块 | 881 |
| 3 | pitted_surface | 点蚀 | 432 |
| 4 | rolled-in_scale | 氧化铁皮压入 | 628 |
| 5 | scratches | 划痕 | 548 |

## 获取方式与来源

- **官方发布页**：[NEU surface defect database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)（官方直链为 Google Drive / 百度网盘）
- **本仓库实际获取途径**：GitHub 镜像 [siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection](https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection)
  （浅克隆后合并其 `IMAGES/` + `Validation_Images/`，恢复为完整 1800 张；标注同理）
- 数据集内容本身不入库（`demo/neu_det/dataset/` 已加入 .gitignore），复现者按上述途径获取后放到本目录。

## 使用条款

数据集作者未指定显式许可证，官方页面注明供**学术研究使用**并要求引用以下论文
（见官方页面 Citation 一节）。本项目仅用于非商业的竞赛与学术研究，不随仓库再分发数据集本体。

- K. Song and Y. Yan, "A noise robust method based on completed local binary patterns
  for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858-864, Nov. 2013.
- Yu He, Kechen Song, Qinggang Meng, Yunhui Yan, "An End-to-end Steel Surface Defect
  Detection Approach via Fusing Multiple Hierarchical Features," *IEEE TIM*, 2020.

## 目录布局

```
demo/neu_det/dataset/
├── IMAGES/              # 1800 张 {class}_{n}.jpg
├── ANNOTATIONS/         # 1800 个 VOC XML
├── yolo_labels/         # analyze_neudet.py 生成的 YOLO 格式候选（class_id xc yc w h）
│   └── classes.txt
└── analysis_report.json # 统计报告（每类数量/尺寸分布/每图目标数/框数）
```

## 复现统计

```bash
python scripts/analyze_neudet.py          # 统计 + 生成 YOLO 标签
python scripts/analyze_neudet.py --no-convert   # 只统计
```

校验要点（T0.3 验收）：1800 张图、6 类×300、标注齐全、bbox 全部合法，脚本退出码 0。
