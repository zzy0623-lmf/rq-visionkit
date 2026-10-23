# RQ-VisionKit

面向睿擎工业开发平台（RK3506 开发板）的视觉模型全流程工具链：
**采集标注 → 模型适配转换 → 低代码部署 → 端侧推理**。

睿擎平台原生仅提供固定的 YOLOv3 示例，缺少数据采集标注、自定义模型导入与低代码部署能力。
RQ-VisionKit 补齐这条链路，让自定义视觉模型不写代码即可在 RK3506（无 NPU，3×Cortex-A7）上跑起来。

> 2026 上海开源大赛 · AI+工业软件赛道 · 睿赛德命题参赛作品。

## 仓库结构

| 目录 | 说明 |
|---|---|
| `annotator/` | M1 采集与标注工具（FastAPI 后端 + Vue3 单页前端） |
| `converter/` | M2 模型适配与一键转换（ONNX → NCNN INT8，含算子兼容扫描） |
| `deployer/` | M3 低代码部署控制台 |
| `runtime/` | M4 端侧推理运行时（`pc_sim/` PC 仿真版，`board/` RK3506 板端版） |
| `demo/` | 主 Demo：NEU-DET 钢材表面缺陷检测 |
| `sorting_arm/` | 扩展 Demo：机械臂智能分拣系统完整源码（RK3588，从软著 PDF 还原） |
| `scripts/` | 训练、基准测试等脚本 |
| `docs/` | 架构、许可证清单、性能基线、路线图、项目流程、板端部署调试手册 |

## 机械臂智能分拣系统（sorting_arm）

`sorting_arm/` 是本作品的扩展 Demo，也是一套完整的**嵌入式 AI 视觉机械臂智能分拣系统**——
论文与软件著作权所述系统的原始实现（RK3588 版本）。它针对人工分拣效率低、成本高，以及
工业级视觉分拣设备售价高昂（5~30 万元）难以向教学与小型产线推广的痛点，用 **1365 元**物料成本
实现 5 类物品的实时视觉识别与机械臂自动抓取分拣。

**核心指标（实测）**

| 指标 | 数值 |
| --- | --- |
| 物品分类准确率 | **96.3%** |
| 综合抓取成功率 | **88.8%** |
| 单次抓取-放置循环 | **6.8 s** |
| 模型单帧推理速度 | **18.1 ms（55.2 FPS）** |
| 整机物料成本 | **1365 元** |
| 坐标映射精度 | **±2 mm** |

**系统架构**：分层模块化设计，自底向上为执行层（四自由度机械臂 + 平行夹爪）、控制层（PCA9685
舵机驱动板）、计算层（香橙派5 / RK3588S）、传感层（罗技 C270 USB 摄像头），数据流为
传感 → 计算 → 控制 → 执行。

**关键技术**

- **视觉识别**：MobileNetV2 轻量级分类网络 + HSV 颜色空间背景分割 + 四点透视变换坐标映射
- **模型量化**：TensorFlow Lite INT8 全整数量化，模型 13.6MB → 3.5MB，推理 20.7 → 55.2 FPS
- **运动学**：四自由度机械臂几何法解析逆运动学，ARM CPU 上亚毫秒级求解
- **软件架构**：Python 3.10 + OpenCV，六状态有限状态机（FSM）主控，模块化分层

详细源码说明见 [sorting_arm/README.md](sorting_arm/README.md)。

## 功能模块导航

四个功能模块（M1–M4）对应赛题完整工具链，各模块独立可测：

| 模块 | 目录 | 交付内容 | 测试入口 |
|---|---|---|---|
| M1 采集与标注 | `annotator/` | FastAPI 后端 + Vue3 标注前端（导入/标注/导出 YOLO 数据集） | `pytest annotator/tests -v` |
| M2 模型转换 | `converter/` | ONNX→NCNN 一键转换 CLI（含算子兼容扫描） | `pytest converter/tests -v` |
| M3 部署控制台 | `deployer/` | 低代码部署（打包 / 本地 / SSH 下发 / reload / 结果看板） | `pytest deployer/tests -v` |
| M4 端侧运行时 | `runtime/` | `pc_sim/` PC 仿真（Python），`board/` RK3506 板端（C/C++，检测+分类） | `pytest runtime/pc_sim/tests -v` |

## 快速开始

环境要求：Windows + Python 3.10+ + Node 18+ + Git。

```bash
# 1. 安装 Python 依赖
pip install fastapi uvicorn onnx onnxsim ncnn pyyaml opencv-python numpy pytest

# 2. 运行全部测试（M1/M2/M3/M4 + 仓库结构）
pytest tests/ annotator/tests converter/tests deployer/tests runtime/pc_sim/tests -v
```

六步主流程（采集→标注→训练→转换→部署→推理查看）的端到端复现记录见
[docs/e2e-run.md](docs/e2e-run.md)；性能基线见 [docs/baselines.md](docs/baselines.md)；
系统架构见 [docs/architecture.md](docs/architecture.md)；
项目完成流程见 [.trae/skills/rq-visionkit-workflow/SKILL.md](.trae/skills/rq-visionkit-workflow/SKILL.md)；
RK3506 板端部署调试排查手册见 [docs/board-bringup-runbook.md](docs/board-bringup-runbook.md)。

## 项目成果（论文 / 软著）

**学术论文**

- 题目：*Design and Implementation of an Embedded AI Vision-Based Robotic Arm Intelligent Sorting System*
- 作者：ZHAO Zheyuan（赵喆远，西安航空学院）
- 会议：2026 年第 5 届先进材料与装备制造国际会议（AMEM 2026）
- 时间地点：武汉，2026 年 12 月 11–13 日
- 录用形式：**发表 + 口头报告**
- 检索：已被 **EI Compendex** 检索

**软件著作权**

- 名称：嵌入式AI视觉机械臂智能分拣系统 V1.0
- 编写人：赵喆远
- 完成日期：2026 年 9 月
- 材料：设计说明书 + 源代码（鉴别材料）

## 许可证

本项目代码以 [Apache-2.0](LICENSE) 开源。第三方依赖许可证清单见 [docs/licenses.md](docs/licenses.md)。

## 参赛说明

- 与已录用论文（AMEM 2026，EI Compendex 检索）的成果复用关系：见 [docs/relation-to-paper.md](docs/relation-to-paper.md)
- 贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)
- 变更记录：[CHANGELOG.md](CHANGELOG.md)
