# 机械臂分拣（扩展 Demo）

RQ-VisionKit 扩展 Demo：把论文的 MobileNetV2 分类模型经工具链转换、部署后，驱动机械臂分拣
5 类物品。任务书 T2.5，时间盒 ≤4 天，超期或受阻即降级。

> **补充（2026-10-23）**：本目录现已补充从《软著鉴别材料·源代码》PDF 还原的完整机械臂分拣系统
> 源代码（RK3588 / TensorFlow Lite / PCA9685 版本），即论文与软件著作权所述系统的原始实现，
> 详见下方「完整源代码（RK3588）」。注意：它与 RQ-VisionKit 的 RK3506 工具链 Demo 属两个平台，
> 前者为 TFLite INT8 + XNNPACK，后者为 NCNN + RT-Thread。

## 当前状态（降级为「分类识别 + 结果显示」）

按任务书 T2.5 的降级条款，本 Demo 已裁剪为**分类识别 + 结果显示**，不做机械臂抓取：

- **论文权重未找回**：论文的五分类模型（TFLite INT8）真实权重未能复现，只有随机权重版，
  无法演示「识别 5 类物品」的真实分类效果。
- **机械臂硬件未到位**：HSV 定位、透视变换、逆运动学、舵机控制依赖真实机械臂 + 摄像头 + 舵机。
- 因此用**现成有效**的 NEU-DET 六分类 MobileNetV2（accuracy 83.61%）替代演示，证明工具链
  **不止适配检测模型（YOLOX），也能适配分类模型（softmax 输出）**——这是「自定义模型导入」
  能力的关键证据。

## 已落地能力（T2.5）

M4 运行时新增 `task=classify` 分支，支持分类模型：

- 预处理：`resize → BGR2RGB → NCHW float32`，mean/norm 由 ncnn 层完成（`mean=127.5, norm=1/127.5`）
- 后处理：模型输出已含 softmax（`Dense(softmax)`），直接取概率降序返回 top-k
- 经 M3 部署控制台下发（`/deploy`）+ 结果看板（`/deploy/infer` 返回类别 + 概率）

验证结果（PC 仿真）：`crazing` 缺陷图 → top-1 `crazing`（score 0.9975），单帧 9.0ms。
详见 [docs/baselines.md](../docs/baselines.md) 的 T2.5 记录。

## 完整源代码（RK3588）

以下为论文/软著所述「嵌入式AI视觉机械臂智能分拣系统」的完整源码，从软著 PDF 还原：

| 文件 | 说明 |
| --- | --- |
| `main.py` | 系统主程序入口（CLI、标定、单次/自动分拣、性能测试） |
| `sorting_controller.py` | 分拣主控编排（有限状态机 + 统计） |
| `vision.py` | 视觉处理（HSV 背景分割、轮廓定位） |
| `classifier.py` | MobileNetV2 INT8 分类推理（TFLite） |
| `ik_solver.py` | 四自由度机械臂几何法逆运动学 |
| `servo_controller.py` | 舵机控制（⚠️ 方法体在 PDF 中被截断） |
| `config.py` | 系统配置（⚠️ 类声明与 __init__ 在 PDF 中缺失） |
| `calibration.py` | 相机标定 / 四点透视变换（重构） |
| `hal.py` | 硬件抽象层：摄像头 + PCA9685（重构） |
| `logger.py` | 日志管理（单例） |
| `utils.py` | 通用工具（数学/坐标/图像/计时/线程安全） |
| `train.py` | 模型训练 + INT8 量化 + 数据采集 |
| `setup.sh` | 香橙派5 一键部署脚本 |
| `requirements.txt` | Python 运行依赖 |

> 缺失说明：`servo_controller.py`（ServoController 方法体）与 `config.py`（class Config 声明
> 及 __init__）因处于软著 PDF 的「前30页+后30页」之间的被截断段，已尽力还原但仍不完整；
> `calibration.py`、`hal.py` 为**依据《软著设计说明书》接口描述 + 调用点重构**（非原始代码，
> 各文件头已注明）。完整原始实现如有需要可向作者索取。

## 目录内容

```
demo/sorting_arm/
├── main.py / sorting_controller.py / vision.py   # RK3588 分拣系统源码（软著PDF还原）
├── classifier.py / ik_solver.py / servo_controller.py / config.py
├── calibration.py / hal.py / logger.py / utils.py / train.py
├── setup.sh / requirements.txt
└── model/                # T1.6 论文 MobileNetV2（五分类）转换产物
    ├── report.json       # 转换报告（onnxsim → pnnx → ncnnoptimize，FP32 8.78MB）
    ├── paper_mobilenetv2.sim_pnnx.py   # pnnx 生成的推理脚本
    └── paper_mobilenetv2.sim_ncnn.py   # ncnn 生成的推理脚本
```

> 说明：`report.json` 记录的 `paper_mobilenetv2.onnx` 为随机权重版（论文真实权重未找回），
> 仅用于验证 Keras→ONNX→NCNN 转换链路，不用于实际识别。模型本体（.param/.bin）不入库。
