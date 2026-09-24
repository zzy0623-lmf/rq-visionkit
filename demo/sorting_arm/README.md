# 机械臂分拣（扩展 Demo）

RQ-VisionKit 扩展 Demo：把论文的 MobileNetV2 分类模型经工具链转换、部署后，驱动机械臂分拣
5 类物品。任务书 T2.5，时间盒 ≤4 天，超期或受阻即降级。

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

## 目录内容

```
demo/sorting_arm/
└── model/                # T1.6 论文 MobileNetV2（五分类）转换产物
    ├── report.json       # 转换报告（onnxsim → pnnx → ncnnoptimize，FP32 8.78MB）
    ├── paper_mobilenetv2.sim_pnnx.py   # pnnx 生成的推理脚本
    └── paper_mobilenetv2.sim_ncnn.py   # ncnn 生成的推理脚本
```

> 说明：`report.json` 记录的 `paper_mobilenetv2.onnx` 为随机权重版（论文真实权重未找回），
> 仅用于验证 Keras→ONNX→NCNN 转换链路，不用于实际识别。模型本体（.param/.bin）不入库。
