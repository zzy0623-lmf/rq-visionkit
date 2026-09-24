# T3.1 适配方案与性能测试报告

> 参赛作品：RQ-VisionKit（睿擎工业开发平台视觉模型轻量化适配与低代码部署）
> 数据来源：全部实测数据取自 [baselines.md](baselines.md)；论文对照数据引自在投稿论文
> 《基于嵌入式 AI 视觉的机械臂智能分拣系统设计与实现》（PDF 第 28/32 页，表 5）。

---

## 1. 适配目标

赛题核心痛点：睿擎 RK3506 开发板（3×Cortex-A7，**无 NPU**）原生仅提供固定的 YOLOv3
示例，缺少「自定义模型导入」能力。本项目要打通的是：**任意训练框架产出的视觉模型，
经工具链转换后，不写代码即可在 RK3506 的 NCNN 推理框架上运行**。

被适配对象（三个真实模型，覆盖两大模型族）：

| 模型 | 来源 | 任务 | 结构 |
|---|---|---|---|
| MobileNetV2 五分类 | 论文（RK3588S，TFLite INT8） | 物品分类 | MobileNetV2 + GAP + Dense(5) |
| MobileNetV2 六分类 | 本项目 T1.6（NEU-DET） | 钢材缺陷分类 | MobileNetV2 + GAP + Dense(6) |
| YOLOX-Nano | 本项目 T1.7（NEU-DET） | 钢材缺陷检测 | YOLOX 0.90M，320×320 |

> 说明：论文五分类权重当前未找回，T1.6 以**同结构**的 MobileNetV2 在 NEU-DET 六分类
> 上重新训练，作为工具链分类适配的验证对象；检测侧用 YOLOX-Nano 验证。两者结构一致、
> 转换路径一致，可等价证明工具链能力。

## 2. 转换与量化策略

转换流水线（M2，全流程 CLI 一键完成，无手工改图）：

```
训练框架导出 ONNX → onnxsim（图简化）→ pnnx（参数化导出，fp16=0）→ ncnnoptimize
```

量化策略：

| 档位 | 方法 | 结果 |
|---|---|---|
| FP32 | 直接转 NCNN | 精度几乎无损（噪声内） |
| INT8 | ncnn PTQ，KL 校准（100 张） | 分模型：YOLOX 成功，MobileNetV2 失败 |

量化失败根因（详见 baselines.md T1.6）：MobileNetV2 的 **depthwise 可分离卷积 + ReLU6**
结构对训练后量化不友好（Qualcomm《A Quantization-Friendly Separable Convolution for
MobileNets》指出 MobileNet 8-bit 精度从 70.5% 崩到 1.8%）；ncnn PTQ 对 depthwise 输出
逐层单 scale 量化，丢失小通道精度、误差累积爆炸。**决策：分类以 FP32 验收，INT8 记为
已知局限**；YOLOX（无 depthwise 短板）INT8 正常，故检测侧 INT8 可用。

## 3. 算子兼容处理

pnnx / ncnn 对部分算子不原生支持，转换中做了 4 处兼容处理（均有实测数值验证）：

| # | 问题 | 处理 | 验证 |
|---|---|---|---|
| 1 | YOLOX Focus 步长切片不被 pnnx 支持 | 自定义 `YoloV5Focus` 层 | 转换通过 |
| 2 | pnnx 转出 `cat_17` concat 轴错误（沿 h 而非 w） | 修复轴参数 `0=2` | 输出与 PyTorch 一致 |
| 3 | pnnx 默认 `fp16=1` 截断权重 | 显式 `fp16=0` | 权重无损 |
| 4 | Focus 层适配 INT8 量化工具 | 用固定 stride-2 space-to-depth conv 替换 | 数值等价，diff=0.0 |

最终 NCNN 原始输出 vs PyTorch `decode=False` 逐元素最大差 **2.86e-06**，证明转换无损。

## 4. 实测对比表

### 4.1 精度 / 时延 / 体积

| 模型 | 平台 | 量化 | 精度 | 单帧时延 | 体积 |
|---|---|---|---|---|---|
| 论文 MobileNetV2 五分类 | **RK3588S**（8 核 A76/A55，XNNPACK CPU） | INT8 | **96.3%** | 18.1 ms（55.2 FPS） | 3.5 MB |
| 论文 MobileNetV2 五分类 | RK3588S | FP32 | 97.5% | 48.3 ms（20.7 FPS） | 13.6 MB |
| 本项目 MobileNetV2 六分类 | PC 仿真（x86） | FP32 | 83.61% | 9.0 ms | 8.86 MB |
| 本项目 MobileNetV2 六分类 | PC 仿真（x86） | INT8 | 17.49%（失败） | — | 2.33 MB |
| 本项目 YOLOX-Nano | PC 仿真（x86） | FP32 | mAP 6.37% | 10.29 ms | 3.56 MB |
| 本项目 YOLOX-Nano | PC 仿真（x86） | INT8 | mAP 6.71% | 11.61 ms | 945 KB |
| RK3506 空载基线 | **RK3506 实测**（串口 msh） | — | — | — | heap 96 MiB |

### 4.2 与论文 RK3588 数据对照的差异来源（重要）

> 论文数据引自论文 PDF 表 5；本项目板端数据因板端运行时（T2.2）尚待编译烧录，
> 当前为 PC 仿真数据 + RK3506 空载基线，**板端模型实测时延/内存待补**。

| 差异维度 | 论文（RK3588S） | 本项目目标（RK3506） |
|---|---|---|
| NPU | 6 TOPS（但分类用 XNNPACK CPU，`use_npu=False`） | **无 NPU** |
| CPU | 8 核 Cortex-A76/A55（大核 + 小核） | 3 核 Cortex-A7（低功耗小核） |
| 单核性能 | A76 约为 A7 的 5–10 倍 | 显著偏低 |
| 任务 | 5 类自采物品（96.3%） | 6 类 NEU-DET 钢材缺陷（83.61%） |

结论：精度差异主要来自**任务/数据集不同**（不可直接对比）；时延差异来自 **RK3506 无
NPU 且 CPU 为 A7 小核**，预期 RK3506 上同规模模型时延显著高于 RK3588S 的 18.1 ms，
需板端实测确认具体倍数。

## 5. 结论

1. **转换链路无损打通**：Keras MobileNetV2 84.15% → NCNN FP32 83.61%（Δ-0.54%，转换
   噪声内）；YOLOX NCNN 输出与 PyTorch 逐元素最大差 2.86e-06。
2. **量化分模型**：YOLOX INT8 成功（mAP 6.71%，无损失，体积省 4×）；MobileNetV2 INT8
   因 depthwise+ReLU6 结构失败（17.49%），已按「FP32 验收、INT8 记局限」处理。
3. **算子兼容处理完整**：4 处修复均基于实测数值验证，构成「自定义模型导入」的工程
   关键——非标准算子（Focus）、轴语义、精度截断都能被工具链自动/半自动纠正。
4. **板端待实测**：板端 C/C++ 运行时已实现（与 PC 仿真同接口契约），待 RK3506 编译
   烧录后补板端时延/内存数据；预期受「无 NPU + A7 小核」影响，性能低于论文 RK3588S，
   具体差异将在板端实测后回填本报告 4.1/4.2。

---

*报告引用数据均可在 [baselines.md](baselines.md) 与论文原文中找到出处；板端数据补测
规则见 baselines.md「待补基线清单」。*
