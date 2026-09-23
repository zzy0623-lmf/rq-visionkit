# 性能基线记录

规则：所有性能数据（时延 / 内存 / mAP）**实测产生后立即追加本文件**，禁止临提交补测。
每条记录须注明：日期、平台（PC 仿真 / RK3506 实测 / QEMU）、模型、量化档位、测试方法。

## 记录表

| 日期 | 平台 | 模型 | 量化 | 单帧时延 | 内存 | mAP/精度 | 备注 |
|---|---|---|---|---|---|---|---|
| 2026-09-16 | PC 仿真（Windows 11, Python 3.10.11, ncnn 1.0.20260526） | squeezenet_v1.1（ncnn 官方示例资产） | FP32 | min 3.85 / mean 5.11 / max 6.80 ms（227×227，预热 3 次计时 20 次） | 未测 | Top-1: rugby ball 0.5969（测试图 messi5.jpg） | T0.2 冒烟，验证 ncnn 链路可用；输出存 docs/images/smoke_ncnn_output.png |
| 2026-09-21 | RK3506 实测（RC-Pi-3506 睿擎派，串口 msh） | 无模型（系统空载基线） | — | — | heap 总量 100,663,296 B（96 MiB），峰值已用 3,672,896 B，可用 97,141,712 B（`free` 命令实测） | — | T0.4：烧录官方 SMP 出厂固件 V1.4.0 后系统空载内存基线；固件版本 8.1.00 / Loader 1.01 / 芯片识别 RK350F；串口 115200 8N1；固件无 `version` 命令（如实记录） |
| 2026-09-21 | RK3506 实测（RC-Pi-3506，RuiChing Studio v1.5.52 构建，BSP 1.4.0 + SMP 固件 V1.4.0） | 官方示例 09_ai_mobilenetv2_yolov3（mobilenetv2_yolov3 NCNN 模型） | 待确认（示例自带） | 待板端运行后补测 | 静态占用：Flash 4,370,481 B（4168 KiB）/ RAM 144,910 B（141.5 KiB）（arm-none-eabi-size 实测，Build Finished 0 errors 0 warnings, 1m20s） | 待运行验证 | app.img 打包成功：start/load 0x04000040，size 0x0042B2B8；此为官方 YOLOv3 基线，后续自定义模型对比用 |
| 2026-09-22 | PC 仿真（Windows 11, Python 3.10.11, ncnn 1.0.20260526） | MobileNetV2 五分类（论文结构：MobileNetV2+GAP+Dropout+Dense(5)，**权重随机**，T1.6 降级方案） | FP32 | 1.81 ms（224×224，单次） | 未测 | 无（随机权重，softmax 均匀分布 [0.2]*5，仅验证链路） | T1.6 降级方案：验证 Keras→ONNX(tf2onnx)→NCNN(pnnx) 全链路可用；模型 8.8MB；真实权重找回后替换重跑即正式基线 |
| 2026-09-22 | PC 训练（Windows 11, PyTorch 2.13.0+cpu，YOLOX） | YOLOX-Nano（NEU-DET 检测 6 类，depth=0.33/width=0.25，0.90M 参数） | FP32（训练权重） | 训练 iter 0.5s（非推理时延，推理待转 NCNN 后补测） | 训练 mem 14GB | **mAP@0.5:0.95 = 6.33%**（best）；per-class AP@0.5:0.95：crazing 4.08 / inclusion 7.59 / patches 3.28 / pitted_surface 14.68 / rolled-in_scale 6.57 / scratches 1.78；AR：crazing 32.9 / pitted_surface 45.3 | T1.7：CPU 降配训练（320×320，20 epoch），best_ckpt.pth 已存；mAP 偏低属降配预期（正常 300 epoch，此处 20 epoch 仅验证流程）；训练/eval 的 CUDA 硬编码已修（trainer/data_prefetcher/coco_evaluator） |
| 2026-09-22 | PC 仿真（Windows 11, Python 3.10.11, ncnn 1.0.20260526） | YOLOX-Nano（NEU-DET 检测 6 类，NCNN） | FP32 | mean 10.29 ms（320×320，360 张 val 逐张计时，min 8.73 / max 12.40） | 未测 | **mAP@0.5:0.95 = 6.37%**，mAP@0.5 = 19.94%（与训练 6.33% 一致，FP32 噪声内） | T1.7 NCNN 推理验证通过（360 张 val 全量）；ncnn 原始输出 vs PyTorch decode=False 逐元素最大差 2.86e-06。转换 3 处修复：① Focus 步长切片不被 pnnx 支持→自定义 YoloV5Focus 层 ② pnnx 转出 cat_17 concat 轴错误（沿 h 非 w）→改 0=2 ③ pnnx 默认 fp16=1 截断权重→加 fp16=0 |
| 2026-09-22 | PC 仿真（Windows 11, Python 3.10.11, ncnn 1.0.20260526） | YOLOX-Nano（NEU-DET，space-to-depth conv 替换 Focus 版） | FP32 vs INT8 | FP32 mean 11.29 ms / INT8 mean 11.61 ms（320×320，360 张 val 逐张） | 未测 | FP32 **6.37%** / INT8 **6.71%**（mAP@0.5:0.95，INT8 无精度损失，Δ+0.34% 噪声内） | T1.7 INT8：KL 校准（100 张 train）；**PC x86 上 INT8 无提速（0.97×，反慢 3%，requantize 开销抵消）**；INT8 体积 945KB vs FP32 3.56MB（省 4×）；Focus 用固定 stride-2 space-to-depth conv 替换（数值等价，diff=0.0）以适配量化工具；RK3506(Cortex-A7 无 INT8 dot 指令)预期也无提速，待板端实测 |
| 2026-09-22 | PC 训练（Windows 11, Python 3.10.11, TensorFlow 2.21.0 + Keras 3.12.4） | MobileNetV2 六分类（NEU-DET 缺陷：crazing/inclusion/patches/pitted_surface/rolled-in_scale/scratches；MobileNetV2(imagenet)+GAP+Dropout(0.2)+Dense(6, softmax)） | FP32（训练） | 未测（训练基线） | 未测 | **test accuracy 84.15%**（loss 1.0041） | T1.6：两阶段训练（stage1 冻结骨干 10ep lr=1e-3；stage2 解冻 block13-16 5ep lr=1e-5）；NEU-DET 1800 张 200×200 钢材缺陷图，划分 1438/179/183 |
| 2026-09-22 | PC 仿真（Windows 11, Python 3.10.11, ncnn 1.0.20260526） | MobileNetV2 六分类（NEU-DET，NCNN） | FP32 | 未测 | 未测（模型 8.86MB） | **accuracy 83.61%**（test 183 张） | T1.6 NCNN 转换验证：Keras 84.15% → NCNN 83.61%（Δ-0.54% 转换噪声内）；去 Lambda 预处理改用 ncnn mean/norm（mean=127.5, norm=1/127.5） |
| 2026-09-22 | PC 仿真（Windows 11, Python 3.10.11, ncnn 1.0.20260526） | MobileNetV2 六分类（NEU-DET，NCNN） | INT8（KL 校准） | 未测 | 未测（模型 2.33MB） | **accuracy 17.49%**（≈ 6 类随机 16.67%，未达 ≤2% 验收线） | T1.6 INT8 量化失败：根因是 MobileNetV2 深度可分离卷积+ReLU6 的量化困难结构（Qualcomm《A Quantization-Friendly Separable Convolution for MobileNets》：MobileNet 8-bit 从 70.5% 崩到 1.8%），ncnn PTQ 对 depthwise 输出逐层单 scale 量化丢失小通道精度、误差累积爆炸；已排除 mean/norm 语义、输入预处理、权重 NaN、环境损坏（T1.7 YOLOX INT8 正常 6.71%）；ReLU6→ReLU 仅 17%→26% 部分缓解；ncnn method=eq 因 Keras MobileNetV2 Conv1 天生 7 个 near-zero 通道崩溃。**方案 1：T1.6 以 FP32 83.61% 验收，INT8 记为已知局限；如需达标需走 QAT 或换结构** |

## 待补基线清单

- [ ] RK3506 官方 YOLOv3 示例时延/内存（板到手当天，T0.4）
- [ ] PC 仿真：主 Demo 模型 FP32 vs INT8 时延/mAP（T1.4/T1.7）
- [x] 论文 MobileNetV2：FP32 基线 84.15% → NCNN FP32 83.61%；INT8 17.49%（量化失败未达 ≤2%，已按方案 1 以 FP32 验收，详见上表）
- [ ] RK3506 实测：自定义模型时延/内存（T2.2）
