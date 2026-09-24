# T3.3 演示视频录屏操作清单

> 配套 [video-script.md](video-script.md)（分镜 + 台词）使用。本清单给出「照着点、照着敲」的
> 具体操作，录制六步主流程约 2 分钟，训练/转换两步可快进。

---

## 零、录制前准备（一次性）

### 1. 依赖
```bash
pip install fastapi uvicorn onnx onnxsim ncnn pyyaml opencv-python numpy
cd annotator/web && npm install
```

### 2. 数据
- 准备 **10~20 张 NEU-DET 缺陷图**（crazing/inclusion 等）放一个文件夹，作为步骤 1「采集」的导入源。
- 训练用 COCO 数据（一次性，录屏前先跑好）：
  ```bash
  python scripts/prepare_neudet_coco.py
  ```

### 3. 启动服务（开两个终端）
终端 A —— 后端：
```bash
python -m annotator.server.main
```
看到 `Uvicorn running on http://127.0.0.1:8000`。

终端 B —— 前端：
```bash
cd annotator/web
npm run dev
```
看到 `Local: http://localhost:5173`。

### 4. 开录屏
OBS / Windows 录屏，1920×1080，同时录屏幕 + 麦克风（或后期配音）。

---

## 一、六步操作（照此录制）

### 步骤 1 · 采集（约 25 秒）
1. 浏览器打开 `http://localhost:5173`
2. 点击「批量导入 / 导入图片」，选择准备好的图片文件夹
3. 网格列表显示导入的图片（截图停顿 2 秒给镜头）
- **台词**：「第一步采集。M1 工具批量导入工业缺陷图片，自动建立数据集。」

### 步骤 2 · 标注（约 25 秒）
1. 点进一张图 → 进入 Canvas 标注页
2. 拖拽画一个矩形框，右侧选类别（如 `crazing`）
3. 保存，按 `N` 切下一张（演示标 2~3 张即可）
- **台词**：「第二步标注。网页画布上框出缺陷位置并标注类别，框和类别实时写回。」

### 步骤 3 · 导出（约 15 秒）
1. 回列表页，点「导出数据集」
2. 生成 YOLO 格式 zip（train/val 划分 + classes.txt）
- **台词**：「第三步导出。一键导出 YOLO 训练格式，自动划分训练集和验证集。」

### 步骤 4 · 训练（约 30 秒，**快进**）
1. 终端执行：
   ```bash
   python scripts/train_neudet.py --batch 4
   ```
2. 训练日志滚动（快进），定格在 `best_ckpt.pth` 生成
- **台词**：「第四步训练。用 YOLOX-Nano 轻量检测模型训练，这里快进展示。」

### 步骤 5 · 转换（约 25 秒）
1. 终端执行：
   ```bash
   python scripts/export_neudet.py --mode fp32
   ```
2. 显示 ONNX 导出 → pnnx → ncnnoptimize 流水线，定格在 `yolox_neudet.opt.param/.bin`
- **台词**：「第五步转换。M2 工具把模型一键转成 NCNN 格式，自动处理 Focus、concat 轴等算子兼容问题。」

### 步骤 6 · 部署 + 推理查看（约 20 秒）
1. 前端切到「模型部署」页
2. 选模型目录（或默认 NCNN 输出目录），点「部署」
3. 看板返回检测框叠加图 + 类别 + 置信度 + 单帧时延
- **台词**：「第六步部署推理。低代码控制台一键下发模型并 reload，端侧立即出框，看板显示检测结果和推理时延。」

---

## 二、注意事项

1. **训练/转换耗时**：CPU 训练 20 epoch 较久，录屏时**快进**，只保留「命令 + 开头日志 + 结果文件」三帧即可。
2. **结果镜头**：步骤 6 的检测框叠加图是「六步闭环成立」的视觉证据，务必定格清晰。
3. **机械臂片段**：视频结尾 30 秒扩展 Demo 用 PC 仿真「分类识别 + 结果显示」降级（机械臂硬件未到），
   操作同步骤 6 但选分类模型，台词见 [video-script.md](video-script.md)。
4. **配音**：可录屏时同步念台词，或录完后用剪映/PR 叠加配音 + 字幕。
5. **可复现性**：六步接口与耗时见 [e2e-run.md](e2e-run.md)（T2.4 端到端联调记录），评委可对照复现。
