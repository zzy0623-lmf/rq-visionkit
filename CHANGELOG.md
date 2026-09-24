# Changelog

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循语义化版本。

## [1.0.0] - 2026-09-24

正式发布：全流程工具链 + 板端运行时 + 材料冲刺。

### Added

- M4 板端运行时（RK3506，C/C++ + NCNN）：检测 + 分类，与 PC 仿真同接口契约（T2.2）。
- 适配方案与性能测试报告 `docs/adaptation-report.md`（T3.1）。
- 作品介绍文档 `docs/project-intro.md` + PDF（T3.2）。
- 演示视频分镜与配音脚本 `docs/video-script.md`（T3.3）。

### Changed

- README 新增「功能模块导航」与实测可用的「快速开始」（全量测试 51 passed）。
- M4 板端运行时已在 RK3506 官方 SDK 工程交叉编译通过（`app.elf` 10.4MB / `app.bin` 4.6MB），
  修复 imgproc/highgui 头文件、不存在的 `rt_memory_info`、回调返回类型等编译问题。

## [1.0.0-rc] - 2026-09-24

Phase 2 出口：低代码部署控制台 + 端侧运行时 + 端到端联调。

### Added

- M4 端侧推理运行时（PC 仿真版，ncnn-python）：`POST /infer`、`GET /health`、`POST /model/reload`，模型文件与代码分离（T2.1）。
- M3 低代码部署控制台：`/deploy` 路由组 + 前端部署页（配置表单 + 结果看板），本地 / SSH（OpenSSH subprocess）下发（T2.3）。
- M4 运行时分类任务支持：`task=classify`（resize + BGR2RGB + mean/norm，softmax 输出 top-k），扩展 Demo 降级为「分类识别 + 结果显示」（T2.5）。
- 端到端联调记录 `docs/e2e-run.md`（六步主流程，T2.4）。

### Changed

- scripts 训练/导出路径改为环境变量驱动（`RQ_TOOLS_DIR` / `RQ_YOLOX_OUTPUT_DIR` / `RQ_NEUDET_COCO_DIR` / `RQ_SORTING_DATASET_DIR`）。
- 部署下发改用系统 OpenSSH 子进程，替代 LGPL 许可证的 paramiko（详见 docs/licenses.md）。

### Fixed

- 换模型部署时旧模型残留导致运行时 `glob("*.opt.param")` 加载到旧模型：部署前清理目标目录旧模型文件。

## [0.2.0] - 2026-09-23

Phase 1 出口：采集标注工具 + 模型转换流水线。

### Added

- M1 采集标注工具：FastAPI 后端（导入/采集/标注/导出 YOLO 数据集）+ Vue3 Canvas 标注前端（T1.1–T1.3）。
- M2 模型转换流水线：ONNX → NCNN INT8 一键转换 CLI + 算子兼容扫描（T1.4–T1.5）。
- NEU-DET 数据分析与 YOLO 转换脚本（T0.3）、NCNN 环境冒烟测试（T0.2）。
- YOLOX 检测训练与导出脚本（T1.7）、论文 MobileNetV2 六分类模型适配（T1.6）。

### Fixed

- pnnx 对检测模型 decode-head concat 轴映射错误（0=1 → 0=2）。

## [0.1.0] - 2026-09-15

### Added

- 初始化仓库目录结构（annotator / converter / deployer / runtime / demo / scripts / tests）。
- 合规文件：Apache-2.0 LICENSE、README、CONTRIBUTING、docs/licenses.md、docs/architecture.md、docs/roadmap.md、docs/relation-to-paper.md、docs/baselines.md。
- GitHub Actions CI：Python 包跑 pytest；前端存在 package.json 时跑构建检查。
- tests/test_repo_layout.py 目录结构自检。
