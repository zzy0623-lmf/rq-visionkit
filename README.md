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
| `demo/` | 主 Demo：NEU-DET 钢材表面缺陷检测；扩展 Demo：机械臂分拣 |
| `scripts/` | 训练、基准测试等脚本 |
| `docs/` | 架构、许可证清单、性能基线、路线图 |

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
系统架构见 [docs/architecture.md](docs/architecture.md)。

## 许可证

本项目代码以 [Apache-2.0](LICENSE) 开源。第三方依赖许可证清单见 [docs/licenses.md](docs/licenses.md)。

## 参赛说明

- 与在投稿论文的成果复用关系：见 [docs/relation-to-paper.md](docs/relation-to-paper.md)
- 贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)
- 变更记录：[CHANGELOG.md](CHANGELOG.md)
