# 贡献指南

感谢关注 RQ-VisionKit！本项目当前为 2026 上海开源大赛参赛工程，按 Phase 计划推进。

## 开发环境

- Windows + Python 3.10+ + Node 18+ + Git
- Python 依赖：`pip install fastapi uvicorn onnx onnxsim ncnn pyyaml opencv-python numpy pytest`

## 基本流程

1. Fork 并克隆仓库，从 `main` 切功能分支。
2. 提交前本地跑通：`pytest tests/ -v`；前端改动需在 `annotator/web/` 下通过构建检查。
3. 提交信息使用 Conventional Commits（如 `feat: ...` / `fix: ...` / `docs: ...`）。
4. 发起 Pull Request，CI 绿后方可合并。

## 硬性约定

- **许可证**：新增第三方依赖仅接受 Apache-2.0 / MIT / BSD 类许可证，且必须先登记到
  [docs/licenses.md](docs/licenses.md)。
- **平台接口**：RK3506 / 睿擎平台的接口、命令、路径一律以官方 SDK 文档为准，禁止臆造 API。
- **性能数据**：时延 / 内存 / mAP 等实测数据产生后立即追加到
  [docs/baselines.md](docs/baselines.md)，禁止临发布补测。
- **范围红线**：多用户权限、标注多人协同、移动端 App、模型训练 Web 化不在本期范围；
  相关想法请写到 [docs/roadmap.md](docs/roadmap.md)。

## 行为准则

保持友善、就事论事。欢迎 issue 讨论与问题复现报告。
