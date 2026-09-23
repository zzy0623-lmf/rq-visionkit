# 第三方依赖许可证清单

规则：**仅允许 Apache-2.0 / MIT / BSD 类许可证**。新增依赖必须先登记本表再引入。
状态含义：✅ 已引入 / 📌 计划引入（任务书指定，尚未安装）/ ⚠️ 许可证不符合要求，待决策。

## Python 依赖

| 依赖 | 版本 | 许可证 | 用途 | 状态 |
|---|---|---|---|---|
| pytest | 9.1.1 | MIT | 测试框架（CI） | ✅ |
| ncnn / ncnn-python | 1.0.20260526 | BSD-3-Clause | PC 仿真推理框架 | ✅ |
| opencv-python | 5.0.0 | MIT（封装层；OpenCV 4.x 核心为 Apache-2.0） | 图像读写/预处理 | ✅ |
| numpy | 2.2.6 | BSD-3-Clause | 数值计算 | ✅ |
| portalocker | 4.3.2 | BSD-3-Clause | ncnn 传递依赖（文件锁） | ✅（传递） |
| fastapi | 0.141.1 | MIT | M1/M3 后端框架 | ✅ |
| uvicorn | 0.53.0 | BSD-3-Clause | ASGI 服务器 | ✅ |
| python-multipart | 0.0.32 | Apache-2.0 | multipart 上传解析（导入/采集接口） | ✅ |
| pydantic / pydantic-core | 2.13.5 / 2.46.5 | MIT | fastapi 传递依赖（数据校验） | ✅（传递） |
| starlette | 1.6.0 | BSD-3-Clause | fastapi 传递依赖（ASGI 工具） | ✅（传递） |
| httpx | 0.28.x | BSD-3-Clause | 测试用 TestClient 依赖 | ✅（测试） |
| onnx | 1.23.0 | Apache-2.0 | 模型图解析/转换 | ✅ |
| onnxsim (onnx-simplifier) | 0.7.3 | MIT | ONNX 图简化 | ✅ |
| tf2onnx | 1.17.0 | Apache-2.0 | Keras/SavedModel → ONNX（T1.6 论文模型导出） | ✅ |
| tensorflow | 2.21.0 | Apache-2.0 | 加载 Keras 模型（CPU，T1.6） | ✅ |
| keras | 3.12.4 | Apache-2.0 | tensorflow 传递依赖（Keras 3 多后端） | ✅（传递） |
| torch (PyTorch) | 2.13.0+cpu | BSD-3-Clause | YOLOX 训练框架（T1.7，CPU） | ✅ |
| torchvision | 0.26 | BSD-3-Clause | torch 图像变换（传递） | ✅（传递） |
| loguru | 0.7.3 | MIT | YOLOX 日志 | ✅ |
| pycocotools | 2.0.11 | BSD-2-Clause | COCO 标注解析/评估（YOLOX） | ✅ |
| tqdm | 4.67 | MIT（与 MPL-2.0 双许可，取 MIT 侧） | YOLOX 进度条 | ✅ |
| thop | 0.1.1 | MIT | 模型 FLOPs 统计（YOLOX） | ✅ |
| ninja | 1.13.2 | Apache-2.0 | torch 编译加速（传递） | ✅（传递） |
| tabulate | 0.10.0 | MIT | YOLOX 表格输出 | ✅ |
| psutil | 7.0 | BSD-3-Clause | YOLOX 资源监控 | ✅ |
| tensorboard | 2.21.0 | Apache-2.0 | YOLOX 训练日志 | ✅ |
| pyyaml | — | MIT | 配置文件解析 | 📌 |
| paramiko | — | **LGPL-2.1-or-later** | SSH 下发部署包（T2.3） | ⚠️ 见下方决策 |

## 前端依赖（T1.2 起）

| 依赖 | 版本 | 许可证 | 用途 | 状态 |
|---|---|---|---|---|
| Vue 3 | ^3.4.21 | MIT | 标注前端框架 | ✅ |
| Vite | ^7.3.6 | MIT | 前端构建工具 | ✅ |
| @vitejs/plugin-vue | ^6.0.9 | MIT | Vite 的 Vue SFC 编译插件 | ✅ |

## 模型与工具链

| 项 | 许可证 | 说明 | 状态 |
|---|---|---|---|
| YOLOX | Apache-2.0 | 主 Demo 检测模型训练选型（T1.7 已引入，clone 到 tools/yolox） | ✅ |
| Ultralytics YOLO | AGPL-3.0 | **默认不使用**；若使用须在此声明并给出理由 | ❌ 规避 |
| NCNN 命令行工具（pnnx / ncnnoptimize / ncnn2table / ncnn2int8） | BSD-3-Clause | M2 转换流水线（ONNX→NCNN、优化、量化） | ✅ |

## 测试用资产（不入库，scripts/assets/）

| 资产 | 来源 | 说明 |
|---|---|---|
| squeezenet_v1.1.param/.bin | github.com/nihui/ncnn-assets（Tencent NCNN 官方资产仓库） | T0.2 冒烟模型 |
| synset_words.txt | github.com/Tencent/ncnn examples/ | ImageNet 1000 类标签 |
| messi5.jpg | github.com/opencv/opencv samples/data/ | T0.2 测试图片 |

## 数据集

| 数据集 | 来源 | 使用条款 | 状态 |
|---|---|---|---|
| NEU-DET（东北大学钢材表面缺陷数据集） | 官方页：faculty.neu.edu.cn/songkechen（直链 Google Drive/百度网盘）；本仓库经 GitHub 镜像 siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection 获取完整 1800 张 | 无显式许可证；作者注明供学术研究使用并要求引用其论文。本项目仅非商业竞赛/研究用途，**数据集本体不入库、不再分发**，详见 demo/neu_det/README.md | ✅ |

## ⚠️ 待决策：paramiko 许可证冲突

任务书第 1 节依赖清单含 `paramiko`，但其许可证为 **LGPL-2.1-or-later**，不在
Apache-2.0 / MIT / BSD 允许范围内。候选替代（T2.3 前必须定）：

1. **改用 OpenSSH 命令行**：通过 `subprocess` 调用系统 `ssh`/`scp` 下发部署包。零依赖、零许可证风险。Windows 10+ 自带 OpenSSH 客户端。
2. **换库**：如 `asyncssh`（EPL-2.0，同样不在白名单）——无合适的 MIT/BSD 纯 Python SSH 库。
3. **为例外申请**：向评审说明 LGPL 动态链接用法。不推荐，徒增合规解释成本。

**推荐方案 1**（OpenSSH + subprocess）。
