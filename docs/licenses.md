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
| fastapi | — | MIT | M1/M3 后端框架 | 📌 |
| uvicorn | — | BSD-3-Clause | ASGI 服务器 | 📌 |
| onnx | — | Apache-2.0 | 模型图解析/转换 | 📌 |
| onnxsim (onnx-simplifier) | — | MIT | ONNX 图简化 | 📌 |
| pyyaml | — | MIT | 配置文件解析 | 📌 |
| paramiko | — | **LGPL-2.1-or-later** | SSH 下发部署包（T2.3） | ⚠️ 见下方决策 |

## 前端依赖（T1.2 起）

| 依赖 | 许可证 | 用途 | 状态 |
|---|---|---|---|
| Vue 3 | MIT | 标注前端框架 | 📌 |
| Vite | MIT | 前端构建 | 📌 |

## 模型与工具链

| 项 | 许可证 | 说明 | 状态 |
|---|---|---|---|
| YOLOX | Apache-2.0 | 主 Demo 检测模型训练默认选型 | 📌 |
| Ultralytics YOLO | AGPL-3.0 | **默认不使用**；若使用须在此声明并给出理由 | ❌ 规避 |
| NCNN 命令行工具（onnx2ncnn / pnnx / ncnn2int8） | BSD-3-Clause | M2 转换流水线 | 📌 |

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
