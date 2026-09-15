# 第三方依赖许可证清单

规则：**仅允许 Apache-2.0 / MIT / BSD 类许可证**。新增依赖必须先登记本表再引入。
状态含义：✅ 已引入 / 📌 计划引入（任务书指定，尚未安装）/ ⚠️ 许可证不符合要求，待决策。

## Python 依赖

| 依赖 | 版本 | 许可证 | 用途 | 状态 |
|---|---|---|---|---|
| pytest | — | MIT | 测试框架（CI） | ✅ |
| fastapi | — | MIT | M1/M3 后端框架 | 📌 |
| uvicorn | — | BSD-3-Clause | ASGI 服务器 | 📌 |
| onnx | — | Apache-2.0 | 模型图解析/转换 | 📌 |
| onnxsim (onnx-simplifier) | — | MIT | ONNX 图简化 | 📌 |
| ncnn / ncnn-python | — | BSD-3-Clause | PC 仿真推理框架 | 📌 |
| pyyaml | — | MIT | 配置文件解析 | 📌 |
| opencv-python | — | MIT（封装层；OpenCV 4.x 核心为 Apache-2.0） | 图像读写/预处理 | 📌 |
| numpy | — | BSD-3-Clause | 数值计算 | 📌 |
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

## 数据集

| 数据集 | 来源 | 使用条款 | 状态 |
|---|---|---|---|
| NEU-DET（东北大学钢材表面缺陷数据集） | T0.3 时登记具体来源链接 | 学术研究用途，T0.3 核实并补充 | 📌 T0.3 完成登记 |

## ⚠️ 待决策：paramiko 许可证冲突

任务书第 1 节依赖清单含 `paramiko`，但其许可证为 **LGPL-2.1-or-later**，不在
Apache-2.0 / MIT / BSD 允许范围内。候选替代（T2.3 前必须定）：

1. **改用 OpenSSH 命令行**：通过 `subprocess` 调用系统 `ssh`/`scp` 下发部署包。零依赖、零许可证风险。Windows 10+ 自带 OpenSSH 客户端。
2. **换库**：如 `asyncssh`（EPL-2.0，同样不在白名单）——无合适的 MIT/BSD 纯 Python SSH 库。
3. **为例外申请**：向评审说明 LGPL 动态链接用法。不推荐，徒增合规解释成本。

**推荐方案 1**（OpenSSH + subprocess）。
