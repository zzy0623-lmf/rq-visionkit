# M4 板端运行时（RK3506 · C/C++）

与 PC 仿真版 `runtime/pc_sim/` 暴露**同一 HTTP 接口契约**，用于在 RK3506 开发板
（无 NPU，3×Cortex-A7）上跑自定义 NCNN 模型，供 M3 部署控制台 SSH 下发后调用。

## 文件

| 文件 | 说明 |
|---|---|
| `yolox_inferencer.h/.cpp` | YOLOX 检测推理封装（ncnn 加载 + letterbox 预处理 + decode + 按类 NMS），逻辑与 PC 版 `inferencer.py` 严格一致 |
| `classify_inferencer.h/.cpp` | 分类推理封装（resize + BGR2RGB + mean/norm + softmax 输出 top-k），对应 PC 版 classify 分支（T2.5 扩展） |
| `runtime_server.cpp` | webnet HTTP 服务器：`/health`、`/reload`、`/infer` 三端点 + 自动初始化 + 任务类型切换 |

## 任务类型切换

`runtime_server.cpp` 顶部 `RUNTIME_TASK` 宏选择编译目标：

| 宏值 | 任务 | 模型 | input_size |
|---|---|---|---|
| `TASK_DETECT`（默认） | 检测（T2.2） | YOLOX-Nano | 320 |
| `TASK_CLASSIFY` | 分类（T2.5 扩展） | MobileNetV2 六分类 | 224 |

## 接口契约（与 PC 仿真版对应）

| 板端 | PC 仿真版 | 说明 |
|---|---|---|
| `GET /cgi-bin/health` | `GET /health` | 返回 `{model_name, version, uptime_s, quant, task, num_classes}` |
| `POST /cgi-bin/reload` | `POST /model/reload` | 重新加载模型，返回 `{reloaded, model_name, version}` |
| `POST /infer`（multipart 图片） | `POST /infer` | 检测返回 `{boxes:[...], inference_ms, mem_kb}`；分类返回 `{classes:[...], inference_ms, mem_kb}` |

> 说明：webnet 的 CGI 模块固定把 handler 挂在 `/cgi-bin/` 前缀下，因此 `/health`、
> `/reload` 实际 URL 带 `/cgi-bin/` 前缀；`/infer` 走 upload 模块，URL 不带前缀。
> 部署控制台（M3）在 SSH 模式下按上表 URL 调用，与本地 PC 仿真模式的目标地址不同，
> 由 `deployer/routes.py` 的 target 配置区分。

## 集成到 RuiChing Studio（官方示例 09_ai_mobilenetv2_yolov3 基础上）

1. 在 RuiChing Studio 里按官方文档新建/打开 `09_ai_mobilenetv2_yolov3` 示例工程。
2. 将本目录 5 个文件复制到工程的 `applications/` 目录：
   `yolox_inferencer.h/.cpp`、`classify_inferencer.h/.cpp`、`runtime_server.cpp`。
3. **删除或改名** `applications/main.c`（`runtime_server.cpp` 已含入口，
   避免 `main` 重定义）；官方 `yolov3.cpp` 可保留（其 `mnet_yolov3_test` 命令仍可用）。
4. 在工程配置里确认 webnet 的 **CGI** 与 **Upload** 模块已启用
   （`WEBNET_USING_CGI`、`WEBNET_USING_UPLOAD`；Upload 未启用时需在 menuconfig 打开）。
5. 编译 → 固化 APP。

## 模型放置

运行时按 `PARAM_PATH`/`BIN_PATH`（默认 `/data/model/model.opt.param` + `.opt.bin`）加载。
`/data` 为板载 nandfs 挂载点（T2.2 摸底确认，无需 TF 卡）。可用 msh 的 `wget`/`tftp`
把 rq-convert 产物放到 `/data/model/`，或由部署控制台 SSH 下发。

## 已知限制（上板验证重点）

1. **类别名硬编码**：`CLASS_NAMES` 在此硬编码为 NEU-DET 六类，与 deployer 下发的
   `config.yaml` 对应；C 侧不解析 YAML（避免引入依赖），换模型类别时需同步修改。
2. **模型名/参数**：`MODEL_NAME`、`INPUT_SIZE`、`CONF_THRES` 等为编译期宏，与
   `config.yaml` 对应；如需运行时可配，需在板端引入配置解析（本期不做）。
3. **webnet upload 响应**：`webnet_session_printf`/`set_header` 直接 `send()` 到 socket，
   不依赖 phase 状态机，故 `upload_done` 回调内写响应体可用（已对照 wn_session.c 确认）；
   仍需上板实测确认 multipart 图片完整接收。

## 与任务书的关系

- 任务书 T2.2「按官方 SDK 用 C/C++ + NCNN 实现与 T2.1 完全相同的 HTTP 接口」。
- 所有 API（webnet / ncnn / OpenCV / RT-Thread 宏）均来自官方 SDK 源码
  `09_ai_mobilenetv2_yolov3_2` 工程，未臆造；分类预处理/后处理对照 `tools/verify_neudet_cls.py`。
