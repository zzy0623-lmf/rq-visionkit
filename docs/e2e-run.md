# T2.4 端到端联调记录

> 执行日期：2026-09-24 · 平台：PC 仿真（Windows 11, Python 3.10.11, ncnn 1.0.20260526）
> 结论：**六步主流程全链路打通**，全程零命令行操作（除训练/转换为脚本内子进程，属工具链内部步骤）。

## 目的

验证任务书图 2 六步主流程：采集 → 标注 → 训练 → 转换 → 部署 → 推理查看，在 PC 仿真
环境下可重复走通，并记录每步耗时。RK3506 到手后，本流程的「部署」「推理查看」两步
目标设备从本地换成 SSH 即可复用，无需改工具链代码。

## 环境

- 数据：NEU-DET 原始图（`demo/neu_det/dataset/IMAGES/`）+ 原始 XML 真值标注（`ANNOTATIONS/`）
- 联调规模：6 类 × 每类 5 张 = 30 张（小样本，仅验证闭环，非精度评测）
- 训练：YOLOX-Nano（0.90M 参数，320×320，5 epoch），exp 配置 `tools/yolox/exps/neudet/yolox_nano_neudet_e2e.py`
- 转换：`scripts/export_neudet.py`（Focus 替换 + 导出 ONNX）→ `rq-convert`（pnnx → ncnnoptimize）
- 运行时：`runtime/pc_sim`（ncnn-python），部署控制台：`deployer`（`/deploy` 路由组）

## 六步流程与耗时

| 步骤 | 说明 | 耗时 | 关键结果 |
|---|---|---|---|
| 1 采集 | M1 批量导入 30 张（`POST /api/images/import`） | 0.17s | 导入 30 张 |
| 2 标注 | 写回 XML 真值框（`PUT /api/annotations/{id}`） | 0.59s | 30 张全部标注，共 76 框 |
| 3 导出 | YOLO 数据集 zip（`POST /api/dataset/export`） | 0.04s | train 24 / val 6，zip 438KB |
| 3b 转 COCO | 导出包 → YOLOX 训练格式（train/val.json） | 0.02s | train 62 框 / val 14 框 |
| 4 训练 | YOLOX-Nano 5 epoch（CPU） | 90.69s | best AP 0.00（30 张小样本预期），`latest_ckpt.pth` 7.57MB |
| 5 导出 ONNX | Focus→space-to-depth 替换 + ONNX 内联权重 | 18.28s | ONNX 4.51MB |
| 5b 转 NCNN | `rq-convert`（onnxsim→pnnx→patch_concat_axis→ncnnoptimize） | 3.14s | `e2e_nano.opt.bin` 3.56MB |
| 6 部署 | `/deploy` 打包 + 下发 + reload | 0.06~0.10s | 模型文件 + config.yaml 下发，reload 生效 |
| 6 推理查看 | `/deploy/infer` 叠加检测框返回看板数据 | 0.05~0.07s | 单帧 23.4ms，内存 ~96MB |

## 两个闭环验证

| 闭环 | 模型 | 推理结果 |
|---|---|---|
| 闭环 A | 联调训练的 `e2e_nano`（30 张 × 5 epoch） | 0 框（小样本弱模型，仅验证链路可用） |
| 闭环 B | T1.7 正式模型 `yolox_neudet`（1800 张 × 20 epoch） | **31 框**，单帧 23.38ms |

闭环 B 用于证明「换一个模型重复流程仍成立」：在同一个部署控制台上，将源模型目录从
联调产物换成 T1.7 正式模型，重新下发并 reload，端侧立即切换到新模型并正确出框。

截图：`docs/images/e2e_loopA_result.jpg`（无框）、`docs/images/e2e_loopB_result.jpg`（31 框叠加）。

## 本次联调发现并修复的问题

**deployer 换模型部署时旧模型残留**（真实 bug，已修复）：

- 现象：闭环 A → 闭环 B 连续部署到同一 `model_dir` 时，闭环 B 推理出 **0 框**。
- 根因：`deployer/ssh.py` 的 `deploy_local` 只复制不清理，目标目录残留闭环 A 的
  `e2e_nano.opt.param`；运行时按 `glob("*.opt.param")` 排序取第一个，加载到残留的
  弱模型而非闭环 B 的正式模型。
- 修复：`deploy_local` 复制前调用新增的 `clear_model_files()` 清空目标目录旧 `*.param/*.bin`
  （替换语义）；`deploy_ssh` 同样在 scp 前 `ssh rm -f` 清理远端旧模型。
- 验证：修复后闭环 B 推理 31 框；新增 2 个单测（`test_clear_model_files_replaces_old`、
  `test_deploy_local_clears_stale_models`），`deployer/tests` 15 passed。

## 备注

- 训练/转换/部署/推理均通过 HTTP 接口或脚本子进程完成，无手工改图、无手工命令行。
- `best AP = 0.00` 为 30 张小样本 + 5 epoch 的预期结果，不影响闭环验证结论；
  正式精度基线见 `docs/baselines.md`（T1.7 全量训练 mAP 6.37%）。
- 联调用临时脚本未入库（一次性执行脚本），数据与产物目录（`C:\e2e_t24`、
  `C:\neudet_e2e`、`C:\yolox_outputs_e2e`）在仓库外，不入库。
