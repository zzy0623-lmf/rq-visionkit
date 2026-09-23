# -*- coding: utf-8 -*-
"""T1.7 训练入口：调用 YOLOX 训练 NEU-DET 钢材缺陷检测模型（CPU 降配）。

用法：
    python scripts/train_neudet.py            # 默认 batch=8, max_epoch=20（见 exp 配置）
    python scripts/train_neudet.py --batch 4

前置：
    1. 数据已转 COCO：python scripts/prepare_neudet_coco.py（生成 C:\\neudet_coco）
    2. YOLOX 代码在 tools/yolox（已做 CPU 兼容修改，见下）
    3. 依赖：torch(cpu) / loguru / pycocotools / tensorboard 等

YOLOX CPU 兼容修改（tools/yolox，仓库外，为让 YOLOX 在无 CUDA 环境训练）：
    - yolox/core/trainer.py：device 按 torch.cuda.is_available() 选择 cpu；set_device 加判断
    - yolox/data/data_prefetcher.py：无 CUDA 时跳过 Stream/cuda 移动
    - exps/neudet/yolox_nano_neudet.py：覆盖 random_resize 禁用 multiscale；output_dir 用绝对路径

训练规模（降配，CPU 可接受）：
    YOLOX-Nano（depth=0.33/width=0.25，0.90M 参数），320×320，max_epoch=20
"""

import os
import subprocess
import sys
from pathlib import Path

YOLOX = Path(r"c:\Users\zzyly\Desktop\2026上海开源大赛\tools\yolox")
EXP = "exps/neudet/yolox_nano_neudet.py"


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="训练 NEU-DET 检测模型（YOLOX-Nano，CPU）")
    p.add_argument("--batch", type=int, default=8, help="batch size")
    p.add_argument("--cache", action="store_true", help="缓存图片到 RAM（加速，但占内存）")
    args = p.parse_args()

    cmd = [sys.executable, "tools/train.py", "-f", EXP, "-b", str(args.batch), "-d", "0"]
    if args.cache:
        cmd.append("--cache")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(YOLOX) + os.pathsep + env.get("PYTHONPATH", "")

    print("训练命令:", " ".join(cmd))
    print("工作目录:", YOLOX)
    return subprocess.run(cmd, cwd=YOLOX, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
