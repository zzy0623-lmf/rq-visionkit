"""T0.2 NCNN 环境验证（PC 仿真基座）。

用公开 NCNN 示例模型 squeezenet_v1.1 对一张图片做推理，
控制台打印 Top-3 类别与单帧毫秒数（预热 3 次，计时 20 次取 min/mean）。

用法：
    python scripts/smoke_ncnn.py
    python scripts/smoke_ncnn.py --image path/to/img.jpg --runs 20

资产默认位于 scripts/assets/（.gitignore 已排除，需先下载，来源见文件头注释）：
    squeezenet_v1.1.param/.bin  https://github.com/nihui/ncnn-assets/tree/master/models
    synset_words.txt            https://github.com/Tencent/ncnn/blob/master/examples/synset_words.txt
    messi5.jpg                  https://github.com/opencv/opencv/blob/4.x/samples/data/messi5.jpg

退出码：0 成功；非 0 失败。
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"

# 与 ncnn 官方 squeezenet 示例一致的预处理：227x227，BGR，仅减均值
INPUT_SIZE = 227
MEAN_VALS = [104.0, 117.0, 123.0]
NORM_VALS: list = []


def main() -> int:
    parser = argparse.ArgumentParser(description="NCNN smoke test (squeezenet_v1.1)")
    parser.add_argument("--param", default=str(ASSETS / "squeezenet_v1.1.param"))
    parser.add_argument("--bin", default=str(ASSETS / "squeezenet_v1.1.bin"))
    parser.add_argument("--image", default=str(ASSETS / "messi5.jpg"))
    parser.add_argument("--labels", default=str(ASSETS / "synset_words.txt"))
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    for p in (args.param, args.bin, args.image, args.labels):
        if not Path(p).exists():
            print(f"[FAIL] 缺少文件: {p}")
            return 2

    labels = Path(args.labels).read_text(encoding="utf-8").splitlines()

    net = ncnn.Net()
    net.load_param(args.param)
    net.load_model(args.bin)

    # Windows 下 cv2.imread 不支持非 ASCII 路径，改用 imdecode
    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[FAIL] 无法读取图片: {args.image}")
        return 2
    h, w = img.shape[:2]

    mat = ncnn.Mat.from_pixels_resize(
        img, ncnn.Mat.PixelType.PIXEL_BGR, w, h, INPUT_SIZE, INPUT_SIZE
    )
    mat.substract_mean_normalize(MEAN_VALS, NORM_VALS)

    def infer_once():
        ex = net.create_extractor()
        ex.input("data", mat)
        ret, out = ex.extract("prob")
        if ret != 0:
            raise RuntimeError(f"extract 失败, ret={ret}")
        return np.array(out)

    # 预热
    for _ in range(3):
        infer_once()

    # 计时
    times = []
    probs = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        probs = infer_once()
        times.append((time.perf_counter() - t0) * 1000.0)

    top_idx = np.argsort(probs)[::-1][:3]

    print(f"[OK] ncnn={ncnn.__version__}")
    print(f"[OK] model={Path(args.param).name} ({Path(args.param).stat().st_size} B) "
          f"+ {Path(args.bin).name} ({Path(args.bin).stat().st_size} B)")
    print(f"[OK] image={Path(args.image).name} ({w}x{h}) -> {INPUT_SIZE}x{INPUT_SIZE}")
    for rank, i in enumerate(top_idx, 1):
        name = labels[i].strip() if i < len(labels) else str(i)
        print(f"[TOP{rank}] class={i} prob={probs[i]:.4f} label={name}")
    print(f"[TIME] runs={args.runs} min={min(times):.2f} ms "
          f"mean={sum(times) / len(times):.2f} ms max={max(times):.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
