# -*- coding: utf-8 -*-
"""T1.7 导出：YOLOX checkpoint → ONNX → NCNN（FP32 / INT8）。

用法：
    python scripts/export_neudet.py --mode fp32
    python scripts/export_neudet.py --mode int8 [--calib 校准集目录]
    # T2.4 联调等场景可覆盖默认路径（均带默认值，向后兼容）：
    python scripts/export_neudet.py --exp <exp文件> --ckpt <权重> --out <输出目录> --stem <前缀>

流程（含三处已实测验证的转换修复）：
    1. 加载模型，用固定 stride-2 space-to-depth 卷积替换 stem 的 Focus（数值等价 diff=0.0）
    2. 导出 ONNX（decode_in_inference=False，端侧 Python 端后处理）+ 内联权重
    3. pnnx 转 ncnn（fp16=0，避免 pnnx 默认 fp16 截断权重）
    4. 修复 decode head 的 concat 轴（pnnx 转出 0=1 错误，应为 0=2）
    5. ncnnoptimize（FP32 优化）
    6. （--mode int8）ncnn2table + ncnn2int8 量化

三处修复的根因：
    - Focus 的 ::2 步长切片 pnnx 不支持（"slice with step 2 is not supported"），
      且 ncnn 量化工具不认识自定义层，故在 PyTorch 侧用等效卷积替换，全模型变原生层。
    - pnnx 默认 fp16=1 会把权重截断成 FP16，即使 ncnnoptimize 存成 FP32 也找不回精度。
    - pnnx 把 3 个检测头拼成 (2100,11) 时 concat 轴选错（沿 h 而非 w），输出错位。
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

# 路径全部环境变量驱动（默认值向后兼容，可在其他机器上覆盖）：
#   RQ_TOOLS_DIR          tools 目录根（默认仓库同级 tools/）
#   RQ_YOLOX_OUTPUT_DIR   YOLOX 训练/导出输出目录（默认 C:\yolox_outputs）
REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(os.environ.get("RQ_TOOLS_DIR", str(REPO_ROOT.parent / "tools")))
YOLOX = TOOLS_DIR / "yolox"
CONVERTER = REPO_ROOT / "converter"
NCNN_TOOLS = TOOLS_DIR / "ncnn-bin"

YOLOX_OUTPUT = Path(os.environ.get("RQ_YOLOX_OUTPUT_DIR", r"C:\yolox_outputs"))
EXP = str(YOLOX / "exps/neudet/yolox_nano_neudet.py")
CKPT = str(YOLOX_OUTPUT / "yolox_nano_neudet" / "best_ckpt.pth")
NCNN_OUT = YOLOX_OUTPUT / "ncnn"
STEM = "yolox_neudet"
INPUT_SHAPE = "1,3,320,320"


def make_space_to_depth(in_channels=3):
    """space-to-depth 固定卷积：3→12，kernel 2×2 stride 2，输出通道按 [tl, bl, tr, br] 排列（与 YOLOX Focus cat 顺序一致）。"""
    patch_map = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}  # tl, bl, tr, br -> (kh, kw)
    out_channels = in_channels * 4
    weight = torch.zeros(out_channels, in_channels, 2, 2)
    for out_ch in range(out_channels):
        patch_idx = out_ch // in_channels
        c_in = out_ch % in_channels
        kh, kw = patch_map[patch_idx]
        weight[out_ch, c_in, kh, kw] = 1.0
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
    conv.weight = nn.Parameter(weight, requires_grad=False)
    return conv


def load_model(exp_file: str = EXP, ckpt_path: str = CKPT):
    """加载训练好的模型，替换 Focus，设置 decode_in_inference=False，返回 model。"""
    sys.path.insert(0, str(YOLOX))
    from yolox.exp import get_exp
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    exp = get_exp(exp_file)
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.eval()
    model.head.decode_in_inference = False  # 端侧 Python 端后处理

    # 替换 stem 的 Focus，并校验数值等价
    stem = model.backbone.backbone.stem
    assert "Focus" in type(stem).__name__, f"预期 Focus，实际 {type(stem).__name__}"
    new_stem = nn.Sequential(make_space_to_depth(3), stem.conv)
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        ref = stem(x)
    model.backbone.backbone.stem = new_stem
    with torch.no_grad():
        got = new_stem(x)
    diff = (ref - got).abs().max().item()
    assert diff < 1e-4, f"Focus 替换不等价，diff={diff}"
    print(f"Focus 替换校验通过（diff={diff:.2e}）")

    return replace_module(model, nn.SiLU, SiLU)


def export_onnx(model, out_path: Path):
    """导出 ONNX 并内联权重（PyTorch 2.13 默认外置权重，pnnx 不支持）。"""
    import onnx

    dummy = torch.randn(1, 3, 320, 320)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["images"], output_names=["output"], opset_version=13,
    )
    m = onnx.load(str(out_path))
    onnx.save(m, str(out_path), save_as_external_data=False)
    data_file = Path(str(out_path) + ".data")
    if data_file.exists():
        data_file.unlink()
    print("ONNX 已导出（内联权重）:", out_path)


def patch_concat_axis(src: Path, dst: Path):
    """修复 decode head 的 concat 轴（0=1 → 0=2，沿 w 拼 num_preds）。"""
    lines = src.read_text(encoding="utf-8").splitlines()
    fixed = 0
    for i, ln in enumerate(lines):
        t = ln.split()
        if t and t[0] == "Concat" and "0=1" in t:
            cols = ln.split()
            for j, tok in enumerate(cols):
                if tok == "0=1":
                    cols[j] = "0=2"
                    fixed += 1
            lines[i] = " ".join(cols)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"concat 轴修复 {fixed} 处 -> {dst.name}")
    return fixed


def run(cmd, cwd):
    print(">>", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"命令失败: {cmd[0]}")
    return r


def prepare_calib(calib_dir: Path, out_dir: Path, n: int = 100) -> Path:
    """从 calib_dir 抽 n 张图，写 imagelist.txt，返回 list 路径。"""
    imgs = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not imgs:
        raise SystemExit(f"校准集目录无图片: {calib_dir}")
    rng = random.Random(42)
    picked = rng.sample(imgs, min(n, len(imgs)))
    list_file = out_dir / "calib.list.txt"
    list_file.write_text("\n".join(str(p) for p in picked) + "\n", encoding="utf-8")
    print(f"校准集 {len(picked)} 张 -> {list_file}")
    return list_file


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="YOLOX → ONNX → NCNN")
    p.add_argument("--mode", choices=["fp32", "int8"], default="fp32")
    p.add_argument("--calib", help="INT8 校准集图片目录（默认从 train2017 抽 100 张）")
    p.add_argument("--exp", default=EXP, help="YOLOX exp 配置文件（默认 T1.7 NEU-DET 配置）")
    p.add_argument("--ckpt", default=CKPT, help="训练权重路径（默认 T1.7 best_ckpt.pth）")
    p.add_argument("--out", default=str(NCNN_OUT), help="产物输出目录（默认 RQ_YOLOX_OUTPUT_DIR/ncnn）")
    p.add_argument("--stem", default=STEM, help="产物文件名前缀（默认 yolox_neudet）")
    p.add_argument("--input-shape", default=INPUT_SHAPE, help="输入尺寸 N,C,H,W（默认 1,3,320,320）")
    args = p.parse_args()

    out_dir = Path(args.out)
    stem = args.stem
    input_shape = args.input_shape
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = out_dir / f"{stem}.onnx"

    # 1. 加载 + 替换 Focus + 导出 ONNX（内联）
    model = load_model(args.exp, args.ckpt)
    export_onnx(model, onnx_out)

    # 2. 算子兼容性扫描（报告用途）
    env = os.environ.copy()
    env["PYTHONPATH"] = str(YOLOX) + os.pathsep + str(CONVERTER) + os.pathsep + env.get("PYTHONPATH", "")
    run([sys.executable, "-m", "rq_convert.scan", str(onnx_out)], CONVERTER)

    pnnx = NCNN_TOOLS / "pnnx.exe"
    ncnnoptimize = NCNN_TOOLS / "ncnnoptimize.exe"

    # 3. pnnx 转 ncnn（fp16=0）
    param = out_dir / f"{stem}.param"
    bin_ = out_dir / f"{stem}.bin"
    run([str(pnnx), str(onnx_out), f"inputshape=[{input_shape}]", "fp16=0",
         f"pnnxparam={out_dir / (stem + '.pnnx.param')}",
         f"pnnxbin={out_dir / (stem + '.pnnx.bin')}",
         f"pnnxpy={out_dir / (stem + '_pnnx.py')}",
         f"ncnnpy={out_dir / (stem + '_ncnn.py')}",
         f"ncnnparam={param}", f"ncnnbin={bin_}"], out_dir)

    # 4. 修复 concat 轴（在 ncnnoptimize 之前）
    patched = out_dir / f"{stem}.patched.param"
    patch_concat_axis(param, patched)

    # 5. ncnnoptimize（FP32）
    opt_param = out_dir / f"{stem}.opt.param"
    opt_bin = out_dir / f"{stem}.opt.bin"
    run([str(ncnnoptimize), str(patched), str(bin_), str(opt_param), str(opt_bin), "0"], out_dir)

    # 6. INT8
    if args.mode == "int8":
        calib_dir = Path(args.calib) if args.calib else Path(r"C:\neudet_coco\train2017")
        list_file = prepare_calib(calib_dir, out_dir)
        table = out_dir / f"{stem}.table"
        run([str(NCNN_TOOLS / "ncnn2table.exe"), str(opt_param), str(opt_bin), str(list_file), str(table),
             "mean=[0,0,0]", "norm=[1,1,1]", "shape=[320,320,3]", "pixel=BGR", "method=kl", "thread=8"], out_dir)
        int8_param = out_dir / f"{stem}.int8.param"
        int8_bin = out_dir / f"{stem}.int8.bin"
        run([str(NCNN_TOOLS / "ncnn2int8.exe"), str(opt_param), str(opt_bin), str(int8_param), str(int8_bin), str(table)], out_dir)

    print("\n导出完成，产物在", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
