# -*- coding: utf-8 -*-
"""RQ-VisionKit M2 转换 CLI（T1.4）。

一条命令完成 ONNX → NCNN（FP32 / INT8）：

    rq-convert model.onnx --out dist/ --quant int8 --calib calib_100/ --input-shape 1,3,224,224

内部流水线（各步耗时/产物大小写入 report.json）：
    1. onnxsim 图简化
    2. pnnx → FP32 ncnn param + bin（onnx2ncnn 已被官方移除，改用 pnnx）
    3. ncnnoptimize 优化
    4. （--quant int8 时）ncnn2table 生成校准表 + ncnn2int8 量化 → INT8 param + bin

依赖的 NCNN 命令行工具（BSD-3-Clause，官方 release 预编译）：
    pnnx（github.com/pnnx/pnnx/releases） / ncnnoptimize / ncnn2table / ncnn2int8
    （后三者见 https://github.com/Tencent/ncnn/releases 的 ncnn-*-windows-vs20xx.zip 的 x64/bin/）

工具查找顺序：--ncnn-tools 指定目录 → 系统 PATH。
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PNNX_RELEASES = "https://github.com/pnnx/pnnx/releases"
NCNN_RELEASES = "https://github.com/Tencent/ncnn/releases"
TOOL_NAMES = ["pnnx", "ncnnoptimize", "ncnn2table", "ncnn2int8"]


def find_tool(name: str, tool_dir: str | None) -> str | None:
    """查找工具可执行文件：优先 tool_dir，其次 PATH。"""
    if tool_dir:
        for ext in ("", ".exe"):
            p = Path(tool_dir) / (name + ext)
            if p.is_file():
                return str(p)
    return shutil.which(name) or shutil.which(name + ".exe")


def run_cmd(argv: list, timeout: int | None = None) -> dict:
    """执行外部命令，返回 returncode/stdout/stderr/耗时。

    用 errors="replace" 容错：NCNN 工具在中文 Windows 下可能输出 GBK 编码，
    强制 utf-8 解码会抛 UnicodeDecodeError，这里替换为可读的乱码字符而非崩溃。
    """
    t0 = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, errors="replace")
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return {"returncode": 127, "stdout": "", "stderr": f"命令不存在: {argv[0]}", "elapsed_ms": (time.time() - t0) * 1000}
    except subprocess.TimeoutExpired as e:
        return {"returncode": 124, "stdout": "", "stderr": f"超时: {e}", "elapsed_ms": (time.time() - t0) * 1000}
    return {"returncode": rc, "stdout": out, "stderr": err, "elapsed_ms": (time.time() - t0) * 1000}


def require_tool(name: str, tool_dir: str | None) -> str:
    """查找工具，找不到则抛出带下载指引的错误。"""
    p = find_tool(name, tool_dir)
    if not p:
        if name == "pnnx":
            hint = f"请从 {PNNX_RELEASES} 下载 pnnx-*-windows.zip，解压得到 pnnx.exe"
        else:
            hint = f"请从 {NCNN_RELEASES} 下载 ncnn-*-windows-vs20xx.zip，解压后取 x64/bin"
        raise SystemExit(f"缺少 NCNN 工具 '{name}'。{hint}，加入 PATH 或用 --ncnn-tools 指定该目录。")
    return p


def _nchw_to_hwc(shape: str) -> str:
    """1,3,224,224 (NCHW) → 224,224,3 (HWC)，供 ncnn2table 使用。"""
    parts = [int(x) for x in shape.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--input-shape 需为 N,C,H,W 四维，如 1,3,224,224，收到: {shape}")
    _, c, h, w = parts
    return f"{h},{w},{c}"


def simplify_onnx(src: Path, dst: Path) -> dict:
    """onnxsim 图简化（库调用）。返回耗时。"""
    import onnxsim

    t0 = time.time()
    model, check = onnxsim.simplify(str(src))
    if not check:
        raise SystemExit(f"onnxsim 简化失败（check 未通过）: {src}")
    import onnx

    onnx.save(model, str(dst))
    return {"elapsed_ms": (time.time() - t0) * 1000, "output": str(dst), "size_bytes": dst.stat().st_size}


def pnnx_convert(onnx: Path, param: Path, bin_: Path, tool: str, input_shape: str) -> dict:
    """pnnx 将 ONNX 转为 ncnn param/bin（onnx2ncnn 已被官方移除）。

    注意：必须加 fp16=0，否则 pnnx 默认 fp16=1 会把权重截断成 FP16，
    后续 ncnnoptimize 即使存成 FP32 也找不回已丢失的精度（实测 mAP/数值偏差明显）。
    """
    r = run_cmd([
        tool, str(onnx),
        f"inputshape=[{input_shape}]",
        "fp16=0",
        f"ncnnparam={param}", f"ncnnbin={bin_}",
    ])
    if r["returncode"] != 0:
        raise SystemExit(f"pnnx 失败:\n{r['stderr']}")
    return {"elapsed_ms": r["elapsed_ms"], "size_bytes": bin_.stat().st_size}


def optimize(param: Path, bin_: Path, opt_param: Path, opt_bin: Path, tool: str) -> dict:
    # 最后一个参数 0 = 保持 FP32（65536 = FP16）
    r = run_cmd([tool, str(param), str(bin_), str(opt_param), str(opt_bin), "0"])
    if r["returncode"] != 0:
        raise SystemExit(f"ncnnoptimize 失败:\n{r['stderr']}")
    return {"elapsed_ms": r["elapsed_ms"], "size_bytes": opt_bin.stat().st_size}


def make_calib_table(param: Path, bin_: Path, calib_dir: Path, table: Path, tool: str, input_shape: str, mean: str, norm: str, pixel: str, method: str, thread: int) -> dict:
    """生成校准表：先写 imagelist.txt，再跑 ncnn2table。"""
    imgs = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not imgs:
        raise SystemExit(f"校准集目录为空或无图片: {calib_dir}")
    list_file = table.with_suffix(".list.txt")
    list_file.write_text("\n".join(str(p) for p in imgs) + "\n", encoding="utf-8")
    hwc = _nchw_to_hwc(input_shape)
    r = run_cmd([
        tool, str(param), str(bin_), str(list_file), str(table),
        f"mean={mean}", f"norm={norm}", f"shape={hwc}", f"pixel={pixel}",
        f"thread={thread}", f"method={method}",
    ])
    if r["returncode"] != 0:
        raise SystemExit(f"ncnn2table 失败:\n{r['stderr']}")
    return {"elapsed_ms": r["elapsed_ms"], "size_bytes": table.stat().st_size, "calib_images": len(imgs)}


def quantize(param: Path, bin_: Path, table: Path, int8_param: Path, int8_bin: Path, tool: str) -> dict:
    r = run_cmd([tool, str(param), str(bin_), str(int8_param), str(int8_bin), str(table)])
    if r["returncode"] != 0:
        raise SystemExit(f"ncnn2int8 失败:\n{r['stderr']}")
    return {"elapsed_ms": r["elapsed_ms"], "size_bytes": int8_bin.stat().st_size}


def convert(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.model)
    stem = src.stem

    report = {"model": str(src), "quant": args.quant, "steps": []}
    steps = report["steps"]

    # 1. onnxsim 简化
    sim = out / f"{stem}.sim.onnx"
    r = simplify_onnx(src, sim)
    steps.append({"name": "onnxsim", **r})
    cur_onnx = sim

    # 2. pnnx（onnx → ncnn）
    pnnx = require_tool("pnnx", args.ncnn_tools)
    param = out / f"{stem}.param"
    bin_ = out / f"{stem}.bin"
    r = pnnx_convert(cur_onnx, param, bin_, pnnx, args.input_shape)
    steps.append({"name": "pnnx", **r})

    # 3. ncnnoptimize
    ncnnoptimize = require_tool("ncnnoptimize", args.ncnn_tools)
    opt_param = out / f"{stem}.opt.param"
    opt_bin = out / f"{stem}.opt.bin"
    r = optimize(param, bin_, opt_param, opt_bin, ncnnoptimize)
    steps.append({"name": "ncnnoptimize", **r})

    fp32 = {"param": opt_param.name, "bin": opt_bin.name, "bin_size": opt_bin.stat().st_size}

    if args.quant == "int8":
        # 4a. ncnn2table
        ncnn2table = require_tool("ncnn2table", args.ncnn_tools)
        if not args.calib:
            raise SystemExit("--quant int8 需要 --calib 校准集目录")
        table = out / f"{stem}.table"
        r = make_calib_table(opt_param, opt_bin, Path(args.calib), table, ncnn2table,
                             args.input_shape, args.mean, args.norm, args.pixel, args.method, args.thread)
        steps.append({"name": "ncnn2table", **r})

        # 4b. ncnn2int8
        ncnn2int8 = require_tool("ncnn2int8", args.ncnn_tools)
        int8_param = out / f"{stem}.int8.param"
        int8_bin = out / f"{stem}.int8.bin"
        r = quantize(opt_param, opt_bin, table, int8_param, int8_bin, ncnn2int8)
        steps.append({"name": "ncnn2int8", **r})
        int8 = {"param": int8_param.name, "bin": int8_bin.name, "bin_size": int8_bin.stat().st_size}
    else:
        int8 = None

    report["outputs"] = {"fp32": fp32, "int8": int8}
    # 时延/mAP 需推理验证，T1.5 联调时实测回填，此处不臆造
    report["latency_ms"] = {"fp32": None, "int8": None}
    report["mAP"] = {"fp32": None, "int8": None}
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rq-convert", description="ONNX → NCNN（FP32/INT8）一键转换")
    p.add_argument("model", help="输入 ONNX 模型路径")
    p.add_argument("--out", default="dist", help="输出目录（默认 dist/）")
    p.add_argument("--quant", choices=["fp32", "int8"], default="int8", help="量化档位（默认 int8）")
    p.add_argument("--calib", help="INT8 校准集图片目录（--quant int8 时必需）")
    p.add_argument("--input-shape", default="1,3,224,224", help="输入尺寸 N,C,H,W（默认 1,3,224,224）")
    p.add_argument("--mean", default="104,117,123", help="校准归一化均值（默认 104,117,123）")
    p.add_argument("--norm", default="0.017,0.017,0.017", help="校准归一化系数（默认 0.017,0.017,0.017）")
    p.add_argument("--pixel", default="BGR", choices=["BGR", "RGB", "GRAY"], help="像素格式（默认 BGR）")
    p.add_argument("--method", default="kl", choices=["kl", "aciq"], help="量化算法（默认 kl）")
    p.add_argument("--thread", type=int, default=8, help="校准线程数（默认 8）")
    p.add_argument("--ncnn-tools", help="NCNN 工具目录（含 onnx2ncnn 等 exe）")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.model).is_file():
        print(f"模型文件不存在: {args.model}", file=sys.stderr)
        return 2
    report = convert(args)
    report_path = Path(args.out) / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n转换完成，报告已写入 {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
