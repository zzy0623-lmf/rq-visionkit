# -*- coding: utf-8 -*-
"""RQ-VisionKit M2 扩展：Keras/TensorFlow 模型 → ONNX 导出（T1.6）。

论文模型是 Keras MobileNetV2（.h5/SavedModel），部署端是 TFLite INT8。
任务书明确 TFLite 直转不顺，必须从训练框架重新导出 ONNX，本模块即承担这一步：

    rq-export-keras model.h5 --out model.onnx --input-shape 1,224,224,3

导出后接入 T1.4 的 rq-convert 流水线转 NCNN。
"""

import argparse
from pathlib import Path


def export_keras_to_onnx(
    model_path: str,
    onnx_path: str,
    input_shape: tuple = (1, 224, 224, 3),
    opset: int = 13,
) -> dict:
    """加载 Keras 模型并导出 ONNX。返回元信息。"""
    import tensorflow as tf
    import tf2onnx

    # safe_mode=False：允许反序列化 Lambda 层（模型可能含预处理 Lambda）
    model = tf.keras.models.load_model(model_path, safe_mode=False)

    # 输入签名：N,H,W,C（N=None 表示动态 batch）
    n, h, w, c = input_shape
    spec = (tf.TensorSpec((None, h, w, c), tf.float32, name="input"),)
    model.output_names = ["output"]

    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)

    import onnx

    out = Path(onnx_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(out))

    return {
        "keras_model": str(model_path),
        "onnx": str(out),
        "input_shape": list(input_shape),
        "opset": opset,
        "size_bytes": out.stat().st_size,
        "num_layers": len(model.layers),
    }


def _parse_shape(s: str) -> tuple:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--input-shape 需为 N,H,W,C 四维，如 1,224,224,3，收到: {s}")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rq-export-keras", description="Keras/TensorFlow 模型 → ONNX")
    p.add_argument("model", help="Keras 模型路径（.h5 或 SavedModel 目录）")
    p.add_argument("--out", required=True, help="输出 ONNX 路径")
    p.add_argument("--input-shape", default="1,224,224,3", help="输入尺寸 N,H,W,C（默认 1,224,224,3）")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset（默认 13）")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.model).exists():
        print(f"模型路径不存在: {args.model}")
        return 2
    info = export_keras_to_onnx(args.model, args.out, _parse_shape(args.input_shape), args.opset)
    import json

    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\n导出完成: {info['onnx']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
