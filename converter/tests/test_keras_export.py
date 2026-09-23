# -*- coding: utf-8 -*-
"""T1.6 Keras→ONNX 导出纯逻辑测试（不依赖 tensorflow/tf2onnx）。"""

import pytest

from rq_convert.keras_export import _parse_shape, build_parser


def test_parse_shape():
    assert _parse_shape("1,224,224,3") == (1, 224, 224, 3)


def test_parse_shape_invalid():
    with pytest.raises(SystemExit):
        _parse_shape("224,224,3")  # 三维，非 N,H,W,C


def test_parser_defaults():
    p = build_parser()
    args = p.parse_args(["model.h5", "--out", "model.onnx"])
    assert args.input_shape == "1,224,224,3"
    assert args.opset == 13
    assert args.out == "model.onnx"
