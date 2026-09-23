# -*- coding: utf-8 -*-
"""T1.4 转换 CLI 纯逻辑测试（不依赖 NCNN 工具与真实模型）。"""

import pytest

from rq_convert.cli import _nchw_to_hwc, build_parser, find_tool, require_tool


def test_nchw_to_hwc():
    assert _nchw_to_hwc("1,3,224,224") == "224,224,3"
    assert _nchw_to_hwc("1,3,640,640") == "640,640,3"


def test_nchw_to_hwc_invalid():
    with pytest.raises(SystemExit):
        _nchw_to_hwc("3,224,224")  # 三维，非 NCHW


def test_parser_defaults():
    p = build_parser()
    args = p.parse_args(["model.onnx"])
    assert args.quant == "int8"
    assert args.input_shape == "1,3,224,224"
    assert args.mean == "104,117,123"
    assert args.method == "kl"
    assert args.out == "dist"


def test_find_tool_missing():
    assert find_tool("definitely_not_a_real_tool_xyz", None) is None


def test_require_tool_missing_hint():
    with pytest.raises(SystemExit) as e:
        require_tool("definitely_not_a_real_tool_xyz", None)
    assert "releases" in str(e.value)  # 报错应含下载指引


def test_quant_int8_requires_calib():
    p = build_parser()
    args = p.parse_args(["model.onnx", "--quant", "int8"])
    assert args.calib is None  # 未给校准集，运行时应在 convert() 报错
