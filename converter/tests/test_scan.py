# -*- coding: utf-8 -*-
"""T1.5 算子兼容扫描测试（classify_ops 纯逻辑 + ncnn_ops.json 加载）。"""

from collections import Counter

from rq_convert.scan import classify_ops, load_ops


def test_load_ops():
    ops = load_ops()
    assert ops["ncnn_version"] == "20260526"
    assert len(ops["layers"]) == 110
    assert "convolution" in ops["layers"]
    assert "Conv" in ops["onnx_supported"]
    assert "NonMaxSuppression" in ops["onnx_replace"]


def test_classify_supported_mapping():
    """Conv → Convolution（名字不同，走 onnx_supported 映射）。"""
    r = classify_ops({"Conv": 3}, load_ops())
    assert r["unsupported"] == [] and r["replace"] == []
    assert r["supported"][0]["op"] == "Conv"
    assert r["supported"][0]["ncnn"] == "Convolution"
    assert r["supported"][0]["count"] == 3


def test_classify_same_name():
    """Sigmoid 既是 ONNX 算子也是 NCNN layer 名（同名直接支持）。"""
    r = classify_ops({"Sigmoid": 2}, load_ops())
    assert r["supported"][0]["ncnn"] == "Sigmoid"


def test_classify_replace():
    """NonMaxSuppression → ⚠️ 需替换，给出建议。"""
    r = classify_ops({"NonMaxSuppression": 1}, load_ops())
    assert r["replace"][0]["op"] == "NonMaxSuppression"
    assert "YoloDetectionOutput" in r["replace"][0]["to"]


def test_classify_unsupported():
    """故意含不支持的算子 → ❌。"""
    r = classify_ops({"DefinitelyNotAnOp": 5}, load_ops())
    assert r["unsupported"][0]["op"] == "DefinitelyNotAnOp"
    assert r["unsupported"][0]["count"] == 5


def test_classify_mixed():
    """混合三档，验证分类汇总正确。"""
    r = classify_ops(Counter({"Conv": 4, "Relu": 4, "Gather": 1, "FakeOp": 2}), load_ops())
    assert [s["op"] for s in r["supported"]] == ["Conv", "Relu"]
    assert [x["op"] for x in r["replace"]] == ["Gather"]
    assert [x["op"] for x in r["unsupported"]] == ["FakeOp"]
