# -*- coding: utf-8 -*-
"""T2.5 扩展 Demo：M4 运行时分类模型（task=classify）接口测试。

分类模型（NEU-DET 六分类 MobileNetV2，去 Lambda 版）与测试图片在仓库外
（tools/t16_outputs/ncnn 与 tools/calib 不入库），CI 环境缺失时跳过。
"""

import shutil
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

CLS_NCNN_DIR = Path(r"C:\Users\zzyly\Desktop\2026上海开源大赛\tools\t16_outputs\ncnn")
IMG = Path(r"C:\Users\zzyly\Desktop\2026上海开源大赛\tools\calib\crazing_115.jpg")
STEM = "mobilenetv2_neudet_nolambda_nchw"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

_PARAM = CLS_NCNN_DIR / f"{STEM}.opt.param"
_BIN = CLS_NCNN_DIR / f"{STEM}.opt.bin"

if not _PARAM.exists() or not _BIN.exists() or not IMG.exists():
    pytest.skip("本地分类模型与测试图片缺失（tools/ 不入库），跳过", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    """把去 Lambda 分类模型复制到干净目录（避免多模型目录 glob 取错），再建 app。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shutil.copy2(_PARAM, model_dir / _PARAM.name)
    shutil.copy2(_BIN, model_dir / _BIN.name)

    cfg = {
        "model_dir": str(model_dir),
        "model_name": "mobilenetv2_neudet",
        "version": "1.0.0",
        "task": "classify",
        "input_size": 224,
        "quant": "fp32",
        "input_blob": "in0",
        "output_blob": "out0",
        "classes": CLASSES,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return TestClient(create_app(str(cfg_path)))


def test_classify_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["task"] == "classify"
    assert data["model_name"] == "mobilenetv2_neudet"
    assert data["num_classes"] == 6


def test_classify_infer_returns_classes(client):
    with open(IMG, "rb") as f:
        r = client.post("/infer", files={"file": ("crazing_115.jpg", f.read(), "image/jpeg")})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["task"] == "classify"
    assert "boxes" not in data
    classes = data["classes"]
    assert isinstance(classes, list) and len(classes) >= 1
    for c in classes:
        assert {"class_id", "class_name", "score"} <= set(c)
        assert c["class_name"] in CLASSES
    # 测试图为 crazing 缺陷，top-1 应为 crazing（六分类模型 accuracy 83.61%）
    assert classes[0]["class_name"] == "crazing"
    assert data["inference_ms"] >= 0
    assert data["mem_kb"] > 0
