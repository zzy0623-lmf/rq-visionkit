# -*- coding: utf-8 -*-
"""T2.1 M4 运行时（PC 仿真版）接口测试。"""

import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

MODEL_DIR = Path(r"C:\Users\zzyly\Desktop\2026上海开源大赛\tools\ncnn_nofocus")
IMG = Path(r"C:\Users\zzyly\Desktop\2026上海开源大赛\tools\calib\crazing_115.jpg")

# 模型与测试图片在仓库外（tools/ 不入库），CI 环境缺失时跳过
# （放在 import main 之前，避免 CI 因缺 ncnn 依赖而 ImportError）
if not MODEL_DIR.exists() or not IMG.exists():
    pytest.skip("本地 NCNN 模型与测试图片缺失（tools/ 不入库），跳过", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import create_app  # noqa: E402

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def _write_config(tmp_path: Path, model_name: str = "yolox_neudet", conf_thres: float = 0.01) -> Path:
    cfg = {
        "model_dir": str(MODEL_DIR),
        "model_name": model_name,
        "version": "1.0.0",
        "input_size": 320,
        "conf_thres": conf_thres,
        "nms_thres": 0.65,
        "quant": "fp32",
        "input_blob": "in0",
        "output_blob": "out0",
        "classes": CLASSES,
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture
def client(tmp_path):
    cfg = _write_config(tmp_path)
    return TestClient(create_app(str(cfg)))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "yolox_neudet"
    assert data["version"] == "1.0.0"
    assert data["uptime_s"] >= 0
    assert data["num_classes"] == 6


def test_infer_returns_json(client):
    with open(IMG, "rb") as f:
        r = client.post("/infer", files={"file": ("crazing_115.jpg", f.read(), "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["boxes"], list)
    assert data["inference_ms"] >= 0
    assert data["mem_kb"] > 0
    for b in data["boxes"]:
        assert {"x1", "y1", "x2", "y2", "score", "class_id", "class_name"} <= set(b)
        assert b["class_name"] in CLASSES


def test_reload_switches_model(tmp_path):
    cfg_path = _write_config(tmp_path, model_name="model_a")
    client = TestClient(create_app(str(cfg_path)))
    h1 = client.get("/health").json()
    assert h1["model_name"] == "model_a"

    # 修改 config 后 reload，模型名/阈值应更新，进程不重启（uptime 连续）
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["model_name"] = "model_b"
    cfg["conf_thres"] = 0.5
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    r = client.post("/model/reload")
    assert r.status_code == 200
    assert r.json()["reloaded"] is True

    h2 = client.get("/health").json()
    assert h2["model_name"] == "model_b"
    assert h2["uptime_s"] >= h1["uptime_s"]  # 未重启，运行时长连续
