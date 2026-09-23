# -*- coding: utf-8 -*-
"""M4 端侧运行时（PC 仿真版）FastAPI 服务（T2.1）。

接口：
- POST /infer          multipart 图片 → {boxes:[...], inference_ms, mem_kb}
- GET  /health         模型名、版本、运行时长
- POST /model/reload   重读 config.yaml 并重载模型（无需重启进程）

配置路径：默认 runtime/pc_sim/config.yaml，可用环境变量 RQ_RUNTIME_CONFIG 覆盖（测试隔离用）。
"""

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import yaml
from fastapi import FastAPI, HTTPException, UploadFile

from inferencer import NcnnInferencer

_LOCK = threading.Lock()


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not cfg or "model_dir" not in cfg:
        raise ValueError(f"config 缺少 model_dir: {path}")
    return cfg


class RuntimeState:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.start_time = time.time()
        self.cfg = load_config(self.config_path)
        self.inferencer = NcnnInferencer(self.cfg)
        self.inferencer.load()

    @property
    def uptime_s(self) -> float:
        return time.time() - self.start_time

    def reload(self) -> dict:
        with _LOCK:
            cfg = load_config(self.config_path)
            inf = NcnnInferencer(cfg)
            inf.load()
            self.cfg = cfg
            self.inferencer = inf
        return {"model_name": cfg.get("model_name"), "version": cfg.get("version"),
                "model_dir": cfg.get("model_dir")}


def create_app(config_path: str | None = None) -> FastAPI:
    if config_path is None:
        config_path = os.environ.get("RQ_RUNTIME_CONFIG", str(Path(__file__).parent / "config.yaml"))
    state = RuntimeState(Path(config_path))
    app = FastAPI(title="RQ-VisionKit M4 Runtime (PC sim)", version="0.1.0")

    @app.post("/infer")
    async def infer(file: UploadFile):
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="空图片")
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=415, detail="无法解码图片")
        with _LOCK:
            boxes, inference_ms = state.inferencer.infer(img)
        mem_kb = psutil.Process().memory_info().rss // 1024
        return {"boxes": boxes, "inference_ms": round(inference_ms, 2), "mem_kb": mem_kb}

    @app.get("/health")
    def health():
        return {
            "model_name": state.cfg.get("model_name"),
            "version": state.cfg.get("version"),
            "uptime_s": round(state.uptime_s, 2),
            "quant": state.cfg.get("quant"),
            "num_classes": len(state.cfg.get("classes", [])),
        }

    @app.post("/model/reload")
    def reload():
        info = state.reload()
        return {"reloaded": True, **info}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
