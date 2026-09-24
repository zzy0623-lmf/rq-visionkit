# -*- coding: utf-8 -*-
"""M3 部署控制台 /deploy 路由组（T2.3）。

与 M1 标注后端共用同一个 FastAPI 进程：annotator/server/main.py 通过
`app.include_router(create_deploy_router(...))` 挂载本路由组。

接口：
- GET  /deploy/model-dirs        扫描候选模型目录（供前端下拉选择，零代码选模型）
- POST /deploy                   打包模型 + config.yaml，下发到目标并触发 /model/reload
- GET  /deploy/health            代理端侧运行时 GET /health
- POST /deploy/infer             上传图片，代理端侧运行时 /infer 并叠加检测框返回结果看板数据
"""

from __future__ import annotations

import base64
import json
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from deployer.pack import DEFAULT_CLASSES, build_config, pack
from deployer.ssh import deploy

REPO_ROOT = Path(__file__).resolve().parent.parent

_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
]


class DeployTarget(BaseModel):
    transport: str = "local"  # local | ssh
    host: str = "127.0.0.1"
    username: str = "root"
    port: int = 22
    identity_file: str | None = None
    model_dir: str = ""       # 模型文件落地目录
    config_path: str = ""     # config.yaml 落地路径
    runtime_url: str = "http://127.0.0.1:8001"


class DeployRequest(BaseModel):
    model_dir: str                                # 源模型文件目录（本机）
    model_name: str = "yolox_neudet"
    version: str = "1.0.0"
    task: str = "detect"                          # detect | classify
    input_size: int = 320
    conf_thres: float = 0.01
    nms_thres: float = 0.65
    quant: str = "fp32"
    classes: list[str] = Field(default_factory=lambda: list(DEFAULT_CLASSES))
    target: DeployTarget = Field(default_factory=DeployTarget)


def _resolve_target(target: DeployTarget, default_model_dir: Path, default_config_path: Path) -> DeployTarget:
    """本地传输缺省路径填充（PC 仿真默认落到 runtime/pc_sim 的 model 与 config）。"""
    if target.transport == "local":
        if not target.model_dir:
            target.model_dir = str(default_model_dir)
        if not target.config_path:
            target.config_path = str(default_config_path)
    return target


# ---- 端侧运行时 HTTP 代理（urllib，零第三方依赖） ----

def _http_get_json(url: str, timeout: int = 15) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _multipart_post(url: str, file_bytes: bytes, filename: str, fields: dict | None = None,
                    timeout: int = 30) -> dict:
    boundary = "----rqvisionkit" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for k, v in (fields or {}).items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8")
        )
    chunks.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n".encode("utf-8") + file_bytes + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---- 结果看板：检测框叠加 ----

def draw_overlay(img_bgr: np.ndarray, boxes: list) -> str:
    """在图上叠加检测框与标签，返回 base64 JPEG。"""
    img = img_bgr.copy()
    for b in boxes:
        x1, y1 = int(round(b["x1"])), int(round(b["y1"]))
        x2, y2 = int(round(b["x2"])), int(round(b["y2"]))
        color = _PALETTE[int(b.get("class_id", 0)) % len(_PALETTE)]
        label = f"{b.get('class_name', '')} {b.get('score', 0):.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, max(th, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("叠加图编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def create_deploy_router(
    default_model_dir: Path | None = None,
    default_config_path: Path | None = None,
    model_scan_root: Path | None = None,
) -> APIRouter:
    default_model_dir = default_model_dir or (REPO_ROOT / "runtime" / "pc_sim" / "deployed_model")
    default_config_path = default_config_path or (REPO_ROOT / "runtime" / "pc_sim" / "config.yaml")
    model_scan_root = model_scan_root or (REPO_ROOT.parent / "tools")

    router = APIRouter(prefix="/deploy", tags=["deploy"])

    @router.get("/model-dirs")
    def model_dirs():
        """扫描 model_scan_root 下直接含 *.opt.param 的目录，供前端下拉选择。"""
        root = Path(model_scan_root)
        found = []
        candidates = [root] + ([d for d in root.iterdir() if d.is_dir()] if root.is_dir() else [])
        for d in candidates:
            try:
                params = sorted(d.glob("*.opt.param"))
            except OSError:
                continue
            if params:
                found.append({"name": d.name if d != root else root.name,
                              "path": str(d), "param_count": len(params)})
        return {"root": str(root), "dirs": found}

    @router.post("")
    def deploy_model(body: DeployRequest):
        src = Path(body.model_dir)
        if not src.is_dir():
            raise HTTPException(status_code=422, detail=f"源模型目录不存在: {body.model_dir}")
        target = _resolve_target(body.target, default_model_dir, default_config_path)

        # 打包到临时目录（模型文件 + config.yaml，model_dir 指向目标落地目录）
        tmp = Path(REPO_ROOT / ".deploy-tmp" / uuid.uuid4().hex)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            config = build_config(
                model_name=body.model_name, version=body.version, input_size=body.input_size,
                conf_thres=body.conf_thres, nms_thres=body.nms_thres, quant=body.quant,
                task=body.task, classes=body.classes,
            )
            pack(str(src), config, str(tmp), target_model_dir=target.model_dir)
            result = deploy(tmp, target.model_dump(), target.runtime_url)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

        return {"ok": True, "model_name": body.model_name, "version": body.version, **result}

    @router.get("/health")
    def health(runtime_url: str = "http://127.0.0.1:8001"):
        try:
            return _http_get_json(runtime_url.rstrip("/") + "/health")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"端侧运行时不可达: {exc}")

    @router.post("/infer")
    def infer(file: UploadFile = File(...), runtime_url: str = Form(default="http://127.0.0.1:8001")):
        data = file.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="空图片")
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=415, detail="无法解码图片")

        try:
            result = _multipart_post(runtime_url.rstrip("/") + "/infer", data,
                                     file.filename or "capture.jpg")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"端侧推理失败: {exc}")

        inference_ms = float(result.get("inference_ms", 0))
        base = {
            "task": result.get("task", "detect"),
            "inference_ms": inference_ms,
            "mem_kb": result.get("mem_kb", 0),
            "fps": round(1000 / inference_ms, 2) if inference_ms > 0 else None,
        }

        # 分类任务：返回 top-k 类别 + 原图（不画检测框）
        if base["task"] == "classify":
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                raise RuntimeError("原图编码失败")
            overlay = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
            return {**base, "classes": result.get("classes", []), "overlay": overlay}

        boxes = result.get("boxes", [])
        overlay = draw_overlay(img, boxes)
        return {**base, "boxes": boxes, "overlay": overlay}

    return router
