# -*- coding: utf-8 -*-
"""pytest 公共配置：加仓库根到 sys.path，并提供部署测试所需的 client 与 mock 运行时。"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotator.server.main import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    """独立数据目录的完整应用（含 /deploy 路由组）。"""
    app = create_app(tmp_path / "data")
    with TestClient(app) as c:
        yield c


class _RuntimeHandler(BaseHTTPRequestHandler):
    """mock 端侧运行时：实现 /health /infer /model/reload，无需 ncnn。"""

    def _drain(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)

    def _reply(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self._drain()
        if self.path == "/model/reload":
            self._reply({"reloaded": True, "model_name": "test", "version": "1.0.0"})
        elif self.path == "/infer":
            self._reply({
                "boxes": [{"x1": 10, "y1": 20, "x2": 40, "y2": 50,
                           "score": 0.95, "class_id": 0, "class_name": "crazing"}],
                "inference_ms": 12.5, "mem_kb": 123,
            })
        else:
            self._reply({"detail": "not found"}, code=404)

    def do_GET(self):
        if self.path == "/health":
            self._reply({"model_name": "test", "version": "1.0.0", "quant": "fp32", "num_classes": 6})
        else:
            self._reply({"detail": "not found"}, code=404)

    def log_message(self, *args):
        pass


@pytest.fixture()
def runtime_url():
    """在临时端口启动 mock 运行时，返回其 base URL。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RuntimeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
