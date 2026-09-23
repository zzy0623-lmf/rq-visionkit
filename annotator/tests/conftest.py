# -*- coding: utf-8 -*-
"""pytest 公共配置：把仓库根加入 sys.path，并给每个测试一个独立数据目录。"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotator.server.main import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    """每个用例独立的数据目录（RQ_DATA_DIR 隔离）。"""
    app = create_app(tmp_path / "data")
    with TestClient(app) as c:
        yield c