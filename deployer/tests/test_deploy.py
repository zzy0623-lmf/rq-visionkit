# -*- coding: utf-8 -*-
"""T2.3 M3 部署控制台测试：打包 / SSH 命令构造 / 本地下发 / /deploy 路由组。"""

import cv2
import numpy as np
import yaml
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deployer.pack import DEFAULT_CLASSES, build_config, pack
from deployer.ssh import (
    build_scp_argv,
    build_ssh_argv,
    copy_files,
    deploy_local,
    split_package,
)


# ---- 打包（pack.py） ----

def test_build_config_defaults():
    cfg = build_config()
    assert cfg["model_name"] == "yolox_neudet"
    assert cfg["input_size"] == 320
    assert cfg["classes"] == DEFAULT_CLASSES


def test_pack_rewrites_model_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.opt.param").write_text("param", encoding="utf-8")
    (src / "m.opt.bin").write_bytes(b"bin")

    out = pack(str(src), build_config(), str(tmp_path / "pkg"), target_model_dir="/data/model")

    assert (out / "m.opt.param").exists()
    assert (out / "m.opt.bin").exists()
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model_dir"] == "/data/model"


def test_pack_rejects_missing_model(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    import pytest

    with pytest.raises(FileNotFoundError):
        pack(str(empty), build_config(), str(tmp_path / "pkg"))


# ---- SSH 命令构造（OpenSSH subprocess） ----

def test_build_scp_argv():
    src = Path("m.opt.param")
    argv = build_scp_argv(src, "10.0.0.1", "root", "/data/model", port=22)
    assert argv[0] == "scp"
    assert "StrictHostKeyChecking=accept-new" in argv
    assert str(src) in argv
    assert "root@10.0.0.1:/data/model/" in argv
    assert "-P" in argv and "22" in argv


def test_build_ssh_argv():
    cmd = "curl -s -X POST http://127.0.0.1:8001/model/reload"
    argv = build_ssh_argv("10.0.0.1", "root", cmd)
    assert argv[0] == "ssh"
    assert "root@10.0.0.1" in argv
    assert argv[-1] == cmd


def test_split_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.opt.param").write_text("p", encoding="utf-8")
    (pkg / "m.opt.bin").write_text("b", encoding="utf-8")
    (pkg / "config.yaml").write_text("c", encoding="utf-8")

    model_files, cfg = split_package(pkg)
    assert {f.name for f in model_files} == {"m.opt.param", "m.opt.bin"}
    assert cfg.name == "config.yaml"


def test_copy_files(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x")
    names = copy_files([a], tmp_path / "dst")
    assert names == ["a.bin"]
    assert (tmp_path / "dst" / "a.bin").read_bytes() == b"x"


def test_clear_model_files_replaces_old(tmp_path):
    """换模型部署应清理目标目录旧模型，避免 runtime glob 加载到残留旧模型。"""
    from deployer.ssh import clear_model_files

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "old.opt.param").write_text("p", encoding="utf-8")
    (model_dir / "old.opt.bin").write_text("b", encoding="utf-8")
    (model_dir / "keep.txt").write_text("k", encoding="utf-8")  # 非模型文件保留

    removed = clear_model_files(model_dir)
    assert removed == 2
    assert not (model_dir / "old.opt.param").exists()
    assert not (model_dir / "old.opt.bin").exists()
    assert (model_dir / "keep.txt").exists()


def test_deploy_local_clears_stale_models(tmp_path, runtime_url):
    """两次部署到同一目录：第二次部署后旧模型应被清除，只留新模型。"""
    model_dir = tmp_path / "model"

    def pkg_with(stem):
        p = tmp_path / f"pkg_{stem}"
        p.mkdir()
        (p / f"{stem}.opt.param").write_text("param", encoding="utf-8")
        (p / f"{stem}.opt.bin").write_text("bin", encoding="utf-8")
        (p / "config.yaml").write_text(yaml.safe_dump({"model_dir": "/x"}), encoding="utf-8")
        return p

    config_path = tmp_path / "cfg" / "config.yaml"
    deploy_local(pkg_with("a"), str(model_dir), str(config_path), runtime_url)
    deploy_local(pkg_with("b"), str(model_dir), str(config_path), runtime_url)

    # 旧模型 a 应被清除，只保留 b
    assert not (model_dir / "a.opt.param").exists()
    assert (model_dir / "b.opt.param").exists()
    assert (model_dir / "b.opt.bin").exists()


# ---- 本地下发（含 mock reload） ----

def test_deploy_local_copies_and_reloads(tmp_path, runtime_url):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.opt.param").write_text("param", encoding="utf-8")
    (pkg / "m.opt.bin").write_text("bin", encoding="utf-8")
    (pkg / "config.yaml").write_text(yaml.safe_dump({"model_dir": "/x"}), encoding="utf-8")

    model_dir = tmp_path / "model"
    config_path = tmp_path / "cfg" / "config.yaml"
    res = deploy_local(pkg, str(model_dir), str(config_path), runtime_url)

    assert res["transport"] == "local"
    assert (model_dir / "m.opt.param").exists()
    assert config_path.exists()
    assert res["reload"]["reloaded"] is True


# ---- /deploy 路由组 ----

def test_model_dirs_scan(tmp_path):
    from deployer.routes import create_deploy_router

    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "a.opt.param").write_text("x", encoding="utf-8")
    (tmp_path / "m2").mkdir()  # 无模型

    app = FastAPI()
    app.include_router(create_deploy_router(model_scan_root=tmp_path))
    c = TestClient(app)
    r = c.get("/deploy/model-dirs")
    assert r.status_code == 200, r.text
    names = {d["name"] for d in r.json()["dirs"]}
    assert "m1" in names
    assert "m2" not in names


def test_deploy_api_local(client, tmp_path, runtime_url):
    src = tmp_path / "model_src"
    src.mkdir()
    (src / "m.opt.param").write_text("p", encoding="utf-8")
    (src / "m.opt.bin").write_text("b", encoding="utf-8")

    payload = {
        "model_dir": str(src),
        "model_name": "demo",
        "input_size": 320,
        "classes": ["a", "b"],
        "target": {
            "transport": "local",
            "model_dir": str(tmp_path / "out_model"),
            "config_path": str(tmp_path / "out_cfg" / "config.yaml"),
            "runtime_url": runtime_url,
        },
    }
    r = client.post("/deploy", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["reload"]["reloaded"] is True
    assert (tmp_path / "out_model" / "m.opt.param").exists()
    assert (tmp_path / "out_cfg" / "config.yaml").exists()


def test_deploy_api_missing_source(client, tmp_path, runtime_url):
    payload = {
        "model_dir": str(tmp_path / "nope"),
        "target": {"transport": "local", "runtime_url": runtime_url},
    }
    r = client.post("/deploy", json=payload)
    assert r.status_code == 422


def test_deploy_health(client, runtime_url):
    r = client.get("/deploy/health", params={"runtime_url": runtime_url})
    assert r.status_code == 200, r.text
    assert r.json()["model_name"] == "test"


def test_deploy_infer_returns_overlay(client, runtime_url):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    r = client.post(
        "/deploy/infer",
        files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")},
        data={"runtime_url": runtime_url},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["boxes"] == [{"x1": 10, "y1": 20, "x2": 40, "y2": 50,
                              "score": 0.95, "class_id": 0, "class_name": "crazing"}]
    assert body["overlay"].startswith("data:image/jpeg;base64,")
    assert body["fps"] is not None
    assert body["inference_ms"] == 12.5
    assert body["mem_kb"] == 123
