# -*- coding: utf-8 -*-
"""M3 部署下发传输（T2.3）：本地复制 + OpenSSH subprocess。

任务书原计划用 paramiko 做 SSH 下发，但 paramiko 许可证为 LGPL-2.1-or-later，
不在 Apache-2.0 / MIT / BSD 白名单内。按 docs/licenses.md 的决策改用系统 OpenSSH
客户端（Windows 10+ / Linux 自带）通过 subprocess 调用 scp/ssh，零第三方依赖、
零许可证风险。

统一契约：部署包目录（deployer/pack.py 产物）内含模型文件（*.opt.param / *.opt.bin）
与 config.yaml。传输负责把模型文件放到 target model_dir、把 config.yaml 放到 target
config_path，然后触发端侧运行时 POST /model/reload。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

_MODEL_SUFFIXES = {".param", ".bin"}


def split_package(pkg_dir: Path) -> tuple[list[Path], Path]:
    """把部署包拆成（模型文件列表, config.yaml 路径）。"""
    pkg = Path(pkg_dir)
    model_files = [f for f in sorted(pkg.iterdir()) if f.is_file() and f.suffix.lower() in _MODEL_SUFFIXES]
    config_file = pkg / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"部署包缺少 config.yaml: {pkg_dir}")
    if not model_files:
        raise FileNotFoundError(f"部署包缺少模型文件: {pkg_dir}")
    return model_files, config_file


# ---- 本地传输 ----

def clear_model_files(model_dir: Path) -> int:
    """清理目标目录里旧的 *.param / *.bin 模型文件，返回删除数量。

    部署语义是「替换」而非「追加」：若不清空旧模型，换模型部署时
    runtime 按 `glob("*.opt.param")` 排序取第一个，可能加载到残留的旧模型
    （见 T2.4 联调 loopA→loopB 时 loopB 推理出 0 框的问题）。
    """
    dst = Path(model_dir)
    dst.mkdir(parents=True, exist_ok=True)
    removed = 0
    for f in dst.iterdir():
        if f.is_file() and f.suffix.lower() in _MODEL_SUFFIXES:
            f.unlink()
            removed += 1
    return removed


def copy_files(files: list[Path], dst_dir: Path) -> list[str]:
    """复制一批文件到 dst_dir，返回文件名列表。"""
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for f in files:
        shutil.copy2(f, dst / f.name)
        names.append(f.name)
    return names


# ---- SSH 传输（OpenSSH subprocess） ----

def build_scp_argv(src: Path, host: str, username: str, remote_dir: str,
                   port: int = 22, identity_file: str | None = None) -> list[str]:
    """构造单文件 scp 命令（逐文件下发，避免 shell glob）。"""
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
    if identity_file:
        cmd += ["-i", str(identity_file)]
    cmd += ["-P", str(port), str(src), f"{username}@{host}:{remote_dir}/"]
    return cmd


def build_ssh_argv(host: str, username: str, command: str,
                   port: int = 22, identity_file: str | None = None) -> list[str]:
    """构造远端执行命令的 ssh 命令。"""
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
    if identity_file:
        cmd += ["-i", str(identity_file)]
    cmd += ["-p", str(port), f"{username}@{host}", command]
    return cmd


def run(argv: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


# ---- 触发运行时 reload（本地：直接 HTTP） ----

def trigger_reload(runtime_url: str) -> dict:
    """调用端侧运行时 POST /model/reload（本地传输用）。"""
    url = runtime_url.rstrip("/") + "/model/reload"
    req = urllib.request.Request(url, data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text.strip()}


def deploy_local(pkg_dir: str | Path, model_dir: str, config_path: str, runtime_url: str) -> dict:
    """本地传输：复制模型文件与 config.yaml，再触发 reload（PC 仿真路径）。"""
    model_files, config_file = split_package(Path(pkg_dir))
    removed = clear_model_files(Path(model_dir))
    names = copy_files(model_files, Path(model_dir))
    cfg_dst = Path(config_path)
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, cfg_dst)
    reload_info = trigger_reload(runtime_url)
    return {"transport": "local", "files": names, "model_dir": str(model_dir),
            "config_path": str(cfg_dst), "cleared_old": removed, "reload": reload_info}


def deploy_ssh(pkg_dir: str | Path, target: dict, runtime_url: str) -> dict:
    """SSH 传输：scp 模型文件到 model_dir、config.yaml 到 config_path，再 ssh curl reload。"""
    model_files, config_file = split_package(Path(pkg_dir))
    host = target["host"]
    username = target.get("username", "root")
    port = int(target.get("port", 22))
    identity_file = target.get("identity_file") or None
    model_dir = target["model_dir"]
    config_path = target["config_path"]

    # 先清理远端旧模型文件（替换语义，避免 runtime glob 加载到残留旧模型）
    rm_cmd = f"mkdir -p {model_dir} && rm -f {model_dir}/*.param {model_dir}/*.bin"
    cp = run(build_ssh_argv(host, username, rm_cmd, port, identity_file))
    if cp.returncode != 0:
        raise RuntimeError(f"ssh 清理旧模型失败: {cp.stderr.strip() or cp.stdout.strip()}")

    for f in model_files:
        cp = run(build_scp_argv(f, host, username, model_dir, port, identity_file))
        if cp.returncode != 0:
            raise RuntimeError(f"scp 模型文件失败: {cp.stderr.strip() or cp.stdout.strip()}")

    cp = run(build_scp_argv(config_file, host, username, str(Path(config_path).parent), port, identity_file))
    if cp.returncode != 0:
        raise RuntimeError(f"scp config.yaml 失败: {cp.stderr.strip() or cp.stdout.strip()}")

    reload_cmd = f"curl -s -X POST {runtime_url.rstrip('/')}/model/reload"
    cp = run(build_ssh_argv(host, username, reload_cmd, port, identity_file))
    if cp.returncode != 0:
        raise RuntimeError(f"ssh 触发 reload 失败: {cp.stderr.strip() or cp.stdout.strip()}")

    return {"transport": "ssh", "files": [f.name for f in model_files],
            "model_dir": model_dir, "config_path": config_path,
            "reload": _parse_json(cp.stdout.strip())}


def deploy(pkg_dir: str | Path, target: dict, runtime_url: str) -> dict:
    """统一入口：按 target.transport 分发本地 / SSH 下发。"""
    transport = target.get("transport", "local")
    if transport == "local":
        return deploy_local(pkg_dir, target["model_dir"], target["config_path"], runtime_url)
    if transport == "ssh":
        return deploy_ssh(pkg_dir, target, runtime_url)
    raise ValueError(f"不支持的传输方式: {transport}")
