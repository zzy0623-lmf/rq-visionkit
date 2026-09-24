# -*- coding: utf-8 -*-
"""M3 部署打包（T2.3）：模型文件 + config.yaml → 部署包。

零代码部署的第一步：把用户选定的模型文件和部署参数打包成 runtime 可直接加载的部署包。
"""

import shutil
import yaml
from pathlib import Path

# 默认类别（NEU-DET 六类缺陷），部署表单可覆盖
DEFAULT_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def build_config(model_name: str = "yolox_neudet", version: str = "1.0.0",
                 input_size: int = 320, conf_thres: float = 0.01, nms_thres: float = 0.65,
                 quant: str = "fp32", input_blob: str = "in0", output_blob: str = "out0",
                 task: str = "detect", classes=None,
                 mean: list | None = None, norm: list | None = None) -> dict:
    """由部署表单参数生成 runtime 的 config 字典。

    task=classify 时可选传 mean/norm（分类模型预处理），缺省用 ncnn 默认
    mean=[127.5]*3, norm=[1/127.5]*3（对应 Keras 的 (x-127.5)/127.5 预处理）。
    """
    cfg = {
        "model_name": model_name,
        "version": version,
        "task": task,
        "input_size": int(input_size),
        "conf_thres": float(conf_thres),
        "nms_thres": float(nms_thres),
        "quant": quant,
        "input_blob": input_blob,
        "output_blob": output_blob,
        "classes": list(classes or DEFAULT_CLASSES),
    }
    if task == "classify":
        cfg["mean"] = list(mean) if mean else [127.5, 127.5, 127.5]
        cfg["norm"] = list(norm) if norm else [1 / 127.5, 1 / 127.5, 1 / 127.5]
    return cfg


def pack(model_files_dir: str, config: dict, out_dir: str, target_model_dir: str = "/data/model") -> Path:
    """把模型文件（*.opt.param / *.opt.bin）和 config.yaml 打包到 out_dir。

    config 里的 model_dir 会被改写为 target_model_dir（目标设备上的部署路径），
    因为模型文件在打包时已经复制到部署包内，运行时按目标路径加载。
    """
    src = Path(model_files_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for p in sorted(src.glob("*.opt.param")):
        shutil.copy(p, dst / p.name)
        bin_ = p.with_suffix(".bin")
        if bin_.exists():
            shutil.copy(bin_, dst / bin_.name)
        copied.append(p.name)
    if not copied:
        raise FileNotFoundError(f"{model_files_dir} 下没有 *.opt.param 模型文件")

    cfg = dict(config)
    cfg["model_dir"] = target_model_dir
    (dst / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return dst
