# -*- coding: utf-8 -*-
"""RQ-VisionKit M1 标注工具后端（T1.1 FastAPI 骨架）。

接口（对应任务书 T1.1）：
- POST /api/images/import          批量导入图片（zip 上传 或 本地文件夹路径）
- POST /api/images/capture         接收摄像头帧（PC 摄像头，预留板端回传 device 字段）
- GET  /api/images                图片列表（status=unlabeled|labeled|all）
- GET  /api/annotations/{image_id}  读标注
- PUT  /api/annotations/{image_id}  写标注
- POST /api/dataset/export         导出 YOLO 数据集 zip（images/ labels/ classes.txt）

额外两个最小接口（支撑 T1.2 前端「类别可增删」验收项）：
- GET  /api/categories            类别列表
- POST /api/categories            新增类别
- GET  /api/images/{image_id}/file 读取原图（前端详情页预览必需）

存储：JSON 落盘（任务书允许「JSON 存 SQLite 或直接落盘 JSON 文件」），
数据目录默认 annotator/server/data/，可用环境变量 RQ_DATA_DIR 覆盖（测试隔离用）。
"""

import io
import json
import random
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from deployer.routes import create_deploy_router

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_LOCK = threading.Lock()


def _default_data_dir() -> Path:
    import os

    return Path(os.environ.get("RQ_DATA_DIR", str(Path(__file__).parent / "data")))


class DatasetStore:
    """图片与标注的落盘存储（JSON 元数据 + 每图一个标注 JSON）。"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.annotations_dir = data_dir / "annotations"
        self.meta_path = data_dir / "dataset.json"
        for d in (self.images_dir, self.annotations_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            self._write_meta({"images": {}, "classes": []})

    # ---- 元数据读写（单进程+线程锁即可保证一致） ----

    def _read_meta(self) -> dict:
        with _LOCK:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _write_meta(self, meta: dict) -> None:
        with _LOCK:
            tmp = self.meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.meta_path)

    def _set_image(self, image: dict) -> None:
        meta = self._read_meta()
        meta["images"][image["id"]] = image
        self._write_meta(meta)

    def _get_image(self, image_id: str) -> dict:
        meta = self._read_meta()
        img = meta["images"].get(image_id)
        if img is None:
            raise HTTPException(status_code=404, detail=f"图片不存在: {image_id}")
        return img

    def _img_file(self, image_id: str) -> Path:
        img = self._get_image(image_id)
        return self.images_dir / (image_id + img["ext"])

    # ---- 图片尺寸（中文路径安全：np.fromfile + imdecode） ----

    @staticmethod
    def _img_size(path: Path):
        data = np.fromfile(str(path), dtype=np.uint8)
        arr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError(f"无法解码图片: {path.name}")
        h, w = arr.shape[:2]
        return w, h

    # ---- 业务操作 ----

    def add_image_bytes(self, raw: bytes, filename: str, device: str = "") -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(status_code=415, detail=f"不支持的格式: {filename}（支持 {sorted(ALLOWED_EXTS)}）")
        image_id = uuid.uuid4().hex[:12]
        self.images_dir.joinpath(image_id + ext).write_bytes(raw)
        size = self._img_size(self.images_dir / (image_id + ext))
        image = {
            "id": image_id,
            "filename": filename,
            "ext": ext,
            "width": size[0],
            "height": size[1],
            "device": device,
            "imported_at": time.time(),
            "box_count": 0,
            "complete": False,
        }
        self._set_image(image)
        return image

    def import_folder(self, folder: Path) -> list:
        """递归收集文件夹内图片。返回已导入图片元数据列表。"""
        imported = []
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
                imported.append(self.add_image_bytes(f.read_bytes(), f.name))
        return imported

    def list_images(self, status: str):
        meta = self._read_meta()
        items = []
        for img in meta["images"].values():
            labeled = img["box_count"] > 0 or img.get("complete", False)
            if status == "labeled" and not labeled:
                continue
            if status == "unlabeled" and labeled:
                continue
            items.append(img.copy())
        items.sort(key=lambda x: x["imported_at"])
        return {"total": len(items), "items": items}

    def get_annotation(self, image_id: str) -> dict:
        img = self._get_image(image_id)
        ann_path = self.annotations_dir / f"{image_id}.json"
        if ann_path.exists():
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
        else:
            ann = {"boxes": [], "complete": False}
        meta = self._read_meta()
        return {**ann, "image_id": image_id, "classes": meta["classes"]}

    def put_annotation(self, image_id: str, boxes: list, complete: bool) -> dict:
        img = self._get_image(image_id)
        meta = self._read_meta()
        valid_ids = {c["id"] for c in meta["classes"]}
        w, h = img["width"], img["height"]
        for b in boxes:
            cid = b["class_id"]
            if cid not in valid_ids:
                raise HTTPException(status_code=422, detail=f"类别 id 无效: {cid}（现有 {sorted(valid_ids)}）")
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise HTTPException(status_code=422, detail=f"框坐标越界: {b}")
        ann = {"boxes": boxes, "complete": complete}
        (self.annotations_dir / f"{image_id}.json").write_text(
            json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        img["box_count"] = len(boxes)
        img["complete"] = complete
        self._set_image(img)
        return {**ann, "image_id": image_id, "saved": True}

    def get_categories(self) -> list:
        return self._read_meta()["classes"]

    def add_category(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="类别名不能为空")
        meta = self._read_meta()
        for c in meta["classes"]:
            if c["name"] == name:
                return c["id"]
        cid = len(meta["classes"])
        meta["classes"].append({"id": cid, "name": name})
        self._write_meta(meta)
        return cid

    def export_yolo_zip(self, split: float = 0.8, seed: int = 42) -> bytes:
        """导出 YOLO 数据集 zip（T1.3）：images/{train,val}/ + labels/{train,val}/ + classes.txt。

        - YOLO 标注：class_id x_center y_center w h（0-1 归一化）
        - train/val 按 split 比例划分，固定 seed 保证可复现
        - class_id 重新映射为连续 0..n-1（类别经删除可能留空洞）
        - 仅导出有框的已标注图片
        """
        meta = self._read_meta()
        classes = sorted(meta["classes"], key=lambda c: c["id"])
        if not classes:
            raise HTTPException(status_code=422, detail="还没有类别，请先用 /api/categories 添加")
        id_map = {c["id"]: i for i, c in enumerate(classes)}  # 旧 id -> 连续新 id

        labeled = []
        for img in meta["images"].values():
            if img["box_count"] == 0:
                continue
            if (self.annotations_dir / f"{img['id']}.json").exists():
                labeled.append(img)
        if not labeled:
            raise HTTPException(status_code=422, detail="没有已标注图片可导出")

        rng = random.Random(seed)
        shuffled = labeled[:]
        rng.shuffle(shuffled)
        total = len(shuffled)
        if total >= 2:
            n_train = max(1, min(int(total * split), total - 1))
        else:
            n_train = total  # 仅 1 张时全进 train，val 为空

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("classes.txt", "\n".join(f"{i} {c['name']}" for i, c in enumerate(classes)) + "\n")
            for subset, imgs in (("train", shuffled[:n_train]), ("val", shuffled[n_train:])):
                for img in imgs:
                    ann = json.loads((self.annotations_dir / f"{img['id']}.json").read_text(encoding="utf-8"))
                    src_name = img["id"] + img["ext"]
                    w, h = img["width"], img["height"]
                    lines = []
                    for b in ann["boxes"]:
                        x_c = (b["x1"] + b["x2"]) / 2 / w
                        y_c = (b["y1"] + b["y2"]) / 2 / h
                        bw = (b["x2"] - b["x1"]) / w
                        bh = (b["y2"] - b["y1"]) / h
                        lines.append(f"{id_map[b['class_id']]} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
                    zf.write(self.images_dir / src_name, f"images/{subset}/{src_name}")
                    zf.writestr(f"labels/{subset}/{img['id']}.txt", "\n".join(lines) + "\n")
        return buf.getvalue()


def create_app(data_dir: Path | None = None) -> FastAPI:
    store = DatasetStore(data_dir or _default_data_dir())
    app = FastAPI(title="RQ-VisionKit Annotator", version="0.1.0")

    # M3 部署控制台（与 M1 共用进程，任务书 T2.3）
    app.include_router(create_deploy_router())

    class AnnotationIn(BaseModel):
        boxes: list = Field(default_factory=list, description="框列表：{class_id,x1,y1,x2,y2}")
        complete: bool = False

    @app.post("/api/images/import")
    def import_images(file: UploadFile | None = File(default=None), folder: str | None = Form(default=None)):
        """导入图片：上传 zip（file 字段），或填本地文件夹路径（folder 字段）。"""
        if file is not None:
            data = file.file.read()
            if not zipfile.is_zipfile(io.BytesIO(data)):
                raise HTTPException(status_code=415, detail="上传文件不是 zip")
            imported = []
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in sorted(zf.namelist()):
                    if Path(name).suffix.lower() in ALLOWED_EXTS:
                        imported.append(store.add_image_bytes(zf.read(name), Path(name).name))
        elif folder:
            src = Path(folder)
            if not src.is_dir():
                raise HTTPException(status_code=422, detail=f"文件夹不存在: {folder}")
            imported = store.import_folder(src)
        else:
            raise HTTPException(status_code=422, detail="需要上传 zip 文件或提供 folder 路径")
        return {"imported": len(imported), "images": imported}

    @app.post("/api/images/capture")
    def capture(file: UploadFile = File(...), device: str = Form(default="pc")):
        """接收摄像头帧，落盘为普通图片；device 预留板端回传来源。"""
        return store.add_image_bytes(file.file.read(), file.filename or "capture.jpg", device=device)

    @app.get("/api/images")
    def list_images(status: str = "all"):
        if status not in ("all", "unlabeled", "labeled"):
            raise HTTPException(status_code=422, detail="status 只能是 all|unlabeled|labeled")
        return store.list_images(status)

    @app.get("/api/images/{image_id}/file")
    def get_image_file(image_id: str):
        path = store._img_file(image_id)
        return FileResponse(path, media_type=f"image/{path.suffix.lstrip('.').replace('jpg', 'jpeg')}")

    @app.get("/api/annotations/{image_id}")
    def get_annotation(image_id: str):
        return store.get_annotation(image_id)

    @app.put("/api/annotations/{image_id}")
    def put_annotation(image_id: str, body: AnnotationIn):
        return store.put_annotation(image_id, body.boxes, body.complete)

    @app.get("/api/categories")
    def get_categories():
        return {"classes": store.get_categories()}

    @app.post("/api/categories")
    def add_category(body: dict):
        return {"class_id": store.add_category(str(body.get("name", "")))}

    @app.delete("/api/categories/{class_id}")
    def delete_category(class_id: int):
        """删除未被任何标注框引用的类别；被引用时返回 422（T1.3 导出前可再做重排处理）。"""
        meta = store._read_meta()
        ids = [c["id"] for c in meta["classes"]]
        if class_id not in ids:
            raise HTTPException(status_code=404, detail=f"类别不存在: {class_id}")
        used = [
            img["id"] for img in meta["images"].values()
        ]
        for image_id in used:
            p = store.annotations_dir / f"{image_id}.json"
            if p.exists():
                ann = json.loads(p.read_text(encoding="utf-8"))
                if any(b["class_id"] == class_id for b in ann["boxes"]):
                    raise HTTPException(status_code=422, detail="该类别仍被标注框引用，先修改或删除相关框")
        meta["classes"] = [c for c in meta["classes"] if c["id"] != class_id]
        store._write_meta(meta)
        return {"deleted": class_id}

    @app.post("/api/dataset/export")
    def export_dataset(split: float = 0.8):
        """导出 YOLO 数据集 zip；split 为 train 占比（默认 0.8，即 8:2）。"""
        if not (0.5 <= split <= 0.95):
            raise HTTPException(status_code=422, detail="split 需在 0.5~0.95 之间（train 占比）")
        data = store.export_yolo_zip(split=split)
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=yolo_dataset_{int(time.time())}.zip"},
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("annotator.server.main:app", host="127.0.0.1", port=8000, reload=False)