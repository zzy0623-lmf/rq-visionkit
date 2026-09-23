# -*- coding: utf-8 -*-
"""T1.1 验收自检：annotator/server 五个接口 + 辅助接口。"""

import io
import zipfile
from pathlib import Path

PNG_A = "sample_a.png"
PNG_B = "sample_b.png"


def make_png(path: Path, size=(24, 12)):
    """生成最小合法彩色 PNG（numpy + cv2 编码，中文路径安全）。"""
    import cv2
    import numpy as np

    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    path.write_bytes(buf.tobytes())


def _import_two(client, tmp_path):
    (tmp_path / PNG_A).write_bytes(b"")  # 占位，下面用有效图片覆盖
    make_png(tmp_path / PNG_A)
    make_png(tmp_path / PNG_B)
    (tmp_path / "readme.txt").write_text("not an image", encoding="utf-8")
    r = client.post("/api/images/import", data={"folder": str(tmp_path)})
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 2  # txt 被跳过
    return r.json()["images"]


def test_import_folder(client, tmp_path):
    r = client.post("/api/images/import", data={"folder": str(tmp_path / "no_such_dir")})
    assert r.status_code == 422
    images = _import_two(client, tmp_path)
    assert all(i["width"] == 24 and i["height"] == 12 for i in images)


def test_import_zip(client, tmp_path):
    make_png(tmp_path / "in_zip.png")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(tmp_path / "in_zip.png", "nested/in_zip.png")
    r = client.post("/api/images/import", files={"file": ("pack.zip", buf.getvalue())})
    assert r.status_code == 200, r.text
    assert r.json()["imported"] == 1

    buf2 = io.BytesIO(b"not a zip")
    r = client.post("/api/images/import", files={"file": ("bad.zip", buf2.getvalue())})
    assert r.status_code == 415


def test_capture(client, tmp_path):
    import cv2
    import numpy as np

    arr = np.zeros((12, 24, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    r = client.post(
        "/api/images/capture",
        files={"file": ("frame.jpg", buf.tobytes())},
        data={"device": "pc"},
    )
    assert r.status_code == 200, r.text
    image = r.json()
    assert image["device"] == "pc"

    # 采集帧进列表，且可回读原图
    r = client.get("/api/images?status=unlabeled")
    assert r.json()["total"] == 1
    r = client.get(f"/api/images/{image['id']}/file")
    assert r.status_code == 200
    assert r.content == buf.tobytes()


def test_list_status_filter(client, tmp_path):
    _import_two(client, tmp_path)
    r = client.get("/api/images?status=unlabeled")
    assert r.status_code == 200
    assert r.json()["total"] == 2
    r = client.get("/api/images?status=labeled")
    assert r.json()["total"] == 0
    r = client.get("/api/images?status=bogus")
    assert r.status_code == 422


def test_annotation_roundtrip(client, tmp_path):
    images = _import_two(client, tmp_path)
    image_id = images[0]["id"]

    # 空标注可读
    r = client.get(f"/api/annotations/{image_id}")
    assert r.status_code == 200
    assert r.json()["boxes"] == []

    # 未建类别时写框 → 422（class_id 越界）
    r = client.put(
        f"/api/annotations/{image_id}",
        json={"boxes": [{"class_id": 0, "x1": 1, "y1": 1, "x2": 10, "y2": 8}], "complete": False},
    )
    assert r.status_code == 422

    # 建两个类别后再写
    cid = client.post("/api/categories", json={"name": "crazing"}).json()["class_id"]
    assert cid == 0
    client.post("/api/categories", json={"name": "scratches"})
    box = {"class_id": 0, "x1": 2, "y1": 2, "x2": 22, "y2": 10}
    r = client.put(
        f"/api/annotations/{image_id}",
        json={"boxes": [box], "complete": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["saved"] is True

    # 回读一致
    ann = client.get(f"/api/annotations/{image_id}").json()
    assert ann["boxes"] == [box]
    assert ann["complete"] is True
    assert [c["name"] for c in ann["classes"]] == ["crazing", "scratches"]

    # 越界框被拒
    r = client.put(
        f"/api/annotations/{image_id}",
        json={"boxes": [{"class_id": 0, "x1": 0, "y1": 0, "x2": 999, "y2": 10}], "complete": False},
    )
    assert r.status_code == 422

    # 标注后状态变为 labeled
    r = client.get("/api/images?status=labeled")
    assert r.json()["total"] == 1


def _seed_labeled(client, n, class_id=0):
    """经 capture 导入 n 张图并全部标注（全图框），返回 image id 列表。"""
    import cv2
    import numpy as np

    arr = np.zeros((12, 24, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    ids = []
    for i in range(n):
        r = client.post("/api/images/capture", files={"file": (f"img_{i}.png", buf.tobytes())})
        assert r.status_code == 200, r.text
        iid = r.json()["id"]
        r = client.put(
            f"/api/annotations/{iid}",
            json={"boxes": [{"class_id": class_id, "x1": 0, "y1": 0, "x2": 24, "y2": 12}], "complete": True},
        )
        assert r.status_code == 200, r.text
        ids.append(iid)
    return ids


def _read_zip(r):
    import zipfile

    return zipfile.ZipFile(io.BytesIO(r.content))


def test_export_yolo_format(client, tmp_path):
    """单张导出：验证目录结构 + classes.txt + YOLO 归一化坐标格式。"""
    client.post("/api/categories", json={"name": "defect"})
    _seed_labeled(client, 1)
    r = client.post("/api/dataset/export")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    with _read_zip(r) as zf:
        names = zf.namelist()
        assert "classes.txt" in names
        assert zf.read("classes.txt").decode() == "0 defect\n"
        label_names = [n for n in names if n.startswith("labels/")]
        assert len(label_names) == 1
        # 全图框 → 中心 0.5,0.5 宽高 1.0,1.0
        assert zf.read(label_names[0]).decode().strip() == "0 0.500000 0.500000 1.000000 1.000000"


def test_export_train_val_split(client, tmp_path):
    """默认 8:2 划分：10 张 → images/train 8 + images/val 2。"""
    client.post("/api/categories", json={"name": "defect"})
    _seed_labeled(client, 10)
    r = client.post("/api/dataset/export")
    assert r.status_code == 200, r.text
    with _read_zip(r) as zf:
        names = zf.namelist()
        train_imgs = [n for n in names if n.startswith("images/train/")]
        val_imgs = [n for n in names if n.startswith("images/val/")]
        train_lbls = [n for n in names if n.startswith("labels/train/")]
        val_lbls = [n for n in names if n.startswith("labels/val/")]
        assert len(train_imgs) == 8 and len(val_imgs) == 2
        assert len(train_lbls) == 8 and len(val_lbls) == 2


def test_export_split_configurable(client, tmp_path):
    """split 可配：split=0.5 → train 5 / val 5。"""
    client.post("/api/categories", json={"name": "defect"})
    _seed_labeled(client, 10)
    r = client.post("/api/dataset/export?split=0.5")
    assert r.status_code == 200, r.text
    with _read_zip(r) as zf:
        names = zf.namelist()
        assert len([n for n in names if n.startswith("images/train/")]) == 5
        assert len([n for n in names if n.startswith("images/val/")]) == 5

    # 越界 split 被拒
    assert client.post("/api/dataset/export?split=0.3").status_code == 422
    assert client.post("/api/dataset/export?split=1.0").status_code == 422


def test_export_class_id_remap(client, tmp_path):
    """删除类别留空洞后，导出 class_id 重新映射为连续 0..n-1。"""
    client.post("/api/categories", json={"name": "a"})  # id 0
    client.post("/api/categories", json={"name": "b"})  # id 1（将删除）
    client.post("/api/categories", json={"name": "c"})  # id 2
    client.delete("/api/categories/1")
    # 用 id=2 标注，导出后应映射为 1
    _seed_labeled(client, 1, class_id=2)
    r = client.post("/api/dataset/export")
    assert r.status_code == 200, r.text
    with _read_zip(r) as zf:
        assert zf.read("classes.txt").decode() == "0 a\n1 c\n"
        label = [n for n in zf.namelist() if n.startswith("labels/")][0]
        assert zf.read(label).decode().startswith("1 ")  # 原 id=2 → 连续 id=1


def test_export_reproducible(client, tmp_path):
    """固定 seed：两次导出的 train/val 划分一致（可复现）。"""
    client.post("/api/categories", json={"name": "defect"})
    _seed_labeled(client, 10)

    def split_names():
        r = client.post("/api/dataset/export")
        with _read_zip(r) as zf:
            names = zf.namelist()
            train = sorted(n for n in names if n.startswith("images/train/"))
            val = sorted(n for n in names if n.startswith("images/val/"))
            return train, val

    t1, v1 = split_names()
    t2, v2 = split_names()
    assert t1 == t2
    assert v1 == v2


def test_empty_export_rejected(client, tmp_path):
    _import_two(client, tmp_path)
    r = client.post("/api/dataset/export")
    assert r.status_code == 422


def test_openapi_docs(client):
    """验收标准：接口有 OpenAPI 文档页。"""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for p in (
        "/api/images/import",
        "/api/images/capture",
        "/api/images",
        "/api/annotations/{image_id}",
        "/api/dataset/export",
    ):
        assert p in paths, f"缺少接口 {p}"
    r = client.get("/docs")
    assert r.status_code == 200


def test_category_duplicate(client):
    assert client.post("/api/categories", json={"name": "x"}).json()["class_id"] == 0
    assert client.post("/api/categories", json={"name": "x"}).json()["class_id"] == 0  # 重名复用
    assert len(client.get("/api/categories").json()["classes"]) == 1


def test_category_delete(client, tmp_path):
    images = _import_two(client, tmp_path)
    client.post("/api/categories", json={"name": "used"})
    client.post("/api/categories", json={"name": "unused"})
    client.put(
        f"/api/annotations/{images[0]['id']}",
        json={"boxes": [{"class_id": 0, "x1": 0, "y1": 0, "x2": 24, "y2": 12}], "complete": False},
    )

    # 被引用的类别拒绝删除
    assert client.delete("/api/categories/0").status_code == 422
    # 未被引用的可删
    assert client.delete("/api/categories/1").status_code == 200
    # 删除后，id 空洞存在但引用它的标注仍可正常更新（按 id 存在性校验，不按 len）
    r = client.put(
        f"/api/annotations/{images[0]['id']}",
        json={"boxes": [{"class_id": 0, "x1": 1, "y1": 1, "x2": 23, "y2": 11}], "complete": True},
    )
    assert r.status_code == 200
    # 不存在的 id 404
    assert client.delete("/api/categories/99").status_code == 404