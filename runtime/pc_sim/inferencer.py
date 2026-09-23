# -*- coding: utf-8 -*-
"""M4 端侧运行时（PC 仿真版）推理封装：ncnn-python 加载 + YOLOX 后处理。

模型文件与代码分离：由 config.yaml 指定 model_dir / input_size / 阈值 / 类别，
运行时只负责按配置加载与推理。
"""

from pathlib import Path

import cv2
import ncnn
import numpy as np


class NcnnInferencer:
    def __init__(self, cfg: dict):
        self.model_dir = Path(cfg["model_dir"])
        self.input_size = int(cfg.get("input_size", 320))
        self.conf_thres = float(cfg.get("conf_thres", 0.01))
        self.nms_thres = float(cfg.get("nms_thres", 0.65))
        self.classes = list(cfg.get("classes", []))
        self.input_blob = cfg.get("input_blob", "in0")
        self.output_blob = cfg.get("output_blob", "out0")
        self.num_classes = len(self.classes)
        self.net = None

    # ---- 模型加载 ----

    def load(self) -> None:
        params = sorted(self.model_dir.glob("*.opt.param"))
        if not params:
            raise FileNotFoundError(f"{self.model_dir} 下没有 *.opt.param 模型文件")
        param = params[0]
        bin_ = param.with_suffix(".bin")
        if not bin_.exists():
            raise FileNotFoundError(f"缺少权重文件: {bin_}")

        net = ncnn.Net()
        if net.load_param(str(param)) != 0:
            raise RuntimeError(f"load_param 失败: {param}")
        if net.load_model(str(bin_)) != 0:
            raise RuntimeError(f"load_model 失败: {bin_}")
        self.net = net

    # ---- 预处理 ----

    def _preproc(self, img_bgr):
        size = self.input_size
        h, w = img_bgr.shape[:2]
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        r = min(size / h, size / w)
        rh, rw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded[:rh, :rw] = resized
        return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r

    # ---- YOLOX decode ----

    def _decode(self, preds):
        strides = [8, 16, 32]
        hsizes = [self.input_size // s for s in strides]
        wsizes = [self.input_size // s for s in strides]
        grids, exp_strides = [], []
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            exp_strides.append(np.full((hsize * wsize, 1), stride))
        grids = np.concatenate(grids, 0)
        exp_strides = np.concatenate(exp_strides, 0)
        out = preds.copy()
        out[:, :2] = (out[:, :2] + grids) * exp_strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * exp_strides
        return out

    @staticmethod
    def _nms(boxes, scores, nms_thr):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= nms_thr)[0] + 1]
        return keep

    def _postprocess(self, preds):
        x, y, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        obj = preds[:, 4]
        cls = preds[:, 5:]
        boxes = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
        cls_conf = cls.max(1)
        cls_pred = cls.argmax(1)
        score = obj * cls_conf
        mask = score >= self.conf_thres
        boxes, score, cls_pred = boxes[mask], score[mask], cls_pred[mask]
        if len(boxes) == 0:
            return []
        final = []
        for c in range(self.num_classes):
            cm = cls_pred == c
            if cm.sum() == 0:
                continue
            keep = self._nms(boxes[cm], score[cm], self.nms_thres)
            for k in keep:
                final.append({
                    "x1": float(boxes[cm][k][0]), "y1": float(boxes[cm][k][1]),
                    "x2": float(boxes[cm][k][2]), "y2": float(boxes[cm][k][3]),
                    "score": float(score[cm][k]), "class_id": c,
                    "class_name": self.classes[c] if c < self.num_classes else str(c),
                })
        return final

    # ---- 推理入口 ----

    def infer(self, img_bgr) -> tuple[list, float]:
        """返回 (boxes, inference_ms)。boxes 元素为 {x1,y1,x2,y2,score,class_id,class_name}。"""
        import time

        if self.net is None:
            raise RuntimeError("模型未加载，先调用 load()")

        x, r = self._preproc(img_bgr)
        ex = self.net.create_extractor()
        ex.input(self.input_blob, ncnn.Mat(x))
        t0 = time.perf_counter()
        ret, out = ex.extract(self.output_blob)
        inference_ms = (time.perf_counter() - t0) * 1000
        if ret != 0:
            raise RuntimeError(f"extract 失败，返回码 {ret}")

        arr = np.array(out)
        n_preds = 4 + 1 + self.num_classes
        n_anchors = sum((self.input_size // s) ** 2 for s in (8, 16, 32))
        preds = self._decode(arr.reshape(n_anchors, n_preds))
        boxes = self._postprocess(preds)
        # 框坐标还原到原图尺寸
        for b in boxes:
            b["x1"] /= r
            b["y1"] /= r
            b["x2"] /= r
            b["y2"] /= r
        return boxes, inference_ms
