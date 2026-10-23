# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifier.py - AI物品分类模块
基于嵌入式AI视觉的机械臂智能分拣系统
基于MobileNetV2的TensorFlow Lite INT8量化模型推理
负责物品图像的分类识别、置信度计算和推理性能统计
"""
import time
from typing import Any, Dict, List, Optional
import numpy as np
import cv2
from config import Config, get_config
from logger import get_logger
class ObjectClassifier:
    """

    物品分类器
    加载INT8量化的TFLite模型，对物品ROI图像进行分类推理
    支持XNNPACK代理加速和多线程推理
    """
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("classifier")
        cls_cfg = self.config.get("ai_model.classifier", {})
        self.model_path = cls_cfg.get("model_path", "models/mobilenetv2_int8.tflite")
        self.label_path = cls_cfg.get("label_path", "models/labels.txt")
        self.input_size = tuple(cls_cfg.get("input_size", [224, 224]))
        self.num_classes = cls_cfg.get("num_classes", 5)
        self.class_names = list(cls_cfg.get("class_names", []))
        self.top_k = cls_cfg.get("top_k", 3)
        self.num_threads = cls_cfg.get("num_threads", 4)
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        # 推理性能统计
        self._total_inferences = 0
        self._total_time_ms = 0.0
        self._load_labels()
        self._load_model()
    def _load_labels(self) -> None:
        """加载类别标签文件"""
        try:
            with open(self.label_path, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            if names:
                self.class_names = names
                self.logger.info(f"已加载 {len(names)} 个类别标签")
        except IOError:
            self.logger.warning(f"标签文件不存在: {self.label_path}，使用配置中的类别名")
    def _load_model(self) -> None:
        """加载TFLite量化模型并分配张量"""
        interpreter_cls = None
        try:
            from tflite_runtime.interpreter import Interpreter
            interpreter_cls = Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
                interpreter_cls = Interpreter
            except ImportError:
                self.logger.error("未安装tflite_runtime或TensorFlow，分类器不可用")
                raise ImportError("未安装TFLite运行时")
        self._interpreter = interpreter_cls(

            model_path=self.model_path, num_threads=self.num_threads
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self.logger.info(f"分类模型加载完成: {self.model_path}")
    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """图像预处理：缩放到模型输入尺寸并按量化参数转换类型"""
        w, h = self.input_size
        image = cv2.resize(roi, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_dtype = self._input_details[0]["dtype"]
        if input_dtype == np.uint8 or input_dtype == np.int8:
            # INT8量化模型：按输入张量的量化参数做定点化
            scale, zero_point = self._input_details[0]["quantization"]
            if input_dtype == np.int8:
                image = image.astype(np.float32) / 255.0
                image = image / scale + zero_point
                image = np.clip(image, -128, 127).astype(np.int8)
            else:
                image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)
    def predict(self, roi: np.ndarray) -> Dict[str, Any]:
        """对物品ROI图像进行分类，返回类别、置信度和Top-K结果"""
        if self._interpreter is None:
            raise RuntimeError("分类模型未加载")
        if roi is None or roi.size == 0:
            return {"class_id": -1, "class_name": "unknown",
                    "confidence": 0.0, "top_k": []}
        input_tensor = self._preprocess(roi)
        start = time.perf_counter()
        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._total_inferences += 1
        self._total_time_ms += elapsed_ms
        # 输出为定点量化值时先做反量化
        output_dtype = self._output_details[0]["dtype"]
        if output_dtype != np.float32:
            scale, zero_point = self._output_details[0]["quantization"]
            if scale > 0:
                output = (output.astype(np.float32) - zero_point) * scale
        probs = self._softmax(output)

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        class_name = self._get_class_name(class_id)
        top_indices = np.argsort(probs)[::-1][:self.top_k]
        top_k = [(self._get_class_name(int(i)), float(probs[i])) for i in top_indices]
        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "top_k": top_k,
        }
    def _get_class_name(self, class_id: int) -> str:
        """按类别索引获取类别名称"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"类别{class_id}"
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """数值稳定的softmax"""
        e = np.exp(x - np.max(x))
        return e / e.sum()
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取推理性能统计"""
        return {
            "total_inferences": self._total_inferences,
            "avg_inference_time_ms": round(
                self._total_time_ms / self._total_inferences, 2
            ) if self._total_inferences > 0 else 0.0,
        }
# ============================================================
