# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision.py - 视觉处理模块
基于嵌入式AI视觉的机械臂智能分拣系统
负责图像预处理、背景分割、目标轮廓提取和物品定位
采用HSV颜色空间阈值分割与形态学处理实现工作区物品的可靠检测
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from config import Config, get_config
from logger import get_logger
from utils import BoundingBox, get_contour_center, get_contour_area, get_bounding_rect
class VisionProcessor:
    """
    视觉处理器
    对相机图像进行预处理、背景分割和目标定位
    输出物品的外接矩形与中心坐标，供后续分类和坐标映射使用
    """
    def __init__(self, config: Optional[Config] = None):

        self.config = config or get_config()
        self.logger = get_logger().get_logger("vision")
        pre_cfg = self.config.get("vision.image_preprocessing", {})
        self.blur_kernel = pre_cfg.get("blur_kernel_size", 3)
        self.denoise_enabled = pre_cfg.get("denoise_enabled", True)
        seg_cfg = self.config.get("vision.background_segmentation", {})
        self.hsv_lower = np.array(seg_cfg.get("hsv_lower", [0, 0, 50]), dtype=np.uint8)
        self.hsv_upper = np.array(seg_cfg.get("hsv_upper", [180, 50, 255]), dtype=np.uint8)
        self.morph_kernel_size = seg_cfg.get("morphology_kernel_size", 3)
        self.morph_iterations = seg_cfg.get("morphology_iterations", 1)
        self.min_area = seg_cfg.get("min_contour_area", 100)
        self.max_area = seg_cfg.get("max_contour_area", 50000)
        loc_cfg = self.config.get("vision.object_localization", {})
        self.center_type = loc_cfg.get("center_type", "centroid")
        self.max_objects = loc_cfg.get("max_objects", 5)
        self.sort_order = loc_cfg.get("sort_order", "descending")
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size)
        )
        self.logger.info("视觉处理器初始化完成")
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """图像预处理：高斯滤波降噪"""
        kernel = self.blur_kernel
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(frame, (kernel, kernel), 0)
    def segment_foreground(self, frame: np.ndarray) -> np.ndarray:
        """背景分割：HSV阈值分割加形态学开运算去噪"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self._morph_kernel,
            iterations=self.morph_iterations
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self._morph_kernel,
            iterations=self.morph_iterations
        )
        return mask
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测图像中的物品，返回按面积排序的目标列表"""
        if frame is None or frame.size == 0:
            return []
        processed = self.preprocess(frame)
        mask = self.segment_foreground(processed)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE

        )
        objects = []
        for contour in contours:
            area = get_contour_area(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = get_bounding_rect(contour)
            if self.center_type == "min_area_rect":
                rect = cv2.minAreaRect(contour)
                cx, cy = int(rect[0][0]), int(rect[0][1])
            else:
                cx, cy = get_contour_center(contour)
            objects.append({
                "bbox": BoundingBox(x, y, w, h),
                "center": (cx, cy),
                "area": area,
                "contour": contour,
            })
        reverse = (self.sort_order == "descending")
        objects.sort(key=lambda obj: obj["area"], reverse=reverse)
        return objects[:self.max_objects]
    def release(self) -> None:
        """释放视觉处理资源"""
        self.logger.info("视觉处理器已释放")
# ============================================================
