# ============================================================
# calibration.py - 相机标定与坐标映射模块（重构版）
# 基于嵌入式AI视觉的机械臂智能分拣系统
#
# 【重要说明】本文件为「重构代码」，非《软著鉴别材料·源代码》PDF 中的原始代码。
# 原始 calibration.py 位于软著 PDF 的「前30页+后30页」之间的被截断段，故缺失。
# 本实现严格依据以下两处证据重构，未凭空捏造接口：
#   1) 《软著设计说明书》第 5.4 节「相机标定与坐标映射」：四点透视变换、Hough 圆检测、
#      DLT 求解 3×3 透视矩阵、cv2.getPerspectiveTransform / cv2.findHomography(RANSAC)；
#   2) main.py / sorting_controller.py 中对本模块的实际调用点（方法名与参数签名）。
# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from config import Config, get_config
from logger import get_logger
from utils import apply_homography, compute_homography


class CircleDetector:
    """
    圆形标记检测器
    用于相机标定时定位工作区四角的高对比度圆形标记，
    采用 Hough 圆检测返回各标记的像素坐标（cx, cy）与半径 r。
    """
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("circle_detector")
        # 标定标记直径（mm）仅作记录，不影响检测
        self.marker_diameter_mm = self.config.get(
            "vision.coordinate_mapping.calibration_marker_diameter_mm", 20
        )

    def detect_circles(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        检测图像中的圆形标记，返回 [(cx, cy, r), ...] 列表（按半径降序）。
        """
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        # Hough 圆检测：dp=1.2, minDist 限制相邻圆心距, 限定半径范围
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=60,
            param1=120,
            param2=40,
            minRadius=8,
            maxRadius=60,
        )
        result: List[Tuple[int, int, int]] = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                result.append((int(c[0]), int(c[1]), int(c[2])))
        # 按半径降序，便于取面积最大的 4 个标记
        result.sort(key=lambda c: c[2], reverse=True)
        return result


class CameraCalibrator:
    """
    相机标定器
    通过四点（或更多）像素—世界坐标对应点，求解透视变换矩阵 H，
    将像素坐标 (u, v) 映射到机械臂工作平面世界坐标 (X, Y)。
    与常规手眼标定（AX=XB）不同，本系统相机悬于工作区正上方、
    基座固定，故采用简化的四点透视变换（DLT）。
    """
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("calibration")
        map_cfg = self.config.get("vision.coordinate_mapping", {})
        self.homography_path = map_cfg.get(
            "homography_matrix_path", "config/homography_matrix.npy"
        )
        self.use_ransac = map_cfg.get("use_ransac", True)
        self.ransac_thresh = map_cfg.get("ransac_reproj_threshold", 3.0)
        self.calibration_points = map_cfg.get("calibration_points", 4)
        # 标定点存储
        self._pixel_points: List[Tuple[float, float]] = []
        self._world_points: List[Tuple[float, float]] = []
        self._names: List[str] = []
        self._homography: Optional[np.ndarray] = None
        self._reprojection_error: float = 0.0

    def clear_points(self) -> None:
        """清空已添加的标定点。"""
        self._pixel_points.clear()
        self._world_points.clear()
        self._names.clear()
        self._homography = None
        self._reprojection_error = 0.0
        self.logger.info("标定点已清空")

    def add_calibration_point(self, px: float, py: float,
                              wx: float, wy: float,
                              name: str = "") -> bool:
        """添加一组像素坐标 (px, py) 与对应世界坐标 (wx, wy) 的标定点。"""
        self._pixel_points.append((float(px), float(py)))
        self._world_points.append((float(wx), float(wy)))
        self._names.append(name)
        return True

    def calibrate(self) -> bool:
        """
        用已添加的标定点求解透视变换矩阵 H。
        4 点用 getPerspectiveTransform，>4 点用 findHomography（支持 RANSAC）。
        """
        n = len(self._pixel_points)
        if n < 4:
            self.logger.error(f"标定点不足: {n}/4")
            return False
        src = np.array(self._pixel_points, dtype=np.float32)
        dst = np.array(self._world_points, dtype=np.float32)
        H = compute_homography(src, dst, use_ransac=self.use_ransac,
                               ransac_thresh=self.ransac_thresh)
        if H is None:
            self.logger.error("透视矩阵求解失败")
            return False
        self._homography = H
        self._reprojection_error = self._compute_reprojection_error(src, dst, H)
        self._save_homography(H)
        self.logger.info(f"标定完成，重投影误差: {self._reprojection_error:.3f} mm")
        return True

    def is_calibrated(self) -> bool:
        """是否已完成标定（存在有效透视矩阵）。"""
        return self._homography is not None

    def get_reprojection_error(self) -> float:
        """返回最近一次标定的重投影误差（mm）。"""
        return self._reprojection_error

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        将像素坐标 (u, v) 映射为世界坐标 (X, Y)。
        未标定时按近似线性映射退化处理。
        """
        if self._homography is not None:
            x, y = apply_homography(self._homography, u, v)
            return float(x), float(y)
        # 未标定退化：按 150×200mm 工作区近似线性映射
        x = (u / 1280.0) * 200.0 - 100.0
        y = (v / 720.0) * 150.0 - 75.0
        return float(x), float(y)

    def _compute_reprojection_error(self, src: np.ndarray, dst: np.ndarray,
                                    H: np.ndarray) -> float:
        """计算透视变换的重投影误差（mm）。"""
        pts = np.hstack([src, np.ones((len(src), 1))])
        proj = H @ pts.T
        proj = (proj[:2] / proj[2]).T
        errors = np.linalg.norm(proj - dst, axis=1)
        return float(np.mean(errors))

    def _save_homography(self, H: np.ndarray) -> None:
        """将透视矩阵保存到配置文件指定的路径。"""
        try:
            os.makedirs(os.path.dirname(self.homography_path), exist_ok=True)
            np.save(self.homography_path, H)
        except OSError as e:
            self.logger.warning(f"透视矩阵保存失败: {e}")

    def load_homography(self) -> bool:
        """从文件加载透视矩阵。"""
        try:
            if os.path.exists(self.homography_path):
                self._homography = np.load(self.homography_path)
                self.logger.info(f"已加载透视矩阵: {self.homography_path}")
                return True
        except (IOError, ValueError) as e:
            self.logger.warning(f"透视矩阵加载失败: {e}")
        return False
# ============================================================
