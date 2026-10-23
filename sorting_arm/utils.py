# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - 通用工具函数模块
基于嵌入式AI视觉的机械臂智能分拣系统
提供数学计算、坐标变换、图像处理辅助、文件操作、时间测量等通用工具
"""
import os
import sys
import time
import math
import json
import hashlib
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2
# ==================== 数学工具 ====================
def clamp(value: float, min_val: float, max_val: float) -> float:
    """将值限制在指定范围内"""
    return max(min_val, min(value, max_val))
def lerp(a: float, b: float, t: float) -> float:
    """线性插值"""
    return a + (b - a) * t

def smoothstep(t: float) -> float:
    """平滑阶跃函数（S曲线）"""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
def smootherstep(t: float) -> float:
    """更平滑的阶跃函数"""
    t = clamp(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)
def map_range(value: float, in_min: float, in_max: float,
              out_min: float, out_max: float) -> float:
    """将值从一个范围映射到另一个范围"""
    if in_max == in_min:
        return out_min
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
def normalize_angle_deg(angle: float) -> float:
    """将角度归一化到 [-180, 180]"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle
def normalize_angle_rad(angle: float) -> float:
    """将弧度归一化到 [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
def deg_to_rad(deg: float) -> float:
    """角度转弧度"""
    return deg * math.pi / 180.0
def rad_to_deg(rad: float) -> float:
    """弧度转角度"""
    return rad * 180.0 / math.pi
def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """二维欧氏距离"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def distance_3d(x1: float, y1: float, z1: float,
                x2: float, y2: float, z2: float) -> float:
    """三维欧氏距离"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
def point_in_rect(px: float, py: float, rx: float, ry: float,
                  rw: float, rh: float) -> bool:
    """判断点是否在矩形内"""
    return rx <= px <= rx + rw and ry <= py <= ry + rh
def point_in_circle(px: float, py: float, cx: float, cy: float, r: float) -> bool:
    """判断点是否在圆内"""
    return distance_2d(px, py, cx, cy) <= r
# ==================== 坐标变换工具 ====================
def pixel_to_normalized(u: int, v: int, width: int, height: int) -> Tuple[float, float]:
    """像素坐标转归一化坐标 [-1, 1]"""
    x = (u / width) * 2.0 - 1.0
    y = (v / height) * 2.0 - 1.0
    return x, y
def normalized_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """归一化坐标转像素坐标"""
    u = int((x + 1.0) / 2.0 * width)
    v = int((y + 1.0) / 2.0 * height)
    return clamp(u, 0, width - 1), clamp(v, 0, height - 1)
def apply_homography(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    """应用单应性矩阵进行坐标变换"""
    p = np.array([u, v, 1.0])
    result = H @ p
    w = result[2]
    if abs(w) < 1e-10:
        return 0.0, 0.0
    return result[0] / w, result[1] / w
def compute_homography(src_points: np.ndarray, dst_points: np.ndarray,
                       use_ransac: bool = True,
                       ransac_thresh: float = 3.0) -> np.ndarray:
    """计算单应性矩阵"""
    if use_ransac and len(src_points) >= 4:
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransac_thresh)
    else:
        H = cv2.getPerspectiveTransform(src_points.astype(np.float32),
                                         dst_points.astype(np.float32))
    return H

# ==================== 图像处理工具 ====================
def resize_with_aspect_ratio(image: np.ndarray, target_width: int,
                             target_height: int,
                             interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """保持宽高比的图像缩放（居中填充）"""
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas
def crop_roi(image: np.ndarray, x: int, y: int, w: int, h: int,
             padding_ratio: float = 0.0) -> np.ndarray:
    """裁剪感兴趣区域"""
    img_h, img_w = image.shape[:2]
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return image[y1:y2, x1:x2]
def get_contour_center(contour: np.ndarray) -> Tuple[int, int]:
    """获取轮廓的质心坐标"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy
def get_contour_area(contour: np.ndarray) -> float:
    """获取轮廓面积"""
    return cv2.contourArea(contour)
def get_bounding_rect(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """获取轮廓外接矩形"""
    return cv2.boundingRect(contour)
def draw_detection_overlay(image: np.ndarray, x: int, y: int, w: int, h: int,

                           label: str, confidence: float,
                           color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2) -> np.ndarray:
    """在图像上绘制检测结果叠加层"""
    result = image.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    text = f"{label}: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    cv2.rectangle(result, (x, y - text_h - baseline - 5),
                  (x + text_w, y), color, -1)
    cv2.putText(result, text, (x, y - baseline - 2), font, font_scale,
                (255, 255, 255), font_thickness, cv2.LINE_AA)
    return result
def draw_crosshair(image: np.ndarray, x: int, y: int,
                   size: int = 10, color: Tuple[int, int, int] = (0, 0, 255),
                   thickness: int = 2) -> np.ndarray:
    """绘制十字准星"""
    result = image.copy()
    cv2.line(result, (x - size, y), (x + size, y), color, thickness)
    cv2.line(result, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(result, (x, y), size // 2, color, thickness)
    return result
def add_info_overlay(image: np.ndarray, lines: List[str],
                     position: str = "top_left",
                     font_scale: float = 0.5) -> np.ndarray:
    """在图像上添加信息叠加层"""
    result = image.copy()
    h, w = result.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_height = int(20 * font_scale * 1.5)
    padding = 10
    for i, line in enumerate(lines):
        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        if position == "top_left":
            tx = padding
            ty = padding + text_h + i * line_height
        elif position == "top_right":
            tx = w - text_w - padding
            ty = padding + text_h + i * line_height
        elif position == "bottom_left":
            tx = padding
            ty = h - padding - (len(lines) - 1 - i) * line_height
        else:  # bottom_right

            tx = w - text_w - padding
            ty = h - padding - (len(lines) - 1 - i) * line_height
        cv2.putText(result, line, (tx, ty), font, font_scale,
                    (0, 255, 0), thickness, cv2.LINE_AA)
    return result
# ==================== 文件操作工具 ====================
def ensure_dir(dir_path: str) -> bool:
    """确保目录存在，不存在则创建"""
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except OSError:
        return False
def get_file_extension(file_path: str) -> str:
    """获取文件扩展名（小写，不含点）"""
    return os.path.splitext(file_path)[1].lower().lstrip('.')
def list_files_by_extension(dir_path: str, extensions: List[str]) -> List[str]:
    """按扩展名列出目录下的文件"""
    result = []
    if not os.path.isdir(dir_path):
        return result
    ext_set = set(e.lower() for e in extensions)
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath) and get_file_extension(fname) in ext_set:
            result.append(fpath)
    return sorted(result)
def load_json(file_path: str, default: Any = None) -> Any:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default
def save_json(data: Any, file_path: str, indent: int = 4) -> bool:
    """保存JSON文件"""
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except IOError:

        return False
def compute_file_md5(file_path: str) -> str:
    """计算文件MD5哈希值"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()
# ==================== 时间测量工具 ====================
class Timer:
    """
    高精度计时器
    支持分段计时、平均统计、上下文管理器用法
    """
    def __init__(self, name: str = ""):
        self.name = name
        self._start_time = 0.0
        self._elapsed = 0.0
        self._running = False
        self._history: List[float] = []
    def start(self) -> None:
        """开始计时"""
        self._start_time = time.perf_counter()
        self._running = True
    def stop(self) -> float:
        """停止计时并返回耗时（秒）"""
        if self._running:
            self._elapsed = time.perf_counter() - self._start_time
            self._history.append(self._elapsed)
            self._running = False
        return self._elapsed
    def get_elapsed(self) -> float:
        """获取最近一次耗时（秒）"""
        return self._elapsed
    def get_average(self) -> float:
        """获取历史平均耗时（秒）"""
        return sum(self._history) / len(self._history) if self._history else 0.0
    def reset(self) -> None:
        """清空历史记录"""
        self._history.clear()
        self._elapsed = 0.0
        self._running = False

    def __enter__(self) -> "Timer":
        self.start()
        return self
    def __exit__(self, *args) -> None:
        self.stop()
class MultiStageTimer:
    """
    多阶段计时器
    分别统计流水线各阶段耗时，用于分拣循环的性能分析
    """
    def __init__(self):
        self._stages: Dict[str, float] = {}
        self._stage_start: Dict[str, float] = {}
        self._total = Timer("total")
    def start_total(self) -> None:
        """开始整体计时"""
        self._stages.clear()
        self._total.start()
    def stop_total(self) -> float:
        """停止整体计时并返回总耗时（秒）"""
        return self._total.stop()
    def start_stage(self, name: str) -> None:
        """开始某个阶段计时"""
        self._stage_start[name] = time.perf_counter()
    def stop_stage(self, name: str) -> float:
        """停止某个阶段计时并返回耗时（秒）"""
        if name not in self._stage_start:
            return 0.0
        elapsed = time.perf_counter() - self._stage_start.pop(name)
        self._stages[name] = self._stages.get(name, 0.0) + elapsed
        return elapsed
    def get_summary(self) -> Dict[str, float]:
        """获取各阶段耗时汇总（秒）"""
        summary = dict(self._stages)
        summary["total"] = self._total.get_elapsed()
        return summary
# ==================== 线程安全工具 ====================
class ThreadSafeFlag:
    """线程安全的布尔标志"""
    def __init__(self, initial: bool = False):
        self._value = initial
        self._lock = threading.Lock()

    def set(self, value: bool) -> None:
        """设置标志值"""
        with self._lock:
            self._value = bool(value)
    def get(self) -> bool:
        """读取标志值"""
        with self._lock:
            return self._value
class ThreadSafeCounter:
    """线程安全的计数器"""
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()
    def increment(self, delta: int = 1) -> int:
        """增加计数并返回新值"""
        with self._lock:
            self._value += delta
            return self._value
    def decrement(self, delta: int = 1) -> int:
        """减少计数并返回新值"""
        return self.increment(-delta)
    def get(self) -> int:
        """读取当前计数值"""
        with self._lock:
            return self._value
    def reset(self) -> None:
        """计数归零"""
        with self._lock:
            self._value = 0
# ==================== 分拣数据结构 ====================
class SortingStatus(Enum):
    """分拣系统运行状态枚举"""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"
@dataclass
class BoundingBox:
    """目标外接矩形"""
    x: int

    y: int
    width: int
    height: int
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """转换为 (x, y, w, h) 元组"""
        return (self.x, self.y, self.width, self.height)
    @property
    def center(self) -> Tuple[int, int]:
        """矩形中心坐标"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    @property
    def area(self) -> int:
        """矩形面积"""
        return self.width * self.height
@dataclass
class DetectionResult:
    """物品检测结果"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    center: Tuple[int, int]
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    world_position: Optional[Tuple[float, float, float]] = None
@dataclass
class SortingResult:
    """单次分拣结果"""
    success: bool
    status: str
    message: str
    error_code: str = ""
    detection: Optional[DetectionResult] = None
    joint_angles: Optional[Dict[str, Any]] = None
    cycle_time_seconds: float = 0.0
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "error_code": self.error_code,
            "class_name": self.detection.class_name if self.detection else None,
            "confidence": self.detection.confidence if self.detection else None,
            "cycle_time_seconds": self.cycle_time_seconds,
        }

# ============================================================
