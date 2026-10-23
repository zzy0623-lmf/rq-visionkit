# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sorting_controller.py - 分拣主控编排模块
基于嵌入式AI视觉的机械臂智能分拣系统
协调视觉检测、AI分类、逆运动学求解和舵机控制，实现完整的自动分拣流程
采用有限状态机架构，支持单物品分拣、批量分拣、统计记录和异常处理
"""
import os
import time
import json
import threading
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import cv2
from config import Config, get_config
from logger import get_logger
from utils import (
    Timer, MultiStageTimer, SortingResult, SortingStatus,
    ThreadSafeCounter, ThreadSafeFlag, DetectionResult
)
from classifier import ObjectClassifier
from ik_solver import IKSolver
from servo_controller import ServoController, GripperController
from vision import VisionProcessor
from calibration import CameraCalibrator
from hal import HardwareAbstractionLayer
class SortingState(Enum):
    """分拣状态枚举"""
    IDLE = "IDLE"
    CAPTURE = "CAPTURE"
    LOCATE = "LOCATE"
    CLASSIFY = "CLASSIFY"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    REPORT = "REPORT"
    ERROR = "ERROR"
    STOPPED = "STOPPED"
@dataclass
class SortingStatistics:

    """分拣统计数据"""
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    recognition_failures: int = 0
    grasp_failures: int = 0
    unreachable_targets: int = 0
    low_confidence_skips: int = 0
    total_time_seconds: float = 0.0
    cycle_times: List[float] = field(default_factory=list)
    per_class_success: Dict[str, Dict[str, int]] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "recognition_failures": self.recognition_failures,
            "grasp_failures": self.grasp_failures,
            "unreachable_targets": self.unreachable_targets,
            "low_confidence_skips": self.low_confidence_skips,
            "success_rate": round(self.successful_cycles / self.total_cycles * 100, 2)
            if self.total_cycles > 0 else 0.0,
            "avg_cycle_time": round(sum(self.cycle_times) / len(self.cycle_times), 3)
            if self.cycle_times else 0.0,
            "per_class_success": self.per_class_success,
        }
        return result
class SortingController:
    """
    分拣主控制器
    协调所有子系统，实现完整的智能分拣流程
    采用有限状态机架构，支持自动运行、单步执行、暂停和停止
    """
    def __init__(self, config: Optional[Config] = None,
                 hal: Optional[HardwareAbstractionLayer] = None,
                 simulated: bool = False):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("sorting")
        self.hal = hal
        self.simulated = simulated
        # 子系统
        self.vision: Optional[VisionProcessor] = None
        self.classifier: Optional[ObjectClassifier] = None
        self.ik_solver: Optional[IKSolver] = None
        self.servo: Optional[ServoController] = None
        self.gripper: Optional[GripperController] = None
        self.calibrator: Optional[CameraCalibrator] = None
        # 状态管理
        self._state = SortingState.IDLE

        self._status = SortingStatus.IDLE
        self._running = ThreadSafeFlag(False)
        self._paused = ThreadSafeFlag(False)
        self._stop_requested = ThreadSafeFlag(False)
        self._emergency_stop = ThreadSafeFlag(False)
        # 计时器
        self._cycle_timer = Timer("cycle")
        self._stage_timer = MultiStageTimer()
        # 统计
        self.stats = SortingStatistics()
        # 放置区域配置
        sorting_config = self.config.get("sorting", {})
        self._drop_zones = sorting_config.get("drop_zones", {})
        self._default_zone = sorting_config.get("default_zone", "未知物品")
        self._max_cycles = sorting_config.get("max_cycles", 50)
        self._confidence_threshold = self.config.get(
            "ai_model.classifier.confidence_threshold", 0.8
        )
        # 当前检测结果
        self._current_detection: Optional[DetectionResult] = None
        self._current_frame: Optional[np.ndarray] = None
        # 回调函数
        self._on_state_change: Optional[callable] = None
        self._on_cycle_complete: Optional[callable] = None
        self._on_error: Optional[callable] = None
        self.logger.info("分拣控制器初始化完成")
    def initialize(self) -> bool:
        """初始化所有子系统"""
        self.logger.info("开始初始化分拣系统...")
        success = True
        try:
            # 初始化视觉处理器
            self.vision = VisionProcessor(self.config)
            # 初始化分类器
            model_path = self.config.get("ai_model.classifier.model_path")
            if os.path.exists(model_path):
                self.classifier = ObjectClassifier(self.config)
            else:
                self.logger.warning(f"模型文件不存在: {model_path}，分类器将使用模拟模式")
                self.classifier = None
            # 初始化逆运动学求解器
            self.ik_solver = IKSolver(self.config)

            # 初始化舵机控制器
            if self.hal and self.hal.pca9685:
                self.servo = ServoController(self.hal.pca9685, self.config)
                self.servo.initialize()
            else:
                self.logger.warning("无PCA9685硬件，舵机控制将使用模拟模式")
                self.servo = ServoController(None, self.config)
            # 初始化夹爪控制器
            self.gripper = GripperController(self.servo)
            # 初始化标定器
            self.calibrator = CameraCalibrator(self.config)
            # 移动到归位位置
            if self.servo:
                self.servo.move_to_home()
            self._state = SortingState.IDLE
            self._status = SortingStatus.IDLE
            self.logger.info("分拣系统初始化完成")
            return success
        except Exception as e:
            self.logger.error(f"分拣系统初始化失败: {e}", exc_info=True)
            return False
    def _set_state(self, new_state: SortingState) -> None:
        """设置状态并触发回调"""
        old_state = self._state
        self._state = new_state
        self.logger.debug(f"状态转换: {old_state.value} -> {new_state.value}")
        if self._on_state_change:
            self._on_state_change(old_state, new_state)
    def set_callbacks(self, on_state_change: Optional[callable] = None,
                     on_cycle_complete: Optional[callable] = None,
                     on_error: Optional[callable] = None) -> None:
        """设置回调函数"""
        self._on_state_change = on_state_change
        self._on_cycle_complete = on_cycle_complete
        self._on_error = on_error
    def run_single_cycle(self) -> SortingResult:
        """执行单次分拣循环"""
        if self._emergency_stop.get():
            return SortingResult(
                success=False, status="error",
                message="紧急停止状态", error_code="EMERGENCY_STOP"
            )
        self._cycle_timer.start()
        self._stage_timer.start_total()

        try:
            # ===== 状态1：图像采集 =====
            self._set_state(SortingState.CAPTURE)
            self._stage_timer.start_stage("capture")
            frame = self._capture_image()
            if frame is None:
                return self._cycle_fail("图像采集失败", "CAPTURE_FAIL")
            self._current_frame = frame
            self._stage_timer.stop_stage("capture")
            # ===== 状态2：物品定位 =====
            self._set_state(SortingState.LOCATE)
            self._stage_timer.start_stage("locate")
            detection = self._locate_object(frame)
            if detection is None:
                return self._cycle_fail("未检测到物品", "NO_OBJECT")
            self._current_detection = detection
            self._stage_timer.stop_stage("locate")
            # ===== 状态3：物品分类 =====
            self._set_state(SortingState.CLASSIFY)
            self._stage_timer.start_stage("classify")
            classification = self._classify_object(frame, detection)
            if classification is None:
                self.stats.low_confidence_skips += 1
                return self._cycle_fail("分类置信度过低", "LOW_CONFIDENCE")
            detection.class_name = classification["class_name"]
            detection.class_id = classification["class_id"]
            detection.confidence = classification["confidence"]
            detection.top_k = classification.get("top_k", [])
            self._stage_timer.stop_stage("classify")
            # ===== 状态4：坐标映射与路径规划 =====
            self._set_state(SortingState.PLAN)
            self._stage_timer.start_stage("plan")
            plan_result = self._plan_motion(detection)
            if not plan_result["reachable"]:
                self.stats.unreachable_targets += 1
                return self._cycle_fail("目标不可达", "UNREACHABLE")
            self._stage_timer.stop_stage("plan")
            # ===== 状态5：执行抓取放置 =====
            self._set_state(SortingState.EXECUTE)
            self._stage_timer.start_stage("execute")
            execute_success = self._execute_sort(plan_result, detection)
            self._stage_timer.stop_stage("execute")
            if not execute_success:
                self.stats.grasp_failures += 1
                return self._cycle_fail("抓取执行失败", "GRASP_FAIL")

            # ===== 状态6：结果报告 =====
            self._set_state(SortingState.REPORT)
            result = self._report_success(detection, plan_result)
            total_time = self._cycle_timer.stop()
            self._stage_timer.stop_total()
            result.cycle_time_seconds = round(total_time, 3)
            # 更新统计
            self._update_stats(result)
            self._set_state(SortingState.IDLE)
            return result
        except Exception as e:
            self._cycle_timer.stop()
            self.logger.error(f"分拣循环异常: {e}", exc_info=True)
            if self._on_error:
                self._on_error(e)
            return SortingResult(
                success=False, status="error",
                message=f"异常: {str(e)}", error_code="EXCEPTION"
            )
    def _capture_image(self) -> Optional[np.ndarray]:
        """采集图像"""
        if self.hal and self.hal.camera:
            ret, frame = self.hal.camera.read()
            if ret:
                return frame
        return None
    def _locate_object(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """定位物品"""
        objects = self.vision.detect_objects(frame)
        if not objects:
            return None
        obj = objects[0]
        return DetectionResult(
            class_id=-1,
            class_name="unknown",
            confidence=0.0,
            bbox=obj["bbox"],
            center=obj["center"],
        )
    def _classify_object(self, frame: np.ndarray,
                        detection: DetectionResult) -> Optional[Dict]:
        """分类物品"""
        if self.classifier is None:
            # 模拟分类
            return {
                "class_id": 0,

                "class_name": "模拟物品",
                "confidence": 0.95,
                "top_k": [("模拟物品", 0.95)],
            }
        # 裁剪ROI
        x, y, w, h = detection.bbox.to_tuple()
        padding = int(min(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        roi = frame[y1:y2, x1:x2]
        result = self.classifier.predict(roi)
        if result["confidence"] < self._confidence_threshold:
            return None
        return result
    def _plan_motion(self, detection: DetectionResult) -> Dict[str, Any]:
        """规划运动路径"""
        # 像素坐标转世界坐标
        if self.calibrator and self.calibrator.is_calibrated():
            world_x, world_y = self.calibrator.pixel_to_world(
                detection.center[0], detection.center[1]
            )
        else:
            # 未标定时使用默认映射（模拟）
            world_x = detection.center[0] * 0.3 - 150
            world_y = detection.center[1] * 0.3 - 100
        detection.world_position = (world_x, world_y, 30.0)
        # 求解抓取点逆运动学
        pick_result = self.ik_solver.solve(world_x, world_y, 30.0)
        if not pick_result["reachable"]:
            return {"reachable": False, "message": pick_result["message"]}
        # 获取放置区域
        zone_config = self._drop_zones.get(
            detection.class_name,
            self._drop_zones.get(self._default_zone, {})
        )
        place_x = zone_config.get("x_mm", 200.0)
        place_y = zone_config.get("y_mm", 0.0)
        place_z = zone_config.get("z_mm", 80.0)
        # 求解放置点逆运动学
        place_result = self.ik_solver.solve(place_x, place_y, place_z)
        return {

            "reachable": True,
            "pick_angles": pick_result["angles"],
            "place_angles": place_result["angles"] if place_result["reachable"] else None,
            "pick_position": (world_x, world_y, 30.0),
            "place_position": (place_x, place_y, place_z),
            "class_name": detection.class_name,
        }
    def _execute_sort(self, plan: Dict, detection: DetectionResult) -> bool:
        """执行分拣动作"""
        if self.servo is None or self.gripper is None:
            return True  # 模拟模式
        try:
            pick_angles = plan["pick_angles"]
            place_angles = plan["place_angles"]
            # 1. 移动到抓取点上方
            approach_z = 70.0
            approach_result = self.ik_solver.solve(
                plan["pick_position"][0],
                plan["pick_position"][1],
                approach_z
            )
            if approach_result["reachable"]:
                self.servo.move_all_joints({
                    "J1": approach_result["theta1"],
                    "J2": approach_result["theta2"],
                    "J3": approach_result["theta3"],
                    "J4": approach_result["theta4"],
                })
                time.sleep(0.2)
            # 2. 打开夹爪
            self.gripper.open(delay=0.3)
            # 3. 下降到抓取点
            self.servo.move_all_joints({
                "J1": pick_angles[0],
                "J2": pick_angles[1],
                "J3": pick_angles[2],
                "J4": pick_angles[3],
            })
            time.sleep(0.3)
            # 4. 闭合夹爪抓取
            self.gripper.grip(force_level=3, delay=0.5)
            # 5. 抬起
            lift_result = self.ik_solver.solve(
                plan["pick_position"][0],
                plan["pick_position"][1],
                80.0
            )

            if lift_result["reachable"]:
                self.servo.move_all_joints({
                    "J1": lift_result["theta1"],
                    "J2": lift_result["theta2"],
                    "J3": lift_result["theta3"],
                    "J4": lift_result["theta4"],
                })
                time.sleep(0.2)
            # 6. 移动到放置点
            if place_angles:
                self.servo.move_all_joints({
                    "J1": place_angles[0],
                    "J2": place_angles[1],
                    "J3": place_angles[2],
                    "J4": place_angles[3],
                })
                time.sleep(0.3)
                # 7. 打开夹爪放置
                self.gripper.release(delay=0.3)
            # 8. 归位
            self.servo.move_to_home()
            return True
        except Exception as e:
            self.logger.error(f"执行分拣动作失败: {e}")
            return False
    def _report_success(self, detection: DetectionResult,
                       plan: Dict) -> SortingResult:
        """生成成功结果"""
        return SortingResult(
            success=True,
            status="success",
            message=f"成功分拣: {detection.class_name}",
            detection=detection,
            joint_angles={
                "pick": plan["pick_angles"],
                "place": plan.get("place_angles"),
            },
        )
    def _cycle_fail(self, message: str, error_code: str) -> SortingResult:
        """处理循环失败"""
        self._cycle_timer.stop()
        self.stats.failed_cycles += 1
        self._set_state(SortingState.IDLE)
        return SortingResult(
            success=False,
            status="failed",
            message=message,
            error_code=error_code,

        )
    def _update_stats(self, result: SortingResult) -> None:
        """更新统计数据"""
        self.stats.total_cycles += 1
        if result.success:
            self.stats.successful_cycles += 1
            class_name = result.detection.class_name if result.detection else "unknown"
            if class_name not in self.stats.per_class_success:
                self.stats.per_class_success[class_name] = {"success": 0, "total": 0}
            self.stats.per_class_success[class_name]["success"] += 1
            self.stats.per_class_success[class_name]["total"] += 1
        else:
            self.stats.failed_cycles += 1
        if result.cycle_time_seconds > 0:
            self.stats.cycle_times.append(result.cycle_time_seconds)
            self.stats.total_time_seconds += result.cycle_time_seconds
        if result.detection:
            class_name = result.detection.class_name
            if class_name not in self.stats.per_class_success:
                self.stats.per_class_success[class_name] = {"success": 0, "total": 0}
            self.stats.per_class_success[class_name]["total"] += 1
    def run(self, max_cycles: Optional[int] = None,
            delay_between_cycles: float = 0.5) -> SortingStatistics:
        """连续运行分拣"""
        if max_cycles is None:
            max_cycles = self._max_cycles
        self._running.set(True)
        self._stop_requested.set(False)
        self._status = SortingStatus.RUNNING
        self.logger.info(f"开始自动分拣，最多 {max_cycles} 次循环")
        for i in range(max_cycles):
            if self._stop_requested.get():
                self.logger.info("收到停止请求")
                break
            if self._emergency_stop.get():
                self.logger.warning("紧急停止")
                break
            while self._paused.get():
                time.sleep(0.1)
                if self._stop_requested.get():
                    break
            self.logger.info(f"--- 第 {i+1}/{max_cycles} 次分拣 ---")
            result = self.run_single_cycle()
            if result.success:

                self.logger.info(
                    f"成功: {result.detection.class_name} "
                    f"(置信度={result.detection.confidence:.3f}, "
                    f"耗时={result.cycle_time_seconds:.2f}s)"
                )
            else:
                self.logger.warning(f"失败: {result.message} ({result.error_code})")
            if self._on_cycle_complete:
                self._on_cycle_complete(i, result)
            time.sleep(delay_between_cycles)
        self._running.set(False)
        self._status = SortingStatus.IDLE
        self._set_state(SortingState.STOPPED)
        self.logger.info(
            f"分拣结束: 总计={self.stats.total_cycles}, "
            f"成功={self.stats.successful_cycles}, "
            f"成功率={self.stats.to_dict()['success_rate']}%"
        )
        return self.stats
    def run_async(self, max_cycles: Optional[int] = None) -> threading.Thread:
        """异步运行分拣（在新线程中）"""
        thread = threading.Thread(
            target=self.run,
            args=(max_cycles,),
            daemon=True
        )
        thread.start()
        return thread
    def stop(self) -> None:
        """停止分拣"""
        self._stop_requested.set(True)
        self._running.set(False)
        self.logger.info("请求停止分拣")
    def pause(self) -> None:
        """暂停分拣"""
        self._paused.set(True)
        self._status = SortingStatus.PAUSED
        self.logger.info("分拣已暂停")
    def resume(self) -> None:
        """恢复分拣"""
        self._paused.set(False)
        self._status = SortingStatus.RUNNING
        self.logger.info("分拣已恢复")

    def emergency_stop(self) -> None:
        """紧急停止"""
        self._emergency_stop.set(True)
        self._stop_requested.set(True)
        if self.servo:
            self.servo.emergency_stop()
        self._status = SortingStatus.ERROR
        self.logger.critical("执行紧急停止！")
    def release_emergency_stop(self) -> None:
        """解除紧急停止"""
        self._emergency_stop.set(False)
        if self.servo:
            self.servo.release_emergency_stop()
        self._status = SortingStatus.IDLE
        self.logger.info("紧急停止已解除")
    def get_state(self) -> SortingState:
        """获取当前状态"""
        return self._state
    def get_status(self) -> SortingStatus:
        """获取运行状态"""
        return self._status
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running.get()
    def is_paused(self) -> bool:
        """是否已暂停"""
        return self._paused.get()
    def get_statistics(self) -> SortingStatistics:
        """获取统计数据"""
        return self.stats
    def reset_statistics(self) -> None:
        """重置统计数据"""
        self.stats = SortingStatistics()
        self.logger.info("统计数据已重置")
    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        return self._current_frame
    def get_current_detection(self) -> Optional[DetectionResult]:
        """获取当前检测结果"""
        return self._current_detection
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {

            "state": self._state.value,
            "status": self._status.value,
            "statistics": self.stats.to_dict(),
            "stage_timing": self._stage_timer.get_summary(),
        }
        if self.classifier:
            summary["classifier"] = self.classifier.get_performance_stats()
        if self.ik_solver:
            summary["ik_solver"] = self.ik_solver.get_performance_stats()
        return summary
    def save_statistics(self, file_path: str) -> bool:
        """保存统计数据到文件"""
        try:
            data = {
                "statistics": self.stats.to_dict(),
                "performance": self.get_performance_summary(),
                "timestamp": time.time(),
            }
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"保存统计数据失败: {e}")
            return False
    def shutdown(self) -> None:
        """关闭系统，释放资源"""
        self.logger.info("关闭分拣系统")
        if self._running.get():
            self.stop()
        if self.servo:
            self.servo.move_to_sleep()
            self.servo.disable_all_servos()
        self._status = SortingStatus.STOPPED
        self._set_state(SortingState.STOPPED)
class SortingMonitor:
    """
    分拣监控器
    实时监控分拣系统状态，提供Web界面或终端输出
    """
    def __init__(self, controller: SortingController):
        self.controller = controller
        self.logger = get_logger().get_logger("monitor")
        self._running = False
    def print_status(self) -> None:
        """打印当前状态"""
        stats = self.controller.get_statistics().to_dict()
        print("\n" + "=" * 50)

        print(f"状态: {self.controller.get_state().value}")
        print(f"运行状态: {self.controller.get_status().value}")
        print(f"总循环: {stats['total_cycles']}")
        print(f"成功: {stats['successful_cycles']}")
        print(f"失败: {stats['failed_cycles']}")
        print(f"成功率: {stats['success_rate']}%")
        print(f"平均耗时: {stats['avg_cycle_time']}s")
        print("=" * 50 + "\n")
    def start_monitoring(self, interval: float = 2.0) -> None:
        """开始监控（阻塞）"""
        self._running = True
        try:
            while self._running:
                self.print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            self._running = False
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._running = False
# ============================================================
