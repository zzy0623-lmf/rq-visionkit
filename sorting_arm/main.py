# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - 系统主程序入口
基于嵌入式AI视觉的机械臂智能分拣系统
提供命令行交互界面，支持系统初始化、标定、单次分拣、自动分拣、
性能测试、参数配置和系统监控等功能
"""
import os
import sys
import time
import json
import argparse
import signal
from typing import Optional
from config import Config, get_config
from logger import get_logger, SystemLogger
from hal import HardwareAbstractionLayer, create_hal
from vision import VisionProcessor
from classifier import ObjectClassifier
from ik_solver import IKSolver
from servo_controller import ServoController
from calibration import CameraCalibrator
from sorting_controller import SortingController, SortingMonitor
class SortingSystem:
    """
    分拣系统主类
    整合所有子系统，提供高层操作接口
    """
    def __init__(self, config_path: Optional[str] = None, simulated: bool = False):
        self.config = get_config(config_path)
        self.logger = get_logger(self.config)
        self.simulated = simulated
        self.hal: Optional[HardwareAbstractionLayer] = None
        self.vision: Optional[VisionProcessor] = None
        self.classifier: Optional[ObjectClassifier] = None
        self.ik_solver: Optional[IKSolver] = None
        self.servo: Optional[ServoController] = None
        self.calibrator: Optional[CameraCalibrator] = None
        self.controller: Optional[SortingController] = None
        self.monitor: Optional[SortingMonitor] = None
        self._initialized = False
        self._running = False

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    def _signal_handler(self, signum, frame):
        """信号处理"""
        self.logger.info(f"收到信号 {signum}，正在关闭系统...")
        self.shutdown()
        sys.exit(0)
    def initialize(self) -> bool:
        """初始化整个系统"""
        self.logger.info("=" * 60)
        self.logger.info("基于嵌入式AI视觉的机械臂智能分拣系统")
        self.logger.info(f"版本: {self.config.get('system.version')}")
        self.logger.info(f"作者: {self.config.get('system.author')}")
        self.logger.info(f"模式: {'模拟' if self.simulated else '真实硬件'}")
        self.logger.info("=" * 60)
        try:
            # 1. 初始化硬件抽象层
            self.logger.info("[1/6] 初始化硬件抽象层...")
            self.hal = create_hal(self.config, simulated=self.simulated)
            if not self.hal.initialize():
                self.logger.error("硬件抽象层初始化失败")
                return False
            # 2. 初始化视觉处理器
            self.logger.info("[2/6] 初始化视觉处理器...")
            self.vision = VisionProcessor(self.config)
            # 3. 初始化分类器
            self.logger.info("[3/6] 初始化AI分类器...")
            model_path = self.config.get("ai_model.classifier.model_path")
            if os.path.exists(model_path):
                self.classifier = ObjectClassifier(self.config)
            else:
                self.logger.warning(f"模型文件不存在: {model_path}，分类功能将受限")
            # 4. 初始化逆运动学求解器
            self.logger.info("[4/6] 初始化逆运动学求解器...")
            self.ik_solver = IKSolver(self.config)
            # 5. 初始化舵机控制器
            self.logger.info("[5/6] 初始化舵机控制器...")
            if self.hal.pca9685:
                self.servo = ServoController(self.hal.pca9685, self.config)
                self.servo.initialize()
            else:
                self.logger.warning("无PCA9685硬件，使用模拟舵机控制")
            # 6. 初始化标定器和主控制器
            self.logger.info("[6/6] 初始化标定器和主控制器...")

            self.calibrator = CameraCalibrator(self.config)
            self.controller = SortingController(
                self.config, self.hal, self.simulated
            )
            self.controller.vision = self.vision
            self.controller.classifier = self.classifier
            self.controller.ik_solver = self.ik_solver
            self.controller.servo = self.servo
            self.controller.calibrator = self.calibrator
            if self.servo:
                from servo_controller import GripperController
                self.controller.gripper = GripperController(self.servo)
            self.monitor = SortingMonitor(self.controller)
            self._initialized = True
            self.logger.info("系统初始化完成！")
            return True
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}", exc_info=True)
            return False
    def calibrate(self) -> bool:
        """执行相机标定"""
        if not self._initialized:
            self.logger.error("系统未初始化")
            return False
        self.logger.info("开始相机标定流程")
        self.logger.info("请在工作区四个角点放置圆形标记")
        input("按回车开始采集标定图像...")
        if self.hal and self.hal.camera:
            ret, frame = self.hal.camera.read()
            if not ret:
                self.logger.error("采集标定图像失败")
                return False
            from calibration import CircleDetector
            detector = CircleDetector()
            circles = detector.detect_circles(frame)
            if len(circles) < 4:
                self.logger.error(f"检测到的圆形标记不足: {len(circles)}/4")
                return False
            # 按位置排序并添加标定点
            circles.sort(key=lambda c: c[2], reverse=True)
            circles = circles[:4]
            centers = [(c[0], c[1]) for c in circles]

            # 世界坐标（工作区四角，单位mm）
            world_corners = [
                (-100.0, -75.0),
                (100.0, -75.0),
                (100.0, 75.0),
                (-100.0, 75.0),
            ]
            self.calibrator.clear_points()
            for i, (px, py) in enumerate(centers):
                wx, wy = world_corners[i]
                self.calibrator.add_calibration_point(px, py, wx, wy, f"P{i+1}")
                self.logger.info(f"标定点 {i+1}: pixel=({px},{py}), world=({wx},{wy})")
            if self.calibrator.calibrate():
                self.logger.info(f"标定成功，重投影误差: {self.calibrator.get_reprojection_error():.3f}mm")
                return True
            else:
                self.logger.error("标定计算失败")
                return False
        else:
            self.logger.error("无可用摄像头")
            return False
    def single_sort(self) -> bool:
        """执行单次分拣"""
        if not self._initialized:
            self.logger.error("系统未初始化")
            return False
        self.logger.info("执行单次分拣...")
        result = self.controller.run_single_cycle()
        if result.success:
            self.logger.info(f"分拣成功: {result.detection.class_name} "
                           f"(置信度={result.detection.confidence:.3f}, "
                           f"耗时={result.cycle_time_seconds:.2f}s)")
        else:
            self.logger.warning(f"分拣失败: {result.message} ({result.error_code})")
        return result.success
    def auto_sort(self, max_cycles: int = 50) -> bool:
        """自动连续分拣"""
        if not self._initialized:
            self.logger.error("系统未初始化")
            return False
        self._running = True
        self.logger.info(f"开始自动分拣，最多 {max_cycles} 次")
        try:
            stats = self.controller.run(max_cycles=max_cycles)

            self.logger.info(f"自动分拣完成: 成功={stats.successful_cycles}/{stats.total_cycles}")
            return True
        except KeyboardInterrupt:
            self.logger.info("用户中断分拣")
            self.controller.stop()
            return False
        finally:
            self._running = False
    def performance_test(self, num_cycles: int = 10) -> dict:
        """性能测试"""
        if not self._initialized:
            return {"error": "系统未初始化"}
        self.logger.info(f"开始性能测试，{num_cycles} 次循环")
        times = []
        successes = 0
        for i in range(num_cycles):
            self.logger.info(f"测试循环 {i+1}/{num_cycles}")
            result = self.controller.run_single_cycle()
            times.append(result.cycle_time_seconds)
            if result.success:
                successes += 1
        import numpy as np
        results = {
            "total_cycles": num_cycles,
            "successful": successes,
            "success_rate": successes / num_cycles * 100,
            "avg_time": float(np.mean(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
        }
        self.logger.info(f"性能测试结果: {json.dumps(results, indent=2)}")
        return results
    def show_status(self) -> None:
        """显示系统状态"""
        if not self._initialized:
            print("系统未初始化")
            return
        print("\n" + "=" * 50)
        print("系统状态")
        print("=" * 50)
        print(f"初始化: {'是' if self._initialized else '否'}")
        print(f"运行中: {'是' if self._running else '否'}")
        print(f"模式: {'模拟' if self.simulated else '真实硬件'}")
        print(f"标定状态: {'已标定' if self.calibrator and self.calibrator.is_calibrated() else '未标定'}")
        if self.calibrator and self.calibrator.is_calibrated():

            print(f"标定误差: {self.calibrator.get_reprojection_error():.3f}mm")
        if self.controller:
            stats = self.controller.get_statistics().to_dict()
            print(f"总循环: {stats['total_cycles']}")
            print(f"成功: {stats['successful_cycles']}")
            print(f"成功率: {stats['success_rate']}%")
        print("=" * 50 + "\n")
    def emergency_stop(self) -> None:
        """紧急停止"""
        self.logger.critical("执行系统紧急停止！")
        if self.controller:
            self.controller.emergency_stop()
        if self.hal:
            self.hal.emergency_stop()
        self._running = False
    def shutdown(self) -> None:
        """关闭系统"""
        self.logger.info("正在关闭系统...")
        self._running = False
        if self.controller:
            self.controller.shutdown()
        if self.hal:
            self.hal.release()
        self.logger.info("系统已关闭")
        self._initialized = False
    def save_config(self, path: str) -> bool:
        """保存当前配置"""
        return self.config.save_to_file(path)
    def run_interactive(self) -> None:
        """运行交互式命令行界面"""
        if not self.initialize():
            print("系统初始化失败")
            return
        print("\n基于嵌入式AI视觉的机械臂智能分拣系统")
        print("输入命令进行操作，输入 'help' 查看帮助\n")
        commands = {
            "help": "显示帮助信息",
            "status": "显示系统状态",
            "calibrate": "执行相机标定",
            "sort": "执行单次分拣",
            "auto [N]": "自动分拣N次（默认50）",
            "test [N]": "性能测试N次（默认10）",
            "stop": "停止当前操作",
            "home": "机械臂归位",

            "save [path]": "保存配置",
            "stats": "显示统计信息",
            "exit": "退出系统",
        }
        while True:
            try:
                cmd = input(">>> ").strip().lower()
                if not cmd:
                    continue
                parts = cmd.split()
                action = parts[0]
                if action == "help":
                    print("\n可用命令:")
                    for cmd, desc in commands.items():
                        print(f"  {cmd:15s} - {desc}")
                    print()
                elif action == "status":
                    self.show_status()
                elif action == "calibrate":
                    self.calibrate()
                elif action == "sort":
                    self.single_sort()
                elif action == "auto":
                    n = int(parts[1]) if len(parts) > 1 else 50
                    self.auto_sort(n)
                elif action == "test":
                    n = int(parts[1]) if len(parts) > 1 else 10
                    self.performance_test(n)
                elif action == "stop":
                    if self.controller:
                        self.controller.stop()
                    print("已发送停止命令")
                elif action == "home":
                    if self.servo:
                        self.servo.move_to_home()
                        print("机械臂已归位")
                elif action == "save":
                    path = parts[1] if len(parts) > 1 else "config_saved.json"
                    if self.save_config(path):
                        print(f"配置已保存到 {path}")
                elif action == "stats":

                    if self.controller:
                        stats = self.controller.get_statistics().to_dict()
                        print(json.dumps(stats, indent=2, ensure_ascii=False))
                elif action == "exit":
                    print("正在退出...")
                    break
                else:
                    print(f"未知命令: {action}，输入 'help' 查看帮助")
            except KeyboardInterrupt:
                print("\n输入 'exit' 退出系统")
            except Exception as e:
                print(f"命令执行错误: {e}")
        self.shutdown()
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于嵌入式AI视觉的机械臂智能分拣系统"
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--simulated", action="store_true", help="使用模拟模式（无硬件）")
    parser.add_argument("--mode", choices=["interactive", "calibrate", "sort", "auto", "test"],
                       default="interactive", help="运行模式")
    parser.add_argument("--cycles", type=int, default=50, help="自动分拣/测试循环次数")
    args = parser.parse_args()
    system = SortingSystem(config_path=args.config, simulated=args.simulated)
    if args.mode == "interactive":
        system.run_interactive()
    else:
        if not system.initialize():
            print("系统初始化失败")
            sys.exit(1)
        if args.mode == "calibrate":
            system.calibrate()
        elif args.mode == "sort":
            system.single_sort()
        elif args.mode == "auto":
            system.auto_sort(args.cycles)
        elif args.mode == "test":
            system.performance_test(args.cycles)
        system.shutdown()
if __name__ == "__main__":
    main()

# ============================================================
