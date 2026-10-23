# ============================================================
# servo_controller.py - 舵机控制模块（重构版）
# 基于嵌入式AI视觉的机械臂智能分拣系统
#
# 【重要说明】本文件为「重构代码」，非《软著鉴别材料·源代码》PDF 中的原始代码。
# 原始 servo_controller.py 的 ServoController 方法体与 GripperController 类
# 位于软著 PDF 的「前30页+后30页」之间的被截断段，故缺失。
# 本实现严格依据以下证据重构，未凭空捏造接口：
#   1) 《软著设计说明书》第 6.5 节「舵机控制模块」：SERVO_CALIBRATION（最小脉宽500μs、
#      最大脉宽2500μs）、JOINT_SPEEDS（J1:60°/s，J2/J3:45°/s，J4:50°/s，夹爪:80°/s）、
#      set_servo_angle / move_all_joints(smoothstep) / set_gripper / move_to_home / emergency_stop；
#   2) 第 6.7 节 channel_map（J1~J4:0~3，夹爪:4）；
#   3) main.py / sorting_controller.py 中对本模块的实际调用点（方法名与参数签名）。
# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional
from config import Config, get_config
from logger import get_logger
from utils import clamp, smoothstep


class ServoController:
    """
    舵机控制器
    通过 PCA9685 驱动板控制 J1-J4 关节舵机与夹爪舵机，
    实现角度控制、smoothstep 平滑插值运动、归位/休眠姿态管理与夹爪开合控制。
    """
    # 各舵机标定参数：(最小脉宽μs, 最大脉宽μs, 最小角度, 最大角度)
    # 夹爪以「开合百分比」为角度量纲：0=闭合，100=全开
    SERVO_CALIBRATION: Dict[str, tuple] = {
        "J1": (500, 2500, 0, 180),
        "J2": (500, 2500, 20, 160),
        "J3": (500, 2500, -150, 0),
        "J4": (500, 2500, -30, 60),
        "gripper": (500, 2500, 0, 100),
    }
    # 各关节限速（°/s）
    JOINT_SPEEDS: Dict[str, float] = {
        "J1": 60.0,
        "J2": 45.0,
        "J3": 45.0,
        "J4": 50.0,
        "gripper": 80.0,
    }
    # 关节名称到 PCA9685 通道号的映射
    CHANNEL_MAP: Dict[str, int] = {
        "J1": 0,
        "J2": 1,
        "J3": 2,
        "J4": 3,
        "gripper": 4,
    }
    # 归位姿态
    # 注：原设计说明书 §6.5 写「J3=60°」，但与 J3 关节限位 [-150°,0°] 及
    # ik_solver 输出的负向 θ₃ 相冲突，此处取 J3=-60° 以保持一致性。
    HOME_POSE: Dict[str, float] = {
        "J1": 90.0,
        "J2": 90.0,
        "J3": -60.0,
        "J4": 15.0,
        "gripper": 50.0,
    }
    PWM_FREQ_HZ = 50.0
    PWM_PERIOD_US = 1_000_000.0 / PWM_FREQ_HZ  # 20000 μs
    PWM_MAX_COUNT = 4096  # 12-bit

    def __init__(self, pca9685: Any = None, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("servo")
        self.pca9685 = pca9685
        self._current_angles: Dict[str, float] = {}
        self._enabled = False
        self._pwm_freq = self.config.get("hardware.pca9685.pwm_freq_hz", 50.0)

    def initialize(self) -> None:
        """初始化舵机控制器：设置 PWM 频率并归位。"""
        if self.pca9685 is not None:
            try:
                self.pca9685.set_pwm_freq(self._pwm_freq)
            except Exception as e:
                self.logger.warning(f"设置 PWM 频率失败: {e}")
        self._enabled = True
        self.move_to_home()
        self.logger.info("舵机控制器初始化完成")

    def _angle_to_pwm(self, servo_name: str, angle_deg: float) -> int:
        """将角度换算为 PCA9685 的 off 计数值（0~4095）。"""
        min_pulse, max_pulse, min_angle, max_angle = self.SERVO_CALIBRATION[servo_name]
        angle = clamp(angle_deg, min_angle, max_angle)
        if max_angle == min_angle:
            ratio = 0.0
        else:
            ratio = (angle - min_angle) / (max_angle - min_angle)
        pulse_us = min_pulse + ratio * (max_pulse - min_pulse)
        return int(pulse_us / self.PWM_PERIOD_US * self.PWM_MAX_COUNT)

    def set_servo_angle(self, servo_name: str, angle_deg: float) -> None:
        """单关节定角：将角度写入对应 PWM 通道。"""
        if servo_name not in self.CHANNEL_MAP:
            self.logger.warning(f"未知关节: {servo_name}")
            return
        angle = clamp(angle_deg, *self.SERVO_CALIBRATION[servo_name][2:])
        self._current_angles[servo_name] = angle
        if self.pca9685 is not None and self._enabled:
            off = self._angle_to_pwm(servo_name, angle)
            channel = self.CHANNEL_MAP[servo_name]
            self.pca9685.set_pwm(channel, 0, off)

    def move_all_joints(self, joint_angles: Dict[str, float],
                        step_delay: float = 0.02) -> None:
        """按 smoothstep 平滑曲线多关节同步插补运动。"""
        targets = {
            name: clamp(angle, *self.SERVO_CALIBRATION[name][2:])
            for name, angle in joint_angles.items()
            if name in self.CHANNEL_MAP
        }
        if not targets:
            return
        # 以最大角差估算插补步数（每 5° 一步，至少 1 步）
        max_delta = 0.0
        for name, target in targets.items():
            cur = self._current_angles.get(name, target)
            max_delta = max(max_delta, abs(target - cur))
        steps = max(1, int(max_delta / 5.0))
        for i in range(1, steps + 1):
            s = smoothstep(i / steps)
            for name, target in targets.items():
                cur = self._current_angles.get(name, target)
                self.set_servo_angle(name, cur + (target - cur) * s)
            time.sleep(step_delay)

    def set_gripper(self, open_percent: float) -> None:
        """控制夹爪开合百分比（0=闭合，100=全开）。"""
        self.set_servo_angle("gripper", open_percent)

    def move_to_home(self) -> None:
        """回到初始归位姿态。"""
        self.move_all_joints(self.HOME_POSE)

    def move_to_sleep(self) -> None:
        """进入休眠姿态：各关节回到安全角度、夹爪闭合。"""
        self.move_all_joints({
            "J1": 90.0,
            "J2": 90.0,
            "J3": -60.0,
            "J4": 15.0,
            "gripper": 0.0,
        })

    def disable_all_servos(self) -> None:
        """将全部 16 路 PWM 输出置空（舵机失力）。"""
        if self.pca9685 is not None:
            self.pca9685.set_all_pwm(0, 0)
        self._enabled = False

    def emergency_stop(self) -> None:
        """紧急停止：全部 PWM 输出置空实现急停释放。"""
        self.disable_all_servos()
        self.logger.critical("舵机已急停释放")

    def release_emergency_stop(self) -> None:
        """解除急停：重新使能并归位。"""
        self._enabled = True
        self.move_to_home()
        self.logger.info("急停已解除，机械臂归位")


class GripperController:
    """
    夹爪控制器
    封装平行开合夹爪的开/闭/释放动作，供分拣主控在抓取-放置流程中调用。
    """
    def __init__(self, servo: ServoController):
        self.servo = servo
        self.logger = get_logger().get_logger("gripper")

    def open(self, delay: float = 0.3) -> None:
        """张开夹爪（全开）。"""
        self.servo.set_gripper(100.0)
        time.sleep(delay)

    def grip(self, force_level: int = 3, delay: float = 0.5) -> None:
        """闭合夹爪抓取物品。当前平行夹爪无力度反馈，force_level 仅预留。"""
        # 闭合到接近全闭，保留少量余量避免堵转
        self.servo.set_gripper(5.0)
        time.sleep(delay)

    def release(self, delay: float = 0.3) -> None:
        """释放物品（张开夹爪）。"""
        self.servo.set_gripper(100.0)
        time.sleep(delay)
# ============================================================
