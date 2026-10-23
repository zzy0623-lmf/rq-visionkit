# ============================================================
# hal.py - 硬件抽象层（重构版）
# 基于嵌入式AI视觉的机械臂智能分拣系统
#
# 【重要说明】本文件为「重构代码」，非《软著鉴别材料·源代码》PDF 中的原始代码。
# 原始 hal.py 位于软著 PDF 的「前30页+后30页」之间的被截断段，故缺失。
# 本实现严格依据以下两处证据重构，未凭空捏造接口：
#   1) 《软著设计说明书》第 2.2/3 节：罗技 C270 USB 摄像头（USB 2.0）、
#      PCA9685 舵机驱动板（I²C 总线、器件地址 0x40、16 路 12-bit PWM、50Hz）；
#   2) main.py / sorting_controller.py 中对本模块的实际调用点（方法名与属性）。
# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Any, Optional, Tuple
import numpy as np
from config import Config, get_config
from logger import get_logger


class PCA9685:
    """
    PCA9685 舵机驱动板（I²C，16 通道 12-bit PWM）
    最小化的 smbus2 驱动，寄存器协议遵循 NXP PCA9685 数据手册与 Adafruit 惯例。
    """
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06
    ALL_LED_ON_L = 0xFA
    ALL_LED_OFF_L = 0xFC

    def __init__(self, bus: int = 1, address: int = 0x40):
        import smbus2
        self._bus = smbus2.SMBus(bus)
        self.address = address
        self._freq = None

    def set_pwm_freq(self, freq: float) -> None:
        """设置 PWM 频率（典型 50Hz）。"""
        prescaleval = 25000000.0 / 4096.0 / freq - 1.0
        prescale = int(prescaleval + 0.5)
        oldmode = self._bus.read_byte_data(self.address, self.MODE1)
        newmode = (oldmode & 0x7F) | 0x10  # sleep
        self._bus.write_byte_data(self.address, self.MODE1, newmode)
        self._bus.write_byte_data(self.address, self.PRESCALE, prescale)
        self._bus.write_byte_data(self.address, self.MODE1, oldmode)
        time.sleep(0.005)
        self._bus.write_byte_data(self.address, self.MODE1, oldmode | 0x80)  # restart
        self._freq = freq

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        """设置单路 PWM 的 on/off 计数值（0~4095）。"""
        base = self.LED0_ON_L + 4 * channel
        self._bus.write_byte_data(self.address, base + 0, on & 0xFF)
        self._bus.write_byte_data(self.address, base + 1, on >> 8)
        self._bus.write_byte_data(self.address, base + 2, off & 0xFF)
        self._bus.write_byte_data(self.address, base + 3, off >> 8)

    def set_all_pwm(self, on: int, off: int) -> None:
        """设置全部 16 路 PWM 的 on/off 计数值。"""
        self._bus.write_byte_data(self.address, self.ALL_LED_ON_L, on & 0xFF)
        self._bus.write_byte_data(self.address, self.ALL_LED_ON_L + 1, on >> 8)
        self._bus.write_byte_data(self.address, self.ALL_LED_OFF_L, off & 0xFF)
        self._bus.write_byte_data(self.address, self.ALL_LED_OFF_L + 1, off >> 8)

    def reset(self) -> None:
        """复位：全部通道 PWM 置空（舵机失力）。"""
        self.set_all_pwm(0, 0)


class MockPCA9685:
    """PCA9685 模拟驱动（无硬件时使用），接口与真实驱动一致。"""
    def __init__(self, *args, **kwargs):
        self._freq = None

    def set_pwm_freq(self, freq: float) -> None:
        self._freq = freq

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        pass

    def set_all_pwm(self, on: int, off: int) -> None:
        pass

    def reset(self) -> None:
        pass


class MockCamera:
    """USB 摄像头模拟（无硬件时使用），read() 返回合成测试帧。"""
    def __init__(self, width: int = 1280, height: int = 720):
        self._width = width
        self._height = height

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # 生成带一个圆形目标的合成图，便于无硬件时跑通视觉/标定流程
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        cv2 = _import_cv2()
        cv2.circle(frame, (self._width // 2, self._height // 2), 40,
                   (0, 200, 0), -1)
        return True, frame

    def release(self) -> None:
        pass


def _import_cv2():
    import cv2
    return cv2


class HardwareAbstractionLayer:
    """
    硬件抽象层
    封装 USB 摄像头与 PCA9685 舵机驱动板，向应用层屏蔽底层设备细节。
    """
    def __init__(self, config: Optional[Config] = None, simulated: bool = False):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("hal")
        self.simulated = simulated
        self.camera = None
        self.pca9685 = None
        # 读取硬件配置
        cam_cfg = self.config.get("hardware.camera", {})
        self.camera_id = cam_cfg.get("device_id", 0)
        self.camera_width = cam_cfg.get("width", 1280)
        self.camera_height = cam_cfg.get("height", 720)
        pca_cfg = self.config.get("hardware.pca9685", {})
        self.i2c_bus = pca_cfg.get("bus", 1)
        self.i2c_address = pca_cfg.get("address", 0x40)

    def initialize(self) -> bool:
        """初始化摄像头与舵机驱动板。"""
        try:
            if self.simulated:
                self.camera = MockCamera(self.camera_width, self.camera_height)
                self.pca9685 = MockPCA9685(self.i2c_bus, self.i2c_address)
                self.logger.info("模拟模式：使用 MockCamera / MockPCA9685")
            else:
                cv2 = _import_cv2()
                self.camera = cv2.VideoCapture(self.camera_id)
                if not self.camera.isOpened():
                    self.logger.error(f"无法打开摄像头: {self.camera_id}")
                    return False
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                try:
                    self.pca9685 = PCA9685(self.i2c_bus, self.i2c_address)
                    self.pca9685.set_pwm_freq(50)
                    self.logger.info(
                        f"PCA9685 初始化完成: bus={self.i2c_bus}, addr=0x{self.i2c_address:02X}"
                    )
                except Exception as e:
                    self.logger.warning(f"PCA9685 初始化失败（无 I²C 设备）: {e}")
                    self.pca9685 = None
            return True
        except Exception as e:
            self.logger.error(f"硬件抽象层初始化失败: {e}", exc_info=True)
            return False

    def emergency_stop(self) -> None:
        """紧急停止：全部舵机 PWM 置空。"""
        if self.pca9685 is not None:
            self.pca9685.reset()
            self.logger.critical("已执行急停：全部 PWM 输出置空")

    def release(self) -> None:
        """释放硬件资源。"""
        if self.pca9685 is not None:
            self.pca9685.reset()
        if self.camera is not None and hasattr(self.camera, "release"):
            self.camera.release()
        self.logger.info("硬件抽象层已释放")


def create_hal(config: Optional[Config] = None,
               simulated: bool = False) -> HardwareAbstractionLayer:
    """硬件抽象层工厂函数。"""
    return HardwareAbstractionLayer(config, simulated=simulated)
# ============================================================
