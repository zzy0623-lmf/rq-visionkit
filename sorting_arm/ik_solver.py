# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ik_solver.py - 机械臂逆运动学求解模块
基于嵌入式AI视觉的机械臂智能分拣系统
针对四自由度机械臂（基座旋转+两连杆平面臂+腕部）采用几何解析法
求解期望末端位置对应的各关节角度，并进行工作空间与关节限位校验
"""
import math
import time
from typing import Any, Dict, Optional, Tuple
from config import Config, get_config
from logger import get_logger
from utils import clamp

class IKSolver:
    """
    逆运动学求解器
    采用几何解析法求解四自由度机械臂的关节角度
    基座关节确定朝向，肩肘两关节在竖直平面内求解，腕部保持夹爪竖直向下
    """
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("ik_solver")
        link_cfg = self.config.get("kinematics.link_lengths_mm", {})
        self.L1 = link_cfg.get("L1_base_height", 45.0)
        self.L2 = link_cfg.get("L2_shoulder_to_elbow", 80.0)
        self.L3 = link_cfg.get("L3_elbow_to_wrist", 110.0)
        self.L4 = link_cfg.get("L4_wrist_to_tip", 60.0)
        self.joint_limits = self.config.get("kinematics.joint_limits_deg", {})
        self.workspace = self.config.get("kinematics.workspace_mm", {})
        self.gripper_down = self.config.get("kinematics.gripper_down_constraint", True)
        # 求解性能统计
        self._total_solves = 0
        self._failed_solves = 0
        self._total_time_ms = 0.0
        self.logger.info(
            f"逆运动学求解器初始化完成: L1={self.L1}, L2={self.L2}, "
            f"L3={self.L3}, L4={self.L4}"
        )
    def solve(self, x: float, y: float, z: float) -> Dict[str, Any]:
        """求解末端位置(x, y, z)对应的关节角度，单位mm"""
        start = time.perf_counter()
        self._total_solves += 1
        result = self._solve_geometric(x, y, z)
        self._total_time_ms += (time.perf_counter() - start) * 1000
        if not result["reachable"]:
            self._failed_solves += 1
        return result
    def _solve_geometric(self, x: float, y: float, z: float) -> Dict[str, Any]:
        """几何解析法求解，返回包含角度与可达性的字典"""
        if not self._in_workspace(x, y, z):
            return self._unreachable("目标点超出工作空间")
        # 基座关节：由目标点水平方位直接确定
        theta1 = math.degrees(math.atan2(y, x))
        if theta1 < 0:
            theta1 += 360.0
        if not self._within_limit("J1", theta1):
            return self._unreachable(f"基座角度超限: {theta1:.1f}°")

        # 夹爪竖直向下时，腕部中心位于末端正上方L4处
        r = math.hypot(x, y)
        z_wrist = z + self.L4
        dz = z_wrist - self.L1
        d = math.hypot(r, dz)
        # 可达性：目标在肩肘两连杆构成的圆环内
        if d > self.L2 + self.L3 or d < abs(self.L2 - self.L3):
            return self._unreachable(f"目标距离超出臂展: {d:.1f}mm")
        # 肘关节：余弦定理求内角，取肘下解（角度为负）
        cos_elbow = clamp(
            (self.L2 ** 2 + self.L3 ** 2 - d ** 2) / (2 * self.L2 * self.L3),
            -1.0, 1.0
        )
        elbow_inner = math.degrees(math.acos(cos_elbow))
        theta3 = -(180.0 - elbow_inner)
        # 肩关节：目标方位角与连杆夹角叠加
        cos_shoulder = clamp(
            (self.L2 ** 2 + d ** 2 - self.L3 ** 2) / (2 * self.L2 * d),
            -1.0, 1.0
        )
        theta2 = math.degrees(math.atan2(dz, r)) + math.degrees(math.acos(cos_shoulder))
        # 腕部关节：补偿俯仰角使夹爪尽量保持竖直向下
        # J4为软约束，超出限位时取边界值，夹爪姿态略有倾斜但不影响抓取
        if self.gripper_down:
            theta4 = -90.0 - theta2 - theta3
            j4_limits = self.joint_limits.get("J4", [-30, 60])
            theta4 = clamp(theta4, j4_limits[0], j4_limits[1])
        else:
            theta4 = 0.0
        # J1-J3为硬约束，超限则判定不可达
        angles = {"J1": theta1, "J2": theta2, "J3": theta3}
        for joint, angle in angles.items():
            if not self._within_limit(joint, angle):
                return self._unreachable(f"{joint}角度超限: {angle:.1f}°")
        return {
            "reachable": True,
            "message": "求解成功",
            "angles": (theta1, theta2, theta3, theta4),
            "theta1": theta1,
            "theta2": theta2,
            "theta3": theta3,
            "theta4": theta4,
        }
    def _in_workspace(self, x: float, y: float, z: float) -> bool:
        """检查目标点是否在配置的工作空间内"""
        ws = self.workspace
        return (
            ws.get("x_min", -150) <= x <= ws.get("x_max", 150)

            and ws.get("y_min", -150) <= y <= ws.get("y_max", 150)
            and ws.get("z_min", 10) <= z <= ws.get("z_max", 200)
        )
    def _within_limit(self, joint: str, angle: float) -> bool:
        """检查关节角度是否在限位范围内"""
        limits = self.joint_limits.get(joint)
        if not limits:
            return True
        return limits[0] - 1e-6 <= angle <= limits[1] + 1e-6
    @staticmethod
    def _unreachable(message: str) -> Dict[str, Any]:
        """构造不可达结果"""
        return {"reachable": False, "message": message, "angles": None,
                "theta1": None, "theta2": None, "theta3": None, "theta4": None}
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取求解性能统计"""
        return {
            "total_solves": self._total_solves,
            "failed_solves": self._failed_solves,
            "success_rate": round(
                (self._total_solves - self._failed_solves) / self._total_solves * 100, 2
            ) if self._total_solves > 0 else 0.0,
            "avg_solve_time_ms": round(
                self._total_time_ms / self._total_solves, 3
            ) if self._total_solves > 0 else 0.0,
        }
# ============================================================
