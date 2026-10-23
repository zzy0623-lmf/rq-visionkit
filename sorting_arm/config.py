# ============================================================
# config.py - 系统配置模块（重构版）
# 基于嵌入式AI视觉的机械臂智能分拣系统
#
# 【重要说明】本文件为「重构代码」，非《软著鉴别材料·源代码》PDF 中的原始代码。
# 原始 config.py 的类声明（class Config:）、__init__ 与 _get_default_config 头部
# 位于软著 PDF 的「前30页+后30页」之间的被截断段，故缺失。
# 本实现依据以下证据重构，未凭空捏造配置项：
#   1) 《软著设计说明书》第 2.2/3.1/4.1/6.7 节：硬件清单、关节参数、术语与关键数据结构；
#   2) ik_solver.py / vision.py / classifier.py / sorting_controller.py / train.py / hal.py
#      中对各配置键的实际读取（如 kinematics.link_lengths_mm、hardware.servos.* 等）；
#   3) 软著 PDF 中已还原出的配置字典尾部（vision/ai_model/sorting/logging/performance）。
# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import copy
from typing import Any, Dict, Optional


class Config:
    """
    系统配置类
    提供默认配置、从 JSON 文件加载/合并、按点分路径读写、参数校验等功能。
    各模块通过 get_config() 获取全局单例，或通过 Config(config_path) 构建实例。
    """
    def __init__(self, config_path: Optional[str] = None):
        self._config = self._get_default_config()
        if config_path and os.path.exists(config_path):
            if not self.load_from_file(config_path):
                print(f"[Config] 配置文件加载失败，使用默认配置: {config_path}")

    def _get_default_config(self) -> Dict:
        """返回系统默认配置字典。"""
        return {
            # ==================== 系统信息 ====================
            "system": {
                "version": "V1.0",
                "author": "赵喆远",
            },
            # ==================== 硬件配置 ====================
            "hardware": {
                "camera": {
                    "device_id": 0,
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                },
                "servos": {
                    "J1": {"min_angle_deg": 0, "max_angle_deg": 180},
                    "J2": {"min_angle_deg": 20, "max_angle_deg": 160},
                    "J3": {"min_angle_deg": -150, "max_angle_deg": 0},
                    "J4": {"min_angle_deg": -30, "max_angle_deg": 60},
                },
                "pca9685": {
                    "bus": 1,
                    "address": 0x40,
                    "pwm_freq_hz": 50,
                },
            },
            # ==================== 运动学配置 ====================
            "kinematics": {
                "link_lengths_mm": {
                    "L1_base_height": 45.0,
                    "L2_shoulder_to_elbow": 80.0,
                    "L3_elbow_to_wrist": 110.0,
                    "L4_wrist_to_tip": 60.0,
                },
                "joint_limits_deg": {
                    "J1": [0, 180],
                    "J2": [20, 160],
                    "J3": [-150, 0],
                    "J4": [-30, 60],
                },
                "workspace_mm": {
                    "x_min": -150,
                    "x_max": 150,
                    "y_min": -150,
                    "y_max": 150,
                    "z_min": 10,
                    "z_max": 200,
                },
                "gripper_down_constraint": True,
                "ik_solver_method": "geometric_analytical",
                "ik_max_iterations": 1,
                "ik_tolerance_mm": 0.5,
                "motion_planning": {
                    "approach_height_mm": 40,
                    "pick_height_mm": 30,
                    "place_height_mm": 80,
                    "lift_height_mm": 60,
                    "movement_type": "linear_interpolation",
                    "interpolation_steps": 50,
                    "smooth_acceleration": True,
                    "smooth_factor": 0.5,
                },
            },
            # ==================== 视觉处理配置 ====================
            "vision": {
                "image_preprocessing": {
                    "resize_width": 1280,
                    "resize_height": 720,
                    "blur_kernel_size": 3,
                    "blur_sigma_x": 0,
                    "denoise_enabled": True,
                    "denoise_h": 10,
                    "denoise_template_window_size": 7,
                    "denoise_search_window_size": 21,
                },
                "background_segmentation": {
                    "method": "hsv_threshold",
                    "hsv_lower": [0, 0, 50],
                    "hsv_upper": [180, 50, 255],
                    "background_color": "black",
                    "morphology_kernel_size": 3,
                    "morphology_iterations": 1,
                    "min_contour_area": 100,
                    "max_contour_area": 50000,
                    "contour_approx_epsilon": 0.02,
                    "use_convex_hull": False,
                },
                "object_localization": {
                    "method": "min_area_rect",
                    "center_type": "centroid",
                    "roi_padding_ratio": 0.2,
                    "roi_min_size": 32,
                    "roi_max_size": 512,
                    "multi_object_enabled": False,
                    "max_objects": 5,
                    "sort_by": "area",
                    "sort_order": "descending",
                },
                "coordinate_mapping": {
                    "method": "perspective_transform",
                    "calibration_points": 4,
                    "homography_matrix_path": "config/homography_matrix.npy",
                    "calibration_marker_type": "circle",
                    "calibration_marker_diameter_mm": 20,
                    "calibration_accuracy_mm": 2.0,
                    "use_ransac": True,
                    "ransac_reproj_threshold": 3.0,
                },
            },
            # ==================== AI模型配置 ====================
            "ai_model": {
                "classifier": {
                    "model_type": "MobileNetV2",
                    "framework": "TensorFlow Lite",
                    "quantization": "INT8",
                    "model_path": "models/mobilenetv2_int8.tflite",
                    "label_path": "models/labels.txt",
                    "input_size": [224, 224],
                    "input_channels": 3,
                    "num_classes": 5,
                    "class_names": [
                        "苹果模型",
                        "香蕉模型",
                        "水杯",
                        "书本模型",
                        "积木块",
                    ],
                    "confidence_threshold": 0.8,
                    "top_k": 3,
                    "delegate": "XNNPACK",
                    "delegate_path": "libXNNPACK.so",
                    "num_threads": 4,
                    "use_npu": False,
                    "inference_timeout_ms": 100,
                },
                "training": {
                    "dataset_path": "data/dataset",
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                    "batch_size": 32,
                    "epochs_stage1": 10,
                    "epochs_stage2": 20,
                    "learning_rate_stage1": 0.001,
                    "learning_rate_stage2": 0.00001,
                    "optimizer": "Adam",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_epsilon": 1e-8,
                    "loss_function": "categorical_crossentropy",
                    "metrics": ["accuracy", "top_k_categorical_accuracy"],
                    "lr_scheduler": "cosine_annealing",
                    "warmup_epochs": 2,
                    "weight_decay": 0.0001,
                    "dropout_rate": 0.2,
                    "data_augmentation": {
                        "random_rotation_range": 20,
                        "horizontal_flip": True,
                        "vertical_flip": False,
                        "brightness_range": [0.8, 1.2],
                        "contrast_range": [0.8, 1.2],
                        "saturation_range": [0.8, 1.2],
                        "hue_range": 0.1,
                        "zoom_range": 0.2,
                        "width_shift_range": 0.1,
                        "height_shift_range": 0.1,
                        "shear_range": 0.1,
                        "channel_shift_range": 20,
                        "fill_mode": "nearest",
                    },
                },
                "quantization": {
                    "method": "full_integer",
                    "calibration_samples": 100,
                    "calibration_dataset_path": "data/calibration",
                    "weight_type": "int8",
                    "activation_type": "int8",
                    "bias_type": "int32",
                    "optimizations": ["DEFAULT"],
                    "supported_ops": ["TFLITE_BUILTINS_INT8"],
                    "inference_input_type": "uint8",
                    "inference_output_type": "float32",
                },
            },
            # ==================== 分拣策略配置 ====================
            "sorting": {
                "drop_zones": {
                    "苹果模型": {"x_mm": 200.0, "y_mm": 80.0, "z_mm": 80.0},
                    "香蕉模型": {"x_mm": 200.0, "y_mm": 0.0, "z_mm": 80.0},
                    "水杯": {"x_mm": 200.0, "y_mm": -80.0, "z_mm": 80.0},
                    "书本模型": {"x_mm": 200.0, "y_mm": -160.0, "z_mm": 80.0},
                    "未知物品": {"x_mm": 200.0, "y_mm": -240.0, "z_mm": 80.0},
                },
                "default_zone": "未知物品",
                "max_cycles": 50,
                "cycle_timeout_seconds": 30,
                "retry_on_fail": True,
                "max_retries": 3,
                "skip_low_confidence": True,
                "skip_unreachable": True,
                "return_home_after_cycle": True,
                "delay_between_cycles_seconds": 0.5,
                "gripper_close_delay_seconds": 0.5,
                "gripper_open_delay_seconds": 0.3,
                "settle_delay_seconds": 0.3,
                "state_machine": {
                    "states": [
                        "IDLE",
                        "CAPTURE",
                        "LOCATE",
                        "PLAN",
                        "EXECUTE",
                        "REPORT",
                        "ERROR",
                    ],
                    "initial_state": "IDLE",
                    "transitions": {
                        "IDLE": {"start": "CAPTURE", "object_detected": "CAPTURE"},
                        "CAPTURE": {"success": "LOCATE", "fail": "ERROR"},
                        "LOCATE": {"success": "PLAN", "no_object": "IDLE", "low_confidence": "REPORT"},
                        "PLAN": {"success": "EXECUTE", "unreachable": "REPORT"},
                        "EXECUTE": {"success": "REPORT", "fail": "ERROR"},
                        "REPORT": {"complete": "IDLE", "more_objects": "CAPTURE"},
                        "ERROR": {"recover": "IDLE", "stop": "IDLE"},
                    },
                },
            },
            # ==================== 日志配置 ====================
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S",
                "console_output": True,
                "file_output": True,
                "log_dir": "logs",
                "log_file_prefix": "sorting_system",
                "max_file_size_mb": 10,
                "backup_count": 5,
                "enable_metrics": True,
                "metrics_interval_seconds": 60,
                "save_statistics": True,
                "statistics_file": "logs/statistics.json",
            },
            # ==================== 性能监控配置 ====================
            "performance": {
                "enable_profiling": True,
                "profile_output_path": "logs/profile.json",
                "track_timing": True,
                "timing_stages": [
                    "image_capture",
                    "object_localization",
                    "ai_inference",
                    "coordinate_mapping",
                    "ik_solving",
                    "motion_execution",
                    "total_cycle",
                ],
                "track_memory": True,
                "memory_interval_seconds": 10,
                "track_cpu": True,
                "cpu_interval_seconds": 5,
                "track_temperature": True,
                "temperature_interval_seconds": 30,
                "alert_thresholds": {
                    "cpu_temp_c": 80,
                    "memory_usage_percent": 90,
                    "inference_time_ms": 50,
                    "cycle_time_seconds": 15,
                },
            },
        }

    def load_from_file(self, config_path: str) -> bool:
        """从JSON配置文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            self._merge_config(self._config, user_config)
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Config] 加载配置文件失败: {e}")
            return False

    def _merge_config(self, base: Dict, override: Dict) -> None:
        """递归合并配置字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def save_to_file(self, config_path: str) -> bool:
        """保存当前配置到JSON文件"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"[Config] 保存配置文件失败: {e}")
            return False

    def get(self, key_path: str, default: Any = None) -> Any:
        """按点分路径获取配置值"""
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> bool:
        """按点分路径设置配置值"""
        keys = key_path.split('.')
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        return True

    def get_section(self, section: str) -> Optional[Dict]:
        """获取整个配置节"""
        return self._config.get(section)

    def validate(self) -> Dict[str, str]:
        """校验配置参数的合法性"""
        errors = {}
        # 校验相机配置
        cam_w = self.get("hardware.camera.width")
        cam_h = self.get("hardware.camera.height")
        if cam_w <= 0 or cam_h <= 0:
            errors["hardware.camera.resolution"] = "相机分辨率必须为正数"
        # 校验舵机角度范围
        for joint in ["J1", "J2", "J3", "J4"]:
            min_a = self.get(f"hardware.servos.{joint}.min_angle_deg")
            max_a = self.get(f"hardware.servos.{joint}.max_angle_deg")
            if min_a >= max_a:
                errors[f"hardware.servos.{joint}.range"] = "最小角度必须小于最大角度"
        # 校验连杆长度
        for link in ["L1_base_height", "L2_shoulder_to_elbow", "L3_elbow_to_wrist", "L4_wrist_to_tip"]:
            length = self.get(f"kinematics.link_lengths_mm.{link}")
            if length <= 0:
                errors[f"kinematics.link_lengths_mm.{link}"] = "连杆长度必须为正数"
        # 校验AI模型配置
        num_classes = self.get("ai_model.classifier.num_classes")
        class_names = self.get("ai_model.classifier.class_names")
        if num_classes and class_names and len(class_names) != num_classes:
            errors["ai_model.classifier.class_names"] = "类别名称数量与num_classes不一致"
        # 校验置信度阈值
        conf_thresh = self.get("ai_model.classifier.confidence_threshold")
        if conf_thresh is not None and (conf_thresh < 0 or conf_thresh > 1):
            errors["ai_model.classifier.confidence_threshold"] = "置信度阈值必须在0到1之间"
        return errors

    def export_for_logging(self) -> Dict:
        """导出用于日志记录的配置摘要（隐藏敏感信息）"""
        summary = copy.deepcopy(self._config)
        # 移除可能包含敏感信息的字段
        if "system" in summary:
            summary["system"].pop("author", None)
        return summary

    def reset_to_default(self) -> None:
        """重置为默认配置"""
        self._config = self._get_default_config()

    def __repr__(self) -> str:
        return f"Config(version={self.get('system.version')}, sections={list(self._config.keys())})"


def get_config(config_path: Optional[str] = None) -> Config:
    """获取全局配置单例"""
    return Config(config_path)
# ============================================================
