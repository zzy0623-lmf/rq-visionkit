# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logger.py - 全局日志管理模块
基于嵌入式AI视觉的机械臂智能分拣系统
提供统一的日志配置、控制台与文件双通道输出、日志轮转和按模块命名管理
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from config import Config
class SystemLogger:
    """
    系统日志管理器
    单例模式，集中管理日志级别、输出格式、控制台和文件输出
    各模块通过 get_logger(name) 获取命名日志器
    """
    _instance = None
    def __new__(cls, config: Optional[Config] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self, config: Optional[Config] = None):
        if self._initialized:
            return
        self._initialized = True

        self._config = config
        self._root = logging.getLogger("sorting_system")
        self._root.propagate = False
        self._setup_handlers()
    def _setup_handlers(self) -> None:
        """按配置初始化控制台和文件处理器"""
        cfg = self._config
        level_name = cfg.get("logging.level", "INFO") if cfg else "INFO"
        fmt = cfg.get(
            "logging.format",
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        ) if cfg else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        date_fmt = cfg.get("logging.date_format", "%Y-%m-%d %H:%M:%S") if cfg else "%Y-%m-%d %H:%M:%S"
        self._root.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
        self._root.handlers.clear()
        formatter = logging.Formatter(fmt, datefmt=date_fmt)
        console_output = cfg.get("logging.console_output", True) if cfg else True
        if console_output:
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(formatter)
            self._root.addHandler(console)
        file_output = cfg.get("logging.file_output", True) if cfg else True
        if file_output:
            log_dir = cfg.get("logging.log_dir", "logs") if cfg else "logs"
            prefix = cfg.get("logging.log_file_prefix", "sorting_system") if cfg else "sorting_system"
            max_bytes = (cfg.get("logging.max_file_size_mb", 10) if cfg else 10) * 1024 * 1024
            backup_count = cfg.get("logging.backup_count", 5) if cfg else 5
            try:
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{prefix}.log")
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=max_bytes,
                    backupCount=backup_count, encoding="utf-8"
                )
                file_handler.setFormatter(formatter)
                self._root.addHandler(file_handler)
            except IOError as e:
                self._root.warning(f"日志文件初始化失败: {e}")
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定模块的命名日志器"""
        return self._root.getChild(name)
    def debug(self, message: str, **kwargs) -> None:
        self._root.debug(message, **kwargs)
    def info(self, message: str, **kwargs) -> None:
        self._root.info(message, **kwargs)
    def warning(self, message: str, **kwargs) -> None:
        self._root.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._root.error(message, **kwargs)
    def critical(self, message: str, **kwargs) -> None:
        self._root.critical(message, **kwargs)
    def set_level(self, level: str) -> None:
        """动态调整日志级别"""
        self._root.setLevel(getattr(logging, level.upper(), logging.INFO))
def get_logger(config: Optional[Config] = None) -> SystemLogger:
    """获取全局日志管理器单例"""
    return SystemLogger(config)
# ============================================================
