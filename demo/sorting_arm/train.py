# ============================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - 模型训练与量化脚本
基于嵌入式AI视觉的机械臂智能分拣系统
实现MobileNetV2模型的迁移学习训练、两阶段微调、数据增强、
INT8全整数量化和模型评估导出
"""
import os
import sys
import time
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from config import Config, get_config
from logger import get_logger
class ModelTrainer:
    """
    模型训练器
    基于TensorFlow/Keras的MobileNetV2迁移学习训练
    支持两阶段训练、数据增强、学习率调度和模型保存
    """
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("trainer")
        train_config = self.config.get("ai_model.training", {})
        self.dataset_path = train_config.get("dataset_path", "data/dataset")
        self.batch_size = train_config.get("batch_size", 32)
        self.epochs_stage1 = train_config.get("epochs_stage1", 10)
        self.epochs_stage2 = train_config.get("epochs_stage2", 20)
        self.lr_stage1 = train_config.get("learning_rate_stage1", 0.001)
        self.lr_stage2 = train_config.get("learning_rate_stage2", 0.00001)
        self.num_classes = self.config.get("ai_model.classifier.num_classes", 5)
        self.input_size = tuple(self.config.get("ai_model.classifier.input_size", [224, 224]))
        self.model = None
        self.history = {}
        self.class_names = []
        self.logger.info("模型训练器初始化完成")

    def load_dataset(self) -> Tuple[Any, Any, Any]:
        """加载数据集"""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            self.logger.error("未安装TensorFlow")
            raise
        img_height, img_width = self.input_size
        # 加载训练集
        train_ds = keras.utils.image_dataset_from_directory(
            os.path.join(self.dataset_path, "train"),
            validation_split=0.0,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=self.batch_size,
            label_mode='categorical',
        )
        # 加载验证集
        val_ds = keras.utils.image_dataset_from_directory(
            os.path.join(self.dataset_path, "val"),
            seed=123,
            image_size=(img_height, img_width),
            batch_size=self.batch_size,
            label_mode='categorical',
        )
        # 加载测试集
        test_ds = keras.utils.image_dataset_from_directory(
            os.path.join(self.dataset_path, "test"),
            seed=123,
            image_size=(img_height, img_width),
            batch_size=self.batch_size,
            label_mode='categorical',
        )
        self.class_names = train_ds.class_names
        self.logger.info(f"数据集加载完成: {len(self.class_names)} 类")
        self.logger.info(f"类别: {self.class_names}")
        # 数据增强
        augmentation_config = self.config.get("ai_model.training.data_augmentation", {})
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(augmentation_config.get("random_rotation_range", 20) / 180),
            keras.layers.RandomZoom(augmentation_config.get("zoom_range", 0.2)),
            keras.layers.RandomContrast(augmentation_config.get("contrast_range", [0.8, 1.2])[1] - 1.0),
            keras.layers.RandomBrightness(augmentation_config.get("brightness_range", [0.8, 1.2])[1] - 1.0),
        ])
        # 预处理和缓存

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        return train_ds, val_ds, test_ds
    def build_model(self) -> Any:
        """构建MobileNetV2模型"""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise
        img_height, img_width = self.input_size
        # 输入预处理
        inputs = keras.Input(shape=(img_height, img_width, 3))
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        # 加载预训练MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet',
            alpha=1.0,
        )
        base_model.trainable = False
        x = base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        self.model = model
        self.logger.info("MobileNetV2模型构建完成")
        return model
    def train_stage1(self, train_ds: Any, val_ds: Any) -> Dict:
        """第一阶段训练：冻结骨干，只训练分类头"""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise

        self.logger.info(f"开始第一阶段训练: {self.epochs_stage1} epochs, lr={self.lr_stage1}")
        # 冻结骨干
        for layer in self.model.layers:
            if 'mobilenetv2' in layer.name:
                layer.trainable = False
        optimizer = keras.optimizers.Adam(
            learning_rate=self.lr_stage1,
            beta_1=0.9, beta_2=0.999
        )
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        # 学习率调度
        lr_scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch: self.lr_stage1 * 0.5 * (1 + math.cos(math.pi * epoch / self.epochs_stage1))
        )
        # 早停
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs_stage1,
            callbacks=[lr_scheduler, early_stopping],
            verbose=1,
        )
        self.history['stage1'] = history.history
        self.logger.info("第一阶段训练完成")
        return history.history
    def train_stage2(self, train_ds: Any, val_ds: Any) -> Dict:
        """第二阶段训练：微调最后几层"""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise
        self.logger.info(f"开始第二阶段微调: {self.epochs_stage2} epochs, lr={self.lr_stage2}")
        # 解冻最后3个瓶颈块
        for layer in self.model.layers:

            if 'mobilenetv2' in layer.name:
                layer.trainable = True
                # 只解冻后面的层
                if 'block_1' in layer.name or 'block_2' in layer.name or 'block_3' in layer.name:
                    layer.trainable = False
        optimizer = keras.optimizers.Adam(
            learning_rate=self.lr_stage2,
            beta_1=0.9, beta_2=0.999
        )
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        lr_scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch: self.lr_stage2 * 0.5 * (1 + math.cos(math.pi * epoch / self.epochs_stage2))
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs_stage2,
            callbacks=[lr_scheduler, early_stopping],
            verbose=1,
        )
        self.history['stage2'] = history.history
        self.logger.info("第二阶段微调完成")
        return history.history
    def evaluate(self, test_ds: Any) -> Dict[str, float]:
        """在测试集上评估模型"""
        self.logger.info("在测试集上评估模型")
        results = self.model.evaluate(test_ds, verbose=1)
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
            self.logger.info(f"  {name}: {value:.4f}")
        return metrics
    def save_model(self, output_path: str) -> bool:
        """保存Keras模型"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.model.save(output_path)
            self.logger.info(f"模型已保存: {output_path}")

            return True
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            return False
    def save_labels(self, output_path: str) -> bool:
        """保存类别标签"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for name in self.class_names:
                    f.write(f"{name}\n")
            self.logger.info(f"标签已保存: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存标签失败: {e}")
            return False
    def save_training_history(self, output_path: str) -> bool:
        """保存训练历史"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=4)
            return True
        except Exception as e:
            self.logger.error(f"保存训练历史失败: {e}")
            return False
class ModelQuantizer:
    """
    模型量化器
    将Keras模型转换为TensorFlow Lite INT8量化模型
    """
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("quantizer")
        quant_config = self.config.get("ai_model.quantization", {})
        self.calibration_samples = quant_config.get("calibration_samples", 100)
        self.calibration_dataset_path = quant_config.get("calibration_dataset_path", "data/calibration")
        self.input_size = tuple(self.config.get("ai_model.classifier.input_size", [224, 224]))
    def quantize_model(self, keras_model_path: str, output_tflite_path: str) -> bool:
        """执行INT8全整数量化"""
        try:
            import tensorflow as tf
        except ImportError:
            self.logger.error("未安装TensorFlow")
            return False
        try:

            self.logger.info(f"加载模型: {keras_model_path}")
            model = tf.keras.models.load_model(keras_model_path)
            # 创建转换器
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # 设置优化
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # 代表性数据集（用于激活量化）
            def representative_dataset_gen():
                calibration_images = self._load_calibration_images()
                for img in calibration_images[:self.calibration_samples]:
                    yield [np.expand_dims(img, axis=0).astype(np.float32)]
            converter.representative_dataset = representative_dataset_gen
            # 设置全整数量化
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32
            # 转换
            self.logger.info("开始量化转换...")
            tflite_model = converter.convert()
            # 保存
            os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
            with open(output_tflite_path, 'wb') as f:
                f.write(tflite_model)
            # 计算模型大小
            original_size = os.path.getsize(keras_model_path) / 1024 / 1024
            quantized_size = len(tflite_model) / 1024 / 1024
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            self.logger.info(f"量化完成:")
            self.logger.info(f"  原始模型: {original_size:.2f} MB")
            self.logger.info(f"  量化模型: {quantized_size:.2f} MB")
            self.logger.info(f"  压缩比: {compression_ratio:.1f}:1")
            self.logger.info(f"  输出: {output_tflite_path}")
            return True
        except Exception as e:
            self.logger.error(f"量化失败: {e}", exc_info=True)
            return False
    def _load_calibration_images(self) -> List[np.ndarray]:
        """加载校准图像"""
        import cv2

        images = []
        if not os.path.exists(self.calibration_dataset_path):
            self.logger.warning(f"校准数据集不存在: {self.calibration_dataset_path}")
            # 生成随机校准数据
            for _ in range(self.calibration_samples):
                img = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
                images.append(img)
            return images
        for fname in os.listdir(self.calibration_dataset_path):
            fpath = os.path.join(self.calibration_dataset_path, fname)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.imread(fpath)
                if img is not None:
                    img = cv2.resize(img, self.input_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                if len(images) >= self.calibration_samples:
                    break
        self.logger.info(f"加载了 {len(images)} 张校准图像")
        return images
    def validate_tflite_model(self, tflite_path: str, test_image: Optional[np.ndarray] = None) -> Dict:
        """验证TFLite模型"""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                return {"valid": False, "error": "TFLite未安装"}
        try:
            interpreter = tflite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            result = {
                "valid": True,
                "input_shape": input_details[0]['shape'].tolist(),
                "input_dtype": str(input_details[0]['dtype']),
                "output_shape": output_details[0]['shape'].tolist(),
                "output_dtype": str(output_details[0]['dtype']),
                "model_size_mb": round(os.path.getsize(tflite_path) / 1024 / 1024, 2),
            }
            # 测试推理
            if test_image is None:
                test_image = np.random.randint(0, 255, input_details[0]['shape'][1:], dtype=np.uint8)

            input_tensor = np.expand_dims(test_image, axis=0).astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            start = time.time()
            interpreter.invoke()
            inference_time = (time.time() - start) * 1000
            output = interpreter.get_tensor(output_details[0]['index'])
            result["inference_time_ms"] = round(inference_time, 2)
            result["output_sum"] = float(np.sum(output))
            result["output_max"] = float(np.max(output))
            return result
        except Exception as e:
            return {"valid": False, "error": str(e)}
class DataCollector:
    """
    数据采集器
    从摄像头采集图像用于模型训练
    """
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger().get_logger("data_collector")
        self.output_dir = self.config.get("ai_model.training.dataset_path", "data/dataset")
        self.camera_id = self.config.get("hardware.camera.device_id", 0)
    def collect_class(self, class_name: str, num_images: int = 100,
                     delay: float = 0.5) -> bool:
        """采集一个类别的图像"""
        import cv2
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.logger.error("无法打开摄像头")
            return False
        self.logger.info(f"开始采集类别 '{class_name}'，共 {num_images} 张")
        self.logger.info("按 's' 保存当前帧，'q' 退出")
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            # 显示预览
            display = frame.copy()
            cv2.putText(display, f"Class: {class_name} | Count: {count}/{num_images}",

                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Data Collection", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # 保存图像
                timestamp = int(time.time() * 1000)
                filename = f"{class_name}_{timestamp}_{count}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, frame)
                count += 1
                self.logger.info(f"已保存: {filepath} ({count}/{num_images})")
                time.sleep(delay)
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.logger.info(f"采集完成，共保存 {count} 张图像")
        return count > 0
    def split_dataset(self, train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> bool:
        """划分数据集为train/val/test"""
        import shutil
        self.logger.info("开始划分数据集...")
        for class_name in os.listdir(self.output_dir):
            class_dir = os.path.join(self.output_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            images = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                continue
            np.random.shuffle(images)
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            splits = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:],
            }
            for split_name, split_images in splits.items():
                split_dir = os.path.join(self.output_dir, split_name, class_name)
                os.makedirs(split_dir, exist_ok=True)
                for img in split_images:

                    src = os.path.join(class_dir, img)
                    dst = os.path.join(split_dir, img)
                    shutil.copy2(src, dst)
            self.logger.info(f"  {class_name}: {n_total} 张 -> "
                           f"train={len(splits['train'])}, "
                           f"val={len(splits['val'])}, "
                           f"test={len(splits['test'])}")
        self.logger.info("数据集划分完成")
        return True
def main():
    """主函数：训练+量化完整流程"""
    parser = argparse.ArgumentParser(description="模型训练与量化")
    parser.add_argument("--mode", choices=["train", "quantize", "all", "collect"],
                       default="all", help="运行模式")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--class-name", type=str, help="采集类别名称")
    parser.add_argument("--num-images", type=int, default=100, help="采集图像数量")
    args = parser.parse_args()
    config = get_config(args.config)
    logger = get_logger(config)
    if args.mode == "collect":
        if not args.class_name:
            print("请指定 --class-name")
            return
        collector = DataCollector(config)
        collector.collect_class(args.class_name, args.num_images)
        return
    if args.mode in ["train", "all"]:
        trainer = ModelTrainer(config)
        train_ds, val_ds, test_ds = trainer.load_dataset()
        trainer.build_model()
        trainer.train_stage1(train_ds, val_ds)
        trainer.train_stage2(train_ds, val_ds)
        metrics = trainer.evaluate(test_ds)
        # 保存模型
        model_dir = "models"
        trainer.save_model(os.path.join(model_dir, "mobilenetv2.h5"))
        trainer.save_labels(os.path.join(model_dir, "labels.txt"))
        trainer.save_training_history(os.path.join(model_dir, "training_history.json"))
    if args.mode in ["quantize", "all"]:
        quantizer = ModelQuantizer(config)
        quantizer.quantize_model(
            "models/mobilenetv2.h5",
            "models/mobilenetv2_int8.tflite"

        )
        result = quantizer.validate_tflite_model("models/mobilenetv2_int8.tflite")
        print(f"量化模型验证: {json.dumps(result, indent=2)}")
if __name__ == "__main__":
    main()
# ============================================================
