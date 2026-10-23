# ============================================================
#!/usr/bin/env bash
# setup.sh - 香橙派5 一键部署
# 用法: bash setup.sh
set -e
echo "==> 1/4 装系统依赖"
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv i2c-tools
echo "==> 2/4 确认 I2C 已开启 (舵机驱动板用)"
# 香橙派在 orangepi-config 里开 I2C, 或改 /boot 下的 overlay, 重启生效
i2cdetect -y 1 || echo "!! 没检测到 I2C 总线, 先去配置里打开"
echo "==> 3/4 建虚拟环境装依赖"
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
echo "==> 4/4 检查模型文件"
if [ ! -f models/mobilenetv2_int8.tflite ]; then
    echo "!! 没找到 models/mobilenetv2_int8.tflite"
    echo "   先在PC上运行 python train.py 训练并导出量化模型, 拷到 models/ 目录"
fi
echo "部署完成。"
echo "下一步: python main.py --mode calibrate 做标定, 然后 python main.py --mode auto 开跑"
# ============================================================
