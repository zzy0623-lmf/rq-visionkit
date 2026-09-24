/*
 * RQ-VisionKit M4 板端运行时（RK3506）：YOLOX 推理封装
 *
 * 与 PC 仿真版 runtime/pc_sim/inferencer.py 保持同一接口契约与后处理逻辑：
 *   - 模型文件与代码分离（config 由 deployer 下发，此处只按参数加载）
 *   - letterbox 预处理（pad=114）+ NCHW float32
 *   - YOLOX decode（strides 8/16/32）+ 按类 NMS
 *
 * 依赖：ncnn（官方示例 09_ai_mobilenetv2_yolov3 同款）+ OpenCV（cv::Mat）
 * 本文件不包含任何 RT-Thread / webnet 逻辑，便于单独编译与复用。
 */
#ifndef RQ_VISIONKIT_YOLOX_INFERENCER_H
#define RQ_VISIONKIT_YOLOX_INFERENCER_H

#include <vector>
#include <string>

#include "net.h"                 /* ncnn::Net / ncnn::Mat */
#include <opencv2/core/core.hpp> /* cv::Mat */

/* 检测框（对应 PC 版 boxes 元素的 x1/y1/x2/y2/score/class_id） */
struct Box
{
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

class YoloxInferencer
{
public:
    /* 构造时传入部署参数（对应 config.yaml） */
    YoloxInferencer(int input_size, float conf_thres, float nms_thres,
                    const char* input_blob, const char* output_blob,
                    int num_classes);

    ~YoloxInferencer();

    /* 加载模型文件（param/bin 路径，如 "/data/model/model.opt.param"） */
    int load(const char* param_path, const char* bin_path);

    /* 推理 + decode + NMS，返回 0 成功；boxes 坐标为原图像素坐标 */
    int infer(const cv::Mat& bgr, std::vector<Box>& boxes);

    bool is_loaded() const { return _net != 0; }

private:
    /* letterbox 预处理 → ncnn::Mat（NCHW float32），返回缩放系数 scale */
    float preprocess(const cv::Mat& bgr, ncnn::Mat& out);

    int _input_size;
    float _conf_thres;
    float _nms_thres;
    std::string _input_blob;
    std::string _output_blob;
    int _num_classes;

    ncnn::Net* _net;
};

#endif /* RQ_VISIONKIT_YOLOX_INFERENCER_H */
