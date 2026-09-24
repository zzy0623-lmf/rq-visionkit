/*
 * RQ-VisionKit M4 板端运行时（RK3506）：分类推理封装（task=classify）
 *
 * 对应 PC 仿真版 inferencer.py 的 classify 分支（T2.5 扩展 Demo）。
 * 预处理与后处理严格对照 tools/verify_neudet_cls.py：
 *   resize(input_size) → BGR2RGB → mean=[127.5]*3 / norm=[1/127.5]*3（ncnn 层完成）
 *   → ncnn 推理 → 输出已含 softmax（Dense(softmax)）→ 取 top-k
 */
#ifndef RQ_VISIONKIT_CLASSIFY_INFERENCER_H
#define RQ_VISIONKIT_CLASSIFY_INFERENCER_H

#include <vector>
#include <string>

#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* 分类结果（对应 PC 版 classes 元素的 class_id/class_name/score） */
struct Classification
{
    int class_id;
    float score;
};

class ClassifyInferencer
{
public:
    ClassifyInferencer(int input_size, const char* input_blob, const char* output_blob,
                       int num_classes);

    ~ClassifyInferencer();

    int load(const char* param_path, const char* bin_path);

    /* 推理 → top_k 个类别（按 score 降序） */
    int infer(const cv::Mat& bgr, std::vector<Classification>& topk, int top_k = 5);

    bool is_loaded() const { return _net != 0; }

private:
    int _input_size;
    std::string _input_blob;
    std::string _output_blob;
    int _num_classes;

    float _mean[3];
    float _norm[3];

    ncnn::Net* _net;
};

#endif /* RQ_VISIONKIT_CLASSIFY_INFERENCER_H */
