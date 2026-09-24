/*
 * RQ-VisionKit M4 板端运行时（RK3506）：分类推理实现
 *
 * 预处理/后处理严格对照 tools/verify_neudet_cls.py（NEU-DET 六分类 MobileNetV2）。
 */
#include "classify_inferencer.h"

#include <cmath>
#include <algorithm>

ClassifyInferencer::ClassifyInferencer(int input_size, const char* input_blob,
                                       const char* output_blob, int num_classes)
    : _input_size(input_size),
      _input_blob(input_blob ? input_blob : "in0"),
      _output_blob(output_blob ? output_blob : "out0"),
      _num_classes(num_classes),
      _net(0)
{
    /* 与 Keras 的 (x - 127.5) / 127.5 预处理一致（去 Lambda 后由 ncnn 层完成） */
    for (int i = 0; i < 3; i++)
    {
        _mean[i] = 127.5f;
        _norm[i] = 1.0f / 127.5f;
    }
}

ClassifyInferencer::~ClassifyInferencer()
{
    if (_net)
    {
        delete _net;
        _net = 0;
    }
}

int ClassifyInferencer::load(const char* param_path, const char* bin_path)
{
    if (_net)
    {
        delete _net;
        _net = 0;
    }

    ncnn::Net* net = new ncnn::Net();
    net->opt.use_vulkan_compute = false;

    if (net->load_param(param_path) != 0)
    {
        delete net;
        return -1;
    }
    if (net->load_model(bin_path) != 0)
    {
        delete net;
        return -1;
    }

    _net = net;
    return 0;
}

int ClassifyInferencer::infer(const cv::Mat& bgr, std::vector<Classification>& topk, int top_k)
{
    topk.clear();

    if (!_net)
        return -1;

    /* 1. 预处理：resize → BGR2RGB → NCHW float32 + mean/norm（ncnn 层） */
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(_input_size, _input_size), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB,
                                          _input_size, _input_size);
    in.substract_mean_normalize(_mean, _norm);

    /* 2. 推理 */
    ncnn::Extractor ex = _net->create_extractor();
    ex.input(_input_blob.c_str(), in);

    ncnn::Mat out;
    if (ex.extract(_output_blob.c_str(), out) != 0)
        return -2;

    /* 3. 输出已含 softmax（Dense(softmax)），取 top_k 降序 */
    const int n = out.w * out.h * out.c;
    std::vector<std::pair<float, int> > scores; /* (score, class_id) */
    for (int i = 0; i < n && i < _num_classes; i++)
    {
        float s = out[i];
        scores.push_back(std::make_pair(s, i));
    }

    std::sort(scores.begin(), scores.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });

    int k = top_k < (int)scores.size() ? top_k : (int)scores.size();
    for (int i = 0; i < k; i++)
    {
        Classification c;
        c.class_id = scores[i].second;
        c.score = scores[i].first;
        topk.push_back(c);
    }

    return 0;
}
