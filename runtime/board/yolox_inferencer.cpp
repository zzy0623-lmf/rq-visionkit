/*
 * RQ-VisionKit M4 板端运行时（RK3506）：YOLOX 推理实现
 *
 * 后处理逻辑与 PC 仿真版 runtime/pc_sim/inferencer.py 严格一致：
 *   letterbox(pad=114) → NCHW float32 → ncnn 推理 → YOLOX decode → 按类 NMS。
 * 参考 ncnn 官方 YOLOX 示例（examples/yolox.cpp）的 decode / NMS 写法。
 */
#include "yolox_inferencer.h"

#include <cmath>
#include <algorithm>
#include <cfloat>

/* ---- 构造 / 析构 ---- */

YoloxInferencer::YoloxInferencer(int input_size, float conf_thres, float nms_thres,
                                 const char* input_blob, const char* output_blob,
                                 int num_classes)
    : _input_size(input_size),
      _conf_thres(conf_thres),
      _nms_thres(nms_thres),
      _input_blob(input_blob ? input_blob : "in0"),
      _output_blob(output_blob ? output_blob : "out0"),
      _num_classes(num_classes),
      _net(0)
{
}

YoloxInferencer::~YoloxInferencer()
{
    if (_net)
    {
        delete _net;
        _net = 0;
    }
}

int YoloxInferencer::load(const char* param_path, const char* bin_path)
{
    if (_net)
    {
        delete _net;
        _net = 0;
    }

    ncnn::Net* net = new ncnn::Net();
    /* 无 GPU，关闭 vulkan（RK3506 无 NPU/GPU 加速） */
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

/* ---- letterbox 预处理 ---- */

float YoloxInferencer::preprocess(const cv::Mat& bgr, ncnn::Mat& out)
{
    const int size = _input_size;
    const int img_h = bgr.rows;
    const int img_w = bgr.cols;

    /* pad=114（与 YOLOX 训练预处理一致） */
    cv::Mat padded(size, size, CV_8UC3, cv::Scalar(114, 114, 114));

    /* 等比缩放，短边对齐，长边留 pad */
    float scale = std::min((float)size / img_w, (float)size / img_h);
    int rw = (int)std::round(img_w * scale);
    int rh = (int)std::round(img_h * scale);

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(padded(cv::Rect(0, 0, rw, rh)));

    /* BGR uint8 → NCHW float32（值域 [0,255]，不做 mean/norm，与 PC 版一致） */
    out = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR, size, size);

    return scale;
}

/* ---- YOLOX decode（strides 8/16/32） ---- */

struct GridAndStride
{
    int grid0; /* x */
    int grid1; /* y */
    int stride;
};

static void generate_grids_and_stride(int target_size, const int* strides, int n_stride,
                                      std::vector<GridAndStride>& grid_strides)
{
    for (int s = 0; s < n_stride; s++)
    {
        int num_grid = target_size / strides[s];
        for (int g1 = 0; g1 < num_grid; g1++)
        {
            for (int g0 = 0; g0 < num_grid; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = strides[s];
                grid_strides.push_back(gs);
            }
        }
    }
}

static void generate_yolox_proposals(const std::vector<GridAndStride>& grid_strides,
                                     const ncnn::Mat& feat_blob, float prob_threshold,
                                     int num_classes, std::vector<Box>& objects)
{
    const int num_anchors = (int)grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        /* feat 为该 anchor 的 num_preds 个预测值 */
        const float* feat = feat_blob.row(anchor_idx);

        float x_center = (feat[0] + grid0) * (float)stride;
        float y_center = (feat[1] + grid1) * (float)stride;
        float w = std::exp(feat[2]) * (float)stride;
        float h = std::exp(feat[3]) * (float)stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float obj = feat[4];
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int c = 0; c < num_classes; c++)
        {
            float s = feat[5 + c];
            if (s > class_score)
            {
                class_score = s;
                class_index = c;
            }
        }

        float score = obj * class_score;
        if (score >= prob_threshold)
        {
            Box b;
            b.x1 = x0;
            b.y1 = y0;
            b.x2 = x0 + w;
            b.y2 = y0 + h;
            b.score = score;
            b.class_id = class_index;
            objects.push_back(b);
        }
    }
}

/* ---- 按 score 降序 + IoU NMS（objects 按 (class_id, score) 预排序） ---- */

static float intersection_area(const Box& a, const Box& b)
{
    float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (w <= 0.f || h <= 0.f)
        return 0.f;
    return w * h;
}

static void nms_sorted_bboxes(const std::vector<Box>& objects, std::vector<int>& picked,
                              float nms_threshold)
{
    picked.clear();
    const int n = (int)objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = (objects[i].x2 - objects[i].x1) * (objects[i].y2 - objects[i].y1);

    for (int i = 0; i < n; i++)
    {
        const Box& a = objects[i];
        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const Box& b = objects[picked[j]];
            float inter = intersection_area(a, b);
            float ovr = inter / (areas[i] + areas[picked[j]] - inter);
            if (ovr > nms_threshold)
            {
                keep = 0;
                break;
            }
        }
        if (keep)
            picked.push_back(i);
    }
}

/* ---- 推理入口 ---- */

int YoloxInferencer::infer(const cv::Mat& bgr, std::vector<Box>& boxes)
{
    boxes.clear();

    if (!_net)
        return -1;

    /* 1. 预处理 */
    ncnn::Mat in;
    float scale = preprocess(bgr, in);

    /* 2. 推理 */
    ncnn::Extractor ex = _net->create_extractor();
    ex.input(_input_blob.c_str(), in);

    ncnn::Mat out;
    if (ex.extract(_output_blob.c_str(), out) != 0)
        return -2;

    /* 3. 生成 grid + stride */
    static const int strides[3] = { 8, 16, 32 };
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(_input_size, strides, 3, grid_strides);

    /* 4. decode → 候选框 */
    std::vector<Box> proposals;
    generate_yolox_proposals(grid_strides, out, _conf_thres, _num_classes, proposals);

    /* 5. 按 (class_id, score 降序) 排序后按类 NMS */
    std::sort(proposals.begin(), proposals.end(),
              [](const Box& a, const Box& b) {
                  if (a.class_id != b.class_id)
                      return a.class_id < b.class_id;
                  return a.score > b.score;
              });

    std::vector<Box> picked;
    int begin = 0;
    while (begin < (int)proposals.size())
    {
        int end = begin;
        int cid = proposals[begin].class_id;
        while (end < (int)proposals.size() && proposals[end].class_id == cid)
            end++;

        std::vector<int> keep;
        nms_sorted_bboxes(std::vector<Box>(proposals.begin() + begin, proposals.begin() + end),
                          keep, _nms_thres);
        for (size_t k = 0; k < keep.size(); k++)
            picked.push_back(proposals[begin + keep[k]]);

        begin = end;
    }

    /* 6. 坐标还原到原图尺寸（除以 letterbox 缩放系数） */
    for (size_t i = 0; i < picked.size(); i++)
    {
        picked[i].x1 /= scale;
        picked[i].y1 /= scale;
        picked[i].x2 /= scale;
        picked[i].y2 /= scale;
    }

    boxes.swap(picked);
    return 0;
}
