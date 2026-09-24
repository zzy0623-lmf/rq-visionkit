/*
 * RQ-VisionKit M4 板端运行时（RK3506）：HTTP 服务器（webnet）
 *
 * 与 PC 仿真版 runtime/pc_sim/main.py 暴露同一接口契约：
 *   GET  /health   → {model_name, version, uptime_s, quant, task, num_classes}
 *   POST /reload   → 重新加载模型（无需重启进程）
 *   POST /infer    → multipart 图片 → 检测:{boxes:[...]} / 分类:{classes:[...]}
 *
 * 任务类型由编译期宏 RUNTIME_TASK 选择：
 *   TASK_DETECT   （默认，T2.2 检测运行时，YOLOX）
 *   TASK_CLASSIFY （T2.5 扩展，分类运行时，MobileNetV2 六分类）
 *
 * 依赖（均来自官方示例 09_ai_mobilenetv2_yolov3 的 RT-Thread SDK）：
 *   - webnet 组件（components/net_apps/webnet），CGI + upload 模块
 *   - ncnn（官方示例同款）、OpenCV（cv::imdecode）
 *
 * 【集成方式】见 runtime/board/README.md。
 */
#include <rtthread.h>
#include <webnet.h>
#include <wn_module.h>
#include <wn_session.h>
#include <wn_request.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yolox_inferencer.h"
#include "classify_inferencer.h"

#include <stdio.h>
#include <string.h>
#include <vector>

/* ---- 任务类型（编译期选择） ---- */
#define TASK_DETECT      0
#define TASK_CLASSIFY    1
#define RUNTIME_TASK     TASK_DETECT

/* ---- 部署参数（对应 config.yaml；模型文件由 deployer 经 SSH 下发到 /data/model） ---- */
#define MODEL_DIR           "/data/model"
#define MODEL_VERSION       "1.0.0"
#define MODEL_QUANT         "fp32"

#if RUNTIME_TASK == TASK_DETECT
#define MODEL_NAME          "yolox_neudet"
#define INPUT_SIZE          320
#define CONF_THRES          0.01f
#define NMS_THRES           0.65f
#else /* TASK_CLASSIFY */
#define MODEL_NAME          "mobilenetv2_neudet"
#define INPUT_SIZE          224
#define CONF_THRES          0.0f
#define NMS_THRES           0.0f
#endif

#define NUM_CLASSES         6
#define INPUT_BLOB          "in0"
#define OUTPUT_BLOB         "out0"

/* 模型文件路径（rq-convert 产物的 .opt.param / .opt.bin） */
#define PARAM_PATH          MODEL_DIR "/model.opt.param"
#define BIN_PATH            MODEL_DIR "/model.opt.bin"

static const char* CLASS_NAMES[NUM_CLASSES] = {
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"
};

/* ---- 全局状态（按任务类型二选一） ---- */
#if RUNTIME_TASK == TASK_DETECT
static YoloxInferencer* g_detector = 0;
#else
static ClassifyInferencer* g_classifier = 0;
#endif
static rt_tick_t g_start_tick = 0;

static void reload_model(void)
{
#if RUNTIME_TASK == TASK_DETECT
    if (g_detector)
    {
        delete g_detector;
        g_detector = 0;
    }
    g_detector = new YoloxInferencer(INPUT_SIZE, CONF_THRES, NMS_THRES,
                                     INPUT_BLOB, OUTPUT_BLOB, NUM_CLASSES);
    if (g_detector->load(PARAM_PATH, BIN_PATH) != 0)
    {
        rt_kprintf("[runtime] load model failed: %s\n", PARAM_PATH);
        delete g_detector;
        g_detector = 0;
    }
    else
    {
        rt_kprintf("[runtime] model loaded: %s\n", PARAM_PATH);
    }
#else
    if (g_classifier)
    {
        delete g_classifier;
        g_classifier = 0;
    }
    g_classifier = new ClassifyInferencer(INPUT_SIZE, INPUT_BLOB, OUTPUT_BLOB, NUM_CLASSES);
    if (g_classifier->load(PARAM_PATH, BIN_PATH) != 0)
    {
        rt_kprintf("[runtime] load model failed: %s\n", PARAM_PATH);
        delete g_classifier;
        g_classifier = 0;
    }
    else
    {
        rt_kprintf("[runtime] model loaded: %s\n", PARAM_PATH);
    }
#endif
}

/* ---- /health（CGI GET） ---- */

static void health_handler(struct webnet_session* session)
{
    rt_tick_t uptime = rt_tick_get() - g_start_tick;

    webnet_session_set_header(session, "application/json", 200, "OK", -1);
    webnet_session_printf(session,
        "{\"model_name\":\"%s\",\"version\":\"%s\",\"uptime_s\":%lu,"
        "\"quant\":\"%s\",\"task\":\"%s\",\"num_classes\":%d}",
        MODEL_NAME, MODEL_VERSION,
        (unsigned long)(uptime / RT_TICK_PER_SECOND),
        MODEL_QUANT,
#if RUNTIME_TASK == TASK_DETECT
        "detect",
#else
        "classify",
#endif
        NUM_CLASSES);
}

/* ---- /reload（CGI POST/GET） ---- */

static void reload_handler(struct webnet_session* session)
{
    reload_model();

    webnet_session_set_header(session, "application/json", 200, "OK", -1);
    webnet_session_printf(session,
        "{\"reloaded\":true,\"model_name\":\"%s\",\"version\":\"%s\"}",
        MODEL_NAME, MODEL_VERSION);
}

/* ---- /infer（upload 模块：接收 multipart 图片，upload_done 里推理） ---- */

struct UploadCtx
{
    unsigned char* data;
    int size;
    int capacity;
};

static int infer_upload_open(struct webnet_session* session)
{
    struct UploadCtx* ctx = (struct UploadCtx*)rt_malloc(sizeof(struct UploadCtx));
    if (ctx == RT_NULL)
        return -1;

    ctx->data = RT_NULL;
    ctx->size = 0;
    ctx->capacity = 0;
    return (int)ctx;
}

static int infer_upload_write(struct webnet_session* session, const void* data, rt_size_t length)
{
    struct UploadCtx* ctx = (struct UploadCtx*)webnet_upload_get_userdata(session);
    if (ctx == RT_NULL)
        return 0;

    if (ctx->size + (int)length > ctx->capacity)
    {
        int new_cap = ctx->capacity ? ctx->capacity : 4096;
        while (new_cap < ctx->size + (int)length)
            new_cap *= 2;
        unsigned char* p = (unsigned char*)rt_realloc(ctx->data, new_cap);
        if (p == RT_NULL)
            return 0;
        ctx->data = p;
        ctx->capacity = new_cap;
    }

    rt_memcpy(ctx->data + ctx->size, data, length);
    ctx->size += (int)length;
    return length;
}

static void infer_upload_done(struct webnet_session* session)
{
    struct UploadCtx* ctx = (struct UploadCtx*)webnet_upload_get_userdata(session);
    rt_tick_t t0 = 0, t1 = 0;
    rt_size_t total_mem = 0, used_mem = 0, max_used = 0;
    char tmp[512];

    rt_memory_info(&total_mem, &used_mem, &max_used);
    webnet_session_set_header(session, "application/json", 200, "OK", -1);

    if (ctx == RT_NULL || ctx->data == RT_NULL)
        goto __empty;

#if RUNTIME_TASK == TASK_DETECT
    if (g_detector == 0)
        goto __empty;

    {
        std::vector<unsigned char> buf(ctx->data, ctx->data + ctx->size);
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (img.empty())
            goto __empty;

        std::vector<Box> boxes;
        t0 = rt_tick_get();
        g_detector->infer(img, boxes);
        t1 = rt_tick_get();

        webnet_session_printf(session, "{\"boxes\":[");
        for (size_t i = 0; i < boxes.size(); i++)
        {
            const Box& b = boxes[i];
            rt_snprintf(tmp, sizeof(tmp),
                "%s{\"x1\":%.1f,\"y1\":%.1f,\"x2\":%.1f,\"y2\":%.1f,"
                "\"score\":%.4f,\"class_id\":%d,\"class_name\":\"%s\"}",
                i ? "," : "",
                b.x1, b.y1, b.x2, b.y2, b.score, b.class_id,
                (b.class_id >= 0 && b.class_id < NUM_CLASSES) ? CLASS_NAMES[b.class_id] : "");
            webnet_session_printf(session, "%s", tmp);
        }
        webnet_session_printf(session,
            "],\"inference_ms\":%lu,\"mem_kb\":%lu}",
            (unsigned long)(t1 - t0), (unsigned long)(used_mem / 1024));
        rt_kprintf("[runtime] /infer detect: %d boxes, %lu ms\n",
                   (int)boxes.size(), (unsigned long)(t1 - t0));
        return;
    }
#else /* TASK_CLASSIFY */
    if (g_classifier == 0)
        goto __empty;

    {
        std::vector<unsigned char> buf(ctx->data, ctx->data + ctx->size);
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (img.empty())
            goto __empty;

        std::vector<Classification> topk;
        t0 = rt_tick_get();
        g_classifier->infer(img, topk, 5);
        t1 = rt_tick_get();

        webnet_session_printf(session, "{\"classes\":[");
        for (size_t i = 0; i < topk.size(); i++)
        {
            const Classification& c = topk[i];
            rt_snprintf(tmp, sizeof(tmp),
                "%s{\"class_id\":%d,\"class_name\":\"%s\",\"score\":%.4f}",
                i ? "," : "", c.class_id,
                (c.class_id >= 0 && c.class_id < NUM_CLASSES) ? CLASS_NAMES[c.class_id] : "",
                c.score);
            webnet_session_printf(session, "%s", tmp);
        }
        webnet_session_printf(session,
            "],\"inference_ms\":%lu,\"mem_kb\":%lu}",
            (unsigned long)(t1 - t0), (unsigned long)(used_mem / 1024));
        rt_kprintf("[runtime] /infer classify: top-1 %d, %lu ms\n",
                   topk.empty() ? -1 : topk[0].class_id, (unsigned long)(t1 - t0));
        return;
    }
#endif

__empty:
    webnet_session_printf(session,
        "{\"inference_ms\":0,\"mem_kb\":%lu}", (unsigned long)(used_mem / 1024));
}

static int infer_upload_close(struct webnet_session* session)
{
    struct UploadCtx* ctx = (struct UploadCtx*)webnet_upload_get_userdata(session);
    if (ctx != RT_NULL)
    {
        if (ctx->data)
            rt_free(ctx->data);
        rt_free(ctx);
    }
    return 0;
}

/* ---- 服务器初始化（INIT_APP_EXPORT 自动启动） ---- */

static void runtime_server_init(void)
{
    static const struct webnet_module_upload_entry infer_entry = {
        "/infer",            /* url */
        infer_upload_open,   /* upload_open */
        infer_upload_close,  /* upload_close */
        infer_upload_write,  /* upload_write */
        infer_upload_done,   /* upload_done */
    };

    g_start_tick = rt_tick_get();
    reload_model();

    webnet_init();
    webnet_cgi_register("health", health_handler);
    webnet_cgi_register("reload", reload_handler);
    webnet_upload_add(&infer_entry);

    /* 启动 HTTP 服务：端口 80，webroot /data/www（无 webroot 也可） */
    webnet_start(80, "/data/www");

    rt_kprintf("[runtime] HTTP server started on :80\n");
    rt_kprintf("[runtime]   GET  /cgi-bin/health\n");
    rt_kprintf("[runtime]   POST /cgi-bin/reload\n");
    rt_kprintf("[runtime]   POST /infer  (multipart image)\n");
}
INIT_APP_EXPORT(runtime_server_init);

/* 手动启动/重载命令（msh 里可调用，便于调试） */
static void runtime_reload(int argc, char** argv)
{
    reload_model();
}
MSH_CMD_EXPORT(runtime_reload, reload the runtime model);
