#include "detectors/ncnn_detector.hpp"

namespace Infer
{

NCNNDetector::NCNNDetector()
{

}

NCNNDetector::~NCNNDetector()
{
    
}

std::vector<Object> NCNNDetector::Detect(const cv::Mat &bgr)
{
    if (isInited_ == false)
        return {};

    // --- Preprocessing
    int img_rows = bgr.rows;
    int img_cols = bgr.cols;
    float scale = 1.0f;
    int scaled_rows = img_rows;
    int scaled_cols = img_cols;
    if (scaled_rows > scaled_cols)
    {
        scale = static_cast<float>(target_size_) / scaled_rows;
        scaled_rows = target_size_;
        scaled_cols = scaled_cols * scale;
    }
    else
    {
        scale = static_cast<float>(target_size_) / scaled_cols;
        scaled_cols = target_size_;
        scaled_rows = scaled_rows * scale;
    }
    int pad_rows = (scaled_rows + max_stride_ - 1) / max_stride_ *
        max_stride_ - scaled_rows;
    int pad_cols = (scaled_cols + max_stride_ - 1) / max_stride_ *
        max_stride_ - scaled_cols;
    // letterbox
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_cols, img_rows, scaled_cols, scaled_rows);
    ncnn::Mat resized_pad;
    ncnn::copy_make_border(
        resized, resized_pad,
        pad_rows / 2, pad_rows - pad_rows / 2,
        pad_cols / 2, pad_cols - pad_cols / 2,
        ncnn::BORDER_CONSTANT, 114.0f
    );
    const float norm_values[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};
    resized_pad.substract_mean_normalize(0, norm_values);

    // --- Model inference
    std::vector<Object> proposals, objects;

    ncnn::Extractor ex = net_.create_extractor();
    ex.input("in0", resized_pad);

    int stride;
    // stride 8
    {
        ncnn::Mat out8;
        ex.extract("out0", out8);
        std::vector<Object> proposals8;
        stride = 8;
        GenerateProposals(out8, stride, anchors8_, proposals8);
        proposals.insert(proposals.end(), proposals8.begin(), proposals8.end());
    }
    // stride 16
    {
        ncnn::Mat out16;
        ex.extract("out1", out16);
        std::vector<Object> proposals16;
        stride = 16;
        GenerateProposals(out16, stride, anchors16_, proposals16);
        proposals.insert(proposals.end(), proposals16.begin(), proposals16.end());
    }
    // stride 32
    {
        ncnn::Mat out32;
        ex.extract("out2", out32);
        std::vector<Object> proposals32;
        stride = 32;
        GenerateProposals(out32, stride, anchors32_, proposals32);
        proposals.insert(proposals.end(), proposals32.begin(), proposals32.end());
    }

    // --- Postprocessing
    NMS(proposals, objects, img_rows, img_cols,
        pad_rows / 2, pad_cols / 2, scale, scale);

    return objects;
}

bool NCNNDetector::Initialize(const int threads, const std::string &model_path,
        const float conf_thres, const float nms_thres,
        const int target_size, const int max_stride)
{
    net_.opt.num_threads = threads;

    if (net_.load_param((model_path + ".param").c_str()) ||
        net_.load_model((model_path + ".bin").c_str()))
        return false;

    conf_thres_ = conf_thres;
    nms_thres_ = nms_thres;
    target_size_ = target_size;
    max_stride_ = max_stride;

    isInited_ = true;
    return true;
}

void NCNNDetector::GenerateProposals(const ncnn::Mat &feat_blob, int stride,
    const std::array<float, 6> &anchors, std::vector<Object> &proposals)
{
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;

    const int num_anchors = static_cast<int>(anchors.size()) / 2;
    const int walk = num_w / num_anchors;
    const int num_class = walk - 5;

    for (int i = 0; i < num_grid_y; ++i)
    {
        for (int j = 0; j < num_grid_x; ++j)
        {
            const float *matat = feat_blob.channel(i).row(j);
            for (int k = 0; k < num_anchors; ++k)
            {
                const float anchor_w = anchors[k * 2];
                const float anchor_h = anchors[k * 2 + 1];
                const float *ptr = matat + k * walk;
                float box_conf = ptr[4];
                if (box_conf >= conf_thres_)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int c = 0; c < num_class; ++c)
                    {
                        float score = ptr[5 + c];
                        if (score > class_score)
                        {
                            class_index = c;
                            class_score = score;
                        }
                        float confidence = box_conf * class_score;

                        if (confidence >= conf_thres_)
                        {
                            float dx = ptr[0];
                            float dy = ptr[1];
                            float dw = ptr[2];
                            float dh = ptr[3];

                            float pb_cx = (dx * 2.0f - 0.5f + j) * stride;
                            float pb_cy = (dy * 2.0f - 0.5f + i) * stride;
                            float pb_w = powf(dw * 2.0f, 2) * anchor_w;
                            float pb_h = powf(dh * 2.0f, 2) * anchor_h;

                            float x0 = pb_cx - pb_w * 0.5f;
                            float y0 = pb_cy - pb_h * 0.5f;
                            float x1 = pb_cx + pb_w * 0.5f;
                            float y1 = pb_cy + pb_h * 0.5f;

                            Object obj;
                            obj.rect.x = x0;
                            obj.rect.y = y0;
                            obj.rect.width = x1 - x0;
                            obj.rect.height = y1 - y0;
                            obj.label = class_index;
                            obj.prob = confidence;
                            proposals.emplace_back(obj);
                        }
                    }
                }
            }
        }
    }
}

void NCNNDetector::NMS(std::vector<Object> &proposals, std::vector<Object> &objects, int orig_h, int orig_w,
    float dh, float dw, float ratio_h, float ratio_w)
{
    objects.clear();
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    for (const auto &prop : proposals)
    {
        boxes.emplace_back(prop.rect);
        scores.emplace_back(prop.prob);
        labels.emplace_back(prop.label);
    }

    cv::dnn::NMSBoxes(boxes, scores, conf_thres_, nms_thres_, indices);

    for (const auto i : indices)
    {
        const auto &box = boxes[i];
        float x0 = box.x;
        float y0 = box.y;
        float x1 = box.x + box.width;
        float y1 = box.y + box.height;
        const float &score = scores[i];
        const int &label = labels[i];

        x0 = (x0 - dw) / ratio_w;
        y0 = (y0 - dh) / ratio_h;
        x1 = (x1 - dw) / ratio_w;
        y1 = (y1 - dh) / ratio_h;

        x0 = Clamp(x0, 0.0f, static_cast<float>(orig_w));
        y0 = Clamp(y0, 0.0f, static_cast<float>(orig_h));
        x1 = Clamp(x1, 0.0f, static_cast<float>(orig_w));
        y1 = Clamp(y1, 0.0f, static_cast<float>(orig_h));

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = score;
        obj.label = label;
        objects.emplace_back(obj);
    }
}

}   // namespace Infer