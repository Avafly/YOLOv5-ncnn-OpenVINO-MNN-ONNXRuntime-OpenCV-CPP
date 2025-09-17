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
    float scale;
    int resize_rows, resize_cols, pad_rows, pad_cols;
    GetLetterboxDimensions(
        img_rows, img_cols, true,
        resize_rows, resize_cols, pad_rows, pad_cols, scale
    );
    // letterbox
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_cols, img_rows, resize_cols, resize_rows
    );
    ncnn::Mat letterbox;
    ncnn::copy_make_border(
        resized, letterbox,
        pad_rows / 2, pad_rows - pad_rows / 2,
        pad_cols / 2, pad_cols - pad_cols / 2,
        ncnn::BORDER_CONSTANT, 114.0f
    );
    const float norm_values[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};
    letterbox.substract_mean_normalize(0, norm_values);

    // --- Model inference
    std::vector<Object> proposals, objects;

    ncnn::Extractor ex = net_.create_extractor();
    ex.input("in0", letterbox);

    const char *blob_names[] = {"out0", "out1", "out2"};
    for (size_t i = 0; i < strides_.size(); ++i)
    {
        ncnn::Mat out;
        ex.extract(blob_names[i], out);
        std::vector<Object> temp;
        GenerateProposals(out, strides_[i], anchors_[i], temp);
        proposals.insert(proposals.end(), temp.begin(), temp.end());
    }

    // --- Postprocessing
    NMS(proposals, objects, img_rows, img_cols, pad_rows / 2, pad_cols / 2, scale, scale);

    return objects;
}

bool NCNNDetector::Initialize(const int threads, const std::string &model_path,
        const float conf_thres, const float nms_thres,
        const int target_size, const int max_stride, const int num_class)
{
    net_.opt.num_threads = std::max(1, threads);

    if (net_.load_param((model_path + ".param").c_str()) ||
        net_.load_model((model_path + ".bin").c_str()))
        return false;

    conf_thres_ = conf_thres;
    nms_thres_ = nms_thres;
    target_size_ = target_size;
    max_stride_ = max_stride;
    num_class_ = num_class;

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
                        float confidence = box_conf * score;

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

}   // namespace Infer
