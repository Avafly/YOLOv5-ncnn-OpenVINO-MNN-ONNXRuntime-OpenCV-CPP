#include "detectors/base_detector.hpp"
#include <cmath>

namespace Infer
{

bool BaseDetector::DrawObjects(cv::Mat &image, const std::vector<Object> &objects,
    const std::vector<std::string> &labels, bool isSilent)
{
    for (auto obj : objects)
    {
        if (obj.label >= labels.size())
            return false;

        if (isSilent != true)
            std::printf("%s = %.2f%% at (%.1f, %.1f)  %.1f x %.1f\n", labels[obj.label].c_str(), obj.prob * 100.0f,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", labels[obj.label].c_str(), obj.prob * 100.0f);

        auto scalar = cv::Scalar(114, 114, 114);
        cv::rectangle(image, obj.rect, scalar, 2);

        int baseLine = 5;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.75, 1, &baseLine);

        int x = obj.rect.x - 1;
        int y = obj.rect.y - label_size.height - baseLine;
        y = std::max(0, y);
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            scalar, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height + baseLine / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
    }

    return true;
}

void BaseDetector::GenerateProposals(const float *feat_blob, const std::array<int, 4> nhwc_shape, int stride,
    const std::array<float, 6> &anchors, std::vector<Object> &proposals)
{
    proposals.clear();

    const int batches = nhwc_shape[0];
    const int num_grid_y = nhwc_shape[1];
    const int num_grid_x = nhwc_shape[2];
    const int num_ch = nhwc_shape[3];
    const int num_anchors = anchors.size() / 2;
    const int walk = num_ch / num_anchors;
    const int num_class = walk - 5;
    for (int b = 0; b < batches; ++b)
    {
        const float *ptr1 = feat_blob + b * (num_grid_y * num_grid_x * num_ch);
        for (int i = 0; i < num_grid_y; ++i)
        {
            const float *ptr2 = ptr1 + i * (num_grid_x * num_ch);
            for (int j = 0; j < num_grid_x; ++j)
            {
                const float *ptr3 = ptr2 + j * num_ch;
                for (int k = 0; k < num_anchors; ++k)
                {
                    const float anchor_w = anchors[k * 2];
                    const float anchor_h = anchors[k * 2 + 1];
                    const float *ptr = ptr3 + k * walk;
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
}

void BaseDetector::GetLetterboxDimensions(const int img_rows, const int img_cols, const bool isDynamic,
    int &resize_rows, int &resize_cols, int &pad_rows, int &pad_cols, float &scale)
{
    scale = static_cast<float>(target_size_) / std::max(img_rows, img_cols);
    resize_rows = static_cast<int>(std::round(img_rows * scale));
    resize_cols = static_cast<int>(std::round(img_cols * scale));

    if (isDynamic)
    {
        pad_rows = (resize_rows + max_stride_ - 1) / max_stride_ * max_stride_ - resize_rows;
        pad_cols = (resize_cols + max_stride_ - 1) / max_stride_ * max_stride_ - resize_cols;
    }
    else
    {
        pad_rows = target_size_ - resize_rows;
        pad_cols = target_size_ - resize_cols;
    }
}

void BaseDetector::NMS(std::vector<Object> &proposals, std::vector<Object> &objects,
    const int orig_h, const int orig_w,
    const float dh, const float dw, const float ratio_h, const float ratio_w)
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
