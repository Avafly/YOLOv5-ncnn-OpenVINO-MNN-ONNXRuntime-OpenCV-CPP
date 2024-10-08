#include "detectors/mnn_detector.hpp"

namespace Infer
{

MNNDetector::MNNDetector()
{

}

MNNDetector::~MNNDetector()
{

}

std::vector<Object> MNNDetector::Detect(const cv::Mat &bgr)
{
    if (isInited_ == false)
        return {};

    // --- Preprocessing
    cv::Size2f input_shape(target_size_, target_size_);
    float img_rows = bgr.rows;
    float img_cols = bgr.cols;
    float scale = input_shape.width / std::max(img_rows, img_cols);
    int scaled_cols = static_cast<int>(std::round(img_cols * scale));
    int scaled_rows = static_cast<int>(std::round(img_rows * scale));
    // letterbox with size of target_size_ x target_size_
    cv::Mat resized_pad;
    cv::resize(bgr, resized_pad, cv::Size(scaled_cols, scaled_rows), 0, 0, cv::INTER_AREA);
    int pad_cols = input_shape.width - scaled_cols;
    int pad_rows = input_shape.height - scaled_rows;
    cv::copyMakeBorder(
        resized_pad, resized_pad,
        pad_rows / 2, pad_rows - pad_rows / 2,
        pad_cols / 2, pad_cols - pad_cols / 2,
        cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0)
    );
    cv::cvtColor(resized_pad, resized_pad, cv::COLOR_BGR2RGB);
    resized_pad.convertTo(resized_pad, CV_32FC3, 1.0 / 255.0);

    std::vector<int> dims{1, target_size_, target_size_, 3};
    auto nhwc_tensor = MNN::Tensor::create<float>(dims, nullptr, MNN::Tensor::TENSORFLOW); // data format: NHWC
    auto nhwc_data = nhwc_tensor->host<float>();
    auto nhwc_size = nhwc_tensor->size();
    std::memcpy(nhwc_data, resized_pad.data, nhwc_size);

    auto inputTensor = net_->getSessionInput(session_, nullptr);
    inputTensor->copyFromHostTensor(nhwc_tensor);

    // --- Model inference
    net_->runSession(session_);
    // get outputs
    MNN::Tensor *out8 = net_->getSessionOutput(session_, "output0");    // stride 8
    MNN::Tensor *out16 = net_->getSessionOutput(session_, "362");       // stride 16
    MNN::Tensor *out32 = net_->getSessionOutput(session_, "365");       // stride 32
    MNN::Tensor out8_host(out8, out8->getDimensionType());
    MNN::Tensor out16_host(out16, out16->getDimensionType());
    MNN::Tensor out32_host(out32, out32->getDimensionType());
    // copy outputs
    out8->copyToHostTensor(&out8_host);
    out16->copyToHostTensor(&out16_host);
    out32->copyToHostTensor(&out32_host);

    // --- Postprocessing
    std::vector<Object> temp, proposals, objects;
    GenerateProposals(out8_host, 8, {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f}, temp);
    proposals.insert(proposals.end(), temp.begin(), temp.end());
    GenerateProposals(out16_host, 16, {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f}, temp);
    proposals.insert(proposals.end(), temp.begin(), temp.end());
    GenerateProposals(out32_host, 32, {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}, temp);
    proposals.insert(proposals.end(), temp.begin(), temp.end());

    NMS(proposals, objects, img_rows, img_cols, pad_rows / 2, pad_cols / 2, scale, scale);

    return objects;
}

bool MNNDetector::Initialize(const int threads, const std::string &model_path,
    const float conf_thres, const float nms_thres,
    const int target_size, const int max_stride)
{
    net_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(
        (model_path + ".mnn").c_str()
    ));
    if (net_ == nullptr)
        return false;

    MNN::ScheduleConfig config;
    config.numThread = threads;
    // change to MNN_FORWARD_AUTO to enable backend acceleration
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(MNN::BackendConfig::Precision_Normal);
    config.backendConfig = &backendConfig;

    session_ = net_->createSession(config);
    if (session_ == nullptr)
        return false;

    conf_thres_ = conf_thres;
    nms_thres_ = nms_thres;
    target_size_ = target_size;
    max_stride_ = max_stride;

    isInited_ = true;
    return true;
}

void MNNDetector::GenerateProposals(MNN::Tensor &data, int stride,
    const std::array<float, 6> &anchors, std::vector<Object> &proposals)
{
    proposals.clear();
    const auto feat_blob = data.host<float>();
    const int batches = data.shape()[0];
    const int num_w = data.shape()[3];
    const int num_grid_y = data.shape()[1];
    const int num_grid_x = data.shape()[2];
    const int num_anchors = anchors.size() / 2;
    const int walk = num_w / num_anchors;
    const int num_class = walk - 5;
    for (int b = 0; b < batches; ++b)
    {
        const float *ptr1 = feat_blob + b * (num_grid_y * num_grid_x * num_w);
        for (int i = 0; i < num_grid_y; ++i)
        {
            const float *ptr2 = ptr1 + i * (num_grid_x * num_w);
            for (int j = 0; j < num_grid_x; ++j)
            {
                const float *matat = ptr2 + j * num_w;
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
}

void MNNDetector::NMS(std::vector<Object> &proposals, std::vector<Object> &objects, int orig_h, int orig_w,
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