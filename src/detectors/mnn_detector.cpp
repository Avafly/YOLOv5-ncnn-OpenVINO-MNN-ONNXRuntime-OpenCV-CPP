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
    // letterbox with size of target_size_ x target_size_
    int img_rows = bgr.rows;
    int img_cols = bgr.cols;
    float scale;
    int resize_rows, resize_cols, pad_rows, pad_cols;
    GetLetterboxDimensions(
        img_rows, img_cols, true,
        resize_rows, resize_cols, pad_rows, pad_cols, scale
    );
    cv::Mat letterbox;
    cv::resize(bgr, letterbox, cv::Size(resize_cols, resize_rows), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(
        letterbox, letterbox,
        pad_rows / 2, pad_rows - pad_rows / 2,
        pad_cols / 2, pad_cols - pad_cols / 2,
        cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0)
    );
    cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
    letterbox.convertTo(letterbox, CV_32FC3, 1.0 / 255.0);
    // create input tensor
    std::vector<int> dims{1, letterbox.rows, letterbox.cols, 3};
    auto nhwc_tensor = MNN::Tensor::create<float>(dims, nullptr, MNN::Tensor::TENSORFLOW); // data format: NHWC
    auto nhwc_data = nhwc_tensor->host<float>();
    auto nhwc_size = nhwc_tensor->size();
    std::memcpy(nhwc_data, letterbox.data, nhwc_size);

    auto input_tensor = net_->getSessionInput(session_, nullptr);
    net_->resizeTensor(input_tensor, {1, 3, letterbox.rows, letterbox.cols});
    net_->resizeSession(session_);
    input_tensor->copyFromHostTensor(nhwc_tensor);

    // --- Model inference
    net_->runSession(session_);

    std::vector<Object> proposals, objects;
    for (size_t i = 0; i < strides_.size(); ++i)
    {
        // get outputs
        MNN::Tensor *out = net_->getSessionOutput(session_, output_names_[i].c_str());
        // create tensions with the same shape as given tensions
        MNN::Tensor out_host(out, out->getDimensionType());
        // save outputs
        out->copyToHostTensor(&out_host);

        std::vector<Object> temp;
        GenerateProposals(
            out_host.host<float>(),
            {out_host.shape()[0], out_host.shape()[1], out_host.shape()[2], out_host.shape()[3]},
            strides_[i], anchors_[i], temp
        );
        proposals.insert(proposals.end(), temp.begin(), temp.end());
    }

    // --- Postprocessing
    NMS(proposals, objects, img_rows, img_cols, pad_rows / 2, pad_cols / 2, scale, scale);

    return objects;
}

bool MNNDetector::Initialize(const int threads, const std::string &model_path,
    const float conf_thres, const float nms_thres,
    const int target_size, const int max_stride, const int num_class)
{
    net_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(
        (model_path + ".mnn").c_str()
    ));
    if (net_ == nullptr)
        return false;

    MNN::ScheduleConfig config;
    config.numThread = std::max(1, threads);
    // change to MNN_FORWARD_AUTO to enable backend acceleration
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(MNN::BackendConfig::Precision_Normal);
    config.backendConfig = &backendConfig;

    session_ = net_->createSession(config);
    if (session_ == nullptr)
        return false;

    // get and sort output names
    for (const auto &[key, value] : net_->getSessionOutputAll(session_))
        output_names_.push_back(key);
    // ensure they are in descending order of size: 80, 40, 20
    std::sort(output_names_.begin(), output_names_.end(), [](const std::string &a, const std::string &b) {
        return std::atoi(a.c_str()) < std::atoi(b.c_str());
    });

    conf_thres_ = conf_thres;
    nms_thres_ = nms_thres;
    target_size_ = target_size;
    max_stride_ = max_stride;
    num_class_ = num_class;

    isInited_ = true;
    return true;
}

}   // namespace Infer