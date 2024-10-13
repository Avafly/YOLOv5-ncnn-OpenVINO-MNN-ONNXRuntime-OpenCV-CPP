#include "detectors/cv_detector.hpp"

namespace Infer
{

CVDetector::CVDetector()
{

}

CVDetector::~CVDetector()
{

}

std::vector<Object> CVDetector::Detect(const cv::Mat &bgr)
{
    if (isInited_ == false)
        return {};

    // --- preprocessing
    int img_rows = bgr.rows;
    int img_cols = bgr.cols;
    float scale;
    int resize_rows, resize_cols, pad_rows, pad_cols;
    // OpenCV need to know input shapes ahead of inference so that memory can be allocated.
    // Therefore, dynamic height and width is not supported by the current dnn engine.
    // Reference: https://github.com/opencv/opencv/issues/19347#issuecomment-1868227401
    GetLetterboxDimensions(
        img_rows, img_cols, false,
        resize_rows, resize_cols, pad_rows, pad_cols, scale
    );
    cv::Mat letterbox, blob;
    cv::resize(bgr, letterbox, cv::Size(resize_cols, resize_rows), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(
        letterbox, letterbox,
        pad_rows / 2, pad_rows - pad_rows / 2,
        pad_cols / 2, pad_cols - pad_cols / 2,
        cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0)
    );
    cv::dnn::blobFromImage(letterbox, blob, 1.0 / 255.0, cv::Size(letterbox.cols, letterbox.rows), cv::Scalar(), true, false);

    net_.setInput(blob);

    // --- Model inference
    std::vector<cv::String> outputNames = net_.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, outputNames);

    // --- Postprocessing
    // ensure they are in descending order of size: 80x80, 40x40, 20x20
    std::sort(outputs.begin(), outputs.end(), [](const cv::Mat &a, const cv::Mat &b) {
        return std::max(a.size[1], a.size[2]) > std::max(b.size[1], b.size[2]);
    });

    std::vector<Object> proposals, objects;
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        cv::Mat &output = outputs[i];
        std::vector<Object> temp;
        GenerateProposals(
            (float *)output.data,
            {output.size[0], output.size[1], output.size[2], output.size[3]},
            strides_[i], anchors_[i], temp
        );
        proposals.insert(proposals.end(), temp.begin(), temp.end());
    }

    NMS(proposals, objects, img_rows, img_cols, pad_rows / 2, pad_cols / 2, scale, scale);

    return objects;
}

bool CVDetector::Initialize(const int threads, const std::string &model_path,
    const float conf_thres, const float nms_thres,
    const int target_size, const int max_stride, const int num_class)
{
    try
    {
        net_ = cv::dnn::readNet(model_path + ".onnx");
    }
    catch (const cv::Exception &e)
    {
        std::cout << "Failed to load model: " << e.what() << "\n";
        return false;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    cv::setNumThreads(threads);

    conf_thres_ = conf_thres;
    nms_thres_ = nms_thres;
    target_size_ = target_size;
    max_stride_ = max_stride;
    num_class_ = num_class;

    isInited_ = true;
    return true;
}

}   // namespace Infer