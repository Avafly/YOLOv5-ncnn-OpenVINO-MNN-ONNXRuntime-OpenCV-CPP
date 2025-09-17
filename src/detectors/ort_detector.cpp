#include "detectors/ort_detector.hpp"

namespace Infer
{

ORTDetector::ORTDetector()
{

}

ORTDetector::~ORTDetector()
{

}

std::vector<Object> ORTDetector::Detect(const cv::Mat &bgr)
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
    cv::Mat letterbox, blob;
    cv::resize(bgr, letterbox, cv::Size(resize_cols, resize_rows), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(
        letterbox, letterbox,
        pad_rows / 2, pad_rows - pad_rows / 2,
        pad_cols / 2, pad_cols - pad_cols / 2,
        cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0)
    );
    cv::dnn::blobFromImage(letterbox, blob, 1.0 / 255.0, cv::Size(letterbox.cols, letterbox.rows), cv::Scalar(0, 0, 0), true, false);

    std::vector<int64_t> input_tensor_shape = {1, 3, letterbox.rows, letterbox.cols};
    int64_t data_element_count = 1;
    for (const auto &element : input_tensor_shape)
        data_element_count *= element;
    Ort::Value input_tensors = Ort::Value::CreateTensor<float>(
        memory_info_, (float *)blob.data, static_cast<size_t>(data_element_count),
        input_tensor_shape.data(), input_tensor_shape.size()
    );

    // -- Model inference
    std::vector<Ort::Value> output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_ptr_.data(),
        &input_tensors,
        input_names_ptr_.size(),
        output_names_ptr_.data(),
        output_names_ptr_.size()
    );

    // --- Postprocessing
    std::vector<Object> proposals, objects;
    for (size_t i = 0; i < strides_.size(); ++i)
    {
        std::vector<Object> temp;
        auto output_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        GenerateProposals(
            output_tensors[i].GetTensorData<float>(),
            {
                static_cast<int>(output_shape[0]), static_cast<int>(output_shape[1]),
                static_cast<int>(output_shape[2]), static_cast<int>(output_shape[3])
            },
            strides_[i], anchors_[i], temp
        );
        proposals.insert(proposals.end(), temp.begin(), temp.end());
    }

    NMS(proposals, objects, img_rows, img_cols, pad_rows / 2, pad_cols / 2, scale, scale);

    return objects;
}

bool ORTDetector::Initialize(const int threads, const std::string &model_path,
    const float conf_thres, const float nms_thres,
    const int target_size, const int max_stride, const int num_class)
{
    // create env, session, and memory
    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "YOLOV5_ONNXRUNTIME");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(std::max(1, threads));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    try
    {
        session_ = Ort::Session(env_, (model_path + ".onnx").c_str(), session_options);
    }
    catch (const Ort::Exception& e)
    {
        std::cout << "Failed to load model: " << e.what() << "\n";
        return false;
    }

    memory_info_ = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    // get input & output names for inference
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_count = session_.GetInputCount(), out_count = session_.GetOutputCount();
    for (size_t i = 0; i < in_count; ++i)
        input_names_.emplace_back(std::string(session_.GetInputNameAllocated(i, allocator).get()));
    for (size_t i = 0; i < out_count; ++i)
        output_names_.emplace_back(std::string(session_.GetOutputNameAllocated(i, allocator).get()));
    for (const auto &name : input_names_)
        input_names_ptr_.emplace_back(name.c_str());
    for (const auto &name : output_names_)
        output_names_ptr_.emplace_back(name.c_str());

    conf_thres_ = conf_thres;
    nms_thres_ = nms_thres;
    target_size_ = target_size;
    max_stride_ = max_stride;
    num_class_ = num_class;

    isInited_ = true;
    return true;
}

}   // namespace Infer