#include "detectors/ov_detector.hpp"

namespace Infer
{

OVDetector::OVDetector()
{

}
OVDetector::~OVDetector()
{
    
}

std::vector<Object> OVDetector::Detect(const cv::Mat &bgr)
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
    // create input
    ov::Shape input_shape = {1,
        static_cast<unsigned long>(letterbox.rows),
        static_cast<unsigned long>(letterbox.cols),
        static_cast<unsigned long>(letterbox.channels())
    };
    ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), input_shape, letterbox.data);
    infer_request_.set_input_tensor(input_tensor);

    // --- Model inference
    infer_request_.infer();

    // --- Postprocessing
    std::vector<Object> proposals, objects;
    for (size_t i = 0; i < net_->outputs().size(); ++i)
    {
        const auto &output_tensor = infer_request_.get_output_tensor(i);
        std::vector<Object> temp;
        GenerateProposals(
            output_tensor.data<float>(),
            {1, letterbox.rows / strides_[i], letterbox.cols / strides_[i], (num_class_ + 5) * 3},
            strides_[i], anchors_[i], temp
        );
        proposals.insert(proposals.end(), temp.begin(), temp.end());
    }

    NMS(proposals, objects, img_rows, img_cols, pad_rows / 2, pad_cols / 2, scale, scale);

    return objects;
}

bool OVDetector::Initialize(const int threads, const std::string &model_path,
    const float conf_thres, const float nms_thres,
    const int target_size, const int max_stride, const int num_class)
{
    // --- Load model
    try
    {
        net_ = core_.read_model(model_path + ".xml", model_path + ".bin");
    }
    catch (const ov::Exception& e)
    {
        std::cout << "Failed to load model: " << e.what() << "\n";
        return false;
    }

    // --- Use PrePostProcessor API
    // instance PrePostProcessor object
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(net_);
    for (const auto &input : net_->inputs())
    {
        std::string input_name = input.get_any_name();
        // declare input data information (opencv style)
        ppp.input(input_name).tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        // specify actual model layout (pytorch style)
        ppp.input(input_name).model().set_layout("NCHW");
        // apply preprocessing modifing the original 'model'
        ppp.input(input_name).preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .mean({0.0f, 0.0f, 0.0f})
            .scale({255.0f, 255.0f, 255.0f});
    }
    for (const auto &output : net_->outputs())
    {
        std::string output_name = output.get_any_name();
        // set output tensor information
        ppp.output(output_name).tensor().set_element_type(ov::element::f32);
    }
    // integrate above steps into net
    net_ = ppp.build();
    if (net_ == nullptr)
        return false;
    // change CPU to GPU to enable GPU acceleration
    compiled_model_ = core_.compile_model(net_, "CPU", ov::inference_num_threads(threads));
    infer_request_ = compiled_model_.create_infer_request();

    conf_thres_ = conf_thres;
    nms_thres_ = nms_thres;
    target_size_ = target_size;
    max_stride_ = max_stride;
    num_class_ = num_class;

    isInited_ = true;
    return true;
}

}   // namespace Infer