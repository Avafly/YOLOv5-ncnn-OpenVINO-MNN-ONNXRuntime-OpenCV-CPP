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
    cv::copyMakeBorder(resized_pad, resized_pad, 0, pad_rows, 0, pad_cols, cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0));

    // create tensor from image
    ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), resized_pad.data);
    infer_request_.set_input_tensor(input_tensor);

    // --- Model inference
    infer_request_.infer();
    const auto &output_tensor = infer_request_.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    float *output_data = output_tensor.data<float>();

    // --- Postprocessing
    std::vector<Object> objects;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels;
    std::vector<float> scores;
    // parse output
    for (int i = 0; i < output_shape[1]; ++i){
        float *grid = &output_data[i * output_shape[2]];
        float box_conf = grid[4];
        if (box_conf >= conf_thres_){
            float *classes_scores = &grid[5];
            cv::Mat classes_scores_mat(1, output_shape[2] - 5, CV_32FC1, classes_scores);
            cv::Point label;
            double max_class_score;
            cv::minMaxLoc(classes_scores_mat, 0, &max_class_score, 0, &label);
            if (max_class_score > conf_thres_)
            {
                float x = grid[0];
                float y = grid[1];
                float w = grid[2];
                float h = grid[3];
                float xmin = x - w * 0.5f;
                float ymin = y - h * 0.5f;
                boxes.emplace_back(cv::Rect(xmin, ymin, w, h));
                scores.emplace_back(box_conf);
                labels.emplace_back(label.x);
            }
        }
    }
    // nms
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thres_, nms_thres_, indices);
    for (const auto i : indices)
    {
        Object obj;
        obj.label = labels[i];
        obj.prob = scores[i];
        obj.rect.x = Clamp(boxes[i].x / scale, 0.0f, static_cast<float>(img_cols));
        obj.rect.y = Clamp(boxes[i].y / scale, 0.0f, static_cast<float>(img_rows));
        obj.rect.width = Clamp(boxes[i].width / scale, 0.0f, static_cast<float>(img_cols) - obj.rect.x);
        obj.rect.height = Clamp(boxes[i].height / scale, 0.0f, static_cast<float>(img_rows) - obj.rect.y);
        objects.emplace_back(obj);
    }

    return objects;
}

bool OVDetector::Initialize(const int threads, const std::string &model_path,
    const float conf_thres, const float nms_thres,
    const int target_size, const int max_stride)
{
    // --- Load model
    net_ = core_.read_model(model_path + ".xml", model_path + ".bin");
    if (net_ == nullptr)
        return false;

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

    isInited_ = true;
    return true;
}

}   // namespace Infer