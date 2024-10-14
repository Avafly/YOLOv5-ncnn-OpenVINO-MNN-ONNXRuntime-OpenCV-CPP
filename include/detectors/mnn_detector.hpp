#ifndef MNN_DETECTOR_HPP_
#define MNN_DETECTOR_HPP_

#include "detectors/base_detector.hpp"
#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

namespace Infer
{

class MNNDetector : public BaseDetector
{
public:
    MNNDetector();
    ~MNNDetector();

    std::vector<Object> Detect(const cv::Mat &bgr) override;
    bool Initialize(const int threads, const std::string &model_path,
        const float conf_thres, const float nms_thres,
        const int target_size, const int max_stride, const int num_class) override;

private:
    std::unique_ptr<MNN::Interpreter> net_ = nullptr;
    MNN::Session *session_ = nullptr;
    std::vector<std::string> output_names_;
};

}   // namespace Infer

#endif  // MNN_DETECTOR_HPP_