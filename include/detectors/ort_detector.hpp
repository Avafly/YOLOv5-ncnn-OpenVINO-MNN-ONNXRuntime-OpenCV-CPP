#ifndef ORT_DETECTOR_HPP_
#define ORT_DETECTOR_HPP_

#include "detectors/base_detector.hpp"
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace Infer
{

class ORTDetector : public BaseDetector
{
public:
    ORTDetector();
    ~ORTDetector();

    std::vector<Object> Detect(const cv::Mat &bgr) override;
    bool Initialize(const int threads, const std::string &model_path,
        const float conf_thres, const float nms_thres,
        const int target_size, const int max_stride, const int num_class) override;

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<std::string> input_names_, output_names_;
    std::vector<const char *> input_names_ptr_, output_names_ptr_;
};

}   // namespace Infer

#endif  // ORT_DETECTOR_HPP_