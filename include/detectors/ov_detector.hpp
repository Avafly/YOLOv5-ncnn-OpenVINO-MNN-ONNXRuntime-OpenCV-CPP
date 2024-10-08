#ifndef OV_DETECTOR_HPP_
#define OV_DETECTOR_HPP_

#include "detectors/base_detector.hpp"
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace Infer
{

class OVDetector : public BaseDetector
{
public:
    OVDetector();
    ~OVDetector();

    std::vector<Object> Detect(const cv::Mat &bgr) override;
    bool Initialize(const int threads, const std::string &model_path,
        const float conf_thres, const float nms_thres,
        const int target_size, const int max_stride) override;

private:
    ov::Core core_;
    std::shared_ptr<ov::Model> net_ = nullptr;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
};

}   // namespace Infer

#endif  // OV_DETECTOR_HPP_