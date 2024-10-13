#ifndef CV_DETECTOR_HPP_
#define CV_DETECTOR_HPP_

#include "detectors/base_detector.hpp"
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

namespace Infer
{

class CVDetector : public BaseDetector
{
public:
    CVDetector();
    ~CVDetector();

    std::vector<Object> Detect(const cv::Mat &bgr) override;
    bool Initialize(const int threads, const std::string &model_path,
        const float conf_thres, const float nms_thres,
        const int target_size, const int max_stride, const int num_class) override;

private:
    cv::dnn::Net net_;
};

}

#endif  // CV_DETECTOR_HPP_