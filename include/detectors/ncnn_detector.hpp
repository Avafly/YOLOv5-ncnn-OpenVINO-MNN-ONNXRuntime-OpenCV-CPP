#ifndef NCNN_DETECTOR_HPP_
#define NCNN_DETECTOR_HPP_

#include "detectors/base_detector.hpp"
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "net.h"

namespace Infer
{

class NCNNDetector : public BaseDetector
{
public:
    NCNNDetector();
    ~NCNNDetector();

    std::vector<Object> Detect(const cv::Mat &bgr) override;
    bool Initialize(const int threads, const std::string &model_path,
        const float conf_thres, const float nms_thres,
        const int target_size, const int max_stride) override;

private:
    ncnn::Net net_;

    void GenerateProposals(const ncnn::Mat &feat_blob, int stride,
        const std::array<float, 6> &anchors, std::vector<Object> &proposals);

    void NMS(std::vector<Object> &proposals, std::vector<Object> &objects, int orig_h, int orig_w,
        float dh, float dw, float ratio_h, float ratio_w);
};

}   // namespace Infer

#endif  // NCNN_DETECTOR_HPP_