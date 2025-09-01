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
        const int target_size, const int max_stride, const int num_class) override;

private:
    ncnn::Net net_;

    /**
     * @brief generate proposals with ncnn specific way
     * @param feat_blob     feature blob
     * @param stride        downsampling stride
     * @param anchors       anchors for the current stride
     * @param proposals     generated proposals from the blob
     */
    void GenerateProposals(const ncnn::Mat &feat_blob, int stride,
        const std::array<float, 6> &anchors, std::vector<Object> &proposals);
    using BaseDetector::GenerateProposals;
};

}   // namespace Infer

#endif  // NCNN_DETECTOR_HPP_