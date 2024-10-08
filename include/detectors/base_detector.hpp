#ifndef BASE_DETECTOR_HPP_
#define BASE_DETECTOR_HPP_

#include <opencv2/opencv.hpp>

namespace Infer
{

struct Object
{
    int label;
    float prob;
    cv::Rect_<float> rect;
};

class BaseDetector
{
public:
    BaseDetector() = default;
    virtual ~BaseDetector() = default;

    // disable copy and move since some frameworks do not allow
    BaseDetector(const BaseDetector &) = delete;
    BaseDetector & operator=(const BaseDetector &) = delete;
    BaseDetector(BaseDetector &&) = delete;
    BaseDetector & operator=(BaseDetector &&) = delete;

    virtual std::vector<Object> Detect(const cv::Mat &bgr) = 0;
    virtual bool Initialize(const int threads, const std::string &model_path,
        const float prob_thres, const float nms_thres,
        const int target_size, const int max_stride) = 0;
    virtual bool DrawObjects(cv::Mat &image, const std::vector<Object> &objects,
        const std::vector<std::string> &labels, bool isSilent=true);

protected:
    float conf_thres_;
    float nms_thres_;
    int target_size_;
    int max_stride_;
    bool isInited_ = false;
    std::array<float, 6> anchors8_ = {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f};
    std::array<float, 6> anchors16_ = {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f};
    std::array<float, 6> anchors32_ = {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f};

    template <typename T>
    T Clamp(T x, T min_x, T max_x)
    {
        return x > min_x ? (x < max_x ? x : max_x) : min_x;
    }
};

}   // namespace Infer

#endif  // BASE_DETECTOR_HPP_