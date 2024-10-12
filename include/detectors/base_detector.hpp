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

    
    /**
     * @brief detect objects in an image
     * @param bgr   BGR image to be detected
     * @return vector of detected objects
     */
    virtual std::vector<Object> Detect(const cv::Mat &bgr) = 0;

    /**
     * @brief initialize the inference framework
     * @param threads                   number of inference threads
     * @param model_path                model file path without file extension
     * @param conf_thres, nms_thres     confidence and NMS thresholds
     * @param target_size               model input dimensions
     * @param max_stride                maximum stride, typically 32
     * @param num_class                 number of classes
     * @return whether initialization was successful
     */
    virtual bool Initialize(const int threads, const std::string &model_path,
        const float conf_thres, const float nms_thres,
        const int target_size, const int max_stride, const int num_class) = 0;
    
    /**
     * @brief draw detected objects on an image
     * @param image     image to draw
     * @param objects   detected objects
     * @param labels    class names
     * @param isSilent  whether to suppress printing object information
     * @return whether drawing was successful
     */
    virtual bool DrawObjects(cv::Mat &image, const std::vector<Object> &objects,
        const std::vector<std::string> &labels, bool isSilent=true);

protected:
    float conf_thres_;
    float nms_thres_;
    int target_size_;
    int max_stride_;
    int num_class_;
    bool isInited_ = false;

    std::array<int, 3> strides_ = {8, 16, 32};
    std::array<std::array<float, 6>, 3> anchors_ = {{
        {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f},
        {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f},
        {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}
    }};

    template <typename T>
    T Clamp(T x, T min_x, T max_x)
    {
        return x > min_x ? (x < max_x ? x : max_x) : min_x;
    }

    /**
     * @brief get resize and padding sizes required for creating letterbox
     * @param img_rows, img_cols        input image size
     * @param isDynamic                 whether the model supports dynamic input size
     * @param resize_rows, resize_cols  letterbox resize size
     * @param pad_rows, pad_cols        letterbox padding size
     */
    virtual void GetLetterboxDimensions(const int img_rows, const int img_cols, const bool isDynamic,
        int &resize_rows, int &resize_cols, int &pad_rows, int &pad_cols, float &scale);

    /**
     * @brief generate proposals from feature blob
     * @param feat_blob     feature blob
     * @param nhwc_shape    blob shape in NHWC layout
     * @param stride        downsampling stride
     * @param anchors       anchors for the current stride
     * @param proposals     generated proposals from the blob
     */
    virtual void GenerateProposals(const float *feat_blob, const std::array<int, 4> nhwc_shape, int stride,
        const std::array<float, 6> &anchors, std::vector<Object> &proposals);

    /**
     * @brief perform non-maximum suppression
     * @param proposals         raw proposals
     * @param objects           object detection results
     * @param orig_h, orig_w    original image size
     * @param dh, dw            padding size applied to the height and width
     * @param ratio_h, ratio_w  scaling ratios applied to height and width
     */
    virtual void NMS(std::vector<Object> &proposals, std::vector<Object> &objects,
        const int orig_h, const int orig_w,
        const float dh, const float dw,
        const float ratio_h, const float ratio_w);
};

}   // namespace Infer

#endif  // BASE_DETECTOR_HPP_