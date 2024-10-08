#include "detectors/base_detector.hpp"
#include <cmath>

namespace Infer
{

bool BaseDetector::DrawObjects(cv::Mat &image, const std::vector<Object> &objects,
    const std::vector<std::string> &labels, bool isSilent)
{
    for (auto obj : objects)
    {
        if (obj.label >= labels.size())
            return false;

        if (isSilent != true)
            std::printf("%s = %.2f%% at (%.1f, %.1f)  %.1f x %.1f\n", labels[obj.label].c_str(), obj.prob * 100.0f,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", labels[obj.label].c_str(), obj.prob * 100.0f);

        auto scalar = cv::Scalar(114, 114, 114);
        cv::rectangle(image, obj.rect, scalar, 2);

        int baseLine = 5;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.75, 1, &baseLine);

        int x = obj.rect.x - 1;
        int y = obj.rect.y - label_size.height - baseLine;
        y = std::max(0, y);
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            scalar, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height + baseLine / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
    }

    return true;
}

}   // namespace Infer