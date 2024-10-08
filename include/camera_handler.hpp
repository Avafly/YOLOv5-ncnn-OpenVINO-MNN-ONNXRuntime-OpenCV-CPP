#ifndef CAMERA_HANDLER_HPP_
#define CAMERA_HANDLER_HPP_

#include <opencv2/opencv.hpp>

class CameraHandler
{
public:
    CameraHandler();
    ~CameraHandler();

    // disable copy and move operations since the camera is an exclusive resource
    CameraHandler(CameraHandler &&) = delete;
    CameraHandler & operator = (CameraHandler &&) = delete;
    CameraHandler(const CameraHandler &) = delete;
    CameraHandler & operator = (const CameraHandler &) = delete;

    bool Open(int id, int width, int height, int fps);
    void Close();
    bool IsOpened() const;
    bool GetFrame(cv::Mat &frame);
    int GetActualWidth() const;
    int GetActualHeight() const;

    void SetResolution(int width, int height);
    void SetFPS(int fps);
private:
    cv::VideoCapture camera_;
    int camera_id_;
    int frame_width_;
    int frame_height_;
    int fps_;

    bool InitCamera();
};

#endif  // CAMERA_HANDLER_HPP_