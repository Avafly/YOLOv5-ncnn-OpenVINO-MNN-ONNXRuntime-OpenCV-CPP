#include "camera_handler.hpp"

CameraHandler::CameraHandler()
{

}

CameraHandler::~CameraHandler()
{
    Close();
}

bool CameraHandler::Open(int id, int width, int height, int fps)
{
    if (camera_.isOpened())
        return true;

    camera_id_ = id;
    frame_width_ = width;
    frame_height_ = height;
    fps_ = fps;
    
    return InitCamera();
}

void CameraHandler::Close()
{
    if (camera_.isOpened())
        camera_.release();
}

bool CameraHandler::IsOpened() const
{
    return camera_.isOpened();
}

bool CameraHandler::GetFrame(cv::Mat &frame)
{
    if (!IsOpened())
        return false;
    return camera_.read(frame);
}

int CameraHandler::GetActualWidth() const
{
    return frame_width_;
}

int CameraHandler::GetActualHeight() const
{
    return frame_height_;
}

void CameraHandler::SetResolution(int width, int height)
{
    frame_width_ = width;
    frame_height_ = height;
    if (IsOpened())
    {
        camera_.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
        camera_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);
    }
}

void CameraHandler::SetFPS(int fps)
{
    fps_ = fps;
    if (IsOpened())
        camera_.set(cv::CAP_PROP_FPS, fps_);
}

bool CameraHandler::InitCamera()
{
    camera_.open(camera_id_);
    if (!camera_.isOpened())
    {
        return false;
    }

    camera_.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
    camera_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);
    camera_.set(cv::CAP_PROP_FPS, fps_);

    // Verify settings
    int actual_width = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_HEIGHT));
    int actual_fps = static_cast<int>(camera_.get(cv::CAP_PROP_FPS));

    if (actual_width != frame_width_ || actual_height != frame_height_ || actual_fps != fps_)
    {
        frame_width_ = actual_width;
        frame_height_ = actual_height;
        fps_ = actual_fps;

        std::cout << "Warning: Camera settings differ from requested parameters\n";
        std::cout << " - Actual resolution: " << actual_width << "x" << actual_height << "\n";
        std::cout << " - Actual FPS: " << actual_fps << "\n";
    }

    return true;
}

