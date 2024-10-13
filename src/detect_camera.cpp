#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include "json.hpp"

#include "camera_handler.hpp"

#include "detectors/base_detector.hpp"
#include "detectors/ncnn_detector.hpp"
#include "detectors/ov_detector.hpp"
#include "detectors/mnn_detector.hpp"
#include "detectors/ort_detector.hpp"
#include "detectors/cv_detector.hpp"

void ShowFPS(cv::Mat &frame, int &frame_count, int &fps, std::chrono::steady_clock::time_point &start)
{
    ++frame_count;
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    if (elapsed.count() >= 1.0)
    {
        fps = frame_count / elapsed.count();
        frame_count = 0;
        start = end;
    }

    std::string fps_text = "FPS: " + std::to_string(fps);
    cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
}

int main(int argc, char *argv[])
{
    // --- Load configs
    std::string config_path = "../Config.json";
    nlohmann::json config;
    if (argc == 2)
        config_path = std::string(argv[1]);
    try
    {
        std::ifstream config_file(config_path);
        config = nlohmann::json::parse(config_file, nullptr, true, true);
    }
    catch(const nlohmann::json::exception &e)
    {
        // std::cout << e.what() << '\n';
        std::cout << "Failed to read JSON config at " << config_path << "\n";
        std::cout << "Use `" << argv[0] << " [path_to_config]` to specify a config file.\n";
        return 1;
    }
    // get model path
    std::vector<std::string> support_frameworks = config.at("Inference").at("Supports").get<std::vector<std::string>>();
    int framework = config.at("Inference").at("Framework").get<int>();
    std::filesystem::path path(config_path);
    std::string model_path = path.parent_path().string() + "/models/" +
        support_frameworks[framework] + "/" +
        config.at("YOLOv5").at("ModelName").get<std::string>();
    // get labels
    auto labels = config.at("YOLOv5").at("Labels").get<std::vector<std::string>>();

    // show configs
    std::cout << "Camera ID: " << config.at("Camera").at("CameraID").get<int>() << "\n";
    std::cout << "Using " << support_frameworks[framework] << "\n";
    std::cout << "Threads: " << config.at("Inference").at("Threads").get<int>() << "\n";
    std::cout << "Classes: " << labels.size() << "\n";
    std::cout << "Model name: " << model_path << "\n";

    // load framework
    std::unique_ptr<Infer::BaseDetector> detector = nullptr;
    switch (framework)
    {
        case 0:
            detector = std::make_unique<Infer::NCNNDetector>();
            break;
        case 1:
            detector = std::make_unique<Infer::OVDetector>();
            break;
        case 2:
            detector = std::make_unique<Infer::MNNDetector>();
            break;
        case 3:
            detector = std::make_unique<Infer::ORTDetector>();
            break;
        case 4:
            detector = std::make_unique<Infer::CVDetector>();
            break;
        default:
            std::cout << "Unknown model: " << framework << "\n";
            return 0;
    }
    if (detector->Initialize(
        config.at("Inference").at("Threads").get<int>(),
        model_path,
        config.at("YOLOv5").at("ConfThreshold").get<float>(),
        config.at("YOLOv5").at("NMSThreshold").get<float>(),
        config.at("YOLOv5").at("TargetSize").get<int>(),
        config.at("YOLOv5").at("MaxStride").get<int>(),
        static_cast<int>(labels.size())
    ) == false)
    {
        std::cout << "Failed to initialize framework\n";
        return 1;
    }

    // --- Open camera
    CameraHandler ch;
    if (ch.Open(
        config.at("Camera").at("CameraID").get<int>(),
        config.at("Camera").at("FrameWidth").get<int>(),
        config.at("Camera").at("FrameHeight").get<int>(),
        config.at("Camera").at("FPS").get<int>()
    ) == false)
    {
        std::cout << "Failed to open camera\n";
        return 1;
    }

    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    std::cout << "* Press [esc] to quit *\n";

    auto start = std::chrono::steady_clock::now();
    int fps = 0, frame_count = 0;

    while (true)
    {
        if (!ch.GetFrame(frame))
        {
            std::cout << "Failed to get frame\n";
            break;
        }

        // detect
        cv::flip(frame, frame, 1);
        auto objects = detector->Detect(frame);
        detector->DrawObjects(frame, objects, labels);

        ShowFPS(frame, frame_count, fps, start);

        cv::imshow("Camera", frame);

        // press esc to quit
        if (cv::waitKey(1) == 27)
            break;
    }

    // releasse
    cv::destroyAllWindows();

    std::cout << "FPS: " << fps << "\n";

    return 0;
}