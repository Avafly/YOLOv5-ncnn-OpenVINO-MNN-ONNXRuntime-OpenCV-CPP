#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include "json.hpp"

#include "detectors/base_detector.hpp"
#include "detectors/ncnn_detector.hpp"
#include "detectors/ov_detector.hpp"
#include "detectors/mnn_detector.hpp"
#include "detectors/ort_detector.hpp"
#include "detectors/cv_detector.hpp"

int main (int argc, char *argv[])
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
    
    // --- Load input image
    cv::Mat image = cv::imread(config.at("Image").at("ImagePath").get<std::string>());
    if (image.empty())
    {
        std::cout << "Failed to load image\n";
        return 1;
    }

    // show configs
    std::cout << "Using " << support_frameworks[framework] << "\n";
    std::cout << "Threads: " << config.at("Inference").at("Threads").get<int>() << "\n";
    std::cout << "Classes: " << labels.size() << "\n";
    std::cout << "Model name: " << model_path << "\n";
    std::cout << "Image path: " << config.at("Image").at("ImagePath").get<std::string>() << "\n";

    // --- Detect
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
    
    // timer
    int64 start_time = cv::getTickCount();

    // detect
    auto objects = detector->Detect(image);

    // show elapsed time
    std::printf("Elapsed time: %.1fms\n", (cv::getTickCount() - start_time) / cv::getTickFrequency() * 1000.0);

    detector->DrawObjects(image, objects, labels, false);
    cv::imwrite(path.parent_path().string() + "/result.jpg", image);

    return 0;
}