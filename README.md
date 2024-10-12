# YOLOv5-Multi-Frameworks-CPP

This project implements YOLOv5 using multiple inference frameworks, including [ncnn](https://github.com/Tencent/ncnn), [OpenVINO](https://github.com/openvinotoolkit/openvino), [MNN](https://github.com/alibaba/MNN), [ONNXRuntime](https://github.com/microsoft/onnxruntime), and [OpenCV](https://github.com/opencv/opencv). Compiling this project will get two executable files: `detect_camera.cpp` for detecting camera frames, and `detect_image.cpp` for image detection. You can configure parameters such as camera settings, image paths, inference framework, model, and threads in `Config.json`.

The code separates the inference into two parts: initialization and detection. You can view this project as an example to understand how to use an inference framework step by step. For me, I use it to evaluate the feasibility, inference speed, and resource usage of various frameworks on the devices.

## Demo

Detect image

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Avafly/ImageHostingService@master/uPic/SCR-20241007-ruzqq.jpg" width = "450">
</p>

Detect camera

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Avafly/ImageHostingService@master/uPic/SCR-20241007-ruzq.png" width="500">
</p>

## Dependencies and Installations

OpenCV: 4.10.0

ncnn: 20240820

- How to install: https://github.com/Tencent/ncnn/wiki/how-to-build

OpenVINO: 2024.4.0

- How to install: https://docs.openvino.ai/2023.3/openvino_docs_install_guides_installing_openvino_from_archive_linux.html

MNN: 2.9.0

- How to install: https://www.yuque.com/mnn/en/build_linux

ONNXRuntime: 1.19.2

* How to install: https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build . --parallel
./detect_[camera|image]
```

## Todo

- [x] Add ONNXRuntime inference
- [ ] Add OpenCV inference

## References

https://github.com/nlohmann/json

https://github.com/opencv/opencv

https://github.com/ultralytics/yolov5

https://github.com/Tencent/ncnn/blob/master/examples/yolov5.cpp

https://github.com/dacquaviva/yolov5-openvino-cpp-python

https://github.com/wangzhaode/mnn-yolo
