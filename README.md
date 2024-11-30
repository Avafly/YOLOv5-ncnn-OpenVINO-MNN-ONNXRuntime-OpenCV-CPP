# YOLOv5-Multi-Frameworks-CPP

This project implements YOLOv5 using multiple inference frameworks, including [ncnn](https://github.com/Tencent/ncnn), [OpenVINO](https://github.com/openvinotoolkit/openvino), [MNN](https://github.com/alibaba/MNN), [ONNXRuntime](https://github.com/microsoft/onnxruntime), and [OpenCV](https://github.com/opencv/opencv). A key feature of this code is its support for dynamic input shapes (except for [OpenCV](https://github.com/opencv/opencv/issues/19347#issuecomment-1868227401)), which is not actively mentioned in other tutorials but can significantly improve inference speed.

The code separates the inference into two parts: initialization and detection. You can view this project as an example to understand how to use an inference framework step by step. For me, I use it to evaluate the feasibility, inference speed, and resource usage of various frameworks on the devices.

All models can be downloaded from https://github.com/ultralytics/yolov5.

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

OpenVINO: 2023.3.0

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

## Simple Benchmarks on M1 Mac and ARM Linux

I ran each framework on my devices and recorded the elapsed time to detect an image with a size of 1878x1030. With only CPU computation, I ran each test three times and took the median time.

### M1 Macbook Air

| Frameworks  | YOLOv5n | YOLOv5s  |
| :---------: | :-----: | :------: |
|    ncnn     | 14.6 ms | 24.8 ms  |
|  OpenVINO   | 47.1 ms | 125.3 ms |
|     MNN     | 45.6 ms | 137.1 ms |
| ONNXRuntime | 20.6 ms | 45.2 ms  |
|   OpenCV    | 53.7 ms | 117.4 ms |

### Oracle Free ARM Server

| Frameworks  | YOLOv5n  | YOLOv5s  |
| :---------: | :------: | :------: |
|    ncnn     | 54.0 ms  | 130.0 ms |
|  OpenVINO   | 168.0 ms | 388.5 ms |
|     MNN     | 161.8 ms | 392.4 ms |
| ONNXRuntime | 120.0 ms | 325.4 ms |
|   OpenCV    | 273.6 ms | 658.3 ms |

<details>
  <summary>CPU Info</summary>
<pre>
$ lscpu
Architecture:             aarch64
  CPU op-mode(s):         32-bit, 64-bit
  Byte Order:             Little Endian
CPU(s):                   1
  On-line CPU(s) list:    0
Vendor ID:                ARM
  Model name:             Neoverse-N1
    Model:                1
    Thread(s) per core:   1
    Core(s) per cluster:  1
    Socket(s):            -
    Cluster(s):           1
    Stepping:             r3p1
    BogoMIPS:             50.00
    Flags:                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asi
                          mddp
NUMA:
  NUMA node(s):           1
  NUMA node0 CPU(s):      0
Vulnerabilities:
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Not affected
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; __user pointer sanitization
  Spectre v2:             Mitigation; CSV2, BHB
  Srbds:                  Not affected
  Tsx async abort:        Not affected
</pre>
</details>

## Todo

- [x] Add ONNXRuntime inference
- [x] Add OpenCV dnn inference

## References

https://github.com/ultralytics/yolov5

https://github.com/Tencent/ncnn/blob/master/examples/yolov5.cpp

https://github.com/dacquaviva/yolov5-openvino-cpp-python

https://github.com/wangzhaode/mnn-yolo
