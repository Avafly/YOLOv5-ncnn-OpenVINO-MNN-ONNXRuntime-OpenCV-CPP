cmake_minimum_required(VERSION 3.20)
project(YOLOv5-Multi-Frameworks-CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# compile options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# libs path
set(LIB_ROOT "$ENV{HOME}/Documents/libs")

if(APPLE)
    set(PLATFORM_OMP_INCLUDE_DIR "$ENV{HOMEBREW_PREFIX}/opt/libomp/include")
    set(PLATFORM_OMP_LIB "$ENV{HOMEBREW_PREFIX}/opt/libomp/lib/libomp.a")
else()
    # linux
    find_package(OpenMP REQUIRED)
    set(PLATFORM_OMP_INCLUDE_DIR "")
    set(PLATFORM_OMP_LIB OpenMP::OpenMP_CXX)

    set(OpenCV_DIR "${LIB_ROOT}/opencv/lib/cmake/opencv4")
    set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")
endif()

# opencv
find_package(OpenCV REQUIRED)
find_package(OpenEXR QUIET)
find_package(Iconv QUIET)
# ncnn
set(ncnn_DIR "${LIB_ROOT}/ncnn/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
# mnn
set(MNN_LIB "${LIB_ROOT}/mnn/lib/libMNN.a")
set(MNN_INCLUDE_DIRS "${LIB_ROOT}/mnn/include")
# onnxruntime
set(onnxruntime_DIR "${LIB_ROOT}/onnxruntime/lib/cmake/onnxruntime")
find_package(onnxruntime REQUIRED)
# openvino
find_package(OpenVINO REQUIRED)

# detectors
set(DETECTOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detectors/base_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detectors/cv_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detectors/mnn_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detectors/ncnn_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detectors/ort_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detectors/ov_detector.cpp
)
add_library(detectors STATIC ${DETECTOR_SOURCES})

target_include_directories(detectors PUBLIC
    ${PLATFORM_OMP_INCLUDE_DIR}
    ${MNN_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(detectors PUBLIC
    ${OpenCV_LIBS}
    ncnn
    ${PLATFORM_OMP_LIB}
    ${MNN_LIB}
    onnxruntime::onnxruntime
    openvino::runtime
)

# detect_image
add_executable(detect_image src/detect_image.cpp)
target_link_libraries(detect_image PRIVATE detectors)
# detect_camera
add_executable(detect_camera src/detect_camera.cpp src/camera_handler.cpp)
target_link_libraries(detect_camera PRIVATE detectors)