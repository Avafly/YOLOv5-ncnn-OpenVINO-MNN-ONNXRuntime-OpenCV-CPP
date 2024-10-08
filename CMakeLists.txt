cmake_minimum_required(VERSION 3.10)
project(YOLOv5-Multi-Frameworks-CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set include directories
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Gather all source files
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Find and link OpenCV
find_package(OpenCV REQUIRED)

# Set paths for other libraries
set(HOMEBREW_PATH "/opt/homebrew/opt")
set(CUSTOM_PATH "/Users/lincx/Documents/github")

# OpenMP
set(OpenMP_LIB_DIR "${HOMEBREW_PATH}/libomp/lib")
set(OpenMP_INCLUDE_DIR "${HOMEBREW_PATH}/libomp/include")
set(OpenMP_LIB "${OpenMP_LIB_DIR}/libomp.a")

# NCNN
set(NCNN_LIB_DIR "${HOMEBREW_PATH}/ncnn/lib")
set(NCNN_INCLUDE_DIR "${HOMEBREW_PATH}/ncnn/include/ncnn")
set(NCNN_LIB "${NCNN_LIB_DIR}/libncnn.dylib")

# MNN
set(MNN_LIB_DIR "${CUSTOM_PATH}/MNN/build/install/lib")
set(MNN_INCLUDE_DIR "${CUSTOM_PATH}/MNN/build/install/include")
set(MNN_LIB "${MNN_LIB_DIR}/libMNN.a")

# ONNX Runtime
set(ORT_LIB_DIR "${HOMEBREW_PATH}/onnxruntime/lib")
set(ORT_INCLUDE_DIR "${HOMEBREW_PATH}/onnxruntime/include/onnxruntime")
set(ORT_LIB "${ORT_LIB_DIR}/libonnxruntime.dylib")

# Find OpenVINO
find_package(OpenVINO REQUIRED)

# Function to create an executable and set its properties
function(create_executable exec_name)
    add_executable(${exec_name} ${ARGN})
    target_include_directories(${exec_name} PRIVATE 
        ${OpenCV_INCLUDE_DIRS}
        ${OpenMP_INCLUDE_DIR}
        ${NCNN_INCLUDE_DIR}
        ${MNN_INCLUDE_DIR}
        ${ORT_INCLUDE_DIR}
        ${OpenVINO_INCLUDE_DIRS}
    )
    target_link_libraries(${exec_name} 
        ${OpenCV_LIBS}
        ${OpenMP_LIB}
        ${NCNN_LIB}
        ${MNN_LIB}
        ${ORT_LIB}
        openvino::runtime
    )
    # MNN specific libraries for Mac
    if(APPLE)
        target_link_libraries(${exec_name}
            "-framework CoreFoundation"
            "-framework Foundation"
            "-framework Metal"
            "-framework OpenGL"
        )
    endif()
endfunction()

# Get all source files except main file
file(GLOB_RECURSE COMMON_SOURCES 
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)
list(FILTER COMMON_SOURCES EXCLUDE REGEX ".*detect_(camera|image)\\.cpp$")

# Create executable files
create_executable(detect_camera src/detect_camera.cpp ${COMMON_SOURCES})
create_executable(detect_image src/detect_image.cpp ${COMMON_SOURCES})

# Optional: Enable optimizations
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")