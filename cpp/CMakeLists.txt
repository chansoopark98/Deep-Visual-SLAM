cmake_minimum_required(VERSION 3.18)
project(dvslam_runtime LANGUAGES CXX)

# C++ frame-loop skeleton (plan v3 item 5.4) and TensorRT depth runner (item 5.2).
# Optional dependencies, each detected independently:
#   CUDA toolkit  -> pinned frame ring, stream priorities      (DVSLAM_WITH_CUDA)
#   OpenCV        -> JPEG decode / resize                      (DVSLAM_WITH_OPENCV)
#   TensorRT      -> depth engine runner + C++ engine builder  (DVSLAM_WITH_TRT; needs CUDA and OpenCV)
# TensorRT must match the CUDA major version of the toolkit; on this box TRT 10.13.2 ships inside CUDA 13.2:
#   cmake -S cpp -B cpp/build -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
#         -DTENSORRT_ROOT=/usr/local/cuda-13.2/targets/x86_64-linux

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit QUIET)
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)

set(TENSORRT_ROOT "" CACHE PATH "Directory with include/NvInfer.h and lib/libnvinfer.so")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT}/include)
find_library(TENSORRT_NVINFER nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(TENSORRT_ONNXPARSER nvonnxparser HINTS ${TENSORRT_ROOT}/lib)

add_executable(dvslam_loop src/main.cpp src/frame_loop.cpp src/depth_engine.cpp src/trt_depth.cpp)
target_include_directories(dvslam_loop PRIVATE include)
target_link_libraries(dvslam_loop PRIVATE Threads::Threads Eigen3::Eigen)
target_compile_options(dvslam_loop PRIVATE -Wall -Wextra)

if(CUDAToolkit_FOUND)
  target_compile_definitions(dvslam_loop PRIVATE DVSLAM_WITH_CUDA)
  target_link_libraries(dvslam_loop PRIVATE CUDA::cudart)
  message(STATUS "dvslam: CUDA ${CUDAToolkit_VERSION} (pinned ring, stream priorities)")
endif()
if(OpenCV_FOUND)
  target_compile_definitions(dvslam_loop PRIVATE DVSLAM_WITH_OPENCV)
  target_include_directories(dvslam_loop PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(dvslam_loop PRIVATE ${OpenCV_LIBS})
  message(STATUS "dvslam: OpenCV ${OpenCV_VERSION}")
endif()
if(CUDAToolkit_FOUND AND OpenCV_FOUND AND TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER AND TENSORRT_ONNXPARSER)
  target_compile_definitions(dvslam_loop PRIVATE DVSLAM_WITH_TRT)
  target_include_directories(dvslam_loop PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(dvslam_loop PRIVATE ${TENSORRT_NVINFER} ${TENSORRT_ONNXPARSER})
  message(STATUS "dvslam: TensorRT ${TENSORRT_NVINFER}")
else()
  message(STATUS "dvslam: TensorRT runner NOT built (needs CUDA, OpenCV, NvInfer.h, libnvinfer, libnvonnxparser)")
endif()
