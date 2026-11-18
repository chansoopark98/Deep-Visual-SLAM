// Depth engine interface for the low-priority depth thread (plan v3 §4.1, item 5.2).
#pragma once

#include <memory>
#include <string>

#include "dvslam/tracker.hpp"

namespace dvslam {

class DepthEngine {
 public:
  virtual ~DepthEngine() = default;
  virtual int out_h() const = 0;
  virtual int out_w() const = 0;
  virtual std::string describe() const = 0;
  // Writes out_h*out_w depth values (metres) into `out`. Returns the GPU (or simulated) time in ms.
  virtual double infer(const FrameView& frame, float* out) = 0;
  virtual void warmup(int iters, int width, int height);
};

// Simulated engine for loop tests without a GPU: sleeps `ms` and writes a constant depth.
std::unique_ptr<DepthEngine> make_sleep_depth_engine(double ms, int out_h = 120, int out_w = 160);

// TensorRT engine (input "image" 1x3xHxW ImageNet-normalised RGB float32, output "depth" 1xHoxWo metres), run on
// a CUDA stream created with `stream_priority` (higher number = lower priority). Returns nullptr and sets *err when
// the binary was built without TensorRT or the engine cannot be loaded.
std::unique_ptr<DepthEngine> make_trt_depth_engine(const std::string& engine_path, int stream_priority,
                                                   std::string* err);

// Builds an engine from ONNX with the TensorRT C++ API (precision: fp32 | fp16 | bf16). Engines are tied to the
// TensorRT version and GPU, so the edge device builds its own (plan 6.1).
bool build_trt_engine(const std::string& onnx_path, const std::string& engine_path, const std::string& precision,
                      std::string* err);

bool trt_available();

}  // namespace dvslam
