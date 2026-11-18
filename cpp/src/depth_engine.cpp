#include "dvslam/depth_engine.hpp"

#include <chrono>
#include <thread>
#include <vector>

namespace dvslam {

void DepthEngine::warmup(int iters, int width, int height) {
  std::vector<unsigned char> blank(static_cast<std::size_t>(width) * height * 3, 0);
  std::vector<float> out(static_cast<std::size_t>(out_h()) * out_w());
  FrameView v{-1, 0.0, blank.data(), width, height};
  for (int i = 0; i < iters; ++i) infer(v, out.data());
}

namespace {

class SleepDepthEngine final : public DepthEngine {
 public:
  SleepDepthEngine(double ms, int h, int w) : ms_(ms), h_(h), w_(w) {}
  int out_h() const override { return h_; }
  int out_w() const override { return w_; }
  std::string describe() const override { return "sleep(" + std::to_string(ms_) + " ms)"; }
  double infer(const FrameView&, float* out) override {
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms_));
    for (int i = 0; i < h_ * w_; ++i) out[i] = 1.0f;
    return ms_;
  }

 private:
  double ms_;
  int h_, w_;
};

}  // namespace

std::unique_ptr<DepthEngine> make_sleep_depth_engine(double ms, int out_h, int out_w) {
  return std::make_unique<SleepDepthEngine>(ms, out_h, out_w);
}

}  // namespace dvslam
