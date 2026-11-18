// Tracker interface for the C++ frame loop (plan v3 §4.1, §4.8) and dummy trackers for loop tests.
#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

#include <cstdint>
#include <string>

namespace dvslam {

using Mat4 = Eigen::Matrix4d;

struct Camera {
  double fx = 525.0, fy = 525.0, cx = 319.5, cy = 239.5;
  int width = 640, height = 480;
};

struct FrameView {
  int index = 0;          // position in the sequence (0-based)
  double t_ms = 0.0;      // capture time
  const std::uint8_t* bgr = nullptr;  // width*height*3, pinned slot (may be null for a synthetic source)
  int width = 0, height = 0;
};

// T^f for a rigid transform via the principal matrix logarithm (the same principal log as dvslam.geometry.se3).
inline Mat4 se3_fraction(const Mat4& T, double f) {
  const Mat4 L = T.log();
  return (L * f).exp();
}

class Tracker {
 public:
  virtual ~Tracker() = default;
  virtual void reset(const Camera& cam) = 0;
  virtual void warmup() {}
  // Measured motion of `cur` relative to the previously tracked frame `prev_index` (camera cur in camera prev):
  // inv(T_wc_prev) * T_wc_cur. prev_index = -1 for the first frame (bootstrap, return identity).
  virtual Mat4 track(const FrameView& cur, int prev_index) = 0;
  virtual std::string name() const = 0;
};

class IdentityTracker final : public Tracker {
 public:
  void reset(const Camera&) override {}
  Mat4 track(const FrameView&, int) override { return Mat4::Identity(); }
  std::string name() const override { return "identity"; }
};

// Constant per-frame motion V: the measured increment over a gap of g frames is V^g.
class ConstantVelocityTracker final : public Tracker {
 public:
  explicit ConstantVelocityTracker(const Mat4& V) : V_(V) {}
  void reset(const Camera&) override {}
  Mat4 track(const FrameView& cur, int prev_index) override {
    Mat4 D = Mat4::Identity();
    if (prev_index < 0) return D;
    for (int k = prev_index; k < cur.index; ++k) D = D * V_;
    return D;
  }
  std::string name() const override { return "constant_velocity"; }

 private:
  Mat4 V_;
};

}  // namespace dvslam
