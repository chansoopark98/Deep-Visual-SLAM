// Capture -> tracker -> depth frame loop (plan v3 §4.1, item 5.4).
//
// Threads: capture publishes frames into a 1-slot drop-oldest mailbox; the tracker thread takes the newest frame;
// every `depth_every`-th tracked frame it posts a latest-wins depth request to a second 1-slot mailbox read by the
// low-priority depth thread. The tracker never waits for depth.
//
// Clocks:
//  - virtual: per-frame tracker latencies come from `proc_ms`; capture and tracker threads are gated so that the
//    processed-frame set is exactly that of dvslam.runtime.replay.simulate_mailbox (same arrival rule
//    j = floor(start/dt + 1e-9)). Depth runs on wall time and never affects the mask.
//  - realtime: capture publishes at wall-clock k/rate; the tracker sleeps `proc_ms[k]` (if given) after its own work.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dvslam/depth_engine.hpp"
#include "dvslam/tracker.hpp"

namespace dvslam {

class FrameSource {
 public:
  virtual ~FrameSource() = default;
  virtual int size() const = 0;
  virtual int width() const = 0;
  virtual int height() const = 0;
  // Decodes frame i as BGR uint8 into dst (width*height*3). May leave dst untouched for a synthetic source.
  virtual void load(int i, unsigned char* dst) = 0;
};

// Image files (OpenCV; `preload` decodes all frames before the clock starts, like a camera delivering decoded
// frames), or `n` synthetic black frames.
std::unique_ptr<FrameSource> make_image_source(const std::vector<std::string>& paths, int width, int height,
                                               bool preload = false);
std::unique_ptr<FrameSource> make_synthetic_source(int n, int width, int height);

struct LoopConfig {
  double rate_hz = 30.0;
  bool virtual_clock = true;
  std::vector<double> proc_ms;  // per-frame tracker latency (required for the virtual clock)
  int depth_every = 0;          // post a depth request every k-th tracked frame (0 = depth thread off)
  int warmup_iters = 10;
  std::string depth_dump_dir;   // if set, write <frame>.bin (float32 out_h*out_w) per finished depth request
};

struct FrameRecord {
  int index = 0;
  bool processed = false;
  double arrival_ms = 0, start_ms = -1, finish_ms = -1;  // virtual: k/rate; realtime: actual publish time
  Mat4 measured = Mat4::Identity();  // valid when processed and index > first processed
};

struct DepthRecord {
  int frame = 0;
  double start_ms = 0, end_ms = 0, gpu_ms = 0;
  float median_m = 0;
};

struct LoopResult {
  std::vector<FrameRecord> frames;
  std::vector<Mat4> scored;          // ΔT_t for t = 1..n-1 after the drop policy
  std::vector<std::string> source;   // "tracked" | "motion_model"
  std::vector<DepthRecord> depth;
  std::size_t frame_drops = 0, depth_drops = 0, ring_waits = 0;
  bool pinned = false;
  double warmup_ms = 0, wall_ms = 0;
};

LoopResult run_frame_loop(FrameSource& src, Tracker& tracker, DepthEngine* depth, const LoopConfig& cfg);

// Drop scoring rule of dvslam.runtime.replay.apply_drop_policy, on the measured increments of tracked frames.
void apply_drop_policy(LoopResult& r);

// increments CSV (one row per frame t >= 1): index,timestamp,source,tracking_state,m00..m33 (row-major).
void write_increments_csv(const std::string& path, const LoopResult& r, double rate_hz);
// frames CSV: index,processed,arrival_ms,start_ms,finish_ms ; depth CSV: frame,start_ms,end_ms,gpu_ms,median_m
void write_frames_csv(const std::string& path, const LoopResult& r);
void write_depth_csv(const std::string& path, const LoopResult& r);

}  // namespace dvslam
