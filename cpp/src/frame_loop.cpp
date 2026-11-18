#include "dvslam/frame_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "dvslam/mailbox.hpp"
#include "dvslam/pinned_ring.hpp"

#ifdef DVSLAM_WITH_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace dvslam {

namespace {

using Clock = std::chrono::steady_clock;

struct FrameMsg {
  int index;
  double t_ms;
  PinnedRing::Ref slot;
};

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

class ImageSource final : public FrameSource {
 public:
  ImageSource(std::vector<std::string> paths, int w, int h, bool preload) : paths_(std::move(paths)), w_(w), h_(h) {
    if (preload) {
      cache_.resize(paths_.size());
      for (std::size_t i = 0; i < paths_.size(); ++i) {
        cache_[i].resize(static_cast<std::size_t>(w_) * h_ * 3);
        decode(static_cast<int>(i), cache_[i].data());
      }
    }
  }
  int size() const override { return static_cast<int>(paths_.size()); }
  int width() const override { return w_; }
  int height() const override { return h_; }
  void load(int i, unsigned char* dst) override {
    if (!cache_.empty()) {
      std::copy(cache_[i].begin(), cache_[i].end(), dst);
      return;
    }
    decode(i, dst);
  }

 private:
  void decode(int i, unsigned char* dst) {
#ifdef DVSLAM_WITH_OPENCV
    cv::Mat img = cv::imread(paths_[i], cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("cannot read " + paths_[i]);
    if (img.cols != w_ || img.rows != h_) cv::resize(img, img, cv::Size(w_, h_), 0, 0, cv::INTER_LINEAR);
    cv::Mat wrap(h_, w_, CV_8UC3, dst);
    img.copyTo(wrap);
#else
    (void)i;
    (void)dst;
#endif
  }

  std::vector<std::string> paths_;
  int w_, h_;
  std::vector<std::vector<unsigned char>> cache_;
};

class SyntheticSource final : public FrameSource {
 public:
  SyntheticSource(int n, int w, int h) : n_(n), w_(w), h_(h) {}
  int size() const override { return n_; }
  int width() const override { return w_; }
  int height() const override { return h_; }
  void load(int, unsigned char*) override {}

 private:
  int n_, w_, h_;
};

}  // namespace

std::unique_ptr<FrameSource> make_image_source(const std::vector<std::string>& paths, int width, int height,
                                               bool preload) {
  return std::make_unique<ImageSource>(paths, width, height, preload);
}

std::unique_ptr<FrameSource> make_synthetic_source(int n, int width, int height) {
  return std::make_unique<SyntheticSource>(n, width, height);
}

LoopResult run_frame_loop(FrameSource& src, Tracker& tracker, DepthEngine* depth, const LoopConfig& cfg) {
  const int n = src.size();
  const double dt = 1000.0 / cfg.rate_hz;
  if (cfg.virtual_clock && static_cast<int>(cfg.proc_ms.size()) < n)
    throw std::invalid_argument("virtual clock needs one tracker latency per frame");
  const bool use_depth = depth != nullptr && cfg.depth_every > 0;

  LoopResult res;
  res.frames.resize(n);
  for (int k = 0; k < n; ++k) {
    res.frames[k].index = k;
    res.frames[k].arrival_ms = k * dt;
  }
  PinnedRing ring(8, static_cast<std::size_t>(src.width()) * src.height() * 3);
  res.pinned = ring.pinned();
  Mailbox<FrameMsg> frames;
  Mailbox<FrameMsg> depth_req;

  // ---- warm-up (before the clock starts): tracker, then depth engine on a blank frame
  auto tw = Clock::now();
  for (int i = 0; i < cfg.warmup_iters; ++i) tracker.warmup();
  if (use_depth) depth->warmup(cfg.warmup_iters, src.width(), src.height());
  res.warmup_ms = ms_since(tw);

  // ---- virtual-clock gate shared by capture and tracker
  std::mutex sm;
  std::condition_variable scv;
  int published = 0;        // frames 0..published-1 have been put into the mailbox
  double free_at = 0.0;     // virtual time at which the tracker finishes its current frame
  bool ready = false;       // tracker found the mailbox empty after all frames arrived by free_at
  auto jmax = [&](double t) { return static_cast<int>(std::floor(t / dt + 1e-9)); };

  const auto t0 = Clock::now();

  std::thread capture([&] {
    for (int k = 0; k < n; ++k) {
      PinnedRing::Ref slot = ring.acquire();
      src.load(k, slot.data());
      if (cfg.virtual_clock) {
        std::unique_lock<std::mutex> lk(sm);
        scv.wait(lk, [&] { return ready ? frames.empty() : k <= jmax(free_at); });
        frames.put(FrameMsg{k, k * dt, std::move(slot)});
        published = k + 1;
        scv.notify_all();
      } else {
        std::this_thread::sleep_until(t0 + std::chrono::duration_cast<Clock::duration>(
                                               std::chrono::duration<double, std::milli>(k * dt)));
        const double now = ms_since(t0);
        res.frames[k].arrival_ms = now;  // actual publish time (late if decode/IO cannot keep up)
        frames.put(FrameMsg{k, now, std::move(slot)});
      }
    }
    frames.close();
  });

  std::thread tracker_thread([&] {
    Camera cam;
    cam.width = src.width();
    cam.height = src.height();
    tracker.reset(cam);
    int last = -1, n_tracked = 0;
    double F = 0.0;
    for (;;) {
      std::optional<FrameMsg> msg;
      double start;
      if (cfg.virtual_clock) {
        std::unique_lock<std::mutex> lk(sm);
        const int need = std::min(n, jmax(F) + 1);
        scv.wait(lk, [&] { return published >= need; });
        msg = frames.try_take();
        if (!msg) {
          if (published == n) break;
          ready = true;
          scv.notify_all();
          scv.wait(lk, [&] { return !frames.empty(); });
          msg = frames.try_take();
          ready = false;
        }
        start = std::max(F, msg->t_ms);
        F = start + cfg.proc_ms[msg->index];
        free_at = F;
        scv.notify_all();
      } else {
        msg = frames.take();
        if (!msg) break;
        start = ms_since(t0);
      }
      FrameView view{msg->index, msg->t_ms, msg->slot.data(), src.width(), src.height()};
      FrameRecord& rec = res.frames[msg->index];
      rec.measured = tracker.track(view, last);
      rec.processed = true;
      rec.start_ms = start;
      if (cfg.virtual_clock) {
        rec.finish_ms = F;
      } else {
        if (static_cast<int>(cfg.proc_ms.size()) > msg->index)
          std::this_thread::sleep_until(t0 + std::chrono::duration_cast<Clock::duration>(
                                                 std::chrono::duration<double, std::milli>(
                                                     start + cfg.proc_ms[msg->index])));
        rec.finish_ms = ms_since(t0);
      }
      last = msg->index;
      if (use_depth && n_tracked % cfg.depth_every == 0) depth_req.put(std::move(*msg));  // latest wins
      ++n_tracked;
    }
    depth_req.close();
  });

  std::thread depth_thread;
  if (use_depth) {
    depth_thread = std::thread([&] {
      std::vector<float> out(static_cast<std::size_t>(depth->out_h()) * depth->out_w());
      while (auto req = depth_req.take()) {
        DepthRecord d;
        d.frame = req->index;
        d.start_ms = ms_since(t0);
        FrameView view{req->index, req->t_ms, req->slot.data(), src.width(), src.height()};
        d.gpu_ms = depth->infer(view, out.data());
        d.end_ms = ms_since(t0);
        std::vector<float> tmp(out);
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        d.median_m = tmp[tmp.size() / 2];
        if (!cfg.depth_dump_dir.empty()) {
          std::ofstream f(cfg.depth_dump_dir + "/" + std::to_string(req->index) + ".bin", std::ios::binary);
          f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * 4));
        }
        res.depth.push_back(d);
      }
    });
  } else {
    depth_req.close();
  }

  capture.join();
  tracker_thread.join();
  if (depth_thread.joinable()) depth_thread.join();
  res.wall_ms = ms_since(t0);
  res.frame_drops = frames.dropped();
  res.depth_drops = depth_req.dropped();
  res.ring_waits = ring.waits();
  apply_drop_policy(res);
  return res;
}

void apply_drop_policy(LoopResult& r) {
  const int n = static_cast<int>(r.frames.size());
  r.scored.assign(n > 0 ? n - 1 : 0, Mat4::Identity());
  r.source.assign(r.scored.size(), "motion_model");
  Mat4 prev = Mat4::Identity();
  int gap = 0;
  for (int t = 1; t < n; ++t) {
    Mat4 dT;
    if (r.frames[t].processed) {
      const Mat4& D = r.frames[t].measured;  // spans gap+1 frames
      dT = gap ? se3_fraction(D, 1.0 / (gap + 1)) : D;
      r.source[t - 1] = "tracked";
      gap = 0;
    } else {
      dT = prev;
      ++gap;
    }
    r.scored[t - 1] = dT;
    prev = dT;
  }
}

void write_increments_csv(const std::string& path, const LoopResult& r, double rate_hz) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "index,timestamp,source,tracking_state";
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) f << ",m" << i << j;
  f << "\n" << std::setprecision(17);
  for (std::size_t k = 0; k < r.scored.size(); ++k) {
    const int t = static_cast<int>(k) + 1;
    f << t << "," << t / rate_hz << "," << r.source[k] << ",OK";
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) f << "," << r.scored[k](i, j);
    f << "\n";
  }
}

void write_frames_csv(const std::string& path, const LoopResult& r) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "index,processed,arrival_ms,start_ms,finish_ms\n" << std::setprecision(10);
  for (const auto& fr : r.frames)
    f << fr.index << "," << int(fr.processed) << "," << fr.arrival_ms << "," << fr.start_ms << "," << fr.finish_ms
      << "\n";
}

void write_depth_csv(const std::string& path, const LoopResult& r) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "frame,start_ms,end_ms,gpu_ms,median_m\n" << std::setprecision(8);
  for (const auto& d : r.depth)
    f << d.frame << "," << d.start_ms << "," << d.end_ms << "," << d.gpu_ms << "," << d.median_m << "\n";
}

}  // namespace dvslam
