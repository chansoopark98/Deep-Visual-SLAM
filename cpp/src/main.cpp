// dvslam_loop: CLI for the C++ frame-loop skeleton (plan v3 item 5.4).
//
//   dvslam_loop info
//   dvslam_loop build-engine <model.onnx> <out.engine> <fp32|fp16|bf16>
//   dvslam_loop run --out-dir DIR (--images-list FILE | --synthetic N) [--latencies FILE | --const-latency MS]
//       [--clock virtual|realtime] [--rate 30] [--tracker identity|cv] [--cv-twist vx,vy,vz,wx,wy,wz]
//       [--depth-engine ENGINE | --depth-sim-ms MS] [--depth-every K] [--depth-dump DIR] [--warmup N]
//       [--preload 1]
//
// Outputs in DIR: increments.csv (-> .inc.npz with dvslam.export.inc_csv), frames.csv, depth.csv, summary.json.
#ifdef DVSLAM_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "dvslam/frame_loop.hpp"

using namespace dvslam;

namespace {

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::vector<std::string> out;
  for (std::string s; std::getline(f, s);)
    if (!s.empty()) out.push_back(s);
  return out;
}

std::vector<double> parse_csv_doubles(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) v.push_back(std::stod(t));
  return v;
}

Mat4 twist_exp(const std::vector<double>& xi) {  // (v, w) as dvslam.geometry.se3.se3_exp
  Mat4 X = Mat4::Zero();
  X(0, 1) = -xi[5];
  X(0, 2) = xi[4];
  X(1, 0) = xi[5];
  X(1, 2) = -xi[3];
  X(2, 0) = -xi[4];
  X(2, 1) = xi[3];
  X(0, 3) = xi[0];
  X(1, 3) = xi[1];
  X(2, 3) = xi[2];
  return X.exp();
}

double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * (v.size() - 1);
  const std::size_t i = static_cast<std::size_t>(pos);
  const double f = pos - i;
  return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
}

int usage() {
  std::fprintf(stderr,
               "usage: dvslam_loop info | build-engine ONNX ENGINE PREC | run --out-dir DIR "
               "(--images-list F | --synthetic N) [options], see src/main.cpp\n");
  return 2;
}

int low_priority() {
  int least = 0;
#ifdef DVSLAM_WITH_CUDA
  int greatest = 0;
  if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess) return 0;
#endif
  return least;  // numerically largest = lowest priority
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string cmd = argv[1];
  if (cmd == "info") {
    std::printf("{\"trt\": %s}\n", trt_available() ? "true" : "false");
    return 0;
  }
  if (cmd == "build-engine") {
    if (argc != 5) return usage();
    std::string err;
    if (!build_trt_engine(argv[2], argv[3], argv[4], &err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    return 0;
  }
  if (cmd != "run") return usage();

  std::map<std::string, std::string> opt;
  for (int i = 2; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
  auto get = [&](const std::string& k, const std::string& d) { return opt.count(k) ? opt[k] : d; };
  if (!opt.count("--out-dir")) return usage();
  const std::string out_dir = opt["--out-dir"];

  try {
    std::unique_ptr<FrameSource> src;
    if (opt.count("--images-list"))
      src = make_image_source(read_lines(opt["--images-list"]), 640, 480, get("--preload", "0") == "1");
    else if (opt.count("--synthetic"))
      src = make_synthetic_source(std::stoi(opt["--synthetic"]), 640, 480);
    else
      return usage();
    const int n = src->size();

    LoopConfig cfg;
    cfg.rate_hz = std::stod(get("--rate", "30"));
    cfg.virtual_clock = get("--clock", "virtual") == "virtual";
    cfg.depth_every = std::stoi(get("--depth-every", "0"));
    cfg.warmup_iters = std::stoi(get("--warmup", "10"));
    cfg.depth_dump_dir = get("--depth-dump", "");
    if (opt.count("--latencies")) {
      for (const auto& s : read_lines(opt["--latencies"])) cfg.proc_ms.push_back(std::stod(s));
    } else if (opt.count("--const-latency")) {
      cfg.proc_ms.assign(n, std::stod(opt["--const-latency"]));
    }

    std::unique_ptr<Tracker> tracker;
    if (get("--tracker", "identity") == "cv")
      tracker = std::make_unique<ConstantVelocityTracker>(
          twist_exp(parse_csv_doubles(get("--cv-twist", "0.01,0,0,0,0.002,0"))));
    else
      tracker = std::make_unique<IdentityTracker>();

    std::unique_ptr<DepthEngine> depth;
    if (opt.count("--depth-engine")) {
      std::string err;
      depth = make_trt_depth_engine(opt["--depth-engine"], low_priority(), &err);
      if (!depth) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 3;
      }
    } else if (opt.count("--depth-sim-ms")) {
      depth = make_sleep_depth_engine(std::stod(opt["--depth-sim-ms"]));
    }

    LoopResult r = run_frame_loop(*src, *tracker, depth.get(), cfg);
    write_increments_csv(out_dir + "/increments.csv", r, cfg.rate_hz);
    write_frames_csv(out_dir + "/frames.csv", r);
    write_depth_csv(out_dir + "/depth.csv", r);

    std::vector<double> lat, dgpu;
    int processed = 0;
    for (const auto& f : r.frames)
      if (f.processed) {
        ++processed;
        lat.push_back(f.finish_ms - f.arrival_ms);
      }
    for (const auto& d : r.depth) dgpu.push_back(d.gpu_ms);
    std::ofstream js(out_dir + "/summary.json");
    js << "{\n \"frames\": " << n << ",\n \"processed\": " << processed << ",\n \"drop_rate\": "
       << (n > 1 ? 1.0 - (processed - 1.0) / (n - 1.0) : 0.0) << ",\n \"clock\": \""
       << (cfg.virtual_clock ? "virtual" : "realtime") << "\",\n \"tracker\": \"" << tracker->name()
       << "\",\n \"depth\": \"" << (depth ? depth->describe() : "off") << "\",\n \"depth_requests_done\": "
       << r.depth.size() << ",\n \"depth_requests_dropped\": " << r.depth_drops
       << ",\n \"depth_gpu_ms_p50\": " << quantile(dgpu, 0.5) << ",\n \"depth_gpu_ms_p95\": " << quantile(dgpu, 0.95)
       << ",\n \"capture_to_increment_ms_p50\": " << quantile(lat, 0.5)
       << ",\n \"capture_to_increment_ms_p95\": " << quantile(lat, 0.95) << ",\n \"frame_drops\": " << r.frame_drops
       << ",\n \"ring_waits\": " << r.ring_waits << ",\n \"pinned\": " << (r.pinned ? "true" : "false")
       << ",\n \"warmup_ms\": " << r.warmup_ms << ",\n \"wall_ms\": " << r.wall_ms << "\n}\n";
    std::printf("processed %d/%d, depth %zu done %zu dropped, wall %.0f ms\n", processed, n, r.depth.size(),
                r.depth_drops, r.wall_ms);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
