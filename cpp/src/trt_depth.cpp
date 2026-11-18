// TensorRT depth engine runner (PyTorch-free path of item 5.2). Compiled with -DDVSLAM_WITH_TRT when NvInfer.h,
// libnvinfer, libnvonnxparser, CUDA and OpenCV are found; otherwise the functions report that TRT is not built.
#include "dvslam/depth_engine.hpp"

#ifdef DVSLAM_WITH_TRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace dvslam {

namespace {

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
  }
};

TrtLogger& logger() {
  static TrtLogger l;
  return l;
}

bool ck(cudaError_t e, std::string* err, const char* what) {
  if (e == cudaSuccess) return true;
  if (err) *err = std::string(what) + ": " + cudaGetErrorString(e);
  return false;
}

class TrtDepthEngine final : public DepthEngine {
 public:
  bool load(const std::string& path, int priority, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      *err = "cannot open engine " + path;
      return false;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    runtime_.reset(nvinfer1::createInferRuntime(logger()));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) {
      *err = "deserializeCudaEngine failed (engine built by another TensorRT version or GPU?) " + path;
      return false;
    }
    ctx_.reset(engine_->createExecutionContext());
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* name = engine_->getIOTensorName(i);
      auto d = engine_->getTensorShape(name);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        in_name_ = name;
        in_h_ = static_cast<int>(d.d[2]);
        in_w_ = static_cast<int>(d.d[3]);
      } else {
        out_name_ = name;
        out_h_ = static_cast<int>(d.d[d.nbDims - 2]);
        out_w_ = static_cast<int>(d.d[d.nbDims - 1]);
      }
    }
    const std::size_t nin = 3ull * in_h_ * in_w_, nout = 1ull * out_h_ * out_w_;
    if (!ck(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority), err, "stream") ||
        !ck(cudaMalloc(&d_in_, nin * 4), err, "cudaMalloc in") ||
        !ck(cudaMalloc(&d_out_, nout * 4), err, "cudaMalloc out") ||
        !ck(cudaHostAlloc(reinterpret_cast<void**>(&h_in_), nin * 4, cudaHostAllocDefault), err, "host in") ||
        !ck(cudaHostAlloc(reinterpret_cast<void**>(&h_out_), nout * 4, cudaHostAllocDefault), err, "host out") ||
        !ck(cudaEventCreate(&e0_), err, "event") || !ck(cudaEventCreate(&e1_), err, "event"))
      return false;
    ctx_->setTensorAddress(in_name_.c_str(), d_in_);
    ctx_->setTensorAddress(out_name_.c_str(), d_out_);
    priority_ = priority;
    return true;
  }

  ~TrtDepthEngine() override {
    if (stream_) cudaStreamSynchronize(stream_);
    cudaFree(d_in_);
    cudaFree(d_out_);
    cudaFreeHost(h_in_);
    cudaFreeHost(h_out_);
    if (e0_) cudaEventDestroy(e0_);
    if (e1_) cudaEventDestroy(e1_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  int out_h() const override { return out_h_; }
  int out_w() const override { return out_w_; }
  std::string describe() const override {
    return "trt(" + std::to_string(in_h_) + "x" + std::to_string(in_w_) + " -> " + std::to_string(out_h_) + "x" +
           std::to_string(out_w_) + ", stream priority " + std::to_string(priority_) + ")";
  }

  // Same preprocessing as dvslam.export.depth_engine.preprocess: INTER_LINEAR resize, BGR->RGB, ImageNet norm.
  double infer(const FrameView& frame, float* out) override {
    static const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    cv::Mat src(frame.height, frame.width, CV_8UC3, const_cast<unsigned char*>(frame.bgr)), rs;
    cv::resize(src, rs, cv::Size(in_w_, in_h_), 0, 0, cv::INTER_LINEAR);
    const std::size_t plane = static_cast<std::size_t>(in_h_) * in_w_;
    for (int y = 0; y < in_h_; ++y) {
      const unsigned char* row = rs.ptr<unsigned char>(y);
      for (int x = 0; x < in_w_; ++x)
        for (int c = 0; c < 3; ++c)  // RGB plane c <- BGR channel 2-c
          h_in_[c * plane + y * in_w_ + x] = (row[3 * x + 2 - c] / 255.0f - mean[c]) / stdv[c];
    }
    const std::size_t nout = static_cast<std::size_t>(out_h_) * out_w_;
    cudaMemcpyAsync(d_in_, h_in_, 3 * plane * 4, cudaMemcpyHostToDevice, stream_);
    cudaEventRecord(e0_, stream_);
    ctx_->enqueueV3(stream_);
    cudaEventRecord(e1_, stream_);
    cudaMemcpyAsync(h_out_, d_out_, nout * 4, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0_, e1_);
    std::copy(h_out_, h_out_ + nout, out);
    return ms;
  }

 private:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  std::string in_name_, out_name_;
  int in_h_ = 0, in_w_ = 0, out_h_ = 0, out_w_ = 0, priority_ = 0;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t e0_ = nullptr, e1_ = nullptr;
  void *d_in_ = nullptr, *d_out_ = nullptr;
  float *h_in_ = nullptr, *h_out_ = nullptr;
};

}  // namespace

bool trt_available() { return true; }

std::unique_ptr<DepthEngine> make_trt_depth_engine(const std::string& engine_path, int stream_priority,
                                                   std::string* err) {
  auto e = std::make_unique<TrtDepthEngine>();
  if (!e->load(engine_path, stream_priority, err)) return nullptr;
  return e;
}

bool build_trt_engine(const std::string& onnx_path, const std::string& engine_path, const std::string& precision,
                      std::string* err) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
  std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, logger()));
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    *err = "ONNX parse failed: " + onnx_path;
    return false;
  }
  std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
  cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ull << 30);
  if (precision == "fp16") {
    cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (precision == "bf16") {
    cfg->setFlag(nvinfer1::BuilderFlag::kBF16);
  } else if (precision == "fp32") {
    cfg->clearFlag(nvinfer1::BuilderFlag::kTF32);  // strict FP32
  } else {
    *err = "precision must be fp32, fp16 or bf16";
    return false;
  }
  std::unique_ptr<nvinfer1::IHostMemory> blob(builder->buildSerializedNetwork(*net, *cfg));
  if (!blob) {
    *err = "buildSerializedNetwork failed";
    return false;
  }
  std::ofstream f(engine_path, std::ios::binary);
  f.write(static_cast<const char*>(blob->data()), static_cast<std::streamsize>(blob->size()));
  return static_cast<bool>(f);
}

}  // namespace dvslam

#else  // !DVSLAM_WITH_TRT

namespace dvslam {

static const char* kNotBuilt =
    "dvslam_loop was built without TensorRT: NvInfer.h / libnvinfer / CUDA / OpenCV were not all found at CMake "
    "configure time. Reconfigure with -DTENSORRT_ROOT=<dir containing include/NvInfer.h and lib/libnvinfer.so>.";

bool trt_available() { return false; }

std::unique_ptr<DepthEngine> make_trt_depth_engine(const std::string&, int, std::string* err) {
  if (err) *err = kNotBuilt;
  return nullptr;
}

bool build_trt_engine(const std::string&, const std::string&, const std::string&, std::string* err) {
  if (err) *err = kNotBuilt;
  return false;
}

}  // namespace dvslam

#endif
