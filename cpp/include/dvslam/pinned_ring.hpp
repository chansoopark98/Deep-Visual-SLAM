// Fixed pool of page-locked frame buffers with reference-counted handles (plan v3 §4.6 "고정 링 버퍼").
// Placeholder: slots hold decoded BGR frames; with CUDA they are cudaHostAlloc'ed so H2D copies can be async.
// Holders at any time: capture (1) + frame mailbox (1) + tracker (1) + depth mailbox (1) + depth thread (1) = 5,
// so the default 8 slots never block.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef DVSLAM_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace dvslam {

class PinnedRing {
 public:
  class Ref {
   public:
    Ref() = default;
    Ref(PinnedRing* ring, int slot) : ring_(ring), slot_(slot) {}
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    Ref(Ref&& o) noexcept : ring_(o.ring_), slot_(o.slot_) { o.ring_ = nullptr; }
    Ref& operator=(Ref&& o) noexcept {
      if (this != &o) {
        reset();
        ring_ = o.ring_;
        slot_ = o.slot_;
        o.ring_ = nullptr;
      }
      return *this;
    }
    ~Ref() { reset(); }

    Ref share() const {
      ring_->refs_[slot_].fetch_add(1);
      return Ref(ring_, slot_);
    }
    std::uint8_t* data() const { return ring_->buf_[slot_]; }
    int slot() const { return slot_; }
    explicit operator bool() const { return ring_ != nullptr; }
    void reset() {
      if (ring_) ring_->release(slot_);
      ring_ = nullptr;
    }

   private:
    PinnedRing* ring_ = nullptr;
    int slot_ = -1;
  };

  PinnedRing(int slots, std::size_t bytes) : refs_(slots), bytes_(bytes) {
    for (int i = 0; i < slots; ++i) {
      void* p = nullptr;
#ifdef DVSLAM_WITH_CUDA
      if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess) {
        pinned_ = true;
      } else {
        p = nullptr;
      }
#endif
      if (!p) p = std::malloc(bytes);
      if (!p) throw std::bad_alloc();
      buf_.push_back(static_cast<std::uint8_t*>(p));
      refs_[i].store(0);
    }
  }
  PinnedRing(const PinnedRing&) = delete;
  PinnedRing& operator=(const PinnedRing&) = delete;
  ~PinnedRing() {
    for (auto* p : buf_) {
#ifdef DVSLAM_WITH_CUDA
      if (pinned_) {
        cudaFreeHost(p);
        continue;
      }
#endif
      std::free(p);
    }
  }

  // Blocks until a free slot exists (never with the default sizing); counts how often it had to wait.
  Ref acquire() {
    std::unique_lock<std::mutex> lk(m_);
    for (;;) {
      for (int i = 0; i < static_cast<int>(buf_.size()); ++i) {
        int zero = 0;
        if (refs_[i].compare_exchange_strong(zero, 1)) return Ref(this, i);
      }
      ++waits_;
      cv_.wait(lk);
    }
  }

  bool pinned() const { return pinned_; }
  std::size_t bytes() const { return bytes_; }
  std::size_t waits() const { return waits_; }

 private:
  void release(int slot) {
    if (refs_[slot].fetch_sub(1) == 1) {
      std::lock_guard<std::mutex> lk(m_);
      cv_.notify_all();
    }
  }

  std::vector<std::uint8_t*> buf_;
  std::vector<std::atomic<int>> refs_;
  std::size_t bytes_;
  bool pinned_ = false;
  std::size_t waits_ = 0;
  std::mutex m_;
  std::condition_variable cv_;
};

}  // namespace dvslam
