// 1-slot mailbox with drop-oldest semantics (plan v3 §4.1): put() replaces an unread item, take() blocks.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

namespace dvslam {

template <class T>
class Mailbox {
 public:
  // Stores `item`, dropping (destroying) an unread older item. Returns true if one was dropped.
  bool put(T item) {
    std::optional<T> old;
    {
      std::lock_guard<std::mutex> lk(m_);
      if (slot_) {
        old = std::move(slot_);
        ++dropped_;
      }
      slot_ = std::move(item);
    }
    cv_.notify_one();
    return old.has_value();
  }

  // Blocks until an item is available or the mailbox is closed (then returns nullopt once empty).
  std::optional<T> take() {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return slot_.has_value() || closed_; });
    return pop_locked();
  }

  std::optional<T> try_take() {
    std::lock_guard<std::mutex> lk(m_);
    return pop_locked();
  }

  void close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    cv_.notify_all();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(m_);
    return !slot_.has_value();
  }

  std::size_t dropped() const {
    std::lock_guard<std::mutex> lk(m_);
    return dropped_;
  }

 private:
  std::optional<T> pop_locked() {
    std::optional<T> out;
    out.swap(slot_);
    return out;
  }

  mutable std::mutex m_;
  std::condition_variable cv_;
  std::optional<T> slot_;
  bool closed_ = false;
  std::size_t dropped_ = 0;
};

}  // namespace dvslam
