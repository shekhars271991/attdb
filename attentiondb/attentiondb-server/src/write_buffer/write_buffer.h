#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "admission/admission.h"
#include "core/config.h"
#include "core/status.h"
#include "core/storage_key.h"

namespace attentiondb {

enum class BackpressureLevel : uint8_t {
    kNormal = 0,      // 0–50%: accept all admitted entries
    kCautious = 1,    // 50–75%: raise cost threshold dynamically
    kSelective = 2,   // 75–90%: only high-value + system prompts
    kEmergency = 3,   // 90–100%: only system prompts
    kFull = 4,        // 100%: reject all
};

class WriteBuffer {
public:
    struct PendingWrite {
        StorageKey key;
        std::vector<uint8_t> data;
        PutOpts opts;
        uint32_t checksum;
    };

    // Called by the flush thread for each batch of entries to persist.
    using FlushCallback = std::function<void(std::vector<PendingWrite>&& batch)>;

    explicit WriteBuffer(const WriteBufferConfig& config);
    ~WriteBuffer();

    WriteBuffer(const WriteBuffer&) = delete;
    WriteBuffer& operator=(const WriteBuffer&) = delete;

    // Enqueue a write. Checks backpressure before accepting.
    Status submit(const StorageKey& key, const void* data, size_t len,
                  const PutOpts& opts, uint32_t checksum);

    BackpressureLevel current_level() const;
    double utilization() const;

    void set_flush_callback(FlushCallback cb);
    void start();
    void stop();

    // Stats
    uint64_t total_submitted() const { return total_submitted_.load(); }
    uint64_t total_rejected_bp() const { return total_rejected_bp_.load(); }
    uint64_t total_flushed() const { return total_flushed_.load(); }

private:
    void flush_thread_fn();
    BackpressureLevel compute_level(double util) const;
    bool should_accept(const PutOpts& opts, BackpressureLevel level) const;

    WriteBufferConfig config_;
    FlushCallback flush_cb_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::queue<PendingWrite> queue_;
    std::atomic<size_t> current_bytes_{0};

    std::thread flush_thread_;
    std::atomic<bool> running_{false};

    std::atomic<uint64_t> total_submitted_{0};
    std::atomic<uint64_t> total_rejected_bp_{0};
    std::atomic<uint64_t> total_flushed_{0};
};

}  // namespace attentiondb
