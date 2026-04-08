#include "write_buffer/write_buffer.h"

#include <chrono>

namespace attentiondb {

WriteBuffer::WriteBuffer(const WriteBufferConfig& config) : config_(config) {}

WriteBuffer::~WriteBuffer() {
    stop();
}

Status WriteBuffer::submit(const StorageKey& key, const void* data, size_t len,
                            const PutOpts& opts, uint32_t checksum) {
    auto level = current_level();

    if (!should_accept(opts, level)) {
        total_rejected_bp_.fetch_add(1, std::memory_order_relaxed);
        if (level == BackpressureLevel::kFull) return Status::kFull;
        return Status::kRejectedBackpressure;
    }

    PendingWrite pw;
    pw.key = key;
    pw.data.assign(static_cast<const uint8_t*>(data),
                   static_cast<const uint8_t*>(data) + len);
    pw.opts = opts;
    pw.checksum = checksum;

    {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.push(std::move(pw));
        current_bytes_.fetch_add(len, std::memory_order_relaxed);
    }
    cv_.notify_one();

    total_submitted_.fetch_add(1, std::memory_order_relaxed);
    return Status::kOk;
}

BackpressureLevel WriteBuffer::current_level() const {
    return compute_level(utilization());
}

double WriteBuffer::utilization() const {
    return static_cast<double>(current_bytes_.load(std::memory_order_relaxed)) /
           config_.size_bytes;
}

BackpressureLevel WriteBuffer::compute_level(double util) const {
    if (util >= 1.0) return BackpressureLevel::kFull;
    if (util >= 0.9) return BackpressureLevel::kEmergency;
    if (util >= 0.75) return BackpressureLevel::kSelective;
    if (util >= 0.5) return BackpressureLevel::kCautious;
    return BackpressureLevel::kNormal;
}

bool WriteBuffer::should_accept(const PutOpts& opts, BackpressureLevel level) const {
    switch (level) {
        case BackpressureLevel::kNormal:
            return true;
        case BackpressureLevel::kCautious:
            // Raise effective cost threshold (2x normal)
            return opts.entry_type == EntryType::kSystemPrompt ||
                   opts.recompute_cost >= 200;
        case BackpressureLevel::kSelective:
            return opts.entry_type == EntryType::kSystemPrompt ||
                   opts.recompute_cost >= 500;
        case BackpressureLevel::kEmergency:
            return opts.entry_type == EntryType::kSystemPrompt;
        case BackpressureLevel::kFull:
            return false;
    }
    return false;
}

void WriteBuffer::set_flush_callback(FlushCallback cb) {
    flush_cb_ = std::move(cb);
}

void WriteBuffer::start() {
    running_ = true;
    flush_thread_ = std::thread(&WriteBuffer::flush_thread_fn, this);
}

void WriteBuffer::stop() {
    running_ = false;
    cv_.notify_all();
    if (flush_thread_.joinable()) flush_thread_.join();

    // Flush remaining entries
    if (flush_cb_) {
        std::vector<PendingWrite> batch;
        std::lock_guard<std::mutex> lock(mu_);
        while (!queue_.empty()) {
            batch.push_back(std::move(queue_.front()));
            queue_.pop();
        }
        if (!batch.empty()) flush_cb_(std::move(batch));
        current_bytes_.store(0);
    }
}

void WriteBuffer::flush_thread_fn() {
    while (running_) {
        std::vector<PendingWrite> batch;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_for(lock,
                         std::chrono::milliseconds(config_.flush_interval_ms),
                         [this] { return !queue_.empty() || !running_; });

            if (!running_ && queue_.empty()) break;

            // Drain up to 64 entries per batch
            constexpr size_t kMaxBatchSize = 64;
            size_t bytes_in_batch = 0;
            while (!queue_.empty() && batch.size() < kMaxBatchSize) {
                bytes_in_batch += queue_.front().data.size();
                batch.push_back(std::move(queue_.front()));
                queue_.pop();
            }
            current_bytes_.fetch_sub(bytes_in_batch, std::memory_order_relaxed);
        }

        if (!batch.empty() && flush_cb_) {
            total_flushed_.fetch_add(batch.size(), std::memory_order_relaxed);
            flush_cb_(std::move(batch));
        }
    }
}

}  // namespace attentiondb
