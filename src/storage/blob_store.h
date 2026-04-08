#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/status.h"
#include "core/storage_key.h"
#include "storage/io_engine.h"
#include "storage/segment.h"

namespace attentiondb {

struct BlobLocation {
    uint32_t segment_id;
    uint64_t offset;        // Byte offset within segment file
    uint32_t length;        // Actual blob size
    uint32_t disk_size;     // Padded on-disk size
};

class BlobStore {
public:
    struct Stats {
        uint32_t num_segments;
        uint64_t total_bytes_on_disk;
        uint64_t live_bytes;
        uint64_t gc_reclaimed_bytes;
        uint32_t gc_cycles;
    };

    explicit BlobStore(const NvmeStoreConfig& config);
    ~BlobStore();

    BlobStore(const BlobStore&) = delete;
    BlobStore& operator=(const BlobStore&) = delete;

    Status open();
    void close();

    // Write a blob, returns its location on disk.
    Status write(const StorageKey& key, const void* data, size_t len,
                 uint32_t checksum, BlobLocation* loc_out);

    // Read a blob into the provided buffer.
    Status read(const BlobLocation& loc, void* buf, size_t buf_len,
                size_t* bytes_read);

    // Mark a blob as dead (for GC accounting).
    void mark_dead(const BlobLocation& loc);

    Stats stats() const;

    // Called by GC: callback receives (key, data, len) for each live entry
    // in a segment being compacted.
    using GcRewriteCallback = std::function<void(const StorageKey& key,
                                                  const void* data, size_t len,
                                                  uint32_t checksum)>;
    void set_gc_rewrite_callback(GcRewriteCallback cb);

private:
    struct SegmentInfo {
        uint32_t id;
        int fd;
        uint64_t write_offset;
        uint32_t entry_count;
        std::atomic<uint32_t> live_count;
        size_t file_size;
        std::string path;
    };

    void create_new_segment();
    int open_segment_fd(const std::string& path, bool create);
    std::string segment_path(uint32_t id) const;
    void gc_thread_fn();
    void gc_compact_segment(SegmentInfo& seg);
    void write_segment_header(int fd, const SegmentInfo& seg);

    NvmeStoreConfig config_;
    std::unique_ptr<IoEngine> io_;

    mutable std::mutex segments_mu_;
    std::vector<std::unique_ptr<SegmentInfo>> segments_;
    SegmentInfo* active_segment_ = nullptr;
    uint32_t next_segment_id_ = 0;
    std::atomic<uint64_t> write_sequence_{0};

    std::mutex write_mu_;  // Serializes writes to active segment

    std::thread gc_thread_;
    std::atomic<bool> gc_running_{false};
    GcRewriteCallback gc_rewrite_cb_;

    // Stats
    std::atomic<uint64_t> total_bytes_on_disk_{0};
    std::atomic<uint64_t> gc_reclaimed_bytes_{0};
    std::atomic<uint32_t> gc_cycles_{0};
};

}  // namespace attentiondb
