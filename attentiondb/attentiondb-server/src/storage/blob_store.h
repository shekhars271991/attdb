#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
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

struct WriteBlock {
    uint8_t* buf = nullptr;
    size_t pos = 0;
    size_t capacity = 0;
    uint32_t segment_id = 0;
    uint64_t segment_offset = 0;
    uint32_t entry_count = 0;
    int segment_fd = -1;

    void reset(uint32_t seg_id, uint64_t seg_off, int fd) {
        pos = 0;
        segment_id = seg_id;
        segment_offset = seg_off;
        entry_count = 0;
        segment_fd = fd;
    }
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

    // Append a blob into the active write block. Returns location immediately;
    // the actual pwrite happens asynchronously on the flush thread.
    Status write(const StorageKey& key, const void* data, size_t len,
                 uint32_t checksum, BlobLocation* loc_out);

    // Read a blob. Checks pending write blocks first (read-through),
    // then falls back to disk.
    Status read(const BlobLocation& loc, void* buf, size_t buf_len,
                size_t* bytes_read);

    void mark_dead(const BlobLocation& loc);

    Stats stats() const;

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

    void flush_thread_fn();
    WriteBlock* acquire_block(std::unique_lock<std::mutex>& lock);
    const uint8_t* find_in_pending_blocks(const BlobLocation& loc) const;

    NvmeStoreConfig config_;
    std::unique_ptr<IoEngine> io_;

    mutable std::mutex segments_mu_;
    std::vector<std::unique_ptr<SegmentInfo>> segments_;
    SegmentInfo* active_segment_ = nullptr;
    uint32_t next_segment_id_ = 0;
    std::atomic<uint64_t> write_sequence_{0};

    // Write block pool (Aerospike-style)
    mutable std::mutex block_mu_;
    std::condition_variable flush_cv_;
    std::condition_variable free_cv_;
    WriteBlock* active_block_ = nullptr;
    std::queue<WriteBlock*> flush_queue_;
    std::vector<WriteBlock*> free_pool_;
    std::vector<std::unique_ptr<WriteBlock>> all_blocks_;
    std::thread flush_thread_;
    std::atomic<bool> flush_running_{false};

    std::thread gc_thread_;
    std::atomic<bool> gc_running_{false};
    GcRewriteCallback gc_rewrite_cb_;

    std::atomic<uint64_t> total_bytes_on_disk_{0};
    std::atomic<uint64_t> gc_reclaimed_bytes_{0};
    std::atomic<uint32_t> gc_cycles_{0};
};

}  // namespace attentiondb
