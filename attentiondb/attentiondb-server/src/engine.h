#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "admission/admission.h"
#include "checkpoint/checkpoint.h"
#include "core/config.h"
#include "core/index_entry.h"
#include "core/status.h"
#include "core/storage_key.h"
#include "eviction/cw_slru.h"
#include "index/concurrent_hashmap.h"
#include "logging/logger.h"
#include "memory/slab_allocator.h"
#include "storage/blob_store.h"

namespace attentiondb {

struct EngineStats {
    // Capacity
    size_t t1_total_bytes;
    size_t t1_used_bytes;
    size_t t2_total_bytes_on_disk;
    uint32_t t2_num_segments;

    // Index
    size_t index_entries;

    // Eviction
    size_t eviction_protected;
    size_t eviction_probationary;

    // Admission
    uint64_t admission_evaluated;
    uint64_t admission_rejected;

    // Checkpoint
    uint64_t last_checkpoint_entries;
    uint64_t last_checkpoint_duration_ms;
};

class AttentionDBEngine {
public:
    AttentionDBEngine();
    ~AttentionDBEngine();

    AttentionDBEngine(const AttentionDBEngine&) = delete;
    AttentionDBEngine& operator=(const AttentionDBEngine&) = delete;

    Status open(const Config& config);
    void close();

    Status put(const StorageKey& key, const void* blob, size_t len,
               const PutOpts& opts);

    Status get(const StorageKey& key, void* buf, size_t buf_len,
               size_t* out_len);

    bool contains(const StorageKey& key);

    Status del(const StorageKey& key);

    EngineStats stats();

private:
    void evict_if_needed();
    double storage_utilization() const;
    uint32_t compute_crc32(const void* data, size_t len);

    Config config_;
    std::unique_ptr<ConcurrentHashMap> index_;
    std::unique_ptr<SlabAllocator> t1_allocator_;
    std::unique_ptr<BlobStore> t2_store_;
    std::unique_ptr<CwSlru> t1_eviction_;
    std::unique_ptr<CwSlru> t2_eviction_;
    std::unique_ptr<AdmissionControl> admission_;
    std::unique_ptr<CheckpointManager> checkpoint_;
    std::unique_ptr<Logger> logger_;
    bool opened_ = false;
};

}  // namespace attentiondb
