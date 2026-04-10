#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace attentiondb {

struct DramCacheConfig {
    bool enabled = true;
    size_t size_bytes = 16ULL * 1024 * 1024 * 1024;  // 16 GB
    std::vector<size_t> size_classes = {
        65536, 131072, 262144, 524288, 1048576, 2097152, 4194304
    };
    bool hugepages = true;
    bool pinned_memory = false;  // cudaMallocHost, requires ATTENTIONDB_CUDA
};

struct NvmeStoreConfig {
    bool enabled = true;
    std::vector<std::string> paths = {"/mnt/nvme0/attentiondb"};
    size_t max_size_per_drive = 1ULL * 1024 * 1024 * 1024 * 1024;  // 1 TB
    size_t segment_size = 1ULL * 1024 * 1024 * 1024;               // 1 GB
    uint32_t fd_pool_size = 256;
    size_t alignment = 4096;

    std::string io_engine = "io_uring";  // "io_uring" | "posix"
    uint32_t uring_queue_depth = 128;
    bool uring_registered_buffers = true;
    uint32_t gds_thread_pool_size = 4;

    uint32_t write_block_size_mb = 8;    // Aerospike-style write block size (MB)
    uint32_t write_block_pool_size = 8;  // Number of pre-allocated write blocks

    bool gc_enabled = true;
    double gc_trigger_fragmentation = 0.5;
    double gc_max_bandwidth_fraction = 0.2;
};

struct AdmissionConfig {
    uint32_t min_recompute_cost = 64;
    uint32_t base_threshold = 100;
};

struct EvictionConfig {
    std::string policy = "cw_slru";
    double protected_ratio = 0.25;
};

struct CheckpointConfig {
    uint32_t interval_s = 30;
};

struct LoggingConfig {
    std::string level = "info";     // "debug" | "info" | "warn" | "error"
    std::string format = "json";
    uint32_t periodic_summary_interval_s = 60;
};

struct Config {
    std::string mode = "local_only";
    DramCacheConfig dram_cache;
    NvmeStoreConfig nvme_store;
    AdmissionConfig admission;
    EvictionConfig eviction;
    CheckpointConfig checkpoint;
    LoggingConfig logging;
    uint32_t get_timeout_ms = 5;

    static Config LoadFromFile(const std::string& path);
    static Config LoadFromString(const std::string& yaml_content);
    static Config Default();
};

}  // namespace attentiondb
