// AttentionDB Engine Benchmark
//
// Measures end-to-end throughput of the full AttentionDB engine (admission,
// write buffer, T1 DRAM slab, T2 NVMe blob store, eviction, checkpointing).
//
// BM_EnginePut: Sustained write throughput at varying KV cache blob sizes
//   (256KB, 1MB, 4MB). Each iteration submits a unique key through the
//   admission control pipeline. Blobs land in T1 DRAM until full, then
//   spill to T2 NVMe via direct synchronous I/O. Reports MB/s and ops/s
//   to characterize the write-path bottleneck.
//
// BM_EngineGet: Read throughput from a pre-populated working set sized to
//   fit in T1 DRAM (~64MB). Iterates over keys round-robin so every get
//   is a T1 hit (memcpy from slab). Reports GB/s and ops/s to show the
//   upper bound on serving latency when the working set is memory-resident.
//   Larger blobs test memcpy bandwidth; smaller blobs test per-op overhead.

#include <benchmark/benchmark.h>

#include <cstring>
#include <filesystem>
#include <unistd.h>

#include "engine.h"

using namespace attentiondb;

static Config bench_config(const std::string& dir) {
    Config cfg;
    cfg.mode = "local_only";
    cfg.dram_cache.enabled = true;
    cfg.dram_cache.size_bytes = 64 * 1024 * 1024;
    cfg.dram_cache.size_classes = {4096, 65536, 262144, 1048576, 4194304};
    cfg.dram_cache.hugepages = false;
    cfg.dram_cache.pinned_memory = false;
    cfg.nvme_store.enabled = true;
    cfg.nvme_store.paths = {dir};
    cfg.nvme_store.segment_size = 64 * 1024 * 1024;
    cfg.nvme_store.io_engine = "posix";
    cfg.nvme_store.gc_enabled = false;
    cfg.admission.min_recompute_cost = 0;
    cfg.checkpoint.interval_s = 3600;
    cfg.logging.level = "error";
    cfg.logging.periodic_summary_interval_s = 0;
    return cfg;
}

static void BM_EnginePut(benchmark::State& state) {
    std::string dir = "/tmp/attentiondb_bench_" + std::to_string(getpid());
    std::filesystem::create_directories(dir);

    AttentionDBEngine engine;
    engine.open(bench_config(dir));

    size_t blob_size = state.range(0);
    std::vector<uint8_t> data(blob_size, 0xAB);
    PutOpts opts;
    opts.recompute_cost = 500;

    uint64_t i = 0;
    for (auto _ : state) {
        StorageKey key{i++, 0, 0, 0, 0};
        engine.put(key, data.data(), data.size(), opts);
    }

    state.SetBytesProcessed(state.iterations() * blob_size);
    state.SetItemsProcessed(state.iterations());

    engine.close();
    std::filesystem::remove_all(dir);
}
BENCHMARK(BM_EnginePut)->Arg(256 * 1024)->Arg(1024 * 1024)->Arg(4 * 1024 * 1024);

static void BM_EngineGet(benchmark::State& state) {
    std::string dir = "/tmp/attentiondb_bench_get_" + std::to_string(getpid());
    std::filesystem::create_directories(dir);

    AttentionDBEngine engine;
    engine.open(bench_config(dir));

    size_t blob_size = state.range(0);
    const uint64_t kEntries = std::min<uint64_t>(1000, 64ULL * 1024 * 1024 / blob_size);

    std::vector<uint8_t> data(blob_size, 0xAB);
    PutOpts opts;
    opts.recompute_cost = 500;
    for (uint64_t i = 0; i < kEntries; ++i) {
        StorageKey key{i, 0, 0, 0, 0};
        engine.put(key, data.data(), data.size(), opts);
    }

    std::vector<uint8_t> buf(blob_size + 4096);
    uint64_t i = 0;
    for (auto _ : state) {
        StorageKey key{i % kEntries, 0, 0, 0, 0};
        size_t out_len = 0;
        engine.get(key, buf.data(), buf.size(), &out_len);
        ++i;
    }

    state.SetBytesProcessed(state.iterations() * blob_size);
    state.SetItemsProcessed(state.iterations());

    engine.close();
    std::filesystem::remove_all(dir);
}
BENCHMARK(BM_EngineGet)->Arg(256 * 1024)->Arg(1024 * 1024)->Arg(4 * 1024 * 1024);
