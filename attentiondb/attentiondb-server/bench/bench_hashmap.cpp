#include <benchmark/benchmark.h>

#include "index/concurrent_hashmap.h"

using namespace attentiondb;

static void BM_HashMapInsert(benchmark::State& state) {
    ConcurrentHashMap map(1 << 20);
    uint64_t i = 0;
    for (auto _ : state) {
        StorageKey key{i++, 0, 0, 0, 0};
        IndexEntry entry{};
        entry.length = 1024;
        map.upsert(key, entry);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HashMapInsert);

static void BM_HashMapFind(benchmark::State& state) {
    ConcurrentHashMap map(1 << 20);
    constexpr uint64_t kEntries = 100000;
    for (uint64_t i = 0; i < kEntries; ++i) {
        StorageKey key{i, 0, 0, 0, 0};
        IndexEntry entry{};
        map.upsert(key, entry);
    }

    uint64_t i = 0;
    for (auto _ : state) {
        StorageKey key{i % kEntries, 0, 0, 0, 0};
        IndexEntry entry{};
        benchmark::DoNotOptimize(map.find(key, entry));
        ++i;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HashMapFind);

static void BM_HashMapContains(benchmark::State& state) {
    ConcurrentHashMap map(1 << 20);
    constexpr uint64_t kEntries = 100000;
    for (uint64_t i = 0; i < kEntries; ++i) {
        StorageKey key{i, 0, 0, 0, 0};
        IndexEntry entry{};
        map.upsert(key, entry);
    }

    uint64_t i = 0;
    for (auto _ : state) {
        StorageKey key{i % kEntries, 0, 0, 0, 0};
        benchmark::DoNotOptimize(map.contains(key));
        ++i;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HashMapContains);
