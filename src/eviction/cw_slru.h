#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "core/index_entry.h"
#include "core/storage_key.h"

namespace attentiondb {

struct EvictionEntry {
    StorageKey key;
    uint32_t recompute_cost;
    uint32_t chunk_index;
    uint64_t last_access_us;
};

// Cost-Weighted Segmented LRU.
// Two segments: Protected (high-value entries) and Probationary (everything else).
// Eviction prefers suffix chunks over prefix chunks within each segment.
class CwSlru {
public:
    struct Config {
        double protected_ratio = 0.25;   // Fraction of capacity for Protected
        uint32_t protection_threshold = 100;  // Min recompute_cost for promotion
        size_t max_entries = 0;           // 0 = unlimited (set at init)
    };

    explicit CwSlru(const Config& config);

    // Notify that a key was accessed (on get). May promote from Probationary to Protected.
    void on_access(const StorageKey& key, uint32_t recompute_cost, uint32_t chunk_index);

    // Add a newly inserted entry to the Probationary segment.
    void on_insert(const StorageKey& key, uint32_t recompute_cost, uint32_t chunk_index);

    // Remove a key entirely (on delete).
    void on_remove(const StorageKey& key);

    // Pick the best eviction candidate. Returns nullopt if both segments empty.
    std::optional<StorageKey> evict();

    size_t protected_size() const;
    size_t probationary_size() const;
    size_t total_size() const;

private:
    using ListType = std::list<EvictionEntry>;
    using IterMap = std::unordered_map<StorageKey, ListType::iterator, StorageKeyHash, StorageKeyEqual>;

    void demote_protected_overflow();
    ListType::iterator find_best_victim(ListType& segment);

    Config config_;

    mutable std::mutex mu_;
    ListType protected_;      // MRU at front, LRU at back
    ListType probationary_;   // MRU at front, LRU at back
    IterMap protected_map_;
    IterMap probationary_map_;
};

}  // namespace attentiondb
