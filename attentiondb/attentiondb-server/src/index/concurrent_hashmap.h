#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include <libcuckoo/cuckoohash_map.hh>

#include "core/index_entry.h"
#include "core/storage_key.h"

namespace attentiondb {

class ConcurrentHashMap {
public:
    using MapType = libcuckoo::cuckoohash_map<StorageKey, IndexEntry,
                                               StorageKeyHash, StorageKeyEqual>;

    explicit ConcurrentHashMap(size_t expected_entries = 1 << 20);

    bool find(const StorageKey& key, IndexEntry& entry) const;

    void upsert(const StorageKey& key, const IndexEntry& entry);

    bool erase(const StorageKey& key);

    bool contains(const StorageKey& key) const;

    size_t size() const;

    // Update an entry in-place via a callback. Returns false if key not found.
    bool update_fn(const StorageKey& key,
                   const std::function<void(IndexEntry&)>& fn);

    // Iterate all entries (takes a shared lock per bucket).
    // Callback returns false to stop early.
    void iterate(const std::function<bool(const StorageKey&, const IndexEntry&)>& fn) const;

    // Serialize all entries for checkpointing.
    std::vector<std::pair<StorageKey, IndexEntry>> snapshot() const;

    // Bulk load entries (for checkpoint recovery).
    void bulk_load(const std::vector<std::pair<StorageKey, IndexEntry>>& entries);

    void clear();

private:
    mutable MapType map_;
};

}  // namespace attentiondb
