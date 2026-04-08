#include "index/concurrent_hashmap.h"

namespace attentiondb {

ConcurrentHashMap::ConcurrentHashMap(size_t expected_entries)
    : map_(expected_entries) {}

bool ConcurrentHashMap::find(const StorageKey& key, IndexEntry& entry) const {
    return map_.find(key, entry);
}

void ConcurrentHashMap::upsert(const StorageKey& key, const IndexEntry& entry) {
    map_.upsert(key, [&](IndexEntry& existing) { existing = entry; }, entry);
}

bool ConcurrentHashMap::erase(const StorageKey& key) {
    return map_.erase(key);
}

bool ConcurrentHashMap::contains(const StorageKey& key) const {
    return map_.contains(key);
}

size_t ConcurrentHashMap::size() const {
    return map_.size();
}

bool ConcurrentHashMap::update_fn(const StorageKey& key,
                                   const std::function<void(IndexEntry&)>& fn) {
    return map_.update_fn(key, fn);
}

void ConcurrentHashMap::iterate(
    const std::function<bool(const StorageKey&, const IndexEntry&)>& fn) const {
    auto locked = map_.lock_table();
    for (const auto& [key, entry] : locked) {
        if (!fn(key, entry)) break;
    }
}

std::vector<std::pair<StorageKey, IndexEntry>> ConcurrentHashMap::snapshot() const {
    std::vector<std::pair<StorageKey, IndexEntry>> result;
    auto locked = map_.lock_table();
    result.reserve(locked.size());
    for (const auto& [key, entry] : locked) {
        result.emplace_back(key, entry);
    }
    return result;
}

void ConcurrentHashMap::bulk_load(
    const std::vector<std::pair<StorageKey, IndexEntry>>& entries) {
    for (const auto& [key, entry] : entries) {
        map_.insert(key, entry);
    }
}

void ConcurrentHashMap::clear() {
    map_.clear();
}

}  // namespace attentiondb
