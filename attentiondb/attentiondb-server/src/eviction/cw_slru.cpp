#include "eviction/cw_slru.h"

#include <algorithm>
#include <cassert>

namespace attentiondb {

CwSlru::CwSlru(const Config& config) : config_(config) {}

void CwSlru::on_access(const StorageKey& key, uint32_t recompute_cost,
                        uint32_t chunk_index) {
    std::lock_guard<std::mutex> lock(mu_);

    // If already in Protected, move to front (MRU)
    auto pit = protected_map_.find(key);
    if (pit != protected_map_.end()) {
        pit->second->recompute_cost = recompute_cost;
        pit->second->last_access_us = now_us();
        protected_.splice(protected_.begin(), protected_, pit->second);
        return;
    }

    // If in Probationary, consider promotion
    auto qit = probationary_map_.find(key);
    if (qit != probationary_map_.end()) {
        qit->second->recompute_cost = recompute_cost;
        qit->second->last_access_us = now_us();

        if (recompute_cost >= config_.protection_threshold) {
            // Promote to Protected
            EvictionEntry entry = *qit->second;
            probationary_.erase(qit->second);
            probationary_map_.erase(qit);

            protected_.push_front(entry);
            protected_map_[key] = protected_.begin();

            demote_protected_overflow();
        } else {
            // Stay in Probationary but move to front
            probationary_.splice(probationary_.begin(), probationary_, qit->second);
        }
    }
}

void CwSlru::on_insert(const StorageKey& key, uint32_t recompute_cost,
                         uint32_t chunk_index) {
    std::lock_guard<std::mutex> lock(mu_);

    // If already tracked, update and move to front
    auto pit = protected_map_.find(key);
    if (pit != protected_map_.end()) {
        pit->second->recompute_cost = recompute_cost;
        pit->second->last_access_us = now_us();
        protected_.splice(protected_.begin(), protected_, pit->second);
        return;
    }

    auto qit = probationary_map_.find(key);
    if (qit != probationary_map_.end()) {
        qit->second->recompute_cost = recompute_cost;
        qit->second->last_access_us = now_us();
        probationary_.splice(probationary_.begin(), probationary_, qit->second);
        return;
    }

    // New entry goes to Probationary front
    EvictionEntry entry{key, recompute_cost, chunk_index, now_us()};
    probationary_.push_front(entry);
    probationary_map_[key] = probationary_.begin();
}

void CwSlru::on_remove(const StorageKey& key) {
    std::lock_guard<std::mutex> lock(mu_);

    auto pit = protected_map_.find(key);
    if (pit != protected_map_.end()) {
        protected_.erase(pit->second);
        protected_map_.erase(pit);
        return;
    }

    auto qit = probationary_map_.find(key);
    if (qit != probationary_map_.end()) {
        probationary_.erase(qit->second);
        probationary_map_.erase(qit);
    }
}

std::optional<StorageKey> CwSlru::evict() {
    std::lock_guard<std::mutex> lock(mu_);

    // First try to evict from Probationary (suffix-before-prefix)
    if (!probationary_.empty()) {
        auto victim = find_best_victim(probationary_);
        StorageKey key = victim->key;
        probationary_map_.erase(key);
        probationary_.erase(victim);
        return key;
    }

    // If Probationary is empty, evict from Protected LRU end
    if (!protected_.empty()) {
        auto victim = find_best_victim(protected_);
        StorageKey key = victim->key;
        protected_map_.erase(key);
        protected_.erase(victim);
        return key;
    }

    return std::nullopt;
}

CwSlru::ListType::iterator CwSlru::find_best_victim(ListType& segment) {
    // Walk from LRU end (back). Among the coldest entries,
    // prefer suffix chunks (higher chunk_index) over prefix chunks.
    // Scan the last N entries (N = min(8, size)) and pick the one with highest chunk_index.
    assert(!segment.empty());

    constexpr int kScanWindow = 8;
    auto best = std::prev(segment.end());
    int scanned = 0;

    for (auto it = std::prev(segment.end()); scanned < kScanWindow; ++scanned) {
        // Prefer higher chunk_index (suffix chunks evicted first)
        if (it->chunk_index > best->chunk_index) {
            best = it;
        } else if (it->chunk_index == best->chunk_index &&
                   it->recompute_cost < best->recompute_cost) {
            best = it;
        }

        if (it == segment.begin()) break;
        --it;
    }
    return best;
}

void CwSlru::demote_protected_overflow() {
    size_t max_protected = 0;
    if (config_.max_entries > 0) {
        max_protected = static_cast<size_t>(config_.max_entries * config_.protected_ratio);
    } else {
        // Use the total tracked entries as the basis
        size_t total = protected_.size() + probationary_.size();
        max_protected = static_cast<size_t>(total * config_.protected_ratio);
    }

    while (protected_.size() > max_protected && !protected_.empty()) {
        // Demote LRU entry from Protected to Probationary head
        EvictionEntry entry = protected_.back();
        protected_map_.erase(entry.key);
        protected_.pop_back();

        probationary_.push_front(entry);
        probationary_map_[entry.key] = probationary_.begin();
    }
}

size_t CwSlru::protected_size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return protected_.size();
}

size_t CwSlru::probationary_size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return probationary_.size();
}

size_t CwSlru::total_size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return protected_.size() + probationary_.size();
}

}  // namespace attentiondb
