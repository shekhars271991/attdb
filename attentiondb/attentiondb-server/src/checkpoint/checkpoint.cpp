#include "checkpoint/checkpoint.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <xxhash.h>

namespace attentiondb {

CheckpointManager::CheckpointManager(const CheckpointConfig& config,
                                       const std::string& checkpoint_dir)
    : config_(config), dir_(checkpoint_dir) {
    std::filesystem::create_directories(dir_);
}

CheckpointManager::~CheckpointManager() {
    stop();
}

Status CheckpointManager::save(const ConcurrentHashMap& index) {
    auto start = std::chrono::steady_clock::now();

    auto entries = index.snapshot();
    if (entries.empty() && !dirty_.load()) return Status::kOk;

    std::string tmp_path = checkpoint_tmp_path();
    std::string final_path = checkpoint_path();

    std::ofstream out(tmp_path, std::ios::binary);
    if (!out.is_open()) return Status::kIOError;

    // Write header placeholder
    CheckpointFileHeader hdr{};
    hdr.magic = kCheckpointMagic;
    hdr.version = kCheckpointVersion;
    hdr.entry_count = entries.size();
    hdr.timestamp = static_cast<uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());

    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) {
                  return std::memcmp(&a.first, &b.first, sizeof(StorageKey)) < 0;
              });

    XXH32_state_t* hash_state = XXH32_createState();
    if (!hash_state) return Status::kIOError;
    XXH32_reset(hash_state, 0);

    for (const auto& [key, entry] : entries) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(StorageKey));
        out.write(reinterpret_cast<const char*>(&entry), sizeof(IndexEntry));
        if (!out.good()) {
            XXH32_freeState(hash_state);
            return Status::kIOError;
        }
        XXH32_update(hash_state, &key, sizeof(StorageKey));
        XXH32_update(hash_state, &entry, sizeof(IndexEntry));
    }

    hdr.checksum = XXH32_digest(hash_state);
    XXH32_freeState(hash_state);

    out.seekp(0);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    out.flush();
    if (!out.good()) return Status::kIOError;
    out.close();

    std::filesystem::rename(tmp_path, final_path);

    dirty_.store(false, std::memory_order_relaxed);

    auto elapsed = std::chrono::steady_clock::now() - start;
    last_entries_.store(entries.size());
    last_duration_ms_.store(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

    return Status::kOk;
}

Status CheckpointManager::load(ConcurrentHashMap& index) {
    std::string path = checkpoint_path();
    if (!std::filesystem::exists(path)) return Status::kNotFound;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return Status::kIOError;

    CheckpointFileHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    if (hdr.magic != kCheckpointMagic || hdr.version != kCheckpointVersion) {
        return Status::kCorruption;
    }

    // Sanity-check entry_count against file size to prevent OOM on corrupt files.
    // Each entry is StorageKey + IndexEntry; cap at 100M entries as a safety limit.
    static constexpr uint64_t kMaxEntries = 100'000'000;
    if (hdr.entry_count > kMaxEntries) return Status::kCorruption;

    auto file_size = std::filesystem::file_size(path);
    size_t expected_body = hdr.entry_count * (sizeof(StorageKey) + sizeof(IndexEntry));
    if (sizeof(CheckpointFileHeader) + expected_body > file_size)
        return Status::kCorruption;

    std::vector<std::pair<StorageKey, IndexEntry>> entries;
    entries.reserve(hdr.entry_count);

    XXH32_state_t* hash_state = XXH32_createState();
    if (!hash_state) return Status::kIOError;
    XXH32_reset(hash_state, 0);

    for (uint64_t i = 0; i < hdr.entry_count; ++i) {
        StorageKey key{};
        IndexEntry entry{};
        in.read(reinterpret_cast<char*>(&key), sizeof(StorageKey));
        in.read(reinterpret_cast<char*>(&entry), sizeof(IndexEntry));

        if (!in.good()) {
            XXH32_freeState(hash_state);
            return Status::kCorruption;
        }

        XXH32_update(hash_state, &key, sizeof(StorageKey));
        XXH32_update(hash_state, &entry, sizeof(IndexEntry));
        entries.emplace_back(key, entry);
    }

    uint32_t computed = XXH32_digest(hash_state);
    XXH32_freeState(hash_state);

    if (computed != hdr.checksum) return Status::kCorruption;

    index.bulk_load(entries);
    return Status::kOk;
}

void CheckpointManager::start_periodic(ConcurrentHashMap& index) {
    running_ = true;
    periodic_thread_ = std::thread(&CheckpointManager::periodic_thread_fn,
                                    this, std::ref(index));
}

void CheckpointManager::stop() {
    running_ = false;
    stop_cv_.notify_all();
    if (periodic_thread_.joinable()) periodic_thread_.join();
}

void CheckpointManager::periodic_thread_fn(ConcurrentHashMap& index) {
    while (running_) {
        std::unique_lock<std::mutex> lock(stop_mu_);
        stop_cv_.wait_for(lock, std::chrono::seconds(config_.interval_s),
                          [this] { return !running_.load(); });
        if (!running_) break;
        if (dirty_.load(std::memory_order_relaxed)) {
            save(index);
        }
    }
    // Final checkpoint on shutdown
    if (dirty_.load()) {
        save(index);
    }
}

std::string CheckpointManager::checkpoint_path() const {
    return dir_ + "/index.checkpoint";
}

std::string CheckpointManager::checkpoint_tmp_path() const {
    return dir_ + "/index.checkpoint.tmp";
}

}  // namespace attentiondb
