#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "core/config.h"
#include "core/index_entry.h"
#include "core/status.h"
#include "core/storage_key.h"
#include "index/concurrent_hashmap.h"

namespace attentiondb {

static constexpr uint32_t kCheckpointMagic = 0x41444249;  // "ADBI" (index)
static constexpr uint32_t kCheckpointVersion = 1;

#pragma pack(push, 1)
struct CheckpointFileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t entry_count;
    uint64_t timestamp;
    uint32_t checksum;        // XXH32 of all entries
    uint8_t  _reserved[36];   // Pad to 64 bytes
};
#pragma pack(pop)

static_assert(sizeof(CheckpointFileHeader) == 64);

class CheckpointManager {
public:
    explicit CheckpointManager(const CheckpointConfig& config,
                                const std::string& checkpoint_dir);
    ~CheckpointManager();

    CheckpointManager(const CheckpointManager&) = delete;
    CheckpointManager& operator=(const CheckpointManager&) = delete;

    // Save the current index to a checkpoint file.
    Status save(const ConcurrentHashMap& index);

    // Load the most recent checkpoint file into the index.
    Status load(ConcurrentHashMap& index);

    // Start periodic checkpoint thread.
    void start_periodic(ConcurrentHashMap& index);
    void stop();

    // Mark that the index has been modified since last checkpoint.
    void mark_dirty() { dirty_.store(true, std::memory_order_relaxed); }

    uint64_t last_checkpoint_entries() const { return last_entries_.load(); }
    uint64_t last_checkpoint_duration_ms() const { return last_duration_ms_.load(); }

private:
    std::string checkpoint_path() const;
    std::string checkpoint_tmp_path() const;
    void periodic_thread_fn(ConcurrentHashMap& index);

    CheckpointConfig config_;
    std::string dir_;

    std::thread periodic_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> dirty_{false};

    std::atomic<uint64_t> last_entries_{0};
    std::atomic<uint64_t> last_duration_ms_{0};
};

}  // namespace attentiondb
