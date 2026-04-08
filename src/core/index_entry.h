#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>

namespace attentiondb {

enum class Tier : uint8_t {
    kT1Dram = 1,
    kT2Nvme = 2,
};

enum class EntryType : uint8_t {
    kConversation = 0,
    kSystemPrompt = 1,
    kRagContext = 2,
};

enum class SlruSegment : uint8_t {
    kProbationary = 0,
    kProtected = 1,
};

#pragma pack(push, 1)
struct alignas(64) IndexEntry {
    // Storage location (19 bytes)
    uint8_t  tier;           // Tier enum
    uint64_t address;        // DRAM pointer or NVMe byte offset
    uint32_t length;         // Blob size in bytes
    uint32_t segment_id;     // Log segment file ID (T2 only, 0 for T1)
    uint16_t flags;          // Codec hint (opaque), entry state

    // Eviction metadata (23 bytes)
    uint32_t recompute_cost; // From LMCache at put() time
    uint8_t  last_access[6]; // 48-bit timestamp (microseconds since epoch)
    uint16_t access_count;   // Saturating counter
    uint8_t  slru_segment;   // SlruSegment enum
    uint32_t num_tokens;     // From LMCache at put() time
    uint8_t  created_at[6];  // 48-bit timestamp

    // Integrity (4 bytes)
    uint32_t checksum;       // CRC32C of blob data

    // Padding (18 bytes → total 64)
    uint8_t  _reserved[18];

    void set_last_access_us(uint64_t us) {
        std::memcpy(last_access, &us, 6);
    }

    uint64_t get_last_access_us() const {
        uint64_t val = 0;
        std::memcpy(&val, last_access, 6);
        return val;
    }

    void set_created_at_us(uint64_t us) {
        std::memcpy(created_at, &us, 6);
    }

    uint64_t get_created_at_us() const {
        uint64_t val = 0;
        std::memcpy(&val, created_at, 6);
        return val;
    }

    void touch() {
        auto now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        set_last_access_us(static_cast<uint64_t>(now));
        if (access_count < UINT16_MAX) {
            ++access_count;
        }
    }
};
#pragma pack(pop)

static_assert(sizeof(IndexEntry) == 64,
              "IndexEntry must be exactly 64 bytes (one cache line)");
static_assert(alignof(IndexEntry) == 64,
              "IndexEntry must be 64-byte aligned");

inline uint64_t now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace attentiondb
