#pragma once

#include <cstdint>
#include <cstring>

namespace attentiondb {

static constexpr uint32_t kSegmentMagic = 0x41444253;  // "ADBS"
static constexpr uint32_t kSegmentVersion = 1;
static constexpr size_t kSegmentHeaderSize = 4096;
static constexpr size_t kEntryHeaderSize = 64;
static constexpr size_t kAlignment = 4096;

#pragma pack(push, 1)
struct SegmentHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t segment_id;
    uint64_t created_at;
    uint32_t entry_count;
    uint32_t live_count;
    uint32_t checksum;
    uint8_t  _reserved[4096 - 28];
};
#pragma pack(pop)

static_assert(sizeof(SegmentHeader) == 4096);

#pragma pack(push, 1)
struct BlobEntryHeader {
    uint32_t key_size;         // Always sizeof(StorageKey)
    uint32_t blob_size;        // Actual blob size
    uint32_t padded_size;      // 4KB-aligned total size (header + key + blob + padding)
    uint32_t checksum;         // CRC32C of blob data
    uint64_t sequence;         // Monotonically increasing write sequence
    uint8_t  key_data[30];     // Inline StorageKey (packed, 30 bytes)
    uint8_t  _reserved[14];
};
#pragma pack(pop)

static_assert(sizeof(BlobEntryHeader) == kEntryHeaderSize);

inline size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

// Total on-disk size for a blob entry including header and alignment padding.
inline size_t entry_disk_size(size_t blob_size) {
    return align_up(kEntryHeaderSize + blob_size, kAlignment);
}

}  // namespace attentiondb
