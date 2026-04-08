#pragma once

#include <cstdint>
#include <cstring>
#include <functional>

#include <xxhash.h>

namespace attentiondb {

#pragma pack(push, 1)
struct StorageKey {
    uint64_t model_id;
    uint64_t tenant_id;
    uint64_t chunk_hash;
    uint16_t layer_group_id;
    uint32_t chunk_index;

    bool operator==(const StorageKey& other) const {
        return std::memcmp(this, &other, sizeof(StorageKey)) == 0;
    }

    bool operator!=(const StorageKey& other) const { return !(*this == other); }
};
#pragma pack(pop)

static_assert(sizeof(StorageKey) == 30,
              "StorageKey must be 30 bytes packed (8+8+8+2+4)");

struct StorageKeyHash {
    std::size_t operator()(const StorageKey& key) const {
        return static_cast<std::size_t>(
            XXH3_64bits(&key, sizeof(StorageKey)));
    }
};

struct StorageKeyEqual {
    bool operator()(const StorageKey& a, const StorageKey& b) const {
        return std::memcmp(&a, &b, sizeof(StorageKey)) == 0;
    }
};

}  // namespace attentiondb

namespace std {
template <>
struct hash<attentiondb::StorageKey> {
    std::size_t operator()(const attentiondb::StorageKey& key) const {
        return attentiondb::StorageKeyHash{}(key);
    }
};
}  // namespace std
