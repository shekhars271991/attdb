#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "core/config.h"

namespace attentiondb {

// Identifies a slab allocation: which size class and which slot within it.
struct SlabHandle {
    uint32_t class_index;
    uint32_t slot_index;

    bool valid() const { return class_index != UINT32_MAX; }
    static SlabHandle Invalid() { return {UINT32_MAX, UINT32_MAX}; }
};

class SlabAllocator {
public:
    struct Stats {
        size_t total_bytes;
        size_t used_bytes;
        size_t num_classes;
        struct ClassStats {
            size_t slot_size;
            size_t total_slots;
            size_t used_slots;
        };
        std::vector<ClassStats> classes;
    };

    explicit SlabAllocator(const DramCacheConfig& config);
    ~SlabAllocator();

    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    // Allocate a slot large enough for `size` bytes.
    // Returns a handle and sets `ptr` to the allocated memory.
    SlabHandle allocate(size_t size, void** ptr);

    // Free a previously allocated slot.
    void free(SlabHandle handle);

    // Get pointer to the data for a given handle.
    void* get_ptr(SlabHandle handle) const;

    // Find the appropriate size class for a given size.
    // Returns the slot size, or 0 if too large.
    size_t slot_size_for(size_t size) const;

    Stats stats() const;
    size_t total_capacity() const { return total_bytes_; }
    size_t used_bytes() const;

private:
    struct SizeClass {
        size_t slot_size = 0;
        size_t num_slots = 0;
        uint8_t* base = nullptr;
        std::vector<uint32_t> free_list;
        mutable std::mutex mu;

        SizeClass() = default;
        SizeClass(SizeClass&& o) noexcept
            : slot_size(o.slot_size), num_slots(o.num_slots),
              base(o.base), free_list(std::move(o.free_list)) {}
        SizeClass& operator=(SizeClass&&) = delete;
        SizeClass(const SizeClass&) = delete;
        SizeClass& operator=(const SizeClass&) = delete;
    };

    void init_classes(const DramCacheConfig& config);
    void* alloc_backing(size_t bytes);
    void free_backing(void* ptr, size_t bytes);

    std::vector<SizeClass> classes_;
    std::vector<void*> backing_allocations_;
    size_t total_bytes_ = 0;
    bool use_hugepages_;
    bool use_pinned_;
};

}  // namespace attentiondb
