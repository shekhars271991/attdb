#include "memory/slab_allocator.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef __linux__
#include <sys/mman.h>
#endif

#ifdef ATTENTIONDB_CUDA
#include <cuda_runtime.h>
#endif

namespace attentiondb {

SlabAllocator::SlabAllocator(const DramCacheConfig& config)
    : use_hugepages_(config.hugepages),
      use_pinned_(config.pinned_memory) {
    init_classes(config);
}

SlabAllocator::~SlabAllocator() {
    for (size_t i = 0; i < backing_allocations_.size(); ++i) {
        free_backing(backing_allocations_[i], classes_[i].slot_size * classes_[i].num_slots);
    }
}

void SlabAllocator::init_classes(const DramCacheConfig& config) {
    auto size_classes = config.size_classes;
    std::sort(size_classes.begin(), size_classes.end());

    if (size_classes.empty()) {
        throw std::invalid_argument("No size classes configured");
    }

    // Distribute total budget across size classes proportionally, favoring
    // the smaller classes which are more frequently used for KV cache blobs.
    // Weight: smaller classes get proportionally more slots.
    size_t total = config.size_bytes;
    size_t num_classes = size_classes.size();

    // Simple strategy: equal bytes per class
    size_t bytes_per_class = total / num_classes;

    classes_.resize(num_classes);
    backing_allocations_.resize(num_classes);

    for (size_t i = 0; i < num_classes; ++i) {
        auto& sc = classes_[i];
        sc.slot_size = size_classes[i];
        sc.num_slots = bytes_per_class / sc.slot_size;
        if (sc.num_slots == 0) sc.num_slots = 1;

        size_t region_size = sc.slot_size * sc.num_slots;
        sc.base = static_cast<uint8_t*>(alloc_backing(region_size));
        backing_allocations_[i] = sc.base;
        total_bytes_ += region_size;

        // Initialize free list with all slots
        sc.free_list.resize(sc.num_slots);
        std::iota(sc.free_list.begin(), sc.free_list.end(), 0);
    }
}

void* SlabAllocator::alloc_backing(size_t bytes) {
#ifdef ATTENTIONDB_CUDA
    if (use_pinned_) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, bytes);
        if (err == cudaSuccess) return ptr;
        // Fall through to non-CUDA path
    }
#endif

#ifdef __linux__
    if (use_hugepages_) {
        void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (ptr != MAP_FAILED) return ptr;
        // Fall through to regular allocation
    }
#endif

    void* ptr = std::aligned_alloc(4096, bytes);
    if (!ptr) throw std::bad_alloc();
    std::memset(ptr, 0, bytes);
    return ptr;
}

void SlabAllocator::free_backing(void* ptr, size_t bytes) {
    if (!ptr) return;

#ifdef ATTENTIONDB_CUDA
    if (use_pinned_) {
        cudaFreeHost(ptr);
        return;
    }
#endif

#ifdef __linux__
    if (use_hugepages_) {
        munmap(ptr, bytes);
        return;
    }
#endif

    std::free(ptr);
}

SlabHandle SlabAllocator::allocate(size_t size, void** ptr) {
    // Find the smallest size class that fits
    for (uint32_t i = 0; i < classes_.size(); ++i) {
        if (classes_[i].slot_size >= size) {
            std::lock_guard<std::mutex> lock(classes_[i].mu);
            if (classes_[i].free_list.empty()) continue;

            uint32_t slot = classes_[i].free_list.back();
            classes_[i].free_list.pop_back();
            *ptr = classes_[i].base + (static_cast<size_t>(slot) * classes_[i].slot_size);
            return {i, slot};
        }
    }

    *ptr = nullptr;
    return SlabHandle::Invalid();
}

void SlabAllocator::free(SlabHandle handle) {
    if (!handle.valid()) return;
    assert(handle.class_index < classes_.size());

    auto& sc = classes_[handle.class_index];
    assert(handle.slot_index < sc.num_slots);

    std::lock_guard<std::mutex> lock(sc.mu);
    sc.free_list.push_back(handle.slot_index);
}

void* SlabAllocator::get_ptr(SlabHandle handle) const {
    if (!handle.valid()) return nullptr;
    assert(handle.class_index < classes_.size());
    const auto& sc = classes_[handle.class_index];
    return sc.base + (static_cast<size_t>(handle.slot_index) * sc.slot_size);
}

size_t SlabAllocator::slot_size_for(size_t size) const {
    for (const auto& sc : classes_) {
        if (sc.slot_size >= size) return sc.slot_size;
    }
    return 0;
}

size_t SlabAllocator::used_bytes() const {
    size_t total = 0;
    for (const auto& sc : classes_) {
        std::lock_guard<std::mutex> lock(sc.mu);
        size_t used_slots = sc.num_slots - sc.free_list.size();
        total += used_slots * sc.slot_size;
    }
    return total;
}

SlabAllocator::Stats SlabAllocator::stats() const {
    Stats s;
    s.total_bytes = total_bytes_;
    s.used_bytes = 0;
    s.num_classes = classes_.size();

    for (const auto& sc : classes_) {
        std::lock_guard<std::mutex> lock(sc.mu);
        size_t used = sc.num_slots - sc.free_list.size();
        s.classes.push_back({sc.slot_size, sc.num_slots, used});
        s.used_bytes += used * sc.slot_size;
    }
    return s;
}

}  // namespace attentiondb
