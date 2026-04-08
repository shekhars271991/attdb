#include <gtest/gtest.h>

#include <cstring>

#include "memory/slab_allocator.h"

using namespace attentiondb;

static DramCacheConfig small_config() {
    DramCacheConfig cfg;
    cfg.enabled = true;
    cfg.size_bytes = 32 * 1024 * 1024;  // 32 MB
    cfg.size_classes = {4096, 8192, 16384, 65536, 262144};
    cfg.hugepages = false;
    cfg.pinned_memory = false;
    return cfg;
}

TEST(SlabAllocatorTest, AllocateAndFree) {
    auto cfg = small_config();
    SlabAllocator alloc(cfg);

    void* ptr = nullptr;
    auto handle = alloc.allocate(4096, &ptr);
    EXPECT_TRUE(handle.valid());
    EXPECT_NE(ptr, nullptr);

    // Write and verify
    std::memset(ptr, 0xAB, 4096);
    EXPECT_EQ(static_cast<uint8_t*>(ptr)[0], 0xAB);

    alloc.free(handle);
}

TEST(SlabAllocatorTest, AllocateBestFit) {
    auto cfg = small_config();
    SlabAllocator alloc(cfg);

    // Request 5000 bytes, should get 8192 slot
    void* ptr = nullptr;
    auto handle = alloc.allocate(5000, &ptr);
    EXPECT_TRUE(handle.valid());
    EXPECT_EQ(alloc.slot_size_for(5000), 8192u);

    alloc.free(handle);
}

TEST(SlabAllocatorTest, GetPtr) {
    auto cfg = small_config();
    SlabAllocator alloc(cfg);

    void* ptr = nullptr;
    auto handle = alloc.allocate(4096, &ptr);
    EXPECT_EQ(alloc.get_ptr(handle), ptr);

    alloc.free(handle);
}

TEST(SlabAllocatorTest, MultipleAllocations) {
    auto cfg = small_config();
    SlabAllocator alloc(cfg);

    std::vector<SlabHandle> handles;
    for (int i = 0; i < 100; ++i) {
        void* ptr = nullptr;
        auto h = alloc.allocate(4096, &ptr);
        EXPECT_TRUE(h.valid());
        handles.push_back(h);
    }

    EXPECT_GT(alloc.used_bytes(), 0u);

    for (auto& h : handles) {
        alloc.free(h);
    }
}

TEST(SlabAllocatorTest, Stats) {
    auto cfg = small_config();
    SlabAllocator alloc(cfg);

    auto s = alloc.stats();
    EXPECT_EQ(s.num_classes, 5u);
    EXPECT_GT(s.total_bytes, 0u);
    EXPECT_EQ(s.used_bytes, 0u);

    void* ptr = nullptr;
    auto h = alloc.allocate(4096, &ptr);

    s = alloc.stats();
    EXPECT_GT(s.used_bytes, 0u);

    alloc.free(h);
}

TEST(SlabAllocatorTest, InvalidHandleFreeIsSafe) {
    auto cfg = small_config();
    SlabAllocator alloc(cfg);
    alloc.free(SlabHandle::Invalid());  // Should not crash
}

TEST(SlabAllocatorTest, TooLargeReturnsInvalid) {
    auto cfg = small_config();
    SlabAllocator alloc(cfg);

    void* ptr = nullptr;
    auto h = alloc.allocate(1024 * 1024, &ptr);  // 1 MB, larger than max class 262144
    EXPECT_FALSE(h.valid());
    EXPECT_EQ(ptr, nullptr);
}

TEST(SlabAllocatorTest, SlotSizeForEdgeCases) {
    auto cfg = small_config();
    SlabAllocator alloc(cfg);

    EXPECT_EQ(alloc.slot_size_for(1), 4096u);
    EXPECT_EQ(alloc.slot_size_for(4096), 4096u);
    EXPECT_EQ(alloc.slot_size_for(4097), 8192u);
    EXPECT_EQ(alloc.slot_size_for(262144), 262144u);
    EXPECT_EQ(alloc.slot_size_for(262145), 0u);
}
