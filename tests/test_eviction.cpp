#include <gtest/gtest.h>

#include "eviction/cw_slru.h"

using namespace attentiondb;

static StorageKey make_key(uint64_t id) {
    return StorageKey{id, 0, 0, 0, 0};
}

static StorageKey make_key_with_chunk(uint64_t id, uint32_t chunk_idx) {
    return StorageKey{id, 0, 0, 0, chunk_idx};
}

TEST(CwSlruTest, InsertAndEvict) {
    CwSlru::Config cfg;
    cfg.protected_ratio = 0.25;
    cfg.protection_threshold = 100;
    CwSlru slru(cfg);

    slru.on_insert(make_key(1), 50, 0);
    slru.on_insert(make_key(2), 50, 0);
    slru.on_insert(make_key(3), 50, 0);

    EXPECT_EQ(slru.total_size(), 3u);

    auto victim = slru.evict();
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(slru.total_size(), 2u);
}

TEST(CwSlruTest, PromotionToProtected) {
    CwSlru::Config cfg;
    cfg.protected_ratio = 0.5;
    cfg.protection_threshold = 100;
    CwSlru slru(cfg);

    slru.on_insert(make_key(1), 50, 0);
    slru.on_insert(make_key(2), 200, 0);  // High cost

    // Access key 2 with high cost → should promote to protected
    slru.on_access(make_key(2), 200, 0);

    EXPECT_EQ(slru.protected_size(), 1u);
    EXPECT_EQ(slru.probationary_size(), 1u);
}

TEST(CwSlruTest, LowCostStaysInProbationary) {
    CwSlru::Config cfg;
    cfg.protected_ratio = 0.5;
    cfg.protection_threshold = 100;
    CwSlru slru(cfg);

    slru.on_insert(make_key(1), 50, 0);  // Low cost
    slru.on_access(make_key(1), 50, 0);  // Access but still low cost

    EXPECT_EQ(slru.protected_size(), 0u);
    EXPECT_EQ(slru.probationary_size(), 1u);
}

TEST(CwSlruTest, SuffixBeforePrefix) {
    CwSlru::Config cfg;
    cfg.protected_ratio = 0.0;  // All probationary
    cfg.protection_threshold = 1000;
    CwSlru slru(cfg);

    // Insert entries with different chunk indices
    slru.on_insert(make_key_with_chunk(1, 0), 50, 0);   // Prefix (chunk 0)
    slru.on_insert(make_key_with_chunk(2, 10), 50, 10);  // Suffix (chunk 10)
    slru.on_insert(make_key_with_chunk(3, 5), 50, 5);    // Middle (chunk 5)

    // Eviction should prefer suffix (highest chunk_index) first
    auto victim = slru.evict();
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->chunk_index, 10u);
}

TEST(CwSlruTest, RemoveKey) {
    CwSlru::Config cfg;
    CwSlru slru(cfg);

    slru.on_insert(make_key(1), 50, 0);
    slru.on_insert(make_key(2), 50, 0);

    slru.on_remove(make_key(1));
    EXPECT_EQ(slru.total_size(), 1u);
}

TEST(CwSlruTest, EvictEmpty) {
    CwSlru::Config cfg;
    CwSlru slru(cfg);

    auto victim = slru.evict();
    EXPECT_FALSE(victim.has_value());
}

TEST(CwSlruTest, ProtectedOverflowDemotes) {
    CwSlru::Config cfg;
    cfg.protected_ratio = 0.1;
    cfg.protection_threshold = 50;
    cfg.max_entries = 10;
    CwSlru slru(cfg);

    // Insert many high-cost entries that all get promoted
    for (uint64_t i = 0; i < 10; ++i) {
        slru.on_insert(make_key(i), 200, 0);
        slru.on_access(make_key(i), 200, 0);
    }

    // Protected should not exceed max_entries * protected_ratio = 1
    EXPECT_LE(slru.protected_size(), 2u);
    EXPECT_GE(slru.probationary_size(), 8u);
}
