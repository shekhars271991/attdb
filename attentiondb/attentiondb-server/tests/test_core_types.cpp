#include <gtest/gtest.h>

#include "core/index_entry.h"
#include "core/storage_key.h"

using namespace attentiondb;

TEST(StorageKeyTest, SizeIs30Bytes) {
    EXPECT_EQ(sizeof(StorageKey), 30);
}

TEST(StorageKeyTest, Equality) {
    StorageKey a{1, 2, 3, 4, 5};
    StorageKey b{1, 2, 3, 4, 5};
    StorageKey c{1, 2, 3, 4, 6};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(StorageKeyTest, HashDeterministic) {
    StorageKey a{1, 2, 3, 4, 5};
    StorageKey b{1, 2, 3, 4, 5};

    StorageKeyHash hasher;
    EXPECT_EQ(hasher(a), hasher(b));
}

TEST(StorageKeyTest, HashDistribution) {
    StorageKeyHash hasher;
    StorageKey a{1, 0, 0, 0, 0};
    StorageKey b{2, 0, 0, 0, 0};
    // Different keys should (very likely) produce different hashes
    EXPECT_NE(hasher(a), hasher(b));
}

TEST(StorageKeyTest, StdHash) {
    StorageKey k{1, 2, 3, 4, 5};
    std::hash<StorageKey> h;
    EXPECT_NE(h(k), 0u);
}

TEST(IndexEntryTest, SizeIs64Bytes) {
    EXPECT_EQ(sizeof(IndexEntry), 64);
}

TEST(IndexEntryTest, AlignmentIs64) {
    EXPECT_EQ(alignof(IndexEntry), 64);
}

TEST(IndexEntryTest, Timestamp48Bit) {
    IndexEntry e{};
    uint64_t ts = 0x0000FFFFFFFFFFFF;  // Max 48-bit value
    e.set_last_access_us(ts);
    EXPECT_EQ(e.get_last_access_us(), ts);

    ts = 1234567890123ULL;
    e.set_created_at_us(ts);
    EXPECT_EQ(e.get_created_at_us(), ts);
}

TEST(IndexEntryTest, Touch) {
    IndexEntry e{};
    e.access_count = 0;
    e.touch();
    EXPECT_EQ(e.access_count, 1);
    EXPECT_GT(e.get_last_access_us(), 0u);
}

TEST(IndexEntryTest, AccessCountSaturates) {
    IndexEntry e{};
    e.access_count = UINT16_MAX;
    e.touch();
    EXPECT_EQ(e.access_count, UINT16_MAX);
}
