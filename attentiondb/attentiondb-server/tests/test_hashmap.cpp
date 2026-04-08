#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "index/concurrent_hashmap.h"

using namespace attentiondb;

static StorageKey make_key(uint64_t id) {
    return StorageKey{id, 0, 0, 0, 0};
}

TEST(ConcurrentHashMapTest, InsertAndFind) {
    ConcurrentHashMap map(1024);

    StorageKey key = make_key(1);
    IndexEntry entry{};
    entry.length = 42;

    map.upsert(key, entry);

    IndexEntry found{};
    EXPECT_TRUE(map.find(key, found));
    EXPECT_EQ(found.length, 42u);
}

TEST(ConcurrentHashMapTest, NotFound) {
    ConcurrentHashMap map(1024);
    IndexEntry found{};
    EXPECT_FALSE(map.find(make_key(99), found));
}

TEST(ConcurrentHashMapTest, Erase) {
    ConcurrentHashMap map(1024);
    StorageKey key = make_key(1);
    IndexEntry entry{};
    map.upsert(key, entry);

    EXPECT_TRUE(map.erase(key));
    EXPECT_FALSE(map.contains(key));
}

TEST(ConcurrentHashMapTest, Contains) {
    ConcurrentHashMap map(1024);
    StorageKey key = make_key(1);
    IndexEntry entry{};
    map.upsert(key, entry);

    EXPECT_TRUE(map.contains(key));
    EXPECT_FALSE(map.contains(make_key(2)));
}

TEST(ConcurrentHashMapTest, Size) {
    ConcurrentHashMap map(1024);
    EXPECT_EQ(map.size(), 0u);

    for (uint64_t i = 0; i < 100; ++i) {
        IndexEntry entry{};
        map.upsert(make_key(i), entry);
    }
    EXPECT_EQ(map.size(), 100u);
}

TEST(ConcurrentHashMapTest, UpdateFn) {
    ConcurrentHashMap map(1024);
    StorageKey key = make_key(1);
    IndexEntry entry{};
    entry.length = 10;
    map.upsert(key, entry);

    bool updated = map.update_fn(key, [](IndexEntry& e) { e.length = 20; });
    EXPECT_TRUE(updated);

    IndexEntry found{};
    map.find(key, found);
    EXPECT_EQ(found.length, 20u);
}

TEST(ConcurrentHashMapTest, Upsert) {
    ConcurrentHashMap map(1024);
    StorageKey key = make_key(1);

    IndexEntry e1{};
    e1.length = 10;
    map.upsert(key, e1);

    IndexEntry e2{};
    e2.length = 20;
    map.upsert(key, e2);

    IndexEntry found{};
    map.find(key, found);
    EXPECT_EQ(found.length, 20u);
    EXPECT_EQ(map.size(), 1u);
}

TEST(ConcurrentHashMapTest, SnapshotAndBulkLoad) {
    ConcurrentHashMap map(1024);
    for (uint64_t i = 0; i < 50; ++i) {
        IndexEntry entry{};
        entry.length = static_cast<uint32_t>(i);
        map.upsert(make_key(i), entry);
    }

    auto snapshot = map.snapshot();
    EXPECT_EQ(snapshot.size(), 50u);

    ConcurrentHashMap map2(1024);
    map2.bulk_load(snapshot);
    EXPECT_EQ(map2.size(), 50u);

    for (uint64_t i = 0; i < 50; ++i) {
        IndexEntry found{};
        EXPECT_TRUE(map2.find(make_key(i), found));
        EXPECT_EQ(found.length, static_cast<uint32_t>(i));
    }
}

TEST(ConcurrentHashMapTest, ConcurrentInserts) {
    ConcurrentHashMap map(1 << 16);
    constexpr int kThreads = 4;
    constexpr int kPerThread = 1000;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&map, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                uint64_t id = t * kPerThread + i;
                IndexEntry entry{};
                entry.length = static_cast<uint32_t>(id);
                map.upsert(make_key(id), entry);
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(map.size(), static_cast<size_t>(kThreads * kPerThread));
}

TEST(ConcurrentHashMapTest, Iterate) {
    ConcurrentHashMap map(1024);
    for (uint64_t i = 0; i < 10; ++i) {
        IndexEntry entry{};
        entry.length = static_cast<uint32_t>(i);
        map.upsert(make_key(i), entry);
    }

    int count = 0;
    map.iterate([&](const StorageKey&, const IndexEntry&) {
        ++count;
        return true;
    });
    EXPECT_EQ(count, 10);
}

TEST(ConcurrentHashMapTest, Clear) {
    ConcurrentHashMap map(1024);
    for (uint64_t i = 0; i < 10; ++i) {
        IndexEntry entry{};
        map.upsert(make_key(i), entry);
    }
    map.clear();
    EXPECT_EQ(map.size(), 0u);
}
