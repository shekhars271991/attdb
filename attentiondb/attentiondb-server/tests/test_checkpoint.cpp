#include <gtest/gtest.h>

#include <filesystem>

#include "checkpoint/checkpoint.h"

using namespace attentiondb;

class CheckpointTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/attentiondb_test_ckpt_" + std::to_string(getpid());
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    std::string test_dir_;
};

TEST_F(CheckpointTest, SaveAndLoad) {
    CheckpointConfig cfg;
    cfg.interval_s = 3600;  // Don't auto-run
    CheckpointManager mgr(cfg, test_dir_);

    // Create and populate an index
    ConcurrentHashMap index(1024);
    for (uint64_t i = 0; i < 100; ++i) {
        StorageKey key{i, 0, 0, 0, 0};
        IndexEntry entry{};
        entry.length = static_cast<uint32_t>(i * 10);
        entry.recompute_cost = static_cast<uint32_t>(i);
        index.upsert(key, entry);
    }

    mgr.mark_dirty();
    EXPECT_EQ(mgr.save(index), Status::kOk);
    EXPECT_EQ(mgr.last_checkpoint_entries(), 100u);

    // Load into a new index
    ConcurrentHashMap index2(1024);
    EXPECT_EQ(mgr.load(index2), Status::kOk);
    EXPECT_EQ(index2.size(), 100u);

    // Verify data
    for (uint64_t i = 0; i < 100; ++i) {
        StorageKey key{i, 0, 0, 0, 0};
        IndexEntry found{};
        EXPECT_TRUE(index2.find(key, found));
        EXPECT_EQ(found.length, static_cast<uint32_t>(i * 10));
        EXPECT_EQ(found.recompute_cost, static_cast<uint32_t>(i));
    }
}

TEST_F(CheckpointTest, LoadNonExistent) {
    CheckpointConfig cfg;
    CheckpointManager mgr(cfg, test_dir_);

    ConcurrentHashMap index(1024);
    EXPECT_EQ(mgr.load(index), Status::kNotFound);
}

TEST_F(CheckpointTest, EmptyIndex) {
    CheckpointConfig cfg;
    CheckpointManager mgr(cfg, test_dir_);

    ConcurrentHashMap index(1024);
    mgr.mark_dirty();
    EXPECT_EQ(mgr.save(index), Status::kOk);
}

TEST_F(CheckpointTest, DirtyTracking) {
    CheckpointConfig cfg;
    CheckpointManager mgr(cfg, test_dir_);

    ConcurrentHashMap index(1024);

    // Not dirty → save is a no-op
    EXPECT_EQ(mgr.save(index), Status::kOk);

    // Mark dirty and save
    StorageKey key{1, 0, 0, 0, 0};
    IndexEntry entry{};
    index.upsert(key, entry);
    mgr.mark_dirty();

    EXPECT_EQ(mgr.save(index), Status::kOk);
    EXPECT_EQ(mgr.last_checkpoint_entries(), 1u);
}
