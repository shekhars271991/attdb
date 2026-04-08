#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <thread>

#include "engine.h"

using namespace attentiondb;

class EngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/attentiondb_test_engine_" + std::to_string(getpid());
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    Config test_config() {
        Config cfg;
        cfg.mode = "local_only";

        cfg.dram_cache.enabled = true;
        cfg.dram_cache.size_bytes = 4 * 1024 * 1024;  // 4 MB
        cfg.dram_cache.size_classes = {4096, 8192, 16384, 65536};
        cfg.dram_cache.hugepages = false;
        cfg.dram_cache.pinned_memory = false;

        cfg.nvme_store.enabled = true;
        cfg.nvme_store.paths = {test_dir_};
        cfg.nvme_store.segment_size = 1024 * 1024;
        cfg.nvme_store.io_engine = "posix";
        cfg.nvme_store.gc_enabled = false;

        cfg.write_buffer.size_bytes = 1 * 1024 * 1024;
        cfg.write_buffer.flush_interval_ms = 50;

        cfg.admission.min_recompute_cost = 0;  // Accept everything
        cfg.admission.base_threshold = 1000;

        cfg.checkpoint.interval_s = 3600;  // Don't auto-checkpoint

        cfg.logging.level = "warn";
        cfg.logging.periodic_summary_interval_s = 0;

        return cfg;
    }

    std::string test_dir_;
};

TEST_F(EngineTest, OpenAndClose) {
    AttentionDBEngine engine;
    EXPECT_EQ(engine.open(test_config()), Status::kOk);
    engine.close();
}

TEST_F(EngineTest, PutAndGet) {
    AttentionDBEngine engine;
    ASSERT_EQ(engine.open(test_config()), Status::kOk);

    StorageKey key{1, 2, 3, 4, 5};
    std::vector<uint8_t> data(1024, 0xAB);
    PutOpts opts;
    opts.num_tokens = 256;
    opts.recompute_cost = 256;
    opts.entry_type = EntryType::kConversation;

    EXPECT_EQ(engine.put(key, data.data(), data.size(), opts), Status::kOk);

    // Wait for flush to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::vector<uint8_t> buf(4096);
    size_t out_len = 0;
    Status s = engine.get(key, buf.data(), buf.size(), &out_len);
    EXPECT_EQ(s, Status::kOk);
    EXPECT_EQ(out_len, 1024u);
    EXPECT_EQ(buf[0], 0xAB);

    engine.close();
}

TEST_F(EngineTest, Contains) {
    AttentionDBEngine engine;
    ASSERT_EQ(engine.open(test_config()), Status::kOk);

    StorageKey key{10, 0, 0, 0, 0};
    EXPECT_FALSE(engine.contains(key));

    std::vector<uint8_t> data(512, 0xFF);
    PutOpts opts;
    opts.recompute_cost = 100;
    engine.put(key, data.data(), data.size(), opts);

    // Wait for flush
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_TRUE(engine.contains(key));

    engine.close();
}

TEST_F(EngineTest, Delete) {
    AttentionDBEngine engine;
    ASSERT_EQ(engine.open(test_config()), Status::kOk);

    StorageKey key{20, 0, 0, 0, 0};
    std::vector<uint8_t> data(512, 0xFF);
    PutOpts opts;
    opts.recompute_cost = 100;
    engine.put(key, data.data(), data.size(), opts);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_TRUE(engine.contains(key));

    EXPECT_EQ(engine.del(key), Status::kOk);
    EXPECT_FALSE(engine.contains(key));

    engine.close();
}

TEST_F(EngineTest, GetMissReturnsNotFound) {
    AttentionDBEngine engine;
    ASSERT_EQ(engine.open(test_config()), Status::kOk);

    StorageKey key{999, 0, 0, 0, 0};
    std::vector<uint8_t> buf(4096);
    size_t out_len = 0;
    EXPECT_EQ(engine.get(key, buf.data(), buf.size(), &out_len), Status::kNotFound);

    engine.close();
}

TEST_F(EngineTest, Stats) {
    AttentionDBEngine engine;
    ASSERT_EQ(engine.open(test_config()), Status::kOk);

    auto s = engine.stats();
    EXPECT_GT(s.t1_total_bytes, 0u);
    EXPECT_EQ(s.index_entries, 0u);

    engine.close();
}

TEST_F(EngineTest, MultiplePutsAndGets) {
    AttentionDBEngine engine;
    ASSERT_EQ(engine.open(test_config()), Status::kOk);

    constexpr int kCount = 20;
    PutOpts opts;
    opts.recompute_cost = 500;

    for (int i = 0; i < kCount; ++i) {
        StorageKey key{static_cast<uint64_t>(i), 0, 0, 0, 0};
        std::vector<uint8_t> data(256, static_cast<uint8_t>(i));
        engine.put(key, data.data(), data.size(), opts);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    for (int i = 0; i < kCount; ++i) {
        StorageKey key{static_cast<uint64_t>(i), 0, 0, 0, 0};
        std::vector<uint8_t> buf(4096);
        size_t out_len = 0;
        Status s = engine.get(key, buf.data(), buf.size(), &out_len);
        EXPECT_EQ(s, Status::kOk);
        EXPECT_EQ(out_len, 256u);
        EXPECT_EQ(buf[0], static_cast<uint8_t>(i));
    }

    engine.close();
}

TEST_F(EngineTest, AdmissionRejectsLowCost) {
    auto cfg = test_config();
    cfg.admission.min_recompute_cost = 100;

    AttentionDBEngine engine;
    ASSERT_EQ(engine.open(cfg), Status::kOk);

    StorageKey key{1, 0, 0, 0, 0};
    std::vector<uint8_t> data(512, 0xFF);
    PutOpts opts;
    opts.recompute_cost = 10;  // Below min

    EXPECT_EQ(engine.put(key, data.data(), data.size(), opts),
              Status::kRejectedAdmission);

    engine.close();
}

TEST_F(EngineTest, SystemPromptBypassesAdmission) {
    auto cfg = test_config();
    cfg.admission.min_recompute_cost = 1000;

    AttentionDBEngine engine;
    ASSERT_EQ(engine.open(cfg), Status::kOk);

    StorageKey key{1, 0, 0, 0, 0};
    std::vector<uint8_t> data(512, 0xFF);
    PutOpts opts;
    opts.recompute_cost = 0;
    opts.entry_type = EntryType::kSystemPrompt;

    EXPECT_EQ(engine.put(key, data.data(), data.size(), opts), Status::kOk);

    engine.close();
}
