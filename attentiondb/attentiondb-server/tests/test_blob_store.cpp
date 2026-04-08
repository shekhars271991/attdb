#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include "storage/blob_store.h"

using namespace attentiondb;

class BlobStoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/attentiondb_test_blob_" + std::to_string(getpid());
        std::filesystem::create_directories(test_dir_);

        NvmeStoreConfig cfg;
        cfg.enabled = true;
        cfg.paths = {test_dir_};
        cfg.segment_size = 1024 * 1024;  // 1 MB for testing
        cfg.io_engine = "posix";
        cfg.gc_enabled = false;
        store_ = std::make_unique<BlobStore>(cfg);
    }

    void TearDown() override {
        store_.reset();
        std::filesystem::remove_all(test_dir_);
    }

    std::string test_dir_;
    std::unique_ptr<BlobStore> store_;
};

TEST_F(BlobStoreTest, OpenAndClose) {
    EXPECT_EQ(store_->open(), Status::kOk);
    store_->close();
}

TEST_F(BlobStoreTest, WriteAndRead) {
    ASSERT_EQ(store_->open(), Status::kOk);

    StorageKey key{1, 2, 3, 4, 5};
    std::vector<uint8_t> data(4096, 0xAB);
    BlobLocation loc{};

    EXPECT_EQ(store_->write(key, data.data(), data.size(), 0x12345678, &loc),
              Status::kOk);
    EXPECT_GT(loc.offset, 0u);
    EXPECT_EQ(loc.length, 4096u);

    std::vector<uint8_t> buf(8192);
    size_t bytes_read = 0;
    EXPECT_EQ(store_->read(loc, buf.data(), buf.size(), &bytes_read),
              Status::kOk);
    EXPECT_EQ(bytes_read, 4096u);
    EXPECT_EQ(buf[0], 0xAB);
    EXPECT_EQ(buf[4095], 0xAB);
}

TEST_F(BlobStoreTest, MultipleWrites) {
    ASSERT_EQ(store_->open(), Status::kOk);

    for (int i = 0; i < 10; ++i) {
        StorageKey key{static_cast<uint64_t>(i), 0, 0, 0, 0};
        std::vector<uint8_t> data(1024, static_cast<uint8_t>(i));
        BlobLocation loc{};

        EXPECT_EQ(store_->write(key, data.data(), data.size(), 0, &loc),
                  Status::kOk);

        std::vector<uint8_t> buf(4096);
        size_t bytes_read = 0;
        EXPECT_EQ(store_->read(loc, buf.data(), buf.size(), &bytes_read),
                  Status::kOk);
        EXPECT_EQ(bytes_read, 1024u);
        EXPECT_EQ(buf[0], static_cast<uint8_t>(i));
    }
}

TEST_F(BlobStoreTest, MarkDead) {
    ASSERT_EQ(store_->open(), Status::kOk);

    StorageKey key{1, 0, 0, 0, 0};
    std::vector<uint8_t> data(1024, 0xFF);
    BlobLocation loc{};

    EXPECT_EQ(store_->write(key, data.data(), data.size(), 0, &loc),
              Status::kOk);

    store_->mark_dead(loc);
    auto s = store_->stats();
    // After mark_dead, live_count should be decremented
    // but the blob should still be readable until GC
}

TEST_F(BlobStoreTest, Stats) {
    ASSERT_EQ(store_->open(), Status::kOk);

    auto s = store_->stats();
    EXPECT_GE(s.num_segments, 1u);

    StorageKey key{1, 0, 0, 0, 0};
    std::vector<uint8_t> data(4096, 0xFF);
    BlobLocation loc{};
    store_->write(key, data.data(), data.size(), 0, &loc);

    s = store_->stats();
    EXPECT_GT(s.total_bytes_on_disk, 0u);
}
