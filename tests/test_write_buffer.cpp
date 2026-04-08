#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "write_buffer/write_buffer.h"

using namespace attentiondb;

TEST(WriteBufferTest, SubmitAndFlush) {
    WriteBufferConfig cfg;
    cfg.size_bytes = 1024 * 1024;  // 1 MB
    cfg.flush_interval_ms = 50;
    WriteBuffer wb(cfg);

    int flushed = 0;
    wb.set_flush_callback([&](std::vector<WriteBuffer::PendingWrite>&& batch) {
        flushed += static_cast<int>(batch.size());
    });
    wb.start();

    StorageKey key{1, 0, 0, 0, 0};
    std::vector<uint8_t> data(1024, 0xAB);
    PutOpts opts;
    opts.recompute_cost = 500;

    EXPECT_EQ(wb.submit(key, data.data(), data.size(), opts, 0), Status::kOk);

    // Wait for flush
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    wb.stop();

    EXPECT_GE(flushed, 1);
    EXPECT_EQ(wb.total_submitted(), 1u);
}

TEST(WriteBufferTest, BackpressureLevels) {
    WriteBufferConfig cfg;
    cfg.size_bytes = 1000;  // Very small buffer
    cfg.flush_interval_ms = 10000;  // Very long flush interval → fills up
    WriteBuffer wb(cfg);

    // Don't set flush callback; don't start flush thread → buffer stays full
    wb.set_flush_callback([](std::vector<WriteBuffer::PendingWrite>&&) {});
    wb.start();

    StorageKey key{1, 0, 0, 0, 0};
    PutOpts opts;
    opts.entry_type = EntryType::kConversation;
    opts.recompute_cost = 100;

    // Fill it up
    std::vector<uint8_t> data(200, 0xFF);
    for (int i = 0; i < 20; ++i) {
        key.chunk_hash = i;
        wb.submit(key, data.data(), data.size(), opts, 0);
    }

    // Buffer should be at high utilization now
    EXPECT_GT(wb.utilization(), 0.0);

    wb.stop();
}

TEST(WriteBufferTest, FullReturnsStatusFull) {
    WriteBufferConfig cfg;
    cfg.size_bytes = 100;  // Tiny buffer
    cfg.flush_interval_ms = 10000;
    WriteBuffer wb(cfg);

    wb.set_flush_callback([](std::vector<WriteBuffer::PendingWrite>&&) {});
    wb.start();

    StorageKey key{1, 0, 0, 0, 0};
    PutOpts opts;
    opts.entry_type = EntryType::kConversation;
    opts.recompute_cost = 50;

    std::vector<uint8_t> data(200, 0xFF);

    // Keep submitting until rejected
    bool got_rejection = false;
    for (int i = 0; i < 100; ++i) {
        key.chunk_hash = i;
        Status s = wb.submit(key, data.data(), data.size(), opts, 0);
        if (s != Status::kOk) {
            got_rejection = true;
            break;
        }
    }

    EXPECT_TRUE(got_rejection);
    wb.stop();
}
