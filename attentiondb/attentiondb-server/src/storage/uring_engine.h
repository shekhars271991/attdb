#pragma once

#include "storage/io_engine.h"

#ifdef __linux__
#include <liburing.h>
#endif

namespace attentiondb {

// io_uring-based async I/O engine (Linux only).
// Falls back to posix engine on non-Linux or when io_uring init fails.
class UringEngine : public IoEngine {
public:
    explicit UringEngine(uint32_t queue_depth = 128);
    ~UringEngine() override;

    UringEngine(const UringEngine&) = delete;
    UringEngine& operator=(const UringEngine&) = delete;

    int submit_read(int fd, void* buf, size_t len, off_t offset,
                    int64_t user_data = 0) override;

    int submit_write(int fd, const void* buf, size_t len, off_t offset,
                     int64_t user_data = 0) override;

    int sync_read(int fd, void* buf, size_t len, off_t offset) override;
    int sync_write(int fd, const void* buf, size_t len, off_t offset) override;

    int poll_completions(IoCompletion* out, int max_completions) override;

    bool is_available() const { return available_; }

private:
#ifdef __linux__
    struct io_uring ring_;
#endif
    bool available_ = false;
};

}  // namespace attentiondb
