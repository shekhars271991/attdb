#pragma once

#include "storage/io_engine.h"

namespace attentiondb {

// Synchronous pread/pwrite fallback engine. Always uses O_DIRECT when available.
class PosixEngine : public IoEngine {
public:
    PosixEngine() = default;

    int submit_read(int fd, void* buf, size_t len, off_t offset,
                    int64_t user_data = 0) override;

    int submit_write(int fd, const void* buf, size_t len, off_t offset,
                     int64_t user_data = 0) override;

    int sync_read(int fd, void* buf, size_t len, off_t offset) override;
    int sync_write(int fd, const void* buf, size_t len, off_t offset) override;

    int poll_completions(IoCompletion* out, int max_completions) override;
};

}  // namespace attentiondb
