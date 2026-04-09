#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

namespace attentiondb {

struct IoCompletion {
    int64_t user_data;
    int result;         // Bytes transferred, or negative errno
};

class IoEngine {
public:
    virtual ~IoEngine() = default;

    // Submit an async read. Returns 0 on success, negative on error.
    virtual int submit_read(int fd, void* buf, size_t len, off_t offset,
                            int64_t user_data = 0) = 0;

    // Submit an async write. Returns 0 on success, negative on error.
    virtual int submit_write(int fd, const void* buf, size_t len, off_t offset,
                             int64_t user_data = 0) = 0;

    // Synchronous read (blocking).
    virtual int sync_read(int fd, void* buf, size_t len, off_t offset) = 0;

    // Synchronous write (blocking).
    virtual int sync_write(int fd, const void* buf, size_t len, off_t offset) = 0;

    // Poll for completed I/O operations. Returns number of completions.
    virtual int poll_completions(IoCompletion* out, int max_completions) = 0;
};

}  // namespace attentiondb
