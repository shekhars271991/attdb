#pragma once

#include "storage/io_engine.h"

namespace attentiondb {

// GPUDirect Storage engine stub. Full implementation requires CUDA toolkit
// and NVMe controllers supporting GDS. Falls back to posix engine.
class GdsEngine : public IoEngine {
public:
    GdsEngine() = default;

    int submit_read(int fd, void* buf, size_t len, off_t offset,
                    int64_t user_data = 0) override;
    int submit_write(int fd, const void* buf, size_t len, off_t offset,
                     int64_t user_data = 0) override;
    int sync_read(int fd, void* buf, size_t len, off_t offset) override;
    int sync_write(int fd, const void* buf, size_t len, off_t offset) override;
    int poll_completions(IoCompletion* out, int max_completions) override;

    bool is_available() const { return false; }  // Stub for Phase 0
};

}  // namespace attentiondb
