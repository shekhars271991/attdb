#include "storage/posix_engine.h"

#include <cerrno>
#include <unistd.h>

namespace attentiondb {

int PosixEngine::sync_read(int fd, void* buf, size_t len, off_t offset) {
    size_t total = 0;
    auto* p = static_cast<uint8_t*>(buf);
    while (total < len) {
        ssize_t n = ::pread(fd, p + total, len - total, offset + total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (n == 0) break;  // EOF
        total += n;
    }
    return static_cast<int>(total);
}

int PosixEngine::sync_write(int fd, const void* buf, size_t len, off_t offset) {
    size_t total = 0;
    auto* p = static_cast<const uint8_t*>(buf);
    while (total < len) {
        ssize_t n = ::pwrite(fd, p + total, len - total, offset + total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        total += n;
    }
    return static_cast<int>(total);
}

// For the posix engine, submit_read/write are synchronous — they block.
int PosixEngine::submit_read(int fd, void* buf, size_t len, off_t offset,
                              int64_t /*user_data*/) {
    return sync_read(fd, buf, len, offset);
}

int PosixEngine::submit_write(int fd, const void* buf, size_t len, off_t offset,
                               int64_t /*user_data*/) {
    return sync_write(fd, buf, len, offset);
}

int PosixEngine::poll_completions(IoCompletion* /*out*/, int /*max*/) {
    return 0;  // Posix engine is synchronous, no pending completions.
}

}  // namespace attentiondb
