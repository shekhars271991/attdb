#include "storage/uring_engine.h"

#include <cerrno>
#include <unistd.h>

namespace attentiondb {

#ifdef __linux__

UringEngine::UringEngine(uint32_t queue_depth) {
    int ret = io_uring_queue_init(queue_depth, &ring_, 0);
    available_ = (ret == 0);
}

UringEngine::~UringEngine() {
    if (available_) {
        io_uring_queue_exit(&ring_);
    }
}

int UringEngine::submit_read(int fd, void* buf, size_t len, off_t offset,
                              int64_t user_data) {
    if (!available_) return sync_read(fd, buf, len, offset);

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return -EAGAIN;

    io_uring_prep_read(sqe, fd, buf, len, offset);
    io_uring_sqe_set_data64(sqe, static_cast<__u64>(user_data));

    int ret = io_uring_submit(&ring_);
    return ret < 0 ? ret : 0;
}

int UringEngine::submit_write(int fd, const void* buf, size_t len, off_t offset,
                               int64_t user_data) {
    if (!available_) return sync_write(fd, buf, len, offset);

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) return -EAGAIN;

    io_uring_prep_write(sqe, fd, buf, len, offset);
    io_uring_sqe_set_data64(sqe, static_cast<__u64>(user_data));

    int ret = io_uring_submit(&ring_);
    return ret < 0 ? ret : 0;
}

int UringEngine::sync_read(int fd, void* buf, size_t len, off_t offset) {
    size_t total = 0;
    auto* p = static_cast<uint8_t*>(buf);
    while (total < len) {
        ssize_t n = ::pread(fd, p + total, len - total, offset + total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (n == 0) break;
        total += n;
    }
    return static_cast<int>(total);
}

int UringEngine::sync_write(int fd, const void* buf, size_t len, off_t offset) {
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

int UringEngine::poll_completions(IoCompletion* out, int max_completions) {
    if (!available_) return 0;

    int count = 0;
    struct io_uring_cqe* cqe;

    while (count < max_completions) {
        int ret = io_uring_peek_cqe(&ring_, &cqe);
        if (ret < 0) break;  // No more completions

        out[count].user_data = static_cast<int64_t>(io_uring_cqe_get_data64(cqe));
        out[count].result = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);
        ++count;
    }
    return count;
}

#else  // non-Linux

UringEngine::UringEngine(uint32_t /*queue_depth*/) : available_(false) {}
UringEngine::~UringEngine() = default;

int UringEngine::submit_read(int fd, void* buf, size_t len, off_t offset, int64_t) {
    return sync_read(fd, buf, len, offset);
}

int UringEngine::submit_write(int fd, const void* buf, size_t len, off_t offset, int64_t) {
    return sync_write(fd, buf, len, offset);
}

int UringEngine::sync_read(int fd, void* buf, size_t len, off_t offset) {
    size_t total = 0;
    auto* p = static_cast<uint8_t*>(buf);
    while (total < len) {
        ssize_t n = ::pread(fd, p + total, len - total, offset + static_cast<off_t>(total));
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (n == 0) break;
        total += n;
    }
    return static_cast<int>(total);
}

int UringEngine::sync_write(int fd, const void* buf, size_t len, off_t offset) {
    size_t total = 0;
    auto* p = static_cast<const uint8_t*>(buf);
    while (total < len) {
        ssize_t n = ::pwrite(fd, p + total, len - total, offset + static_cast<off_t>(total));
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        total += n;
    }
    return static_cast<int>(total);
}

int UringEngine::poll_completions(IoCompletion*, int) { return 0; }

#endif  // __linux__

}  // namespace attentiondb
