#include "storage/blob_store.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "storage/posix_engine.h"
#include "storage/uring_engine.h"

namespace attentiondb {

BlobStore::BlobStore(const NvmeStoreConfig& config) : config_(config) {
    if (config_.io_engine == "io_uring") {
        auto uring = std::make_unique<UringEngine>(config_.uring_queue_depth);
        if (uring->is_available()) {
            io_ = std::move(uring);
        } else {
            io_ = std::make_unique<PosixEngine>();
        }
    } else {
        io_ = std::make_unique<PosixEngine>();
    }
}

BlobStore::~BlobStore() {
    close();
}

Status BlobStore::open() {
    if (config_.paths.empty()) return Status::kInvalidArgument;

    for (const auto& path : config_.paths) {
        std::filesystem::create_directories(path);
    }

    // Scan for existing segments
    for (const auto& base_path : config_.paths) {
        for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
            if (entry.path().extension() == ".seg") {
                auto stem = entry.path().stem().string();
                // Parse segment_XXXX.seg
                if (stem.substr(0, 8) == "segment_") {
                    uint32_t id = std::stoul(stem.substr(8));
                    next_segment_id_ = std::max(next_segment_id_, id + 1);

                    auto seg = std::make_unique<SegmentInfo>();
                    seg->id = id;
                    seg->path = entry.path().string();
                    seg->fd = open_segment_fd(seg->path, false);
                    if (seg->fd < 0) continue;

                    // Read header to get metadata
                    SegmentHeader hdr{};
                    io_->sync_read(seg->fd, &hdr, sizeof(hdr), 0);

                    if (hdr.magic == kSegmentMagic) {
                        seg->entry_count = hdr.entry_count;
                        seg->live_count.store(hdr.live_count);
                        struct stat st;
                        fstat(seg->fd, &st);
                        seg->file_size = st.st_size;
                        seg->write_offset = st.st_size;
                        total_bytes_on_disk_ += seg->file_size;
                    }
                    segments_.push_back(std::move(seg));
                }
            }
        }
    }

    create_new_segment();

    // Start GC thread
    if (config_.gc_enabled) {
        gc_running_ = true;
        gc_thread_ = std::thread(&BlobStore::gc_thread_fn, this);
    }

    return Status::kOk;
}

void BlobStore::close() {
    gc_running_ = false;
    if (gc_thread_.joinable()) gc_thread_.join();

    std::lock_guard<std::mutex> lock(segments_mu_);
    for (auto& seg : segments_) {
        if (seg->fd >= 0) {
            // Update header before closing
            write_segment_header(seg->fd, *seg);
            ::close(seg->fd);
            seg->fd = -1;
        }
    }
    active_segment_ = nullptr;
}

Status BlobStore::write(const StorageKey& key, const void* data, size_t len,
                         uint32_t checksum, BlobLocation* loc_out) {
    size_t disk_size = entry_disk_size(len);

    std::lock_guard<std::mutex> lock(write_mu_);

    if (!active_segment_ ||
        active_segment_->write_offset + disk_size > config_.segment_size) {
        create_new_segment();
    }

    auto* seg = active_segment_;
    uint64_t offset = seg->write_offset;

    // Prepare entry: header + blob in an aligned buffer
    auto buf = std::unique_ptr<uint8_t[], decltype(&std::free)>(
        static_cast<uint8_t*>(std::aligned_alloc(kAlignment, disk_size)),
        &std::free);
    std::memset(buf.get(), 0, disk_size);

    auto* hdr = reinterpret_cast<BlobEntryHeader*>(buf.get());
    hdr->key_size = sizeof(StorageKey);
    hdr->blob_size = static_cast<uint32_t>(len);
    hdr->padded_size = static_cast<uint32_t>(disk_size);
    hdr->checksum = checksum;
    hdr->sequence = write_sequence_.fetch_add(1);
    std::memcpy(hdr->key_data, &key, sizeof(StorageKey));

    std::memcpy(buf.get() + kEntryHeaderSize, data, len);

    int ret = io_->sync_write(seg->fd, buf.get(), disk_size,
                               static_cast<off_t>(offset));
    if (ret < 0) return Status::kIOError;

    seg->write_offset += disk_size;
    seg->entry_count++;
    seg->live_count.fetch_add(1);
    total_bytes_on_disk_ += disk_size;

    loc_out->segment_id = seg->id;
    loc_out->offset = offset;
    loc_out->length = static_cast<uint32_t>(len);
    loc_out->disk_size = static_cast<uint32_t>(disk_size);

    return Status::kOk;
}

Status BlobStore::read(const BlobLocation& loc, void* buf, size_t buf_len,
                        size_t* bytes_read) {
    std::lock_guard<std::mutex> slock(segments_mu_);

    int fd = -1;
    for (const auto& seg : segments_) {
        if (seg->id == loc.segment_id) {
            fd = seg->fd;
            break;
        }
    }
    if (fd < 0) return Status::kNotFound;

    // Read the entry header first to get actual blob size
    auto aligned_buf = std::unique_ptr<uint8_t[], decltype(&std::free)>(
        static_cast<uint8_t*>(std::aligned_alloc(kAlignment, loc.disk_size)),
        &std::free);

    int ret = io_->sync_read(fd, aligned_buf.get(), loc.disk_size,
                              static_cast<off_t>(loc.offset));
    if (ret < 0) return Status::kIOError;

    auto* hdr = reinterpret_cast<const BlobEntryHeader*>(aligned_buf.get());
    if (hdr->blob_size > buf_len) return Status::kBufferTooSmall;

    std::memcpy(buf, aligned_buf.get() + kEntryHeaderSize, hdr->blob_size);
    *bytes_read = hdr->blob_size;

    return Status::kOk;
}

void BlobStore::mark_dead(const BlobLocation& loc) {
    std::lock_guard<std::mutex> lock(segments_mu_);
    for (auto& seg : segments_) {
        if (seg->id == loc.segment_id) {
            seg->live_count.fetch_sub(1);
            return;
        }
    }
}

void BlobStore::create_new_segment() {
    std::lock_guard<std::mutex> lock(segments_mu_);

    auto seg = std::make_unique<SegmentInfo>();
    seg->id = next_segment_id_++;
    seg->path = segment_path(seg->id);
    seg->fd = open_segment_fd(seg->path, true);
    seg->write_offset = kSegmentHeaderSize;  // Skip header
    seg->entry_count = 0;
    seg->live_count.store(0);
    seg->file_size = 0;

    // Write initial header
    write_segment_header(seg->fd, *seg);

    active_segment_ = seg.get();
    segments_.push_back(std::move(seg));
}

int BlobStore::open_segment_fd(const std::string& path, bool create) {
    int flags = O_RDWR;
    if (create) flags |= O_CREAT | O_TRUNC;
#ifdef __linux__
    flags |= O_DIRECT;
#endif
    int fd = ::open(path.c_str(), flags, 0644);
    if (fd < 0 && (flags & O_DIRECT)) {
        // Retry without O_DIRECT (e.g., on tmpfs)
        flags &= ~O_DIRECT;
        fd = ::open(path.c_str(), flags, 0644);
    }
    return fd;
}

std::string BlobStore::segment_path(uint32_t id) const {
    char buf[64];
    snprintf(buf, sizeof(buf), "segment_%04u.seg", id);
    return config_.paths[0] + "/" + buf;
}

void BlobStore::write_segment_header(int fd, const SegmentInfo& seg) {
    SegmentHeader hdr{};
    hdr.magic = kSegmentMagic;
    hdr.version = kSegmentVersion;
    hdr.segment_id = seg.id;
    hdr.created_at = static_cast<uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
    hdr.entry_count = seg.entry_count;
    hdr.live_count = seg.live_count.load();

    // Use aligned write
    auto buf = std::unique_ptr<uint8_t[], decltype(&std::free)>(
        static_cast<uint8_t*>(std::aligned_alloc(kAlignment, kSegmentHeaderSize)),
        &std::free);
    std::memcpy(buf.get(), &hdr, sizeof(hdr));

    io_->sync_write(fd, buf.get(), kSegmentHeaderSize, 0);
}

void BlobStore::gc_thread_fn() {
    while (gc_running_) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (!gc_running_) break;

        std::lock_guard<std::mutex> lock(segments_mu_);

        for (auto& seg : segments_) {
            if (seg.get() == active_segment_) continue;
            if (seg->entry_count == 0) continue;

            double live_ratio = static_cast<double>(seg->live_count.load()) /
                                seg->entry_count;
            if (live_ratio < config_.gc_trigger_fragmentation) {
                gc_compact_segment(*seg);
                gc_cycles_.fetch_add(1);
            }
        }
    }
}

void BlobStore::gc_compact_segment(SegmentInfo& seg) {
    if (!gc_rewrite_cb_) return;

    // Read all entries in the segment and rewrite live ones
    uint64_t offset = kSegmentHeaderSize;

    auto read_buf = std::unique_ptr<uint8_t[], decltype(&std::free)>(
        static_cast<uint8_t*>(std::aligned_alloc(kAlignment, kAlignment)),
        &std::free);

    while (offset < seg.write_offset) {
        // Read entry header
        int ret = io_->sync_read(seg.fd, read_buf.get(), kEntryHeaderSize,
                                  static_cast<off_t>(offset));
        if (ret < 0) break;

        auto* hdr = reinterpret_cast<const BlobEntryHeader*>(read_buf.get());
        if (hdr->padded_size == 0) break;

        // Read full entry if we need the blob data
        size_t disk_size = hdr->padded_size;
        auto full_buf = std::unique_ptr<uint8_t[], decltype(&std::free)>(
            static_cast<uint8_t*>(std::aligned_alloc(kAlignment, disk_size)),
            &std::free);

        ret = io_->sync_read(seg.fd, full_buf.get(), disk_size,
                              static_cast<off_t>(offset));
        if (ret < 0) break;

        auto* entry_hdr = reinterpret_cast<const BlobEntryHeader*>(full_buf.get());
        StorageKey key;
        std::memcpy(&key, entry_hdr->key_data, sizeof(StorageKey));

        gc_rewrite_cb_(key, full_buf.get() + kEntryHeaderSize,
                        entry_hdr->blob_size, entry_hdr->checksum);

        offset += disk_size;
    }

    uint64_t reclaimed = seg.write_offset;
    gc_reclaimed_bytes_ += reclaimed;
    total_bytes_on_disk_ -= reclaimed;

    // Truncate and close old segment
    ::ftruncate(seg.fd, 0);
    ::close(seg.fd);
    std::filesystem::remove(seg.path);
    seg.fd = -1;
    seg.entry_count = 0;
    seg.live_count.store(0);
    seg.write_offset = 0;
}

void BlobStore::set_gc_rewrite_callback(GcRewriteCallback cb) {
    gc_rewrite_cb_ = std::move(cb);
}

BlobStore::Stats BlobStore::stats() const {
    Stats s{};
    std::lock_guard<std::mutex> lock(segments_mu_);
    s.num_segments = static_cast<uint32_t>(segments_.size());
    s.total_bytes_on_disk = total_bytes_on_disk_.load();
    s.gc_reclaimed_bytes = gc_reclaimed_bytes_.load();
    s.gc_cycles = gc_cycles_.load();

    for (const auto& seg : segments_) {
        // Approximate live bytes
        if (seg->entry_count > 0) {
            double ratio = static_cast<double>(seg->live_count.load()) / seg->entry_count;
            s.live_bytes += static_cast<uint64_t>(seg->write_offset * ratio);
        }
    }
    return s;
}

}  // namespace attentiondb
