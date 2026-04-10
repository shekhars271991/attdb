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
                if (stem.substr(0, 8) == "segment_") {
                    uint32_t id = std::stoul(stem.substr(8));
                    next_segment_id_ = std::max(next_segment_id_, id + 1);

                    auto seg = std::make_unique<SegmentInfo>();
                    seg->id = id;
                    seg->path = entry.path().string();
                    seg->fd = open_segment_fd(seg->path, false);
                    if (seg->fd < 0) continue;

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

    // Pre-allocate write block pool
    size_t block_capacity = static_cast<size_t>(config_.write_block_size_mb) * 1024 * 1024;
    for (uint32_t i = 0; i < config_.write_block_pool_size; ++i) {
        auto wb = std::make_unique<WriteBlock>();
        wb->capacity = block_capacity;
        wb->buf = static_cast<uint8_t*>(std::aligned_alloc(kAlignment, block_capacity));
        if (!wb->buf) return Status::kIOError;
        std::memset(wb->buf, 0, block_capacity);
        free_pool_.push_back(wb.get());
        all_blocks_.push_back(std::move(wb));
    }

    // Grab the first block as active
    {
        std::unique_lock<std::mutex> lock(block_mu_);
        active_block_ = acquire_block(lock);
        if (active_block_) {
            active_block_->reset(active_segment_->id,
                                 active_segment_->write_offset,
                                 active_segment_->fd);
        }
    }

    // Start flush thread
    flush_running_ = true;
    flush_thread_ = std::thread(&BlobStore::flush_thread_fn, this);

    if (config_.gc_enabled) {
        gc_running_ = true;
        gc_thread_ = std::thread(&BlobStore::gc_thread_fn, this);
    }

    return Status::kOk;
}

void BlobStore::close() {
    // Signal flush thread to stop and flush remaining blocks
    {
        std::lock_guard<std::mutex> lock(block_mu_);
        if (active_block_ && active_block_->pos > 0) {
            flush_queue_.push(active_block_);
            active_block_ = nullptr;
            flush_cv_.notify_one();
        }
    }

    flush_running_ = false;
    flush_cv_.notify_all();
    free_cv_.notify_all();
    if (flush_thread_.joinable()) flush_thread_.join();

    gc_running_ = false;
    if (gc_thread_.joinable()) gc_thread_.join();

    // Free write block buffers
    for (auto& wb : all_blocks_) {
        if (wb->buf) {
            std::free(wb->buf);
            wb->buf = nullptr;
        }
    }

    std::lock_guard<std::mutex> lock(segments_mu_);
    for (auto& seg : segments_) {
        if (seg->fd >= 0) {
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

    std::unique_lock<std::mutex> lock(block_mu_);

    // If entry is larger than a full block, fall through to direct write
    if (disk_size > active_block_->capacity) {
        lock.unlock();

        // Direct write for oversized entries (rare)
        std::lock_guard<std::mutex> wlock(segments_mu_);
        if (!active_segment_ ||
            active_segment_->write_offset + disk_size > config_.segment_size) {
            create_new_segment();
        }
        auto* seg = active_segment_;
        uint64_t offset = seg->write_offset;

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

    // Entry doesn't fit in current block -- rotate
    if (active_block_->pos + disk_size > active_block_->capacity) {
        if (active_block_->pos > 0) {
            flush_queue_.push(active_block_);
            flush_cv_.notify_one();
        } else {
            free_pool_.push_back(active_block_);
        }

        active_block_ = acquire_block(lock);

        // Check if we need a new segment for the next block
        {
            std::lock_guard<std::mutex> slock(segments_mu_);
            if (!active_segment_ ||
                active_segment_->write_offset + active_block_->capacity > config_.segment_size) {
                create_new_segment();
            }
        }
        active_block_->reset(active_segment_->id,
                             active_segment_->write_offset,
                             active_segment_->fd);
    }

    // Append entry into the active block (single memcpy, no alloc)
    uint8_t* dest = active_block_->buf + active_block_->pos;

    auto* hdr = reinterpret_cast<BlobEntryHeader*>(dest);
    hdr->key_size = sizeof(StorageKey);
    hdr->blob_size = static_cast<uint32_t>(len);
    hdr->padded_size = static_cast<uint32_t>(disk_size);
    hdr->checksum = checksum;
    hdr->sequence = write_sequence_.fetch_add(1);
    std::memcpy(hdr->key_data, &key, sizeof(StorageKey));

    std::memcpy(dest + kEntryHeaderSize, data, len);

    size_t used = kEntryHeaderSize + len;
    if (used < disk_size) {
        std::memset(dest + used, 0, disk_size - used);
    }

    loc_out->segment_id = active_block_->segment_id;
    loc_out->offset = active_block_->segment_offset + active_block_->pos;
    loc_out->length = static_cast<uint32_t>(len);
    loc_out->disk_size = static_cast<uint32_t>(disk_size);

    active_block_->pos += disk_size;
    active_block_->entry_count++;

    // Advance segment bookkeeping
    active_segment_->write_offset += disk_size;
    active_segment_->entry_count++;
    active_segment_->live_count.fetch_add(1);
    total_bytes_on_disk_ += disk_size;

    return Status::kOk;
}

Status BlobStore::read(const BlobLocation& loc, void* buf, size_t buf_len,
                        size_t* bytes_read) {
    // Check pending write blocks first (read-through for unflushed data)
    {
        std::lock_guard<std::mutex> lock(block_mu_);
        const uint8_t* pending = find_in_pending_blocks(loc);
        if (pending) {
            auto* hdr = reinterpret_cast<const BlobEntryHeader*>(pending);
            if (hdr->blob_size > buf_len) return Status::kBufferTooSmall;
            std::memcpy(buf, pending + kEntryHeaderSize, hdr->blob_size);
            *bytes_read = hdr->blob_size;
            return Status::kOk;
        }
    }

    // Fall back to disk read
    std::lock_guard<std::mutex> slock(segments_mu_);

    int fd = -1;
    for (const auto& seg : segments_) {
        if (seg->id == loc.segment_id) {
            fd = seg->fd;
            break;
        }
    }
    if (fd < 0) return Status::kNotFound;

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

const uint8_t* BlobStore::find_in_pending_blocks(const BlobLocation& loc) const {
    // Check active block
    if (active_block_ && active_block_->segment_id == loc.segment_id &&
        loc.offset >= active_block_->segment_offset &&
        loc.offset < active_block_->segment_offset + active_block_->pos) {
        return active_block_->buf + (loc.offset - active_block_->segment_offset);
    }

    // Check flush queue (iterate via a copy -- queue is small, max pool_size entries)
    auto q_copy = flush_queue_;
    while (!q_copy.empty()) {
        auto* wb = q_copy.front();
        q_copy.pop();
        if (wb->segment_id == loc.segment_id &&
            loc.offset >= wb->segment_offset &&
            loc.offset < wb->segment_offset + wb->pos) {
            return wb->buf + (loc.offset - wb->segment_offset);
        }
    }

    return nullptr;
}

WriteBlock* BlobStore::acquire_block(std::unique_lock<std::mutex>& lock) {
    while (free_pool_.empty()) {
        if (!flush_running_) return nullptr;
        free_cv_.wait(lock);
    }
    auto* wb = free_pool_.back();
    free_pool_.pop_back();
    return wb;
}

void BlobStore::flush_thread_fn() {
    while (true) {
        WriteBlock* block = nullptr;
        {
            std::unique_lock<std::mutex> lock(block_mu_);
            flush_cv_.wait(lock, [this] {
                return !flush_queue_.empty() || !flush_running_;
            });

            if (flush_queue_.empty()) {
                if (!flush_running_) break;
                continue;
            }

            block = flush_queue_.front();
            flush_queue_.pop();
        }

        if (!block || block->pos == 0) {
            std::lock_guard<std::mutex> lock(block_mu_);
            if (block) free_pool_.push_back(block);
            free_cv_.notify_one();
            continue;
        }

        // Single pwrite for the entire block -- the key optimization
        size_t write_size = align_up(block->pos, kAlignment);
        io_->sync_write(block->segment_fd, block->buf, write_size,
                        static_cast<off_t>(block->segment_offset));

        // Return block to free pool
        {
            std::lock_guard<std::mutex> lock(block_mu_);
            free_pool_.push_back(block);
            free_cv_.notify_one();
        }
    }

    // Drain remaining blocks on shutdown
    while (true) {
        WriteBlock* block = nullptr;
        {
            std::lock_guard<std::mutex> lock(block_mu_);
            if (flush_queue_.empty()) break;
            block = flush_queue_.front();
            flush_queue_.pop();
        }
        if (block && block->pos > 0) {
            size_t write_size = align_up(block->pos, kAlignment);
            io_->sync_write(block->segment_fd, block->buf, write_size,
                            static_cast<off_t>(block->segment_offset));
        }
    }
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
    // segments_mu_ may or may not be held by caller; use try_lock pattern
    // Actually, all callers should hold segments_mu_ or block_mu_ which
    // implies exclusive access to active_segment_. For safety, always lock.
    // Note: callers from write() already hold block_mu_ and acquire segments_mu_.

    auto seg = std::make_unique<SegmentInfo>();
    seg->id = next_segment_id_++;
    seg->path = segment_path(seg->id);
    seg->fd = open_segment_fd(seg->path, true);
    seg->write_offset = kSegmentHeaderSize;
    seg->entry_count = 0;
    seg->live_count.store(0);
    seg->file_size = 0;

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
#ifdef __linux__
    if (fd < 0 && (flags & O_DIRECT)) {
        flags &= ~O_DIRECT;
        fd = ::open(path.c_str(), flags, 0644);
    }
#endif
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

    uint64_t offset = kSegmentHeaderSize;

    auto read_buf = std::unique_ptr<uint8_t[], decltype(&std::free)>(
        static_cast<uint8_t*>(std::aligned_alloc(kAlignment, kAlignment)),
        &std::free);

    while (offset < seg.write_offset) {
        int ret = io_->sync_read(seg.fd, read_buf.get(), kEntryHeaderSize,
                                  static_cast<off_t>(offset));
        if (ret < 0) break;

        auto* hdr = reinterpret_cast<const BlobEntryHeader*>(read_buf.get());
        if (hdr->padded_size == 0) break;

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
        if (seg->entry_count > 0) {
            double ratio = static_cast<double>(seg->live_count.load()) / seg->entry_count;
            s.live_bytes += static_cast<uint64_t>(seg->write_offset * ratio);
        }
    }
    return s;
}

}  // namespace attentiondb
