#include "engine.h"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace attentiondb {

AttentionDBEngine::AttentionDBEngine() = default;
AttentionDBEngine::~AttentionDBEngine() { close(); }

Status AttentionDBEngine::open(const Config& config) {
    if (opened_) return Status::kAlreadyExists;
    config_ = config;

    logger_ = std::make_unique<Logger>(config_.logging);

    // Index
    index_ = std::make_unique<ConcurrentHashMap>(1 << 20);

    // T1 DRAM slab allocator
    if (config_.dram_cache.enabled) {
        t1_allocator_ = std::make_unique<SlabAllocator>(config_.dram_cache);
    }

    // T2 NVMe blob store
    if (config_.nvme_store.enabled) {
        t2_store_ = std::make_unique<BlobStore>(config_.nvme_store);
        Status s = t2_store_->open();
        if (s != Status::kOk) {
            logger_->error("t2_open_failed", "");
            return s;
        }
    }

    // Eviction policies
    CwSlru::Config ev_cfg;
    ev_cfg.protected_ratio = config_.eviction.protected_ratio;
    t1_eviction_ = std::make_unique<CwSlru>(ev_cfg);
    t2_eviction_ = std::make_unique<CwSlru>(ev_cfg);

    // Admission control
    admission_ = std::make_unique<AdmissionControl>(config_.admission);

    // Checkpoint
    std::string ckpt_dir = config_.nvme_store.enabled
        ? config_.nvme_store.paths[0] + "/checkpoint"
        : "/tmp/attentiondb_checkpoint";
    checkpoint_ = std::make_unique<CheckpointManager>(config_.checkpoint, ckpt_dir);

    // Try loading existing checkpoint
    Status ckpt_status = checkpoint_->load(*index_);
    if (ckpt_status == Status::kOk) {
        char buf[64];
        snprintf(buf, sizeof(buf), "\"entries\":%zu", index_->size());
        logger_->info("checkpoint_loaded", buf);

        // T1 DRAM entries from a previous process have stale pointers.
        // Remove them; only T2 NVMe entries survive cold restart.
        std::vector<StorageKey> stale_keys;
        index_->iterate([&](const StorageKey& k, const IndexEntry& e) {
            if (e.tier == static_cast<uint8_t>(Tier::kT1Dram))
                stale_keys.push_back(k);
            return true;
        });
        for (const auto& k : stale_keys) index_->erase(k);

        if (!stale_keys.empty()) {
            char buf2[128];
            snprintf(buf2, sizeof(buf2),
                     "\"removed\":%zu,\"remaining\":%zu",
                     stale_keys.size(), index_->size());
            logger_->info("checkpoint_scrubbed_t1", buf2);
        }
    }

    checkpoint_->start_periodic(*index_);

    // Start periodic logging summary
    logger_->start_periodic_summary();

    // Log startup
    char startup_buf[256];
    snprintf(startup_buf, sizeof(startup_buf),
             "\"mode\":\"%s\",\"t1_enabled\":%s,\"t1_size\":%zu,"
             "\"t2_enabled\":%s",
             config_.mode.c_str(),
             config_.dram_cache.enabled ? "true" : "false",
             config_.dram_cache.size_bytes,
             config_.nvme_store.enabled ? "true" : "false");
    logger_->info("engine_started", startup_buf);

    opened_ = true;
    return Status::kOk;
}

void AttentionDBEngine::close() {
    if (!opened_) return;
    opened_ = false;

    logger_->info("engine_closing", "");

    checkpoint_->stop();
    logger_->stop();

    if (t2_store_) t2_store_->close();
}

Status AttentionDBEngine::put(const StorageKey& key, const void* blob,
                               size_t len, const PutOpts& opts) {
    if (!opened_) return Status::kShutdown;

    // Admission check
    double util = storage_utilization();
    auto admit = admission_->evaluate(opts, util);
    admission_->record_evaluation(admit);

    if (admit != AdmissionResult::kAdmit) {
        logger_->period_stats().puts_rejected.fetch_add(1, std::memory_order_relaxed);

        char buf[128];
        snprintf(buf, sizeof(buf),
                 "\"recompute_cost\":%u,\"reason\":\"%s\"",
                 opts.recompute_cost,
                 admit == AdmissionResult::kRejectLowCost ? "low_cost" : "dynamic_threshold");
        logger_->debug("put_rejected_admission", buf);

        return Status::kRejectedAdmission;
    }

    if (len > UINT32_MAX) {
        return Status::kInvalidArgument;
    }

    uint32_t crc = compute_crc32(blob, len);

    IndexEntry entry{};
    std::memset(&entry, 0, sizeof(entry));
    entry.recompute_cost = opts.recompute_cost;
    entry.num_tokens = opts.num_tokens;
    entry.access_count = 1;
    entry.slru_segment = static_cast<uint8_t>(SlruSegment::kProbationary);
    entry.checksum = crc;
    entry.length = static_cast<uint32_t>(len);
    entry.set_created_at_us(now_us());
    entry.set_last_access_us(now_us());

    bool stored = false;

    if (t1_allocator_) {
        void* ptr = nullptr;
        auto handle = t1_allocator_->allocate(len, &ptr);
        if (handle.valid()) {
            std::memcpy(ptr, blob, len);
            entry.tier = static_cast<uint8_t>(Tier::kT1Dram);
            entry.address = reinterpret_cast<uint64_t>(ptr);
            entry.segment_id = 0;
            stored = true;
            t1_eviction_->on_insert(key, opts.recompute_cost, key.chunk_index);
        }
    }

    if (!stored && t2_store_) {
        BlobLocation loc{};
        Status s = t2_store_->write(key, blob, len, crc, &loc);
        if (s != Status::kOk) {
            logger_->period_stats().puts_rejected.fetch_add(1, std::memory_order_relaxed);
            logger_->warn("direct_write_failed", "");
            return s;
        }
        entry.tier = static_cast<uint8_t>(Tier::kT2Nvme);
        entry.address = loc.offset;
        entry.segment_id = loc.segment_id;
        stored = true;
        t2_eviction_->on_insert(key, opts.recompute_cost, key.chunk_index);
    }

    if (!stored) {
        logger_->period_stats().puts_rejected.fetch_add(1, std::memory_order_relaxed);
        return Status::kIOError;
    }

    index_->upsert(key, entry);
    checkpoint_->mark_dirty();
    evict_if_needed();

    logger_->period_stats().puts.fetch_add(1, std::memory_order_relaxed);

    return Status::kOk;
}

Status AttentionDBEngine::get(const StorageKey& key, void* buf,
                               size_t buf_len, size_t* out_len) {
    if (!opened_) return Status::kShutdown;

    auto start = std::chrono::steady_clock::now();

    IndexEntry entry{};
    if (!index_->find(key, entry)) {
        logger_->period_stats().misses.fetch_add(1, std::memory_order_relaxed);
        logger_->period_stats().gets.fetch_add(1, std::memory_order_relaxed);
        logger_->debug("get_miss", "");
        return Status::kNotFound;
    }

    Status s;

    if (entry.tier == static_cast<uint8_t>(Tier::kT1Dram) && t1_allocator_) {
        // T1 DRAM hit
        if (entry.length > buf_len) return Status::kBufferTooSmall;
        auto* src = reinterpret_cast<void*>(entry.address);
        std::memcpy(buf, src, entry.length);
        *out_len = entry.length;
        s = Status::kOk;

        logger_->period_stats().t1_hits.fetch_add(1, std::memory_order_relaxed);
        t1_eviction_->on_access(key, entry.recompute_cost, key.chunk_index);
    } else if (entry.tier == static_cast<uint8_t>(Tier::kT2Nvme) && t2_store_) {
        // T2 NVMe hit
        BlobLocation loc;
        loc.segment_id = entry.segment_id;
        loc.offset = entry.address;
        loc.length = entry.length;
        loc.disk_size = entry_disk_size(entry.length);

        size_t bytes_read = 0;
        s = t2_store_->read(loc, buf, buf_len, &bytes_read);
        if (s == Status::kOk) {
            *out_len = bytes_read;
            logger_->period_stats().t2_hits.fetch_add(1, std::memory_order_relaxed);
            t2_eviction_->on_access(key, entry.recompute_cost, key.chunk_index);
        }
    } else {
        s = Status::kNotFound;
        logger_->period_stats().misses.fetch_add(1, std::memory_order_relaxed);
    }

    // Update access timestamp
    index_->update_fn(key, [](IndexEntry& e) { e.touch(); });
    checkpoint_->mark_dirty();

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    logger_->period_stats().gets.fetch_add(1, std::memory_order_relaxed);
    logger_->period_stats().get_latency_sum_us.fetch_add(us, std::memory_order_relaxed);

    // Track max latency (approximate, no CAS loop for performance)
    uint64_t cur_max = logger_->period_stats().get_latency_max_us.load(std::memory_order_relaxed);
    if (static_cast<uint64_t>(us) > cur_max) {
        logger_->period_stats().get_latency_max_us.store(us, std::memory_order_relaxed);
    }

    return s;
}

bool AttentionDBEngine::contains(const StorageKey& key) {
    if (!opened_) return false;
    return index_->contains(key);
}

Status AttentionDBEngine::del(const StorageKey& key) {
    if (!opened_) return Status::kShutdown;

    IndexEntry entry{};
    if (!index_->find(key, entry)) return Status::kNotFound;

    if (entry.tier == static_cast<uint8_t>(Tier::kT1Dram) && t1_allocator_) {
        t1_allocator_->free_by_ptr(reinterpret_cast<void*>(entry.address));
    }

    // Mark dead in blob store for GC
    if (entry.tier == static_cast<uint8_t>(Tier::kT2Nvme) && t2_store_) {
        BlobLocation loc;
        loc.segment_id = entry.segment_id;
        loc.offset = entry.address;
        loc.length = entry.length;
        t2_store_->mark_dead(loc);
    }

    index_->erase(key);
    t1_eviction_->on_remove(key);
    t2_eviction_->on_remove(key);
    checkpoint_->mark_dirty();

    return Status::kOk;
}

EngineStats AttentionDBEngine::stats() {
    EngineStats s{};

    if (t1_allocator_) {
        s.t1_total_bytes = t1_allocator_->total_capacity();
        s.t1_used_bytes = t1_allocator_->used_bytes();
    }

    if (t2_store_) {
        auto bs = t2_store_->stats();
        s.t2_total_bytes_on_disk = bs.total_bytes_on_disk;
        s.t2_num_segments = bs.num_segments;
    }

    s.index_entries = index_->size();

    s.eviction_protected = t1_eviction_->protected_size() +
                           t2_eviction_->protected_size();
    s.eviction_probationary = t1_eviction_->probationary_size() +
                              t2_eviction_->probationary_size();

    s.admission_evaluated = admission_->total_evaluated();
    s.admission_rejected = admission_->total_rejected();

    s.last_checkpoint_entries = checkpoint_->last_checkpoint_entries();
    s.last_checkpoint_duration_ms = checkpoint_->last_checkpoint_duration_ms();

    return s;
}

void AttentionDBEngine::evict_if_needed() {
    if (!t1_allocator_) return;

    while (t1_allocator_->used_bytes() > t1_allocator_->total_capacity() * 0.95) {
        auto victim = t1_eviction_->evict();
        if (!victim) break;

        IndexEntry entry{};
        if (!index_->find(*victim, entry)) continue;
        if (entry.tier != static_cast<uint8_t>(Tier::kT1Dram)) continue;

        auto* src = reinterpret_cast<void*>(entry.address);

        // Demote to T2 if the entry is warm enough
        bool demoted = false;
        if (t2_store_ && entry.recompute_cost >= config_.admission.min_recompute_cost) {
            BlobLocation loc{};
            Status s = t2_store_->write(*victim, src, entry.length,
                                         entry.checksum, &loc);
            if (s == Status::kOk) {
                entry.tier = static_cast<uint8_t>(Tier::kT2Nvme);
                entry.address = loc.offset;
                entry.segment_id = loc.segment_id;
                index_->upsert(*victim, entry);
                t2_eviction_->on_insert(*victim, entry.recompute_cost,
                                         victim->chunk_index);
                demoted = true;
            }
        }

        if (!demoted) {
            index_->erase(*victim);
        }

        // Free the T1 slab slot regardless of demotion outcome
        t1_allocator_->free_by_ptr(src);

        logger_->period_stats().evictions.fetch_add(1, std::memory_order_relaxed);
    }
}

double AttentionDBEngine::storage_utilization() const {
    double util = 0.0;
    if (t1_allocator_ && t1_allocator_->total_capacity() > 0) {
        util = static_cast<double>(t1_allocator_->used_bytes()) /
               t1_allocator_->total_capacity();
    }
    return util;
}

uint32_t AttentionDBEngine::compute_crc32(const void* data, size_t len) {
    // Simple CRC32 using a lookup table (Castagnoli polynomial)
    static constexpr uint32_t kCrc32Table[256] = {
        0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4, 0xC79A971F, 0x35F1141C,
        0x26A1E7E8, 0xD4CA64EB, 0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B,
        0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24, 0x105EC76F, 0xE235446C,
        0xF165B798, 0x030E349B, 0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384,
        0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54, 0x5D1D08BF, 0xAF768BBC,
        0xBC267848, 0x4E4DFB4B, 0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A,
        0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35, 0xAA64D611, 0x580F5512,
        0x4B5FA6E6, 0xB93425E5, 0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA,
        0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45, 0xF779DEAE, 0x05125DAD,
        0x1642AE59, 0xE4292D5A, 0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A,
        0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595, 0x417B1DBC, 0xB3109EBF,
        0xA0406D4B, 0x522BEE48, 0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957,
        0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687, 0x0C38D26C, 0xFE53516F,
        0xED03A29B, 0x1F682198, 0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927,
        0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38, 0xDBFC821C, 0x2997011F,
        0x3AC7F2EB, 0xC8AC71E8, 0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7,
        0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096, 0xA65C047D, 0x5437877E,
        0x4767748A, 0xB50CF789, 0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859,
        0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46, 0x7198540D, 0x83F3D70E,
        0x90A324FA, 0x62C8A7F9, 0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6,
        0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36, 0x3CDB9BDD, 0xCEB018DE,
        0xDDE0EB2A, 0x2F8B6829, 0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C,
        0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93, 0x082F63B7, 0xFA44E0B4,
        0xE9141340, 0x1B7F9043, 0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C,
        0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3, 0x55326B08, 0xA759E80B,
        0xB4091BFF, 0x466298FC, 0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C,
        0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033, 0xA24BB5A6, 0x502036A5,
        0x4370C551, 0xB11B4652, 0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
        0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D, 0xEF087A76, 0x1D63F975,
        0x0E330A81, 0xFC588982, 0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D,
        0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622, 0x38CC2A06, 0xCAA7A905,
        0xD9F75AF1, 0x2B9CD9F2, 0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED,
        0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530, 0x0417B1DB, 0xF67C32D8,
        0xE52CC12C, 0x1747422F, 0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF,
        0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0, 0xD3D3E1AB, 0x21B862A8,
        0x32E8915C, 0xC083125F, 0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540,
        0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90, 0x9E902E7B, 0x6CFBAD78,
        0x7FAB5E8C, 0x8DC0DD8F, 0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE,
        0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1, 0x69E9F0D5, 0x9B8273D6,
        0x88D28022, 0x7AB90321, 0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E,
        0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81, 0x34F4F86A, 0xC69F7B69,
        0xD5CF889D, 0x27A40B9E, 0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E,
        0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351,
    };

    uint32_t crc = 0xFFFFFFFF;
    auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        crc = kCrc32Table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

}  // namespace attentiondb
