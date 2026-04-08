#include "c_api.h"

#include <cstring>
#include <new>

#include "engine.h"

using namespace attentiondb;

struct attentiondb {
    AttentionDBEngine engine;
};

static attentiondb_status_t to_c_status(Status s) {
    return static_cast<attentiondb_status_t>(static_cast<uint8_t>(s));
}

static StorageKey to_storage_key(const attentiondb_key_t* k) {
    StorageKey sk;
    sk.model_id = k->model_id;
    sk.tenant_id = k->tenant_id;
    sk.chunk_hash = k->chunk_hash;
    sk.layer_group_id = k->layer_group_id;
    sk.chunk_index = k->chunk_index;
    return sk;
}

static PutOpts to_put_opts(const attentiondb_put_opts_t* o) {
    PutOpts opts;
    opts.num_tokens = o->num_tokens;
    opts.recompute_cost = o->recompute_cost;
    opts.entry_type = static_cast<EntryType>(o->entry_type);
    opts.ttl_seconds = o->ttl_seconds;
    return opts;
}

attentiondb_status_t attentiondb_open(const attentiondb_config_t* config,
                                       attentiondb_t** handle_out) {
    if (!config || !handle_out) return ATTENTIONDB_INVALID_ARGUMENT;

    Config cfg;
    if (config->config_path) {
        cfg = Config::LoadFromFile(config->config_path);
    } else if (config->config_yaml) {
        cfg = Config::LoadFromString(config->config_yaml);
    } else {
        cfg = Config::Default();
    }

    auto* h = new (std::nothrow) attentiondb();
    if (!h) return ATTENTIONDB_IO_ERROR;

    Status s = h->engine.open(cfg);
    if (s != Status::kOk) {
        delete h;
        return to_c_status(s);
    }

    *handle_out = h;
    return ATTENTIONDB_OK;
}

attentiondb_status_t attentiondb_close(attentiondb_t* handle) {
    if (!handle) return ATTENTIONDB_INVALID_ARGUMENT;
    handle->engine.close();
    delete handle;
    return ATTENTIONDB_OK;
}

attentiondb_status_t attentiondb_put(attentiondb_t* handle,
                                      const attentiondb_key_t* key,
                                      const void* blob, size_t blob_len,
                                      const attentiondb_put_opts_t* opts) {
    if (!handle || !key || !blob || !opts) return ATTENTIONDB_INVALID_ARGUMENT;
    StorageKey sk = to_storage_key(key);
    PutOpts po = to_put_opts(opts);
    return to_c_status(handle->engine.put(sk, blob, blob_len, po));
}

attentiondb_status_t attentiondb_get(attentiondb_t* handle,
                                      const attentiondb_key_t* key,
                                      void* buf, size_t buf_len,
                                      size_t* blob_len_out) {
    if (!handle || !key || !buf || !blob_len_out)
        return ATTENTIONDB_INVALID_ARGUMENT;
    StorageKey sk = to_storage_key(key);
    return to_c_status(handle->engine.get(sk, buf, buf_len, blob_len_out));
}

attentiondb_status_t attentiondb_contains(attentiondb_t* handle,
                                            const attentiondb_key_t* keys,
                                            size_t num_keys, bool* results) {
    if (!handle || !keys || !results) return ATTENTIONDB_INVALID_ARGUMENT;
    for (size_t i = 0; i < num_keys; ++i) {
        StorageKey sk = to_storage_key(&keys[i]);
        results[i] = handle->engine.contains(sk);
    }
    return ATTENTIONDB_OK;
}

attentiondb_status_t attentiondb_delete(attentiondb_t* handle,
                                          const attentiondb_key_t* keys,
                                          size_t num_keys) {
    if (!handle || !keys) return ATTENTIONDB_INVALID_ARGUMENT;
    for (size_t i = 0; i < num_keys; ++i) {
        StorageKey sk = to_storage_key(&keys[i]);
        handle->engine.del(sk);
    }
    return ATTENTIONDB_OK;
}

attentiondb_status_t attentiondb_batched_put(attentiondb_t* handle,
                                               const attentiondb_key_t* keys,
                                               const void* const* blobs,
                                               const size_t* blob_lens,
                                               const attentiondb_put_opts_t* opts,
                                               size_t count) {
    if (!handle || !keys || !blobs || !blob_lens || !opts)
        return ATTENTIONDB_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; ++i) {
        StorageKey sk = to_storage_key(&keys[i]);
        PutOpts po = to_put_opts(&opts[i]);
        Status s = handle->engine.put(sk, blobs[i], blob_lens[i], po);
        if (s != Status::kOk && s != Status::kRejectedAdmission &&
            s != Status::kRejectedBackpressure) {
            return to_c_status(s);
        }
    }
    return ATTENTIONDB_OK;
}

attentiondb_status_t attentiondb_batched_get(attentiondb_t* handle,
                                               const attentiondb_key_t* keys,
                                               void* const* bufs,
                                               const size_t* buf_lens,
                                               size_t* blob_lens_out,
                                               size_t count) {
    if (!handle || !keys || !bufs || !buf_lens || !blob_lens_out)
        return ATTENTIONDB_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; ++i) {
        StorageKey sk = to_storage_key(&keys[i]);
        Status s = handle->engine.get(sk, bufs[i], buf_lens[i], &blob_lens_out[i]);
        if (s == Status::kNotFound) {
            blob_lens_out[i] = 0;
        } else if (s != Status::kOk) {
            return to_c_status(s);
        }
    }
    return ATTENTIONDB_OK;
}

attentiondb_status_t attentiondb_stats(attentiondb_t* handle,
                                        attentiondb_stats_t* stats_out) {
    if (!handle || !stats_out) return ATTENTIONDB_INVALID_ARGUMENT;
    auto s = handle->engine.stats();

    stats_out->t1_total_bytes = s.t1_total_bytes;
    stats_out->t1_used_bytes = s.t1_used_bytes;
    stats_out->t2_total_bytes_on_disk = s.t2_total_bytes_on_disk;
    stats_out->t2_num_segments = s.t2_num_segments;
    stats_out->index_entries = s.index_entries;
    stats_out->eviction_protected = s.eviction_protected;
    stats_out->eviction_probationary = s.eviction_probationary;
    stats_out->admission_evaluated = s.admission_evaluated;
    stats_out->admission_rejected = s.admission_rejected;
    stats_out->wb_submitted = s.wb_submitted;
    stats_out->wb_rejected = s.wb_rejected;
    stats_out->wb_flushed = s.wb_flushed;
    stats_out->wb_utilization = s.wb_utilization;
    stats_out->last_checkpoint_entries = s.last_checkpoint_entries;
    stats_out->last_checkpoint_duration_ms = s.last_checkpoint_duration_ms;

    return ATTENTIONDB_OK;
}
