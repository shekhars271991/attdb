#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ATTENTIONDB_OK = 0,
    ATTENTIONDB_NOT_FOUND = 1,
    ATTENTIONDB_REJECTED_ADMISSION = 2,
    ATTENTIONDB_REJECTED_BACKPRESSURE = 3,
    ATTENTIONDB_FULL = 4,
    ATTENTIONDB_BUFFER_TOO_SMALL = 5,
    ATTENTIONDB_CORRUPTION = 6,
    ATTENTIONDB_IO_ERROR = 7,
    ATTENTIONDB_INVALID_ARGUMENT = 8,
    ATTENTIONDB_ALREADY_EXISTS = 9,
    ATTENTIONDB_SHUTDOWN = 10,
} attentiondb_status_t;

#pragma pack(push, 1)
typedef struct {
    uint64_t model_id;
    uint64_t tenant_id;
    uint64_t chunk_hash;
    uint16_t layer_group_id;
    uint32_t chunk_index;
} attentiondb_key_t;
#pragma pack(pop)

typedef struct {
    uint32_t num_tokens;
    uint32_t recompute_cost;
    uint8_t  entry_type;      /* 0=conversation, 1=system_prompt, 2=rag_context */
    uint32_t ttl_seconds;
} attentiondb_put_opts_t;

typedef struct {
    /* Capacity */
    size_t   t1_total_bytes;
    size_t   t1_used_bytes;
    uint64_t t2_total_bytes_on_disk;
    uint32_t t2_num_segments;

    /* Index */
    size_t   index_entries;

    /* Eviction */
    size_t   eviction_protected;
    size_t   eviction_probationary;

    /* Admission */
    uint64_t admission_evaluated;
    uint64_t admission_rejected;

    /* Write buffer */
    uint64_t wb_submitted;
    uint64_t wb_rejected;
    uint64_t wb_flushed;
    double   wb_utilization;

    /* Checkpoint */
    uint64_t last_checkpoint_entries;
    uint64_t last_checkpoint_duration_ms;
} attentiondb_stats_t;

typedef struct attentiondb_config {
    const char* config_path;   /* Path to YAML config file, or NULL for defaults */
    const char* config_yaml;   /* Inline YAML string, or NULL */
} attentiondb_config_t;

typedef struct attentiondb_handle attentiondb_t;

/* Lifecycle */
attentiondb_status_t attentiondb_open(
    const attentiondb_config_t* config,
    attentiondb_t** handle_out);

attentiondb_status_t attentiondb_close(attentiondb_t* handle);

/* Core operations */
attentiondb_status_t attentiondb_put(
    attentiondb_t* handle,
    const attentiondb_key_t* key,
    const void* blob,
    size_t blob_len,
    const attentiondb_put_opts_t* opts);

attentiondb_status_t attentiondb_get(
    attentiondb_t* handle,
    const attentiondb_key_t* key,
    void* buf,
    size_t buf_len,
    size_t* blob_len_out);

attentiondb_status_t attentiondb_contains(
    attentiondb_t* handle,
    const attentiondb_key_t* keys,
    size_t num_keys,
    bool* results);

attentiondb_status_t attentiondb_delete(
    attentiondb_t* handle,
    const attentiondb_key_t* keys,
    size_t num_keys);

/* Batched variants */
attentiondb_status_t attentiondb_batched_put(
    attentiondb_t* handle,
    const attentiondb_key_t* keys,
    const void* const* blobs,
    const size_t* blob_lens,
    const attentiondb_put_opts_t* opts,
    size_t count);

attentiondb_status_t attentiondb_batched_get(
    attentiondb_t* handle,
    const attentiondb_key_t* keys,
    void* const* bufs,
    const size_t* buf_lens,
    size_t* blob_lens_out,
    size_t count);

/* Stats */
attentiondb_status_t attentiondb_stats(
    attentiondb_t* handle,
    attentiondb_stats_t* stats_out);

#ifdef __cplusplus
}
#endif
