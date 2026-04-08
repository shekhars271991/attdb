# AttentionDB Phase 0 — Local Storage Engine MVP

> This document is a focused extraction of the full [DESIGN.md](DESIGN.md) for the Phase 0
> MVP: a local-only storage engine deployed on a GPU node behind LMCache. For the complete
> architecture including clustering, remote tiers, and security, see DESIGN.md.

---

## 1. Scope and Goals

### What Phase 0 Is

A purpose-built C/C++ storage engine for KV cache offloading that runs entirely on a single
GPU node. It replaces LMCache's built-in storage backends (`LocalCPUBackend`,
`LocalDiskBackend`, `GdsBackend`) with a unified, higher-performance implementation
optimized for large compressed blobs.

Phase 0 ships as an in-process library that implements LMCache's `StorageBackendInterface`.
LMCache continues to own token-to-key mapping, prefix matching, chunking, vLLM integration,
and compression/decompression.

### What Phase 0 Is NOT

- Not a cache orchestration system (LMCache does that)
- No networking, no remote tiers (T3/T4), no cluster
- No placement groups, rendezvous hashing, or replication
- No Prometheus, OpenTelemetry, or metric exporters (structured logs only)
- No security, multi-tenant isolation, or authentication
- No popularity sketches or predictive prefetch
- No compression/decompression (blobs arrive pre-compressed from LMCache)

These are all deferred to Phase 1 (see [PHASE1-SERVER.md](PHASE1-SERVER.md)).

### What LMCache Continues to Own

| Concern | LMCache component | Why it stays in LMCache |
|---|---|---|
| Token-to-key mapping | `ChunkedTokenDatabase` | Tightly coupled to tokenizer and vLLM |
| Prefix matching | `StorageManager.batched_contains` tier walk | Cross-backend orchestration |
| Chunking | `ChunkedTokenDatabase._chunk_tokens` | Chunk boundaries are inference-level |
| vLLM integration | `LMCacheConnectorV1` | Python, model-lifecycle aware |
| Compression/serde | CacheGen, KIVI, TurboQuant | Needs GPU context |
| Inference metrics | `LMCStatsMonitor`, `PrometheusLogger` | Needs request-level context |

### Deployment Model

```
GPU Node:
  ┌──────────────────────────────────────────┐
  │ vLLM / SGLang                            │
  │         │                                │
  │         ▼                                │
  │ LMCache (orchestration)                  │
  │   prefix matching, chunking, serde       │
  │         │                                │
  │         ▼  StorageBackendInterface        │
  │ AttentionDB Storage Engine (in-process)   │
  │ ┌──────────────────────────────────────┐ │
  │ │ T1: CPU DRAM  (pinned, 16-32 GB)    │ │  Index + hot compressed blobs
  │ │ T2: Local NVMe (1-2 TB)             │ │  Warm blobs, GDS-capable
  │ └──────────────────────────────────────┘ │
  └──────────────────────────────────────────┘
```

### Latency Targets (AttentionDB's contribution only, excluding decompression)

| Path | p50 target | p99 target |
|------|-----------|-----------|
| T1 DRAM hit (blob delivered to caller buffer) | < 30 μs | < 80 μs |
| T2 NVMe hit (blob delivered to caller buffer) | < 100 μs | < 300 μs |

Note: GPU decompression (~50–100 μs) happens after AttentionDB returns the blob to LMCache.
The end-to-end latency including decompression is covered in DESIGN.md Section 1.

### Non-Goals

- General-purpose KV storage
- Compression/decompression of any kind
- vLLM or inference engine integration (that's LMCache)
- Metric export infrastructure
- Remote storage or networking

---

## 2. Architecture

### Data Flow

**Store** (LMCache calls `put`):
```
LMCache compresses KV tensor on GPU
  → LMCache calls attentiondb_put(key, compressed_blob, metadata)
  → AttentionDB: admission check → write buffer (pinned DRAM) → return immediately
  → Background: write buffer → T1 DRAM slab → T2 NVMe (io_uring)
```

**Retrieve** (LMCache calls `contains` then `get`):
```
LMCache calls attentiondb_contains(key) → true/false
  → if true: LMCache calls attentiondb_get(key)
    → AttentionDB checks T1 DRAM → T2 NVMe → returns blob
  → LMCache decompresses on GPU → returns KV tensor to vLLM
```

The internal T1 → T2 cascade is transparent to LMCache. From LMCache's perspective, it
calls `get(key)` and either gets a blob or a miss.

### Tier Responsibilities

| Tier | Medium | Contents | Access pattern |
|------|--------|----------|---------------|
| **T1** | CPU DRAM (pinned) | Hash index + hot blobs | Read: every `get()`. Write: from write buffer. |
| **T2** | Local NVMe | Warm blobs, spilled from T1 | Read: T1 miss, via GDS or io_uring. Write: async spill. |

---

## 3. Storage Engine

### 3.1 Storage Key

AttentionDB receives keys from LMCache's adapter, it does not compute them from tokens.

```
StorageKey {
    model_id:        u64,     // Routing hint (unused in Phase 0, needed for Phase 1 PGs)
    tenant_id:       u64,     // Routing hint (can be constant in Phase 0)
    chunk_hash:      u64,     // Identity: from LMCache's CacheEngineKey.chunk_hash
    layer_group_id:  u16,     // Which layer group
    chunk_index:     u32,     // Position within the sequence
}
```

**Key size**: 26 bytes fixed.

**Who computes what**: LMCache's `ChunkedTokenDatabase` computes the rolling prefix hash
and chunk boundaries. The `StorageBackendInterface` adapter maps LMCache's
`CacheEngineKey` fields to the `StorageKey` struct. AttentionDB treats `chunk_hash` as
an opaque identity — it never looks at tokens.

### 3.2 In-Memory Index

Maps `StorageKey → IndexEntry`. Each entry is **64 bytes** (one CPU cache line):

```
IndexEntry {                        // 64 bytes, cache-line aligned
    // Storage location (25 bytes)
    tier:           u8,             // T1 or T2
    address:        u64,            // DRAM pointer or NVMe byte offset
    length:         u32,            // Blob size in bytes
    segment_id:     u32,            // Log segment file ID (T2 only, 0 for T1)
    flags:          u16,            // Codec hint (opaque to engine), entry state

    // Eviction metadata (21 bytes)
    recompute_cost: u32,            // Provided by LMCache at put() time
    last_access:    u48,            // Timestamp of last get() (6 bytes)
    access_count:   u16,            // Saturating counter
    slru_segment:   u8,             // 0 = probationary, 1 = protected
    num_tokens:     u32,            // Provided by LMCache at put() time
    created_at:     u48,            // Creation timestamp (6 bytes)

    // Integrity (4 bytes)
    checksum:       u32,            // CRC32C of the blob data

    // Padding (14 bytes → total 64)
    _reserved:      [u8; 14],
}
```

**Implementation**: Lock-free concurrent hash map (open addressing, Robin Hood hashing).
Phase 0 uses a **single hash map** (no PG partitioning — that's Phase 1).
- Read path: lock-free (atomic load + compare)
- Write path: per-bucket spinlock
- Capacity: pre-allocated, background resize at load factor > 0.7

**Checkpointing**: Serialized to NVMe every 30s or on clean shutdown. On restart, loaded
instead of scanning blob segments.

### 3.3 DRAM Cache (T1)

Stores compressed blobs in CUDA pinned memory.

**The DRAM cache is optional.** When disabled, the index still lives in DRAM, but blob
data is served from NVMe via GDS or io_uring.

**Allocator**: Slab allocator with size classes (64 KB – 4 MB). Pre-allocates at startup
using hugepages. CUDA pinned memory (`cudaMallocHost`) for zero-copy to GPU HBM.

**Configuration**:
```yaml
dram_cache:
  enabled: true
  size: "16GB"
  allocator:
    type: "slab"
    size_classes: [65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
    auto_tune_classes: true
  hugepages:
    enabled: true
    page_size: "2MB"
  memory_type: "pinned"                  # "pinned" (cudaMallocHost) | "standard" (testing)
```

### 3.4 NVMe Blob Store (T2)

Log-structured blob store.

```
Segment file layout (default 1 GB):
┌──────────────────────────────────────────────────────┐
│ Segment Header (4 KB)                                │
│   magic, version, segment_id, created_at,            │
│   entry_count, live_count, checksum                  │
├──────────────────────────────────────────────────────┤
│ Entry 0: [Header 64B] [Blob data, 4KB-aligned]      │
│ Entry 1: [Header 64B] [Blob data, 4KB-aligned]      │
│ ...                                                  │
└──────────────────────────────────────────────────────┘
```

**Write path**: Blobs accumulate in write buffer → flush to active segment via `io_uring`
(`O_DIRECT`) → index updated after durable write → segment sealed at size limit.

**Read path**: Index returns `(segment_id, offset, length)` → GDS `cuFileRead` directly
into caller buffer, or `io_uring` into DRAM.

**GC**: Background thread rewrites live entries from fragmented segments. Rate-limited
to configured fraction of NVMe bandwidth.

**Configuration**:
```yaml
nvme_store:
  enabled: true
  paths: ["/mnt/nvme0/attentiondb"]
  max_size_per_drive: "1TB"
  segments:
    size: "1GB"
    fd_pool_size: 256
  write_buffer:
    size: "8MB"
    flush_interval_ms: 50
  alignment: "4KB"
  io:
    engine: "io_uring"
    io_uring_queue_depth: 128
    io_uring_registered_buffers: true
    gds_thread_pool_size: 4
  gc:
    enabled: true
    trigger_fragmentation: 0.5
    max_bandwidth_fraction: 0.2
    schedule: "continuous"
```

### 3.5 Blob Opacity

AttentionDB stores opaque blobs. It does not compress, decompress, or inspect blob contents.
Compression is handled by LMCache before `put()` and after `get()`.

---

## 4. Admission and Eviction

### 4.1 Admission Control

Not all blobs should be stored. In Phase 0, admission uses metadata provided by LMCache
at `put()` time (no popularity sketch — that requires the cluster in Phase 1).

**Admission criteria** (evaluated in order):

| Priority | Rule | Rationale |
|----------|------|-----------|
| 1 | Always admit `entry_type=system_prompt` | High reuse probability |
| 2 | Reject if `recompute_cost < min_recompute_cost` | Cheaper to recompute |
| 3 | Reject if `recompute_cost < dynamic_threshold` when storage > 70% full | Cost-based filtering |

The `recompute_cost` and `entry_type` are provided by LMCache's adapter in the
`attentiondb_put_opts_t` struct. AttentionDB doesn't know what tokens are or how expensive
prefill is — it trusts the metadata LMCache provides.

**Dynamic threshold** scales with storage pressure:
```
Storage < 70%:   threshold = 0
Storage 70-90%:  threshold scales linearly
Storage > 90%:   threshold = 2× base
```

### 4.2 Cost-Weighted Segmented LRU (CW-SLRU)

Each tier uses a two-segment eviction policy:

```
┌──────────────────────────────────┐
│        Protected Segment          │  20-30% of capacity
│  (high recompute_cost entries)    │
│  Eviction: demote to Probationary │
├──────────────────────────────────┤
│       Probationary Segment        │  70-80% of capacity
│  (everything else + demoted)      │
│  Eviction: discard or demote tier │
│  Promotion: re-accessed + high    │
│             cost → Protected      │
└──────────────────────────────────┘
```

**Within segments**: Evict suffix chunks before prefix chunks (prefix chunks are shared
by more future requests — prefix-cache-friendly ordering).

### 4.3 Cross-Tier Demotion

| From | To | When |
|------|----|------|
| T1 (DRAM) | T2 (NVMe) | T1 full, entry warm but not hot |
| T1 (DRAM) | Discard | Entry cold, below cost threshold |
| T2 (NVMe) | Discard | Entry cold, below cost threshold |

Demotion is async — write to T2 in background, remove from T1 after confirmation.

---

## 5. Storage Engine API

### 5.1 C API

```c
// Lifecycle
attentiondb_status_t attentiondb_open(
    const attentiondb_config_t *config,
    attentiondb_t **handle
);
attentiondb_status_t attentiondb_close(attentiondb_t *handle);

// Core operations (called by LMCache adapter)
attentiondb_status_t attentiondb_put(
    attentiondb_t *handle,
    const attentiondb_key_t *key,       // 26-byte StorageKey
    const void *blob,                   // Opaque compressed blob
    size_t blob_len,
    const attentiondb_put_opts_t *opts  // num_tokens, recompute_cost, entry_type, ttl
);

attentiondb_status_t attentiondb_get(
    attentiondb_t *handle,
    const attentiondb_key_t *key,
    void *buf,                          // Caller-provided buffer
    size_t buf_len,
    size_t *blob_len_out
);

attentiondb_status_t attentiondb_contains(
    attentiondb_t *handle,
    const attentiondb_key_t *keys,
    size_t num_keys,
    bool *results
);

attentiondb_status_t attentiondb_delete(
    attentiondb_t *handle,
    const attentiondb_key_t *keys,
    size_t num_keys
);

// Batched variants
attentiondb_status_t attentiondb_batched_put(...);
attentiondb_status_t attentiondb_batched_get(...);

// Stats
attentiondb_status_t attentiondb_stats(
    attentiondb_t *handle,
    attentiondb_stats_t *stats
);
```

**What this API does NOT include** (handled by LMCache):
- No `lookup` + `retrieve` split — LMCache's `async_lookup_and_prefetch` handles this
- No CUDA stream parameters — blobs are returned to CPU buffers; LMCache handles GPU
- No compression codec selection — blobs arrive pre-compressed
- No prefix matching — LMCache walks `contains()` calls to find the longest cached prefix

### 5.2 Store Metadata

```c
typedef struct {
    uint32_t num_tokens;            // Token count (for eviction cost estimation)
    uint32_t recompute_cost;        // num_tokens × cost_per_token (for CW-SLRU)
    uint8_t  entry_type;            // 0=conversation, 1=system_prompt, 2=rag_context
    uint32_t ttl_seconds;           // 0 = no expiry
} attentiondb_put_opts_t;
```

LMCache computes `recompute_cost` (e.g., `num_tokens × cost_per_token(model_id)`) and
passes it at store time. This is how AttentionDB makes intelligent eviction decisions
without understanding inference semantics.

### 5.3 Python Bindings

PyO3 or cffi bindings exposing the C API to Python. Used by the LMCache adapter.

### 5.4 Internal Write Buffer and Backpressure

`put()` returns immediately after admission check + write buffer copy. Background flush
thread writes to NVMe.

Progressive backpressure levels:

| Buffer fill | Mode | Behavior |
|-------------|------|----------|
| 0–50% | NORMAL | Accept all admitted entries |
| 50–75% | CAUTIOUS | Raise cost threshold dynamically |
| 75–90% | SELECTIVE | Only high-value entries + system prompts |
| 90–100% | EMERGENCY | Only system prompts |
| 100% | FULL | Reject all `put()` calls (return `ATTENTIONDB_FULL`) |

---

## 6. LMCache Integration

AttentionDB implements LMCache's `StorageBackendInterface`:

```python
class AttentionDBBackend(StorageBackendInterface):
    def __init__(self, dst_device, config):
        self._engine = attentiondb.open(config.attentiondb_config)

    def contains(self, key: CacheEngineKey) -> bool:
        storage_key = self._map_key(key)
        return self._engine.contains(storage_key)

    def batched_contains(self, keys, search_range, pin=False):
        storage_keys = [self._map_key(k) for k in keys]
        return self._engine.batched_contains(storage_keys)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        storage_key = self._map_key(key)
        blob = self._engine.get(storage_key)
        if blob is None:
            return None
        return MemoryObj.from_buffer(blob, device=self.dst_device)

    def batched_submit_put_task(self, keys, memory_objs, **kwargs):
        for key, mem in zip(keys, memory_objs):
            storage_key = self._map_key(key)
            opts = self._compute_put_opts(key)
            self._engine.put(storage_key, mem.data, len(mem.data), opts)

    def _map_key(self, key: CacheEngineKey) -> attentiondb.StorageKey:
        """Map LMCache's key format to AttentionDB's 26-byte StorageKey."""
        return attentiondb.StorageKey(
            model_id=hash_u64(key.model_name),
            tenant_id=self._tenant_id,
            chunk_hash=key.chunk_hash,
            layer_group_id=key.worker_id,
            chunk_index=0,  # derived from key ordering
        )

    def _compute_put_opts(self, key: CacheEngineKey) -> attentiondb.PutOpts:
        """Compute eviction metadata for this entry."""
        return attentiondb.PutOpts(
            num_tokens=self._chunk_size,
            recompute_cost=self._chunk_size * self._cost_per_token,
            entry_type=self._classify_entry(key),
            ttl_seconds=0,
        )
```

**Key mapping**: The adapter translates between LMCache's `CacheEngineKey` (model_name,
world_size, worker_id, chunk_hash, dtype) and AttentionDB's compact 26-byte `StorageKey`.
This is a thin, stateless mapping.

**Metadata computation**: The adapter computes `recompute_cost` from model-specific
calibration (measured once during model loading). It classifies entries as system_prompt,
conversation, or rag_context based on LMCache's context.

---

## 7. Observability

Structured logging only. No Prometheus, no OpenTelemetry. JSON to stderr.

| Event | Level | Fields |
|-------|-------|--------|
| `get` hit | DEBUG | `tier`, `key_hash`, `latency_us`, `blob_size` |
| `get` miss | DEBUG | `key_hash` |
| `put` accepted | DEBUG | `key_hash`, `blob_size`, `recompute_cost` |
| `put` rejected (admission) | DEBUG | `key_hash`, `recompute_cost`, `reason` |
| `put` rejected (backpressure) | WARN | `key_hash`, `recompute_cost`, `level` |
| Eviction | INFO | `tier`, `key_hash`, `reason`, `recompute_cost` |
| Index checkpoint | INFO | `entries`, `duration_ms`, `size_bytes` |
| GC cycle | INFO | `segment_id`, `live_ratio`, `reclaimed_bytes` |
| Startup | INFO | `config_summary`, `t1_size`, `t2_size`, `gds_available` |

**Periodic summary** (every 60s at INFO level):
```json
{
  "interval_s": 60,
  "t1_hit_rate": 0.82,
  "t2_hit_rate": 0.11,
  "miss_rate": 0.07,
  "gets": 61555,
  "puts": 3200,
  "puts_rejected": 45,
  "evictions": 1100,
  "t1_utilization": 0.78,
  "t2_utilization": 0.45,
  "avg_get_latency_us": 35,
  "p99_get_latency_us": 120,
  "gc_reclaimed_bytes": 1073741824
}
```

---

## 8. Configuration

```yaml
attentiondb:
  mode: "local_only"

  local:
    t1_dram_size: "16GB"
    t2_nvme_path: "/mnt/nvme0/attentiondb"
    t2_nvme_size: "1TB"
    t2_use_gds: true
    write_buffer_size: "256MB"

  admission:
    min_recompute_cost: 64
    base_threshold: 100

  eviction:
    policy: "cw_slru"
    protected_ratio: 0.25

  timeouts:
    get_ms: 5

  index:
    checkpoint_interval_s: 30

  logging:
    level: "info"
    format: "json"
    periodic_summary_interval_s: 60
```

---

## 9. Implementation Tasks

| # | Task | Dependency | Estimated effort |
|---|------|------------|-----------------|
| 1 | Lock-free concurrent hash map (C/C++) | — | 2 weeks |
| 2 | Slab allocator with CUDA pinned memory | — | 1 week |
| 3 | Log-structured blob store with io_uring | — | 2 weeks |
| 4 | CW-SLRU eviction engine | #1 | 1 week |
| 5 | Cost-aware admission control | #4 | 3 days |
| 6 | GDS integration for T2 reads | #3 | 1 week |
| 7 | Index checkpointing | #1, #3 | 3 days |
| 8 | Write buffer with backpressure | #2, #3 | 1 week |
| 9 | C API surface (open/close/put/get/contains/delete/stats) | #1–#8 | 1 week |
| 10 | Python bindings (PyO3 or cffi) | #9 | 3 days |
| 11 | LMCache `StorageBackendInterface` adapter | #10 | 3 days |
| 12 | Structured logging | #9 | 2 days |
| 13 | Integration test: vLLM + LMCache + AttentionDB backend | #11 | 1 week |
| 14 | Benchmark: compare vs LMCache `LocalDiskBackend` | #13 | 3 days |

**Critical path**: #1 → #4 → #9 → #10 → #11 → #13 (approximately 6 weeks)

**Parallelizable**: #2 and #3 can run in parallel with #1. #6 can run in parallel with #4–#5.

---

## 10. Known Risks (Phase 0 Scope)

**R1. GDS availability**: GPUDirect Storage requires specific NVMe controllers and CUDA
versions. Fallback to `io_uring` + `cudaMemcpy` must be graceful.

**R2. io_uring kernel version**: Requires Linux >= 5.1 (basic) or >= 5.11 (registered
buffers). Fallback to `pread`/`pwrite` with `O_DIRECT`.

**R3. Index checkpoint gap**: On crash, up to 30s of index entries are lost. Blobs exist
on NVMe but are unreachable. Acceptable for a cache (recomputable data).

**R4. LMCache adapter complexity**: Mapping between LMCache's `CacheEngineKey` and
AttentionDB's `StorageKey` must be correct and deterministic. Particularly, the
`chunk_hash` must be the same value that LMCache will use for future lookups.

**R5. Storage engine crash isolation**: A segfault in the in-process C library crashes
the entire LMCache + vLLM process. Mitigation: Rust for memory safety, extensive fuzzing.
