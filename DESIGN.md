# AttentionDB Design Document

## 1. Problem Statement

Large language model inference pipelines are bottlenecked by the KV cache: the key-value
attention state that must be maintained for every token in a conversation's context. KV caches
consume scarce GPU HBM, and recomputing them (prefill) is the primary contributor to
Time-To-First-Token (TTFT) latency.

**KV cache offloading** moves KV state out of GPU HBM and into cheaper, higher-capacity storage
tiers — CPU DRAM and NVMe SSDs — so it can be recalled on cache hit instead of recomputed.
When a new request shares a prefix with a previously-seen conversation (same system prompt,
shared RAG context, multi-turn continuation), the offloaded KV cache can be loaded back into GPU
HBM in microseconds instead of the milliseconds-to-seconds that prefill would take.

AttentionDB is a **purpose-built database for KV cache offloading** in inference pipelines.
It is not a general-purpose key-value store. Every design decision — storage format, eviction
policy, clustering strategy, observability — is optimized for the specific access patterns,
data shapes, and latency requirements of LLM inference.

### Design Goals

**End-to-end latency targets** (from cache lookup initiation to decompressed KV in GPU HBM):

| Path | p50 target | p99 target | Notes |
|------|-----------|-----------|-------|
| T1 local DRAM hit | < 100 μs | < 300 μs | Dominated by GPU decompression |
| T2 local NVMe via GDS | < 300 μs | < 800 μs | Dominated by NVMe read + decompression |
| T3 remote DRAM via RDMA | < 500 μs | < 2 ms | Dominated by RDMA round-trip + decompression |
| T3 remote DRAM via TCP | < 2 ms | < 5 ms | TCP transport adds kernel copies and syscall overhead |
| T4 remote NVMe via RDMA | < 2 ms | < 5 ms | Remote NVMe read + RDMA transfer + decompression |

**Component latency budgets** (realistic, not best-case microbenchmarks):

```
T1 DRAM hit breakdown:
  Index lookup (hash + cache-line read):     ~0.1 μs
  Slab pointer chase:                        ~0.1 μs
  memcpy to pinned buffer:                   ~5–20 μs  (depends on blob size, 256KB–4MB)
  PCIe transfer to GPU HBM:                  ~10–30 μs (PCIe Gen4/5 bandwidth)
  GPU TurboQuant decompress:                 ~50–100 μs (kernel launch + compute)
  Queueing / scheduling overhead:            ~5–20 μs
  ───────────────────────────────────────────────────────
  Total:                                     ~70–170 μs

T2 NVMe via GDS breakdown:
  Index lookup:                              ~0.1 μs
  io_uring submit + NVMe read:               ~50–150 μs (depends on blob size, device load)
  GDS DMA to GPU HBM:                        ~10–30 μs
  GPU TurboQuant decompress:                 ~50–100 μs
  Queueing / contention with other I/O:      ~20–100 μs (varies with concurrent read load)
  ───────────────────────────────────────────────────────
  Total:                                     ~130–380 μs

T3 remote DRAM via RDMA breakdown:
  Local index lookup:                        ~0.1 μs
  RDMA request to server:                    ~2–5 μs (one-sided RDMA write for request)
  Server index lookup:                       ~0.1 μs
  RDMA data transfer (256KB–4MB):            ~20–200 μs (100 Gbps InfiniBand)
  NIC queueing under contention:             ~10–100 μs (multiple GPU nodes competing)
  PCIe transfer to GPU HBM:                  ~10–30 μs
  GPU TurboQuant decompress:                 ~50–100 μs
  ───────────────────────────────────────────────────────
  Total:                                     ~90–435 μs

T3 remote DRAM via TCP breakdown:
  Same as RDMA, but replace RDMA steps with:
  TCP socket send/recv:                      ~100–500 μs (kernel copies, syscalls, Nagle/ACK)
  memcpy from kernel buffer:                 ~5–20 μs
  ───────────────────────────────────────────────────────
  Total:                                     ~200–800 μs (higher variance under load)
```

**Important caveats**:
- All numbers assume a 256 KB compressed blob (~1M tokens × 4 layers at INT4). Larger blobs
  (4 MB for long-context) shift the distribution toward the upper end of each range.
- p99 under production load (32+ concurrent GPU clients) can be 2–3× the isolated benchmark
  numbers due to NIC contention, NVMe queue depth saturation, and PCIe bus contention.
- GPU decompression latency (~50–100 μs) is a hard floor that cannot be reduced by storage
  optimizations. This makes T1 DRAM hits only ~2× faster than T2 NVMe hits — the storage
  read latency is partially masked by the decompression bottleneck.

**Other goals**:

| Goal | Target |
|------|--------|
| Write throughput | Async, non-blocking to inference pipeline |
| Cache hit rate (shared prefixes) | > 95% for system prompts, > 70% for conversations |
| Cold start (new GPU node to serving) | < 30 seconds to first cache-assisted request |
| Failure recovery | No inference downtime; graceful fallback to prefill |
| Compression support | Native TurboQuant / INT2-INT4 quantization, 4-8x reduction |

### Non-Goals

- General-purpose KV storage (no range scans, transactions, or secondary indexes)
- Strong consistency (KV cache entries are immutable and recomputable)
- Support for non-inference workloads

### Relationship to LMCache

AttentionDB is a **storage engine**, not a cache orchestration system. It sits *below*
LMCache in the stack, replacing LMCache's storage backends with purpose-built C/C++
implementations.

```
┌─────────────────────────────────────────────────────────────┐
│ Inference Engine (vLLM / SGLang)                            │
├─────────────────────────────────────────────────────────────┤
│ LMCache (orchestration layer)                               │
│   - Token-to-key mapping (rolling prefix hash, chunking)    │
│   - Prefix matching (longest cached prefix detection)       │
│   - vLLM KV connector integration                           │
│   - Two-phase lookup + retrieve orchestration               │
│   - Tier routing (which backend to query, in what order)    │
│   - Compression / serialization (CacheGen, KIVI, etc.)      │
│   - Inference-correlated observability (TTFT, hit rates)    │
├─────────────────────────────────────────────────────────────┤
│ AttentionDB (storage engine)                                │
│   - Lock-free index (C/C++, 64-byte cache-line entries)     │
│   - Slab allocator (pinned DRAM, RDMA-registered)           │
│   - Log-structured blob store (io_uring, GDS)               │
│   - CW-SLRU eviction with cost-aware admission              │
│   - Internal DRAM ↔ NVMe tiering (transparent to LMCache)  │
│   - Write buffer with progressive backpressure              │
│   - Clustering: PGs, rendezvous hashing, epoch leases       │
│   - RDMA/TCP transport for remote storage                   │
└─────────────────────────────────────────────────────────────┘
```

**What LMCache owns** (AttentionDB does NOT duplicate):
- Token-to-key mapping and prefix hash computation
- Chunk boundary detection and chunk management
- vLLM `KVConnectorBase` integration
- Cross-backend tier orchestration (deciding which backend to query)
- Compression/decompression codec management (TurboQuant, CacheGen, KIVI)
- Inference-level metrics (TTFT savings, prefill time avoided, request-level hit rates)

**What AttentionDB owns** (LMCache does NOT provide):
- High-performance C/C++ storage engine (vs. Python dict + buffered I/O)
- Log-structured blob store with io_uring/GDS (vs. one-file-per-chunk)
- Cost-weighted segmented LRU eviction (vs. basic LRU/LFU)
- Cost-aware admission control with progressive backpressure
- Inference-aware clustering (placement groups, shadow join, epoch leases)
- RDMA transport for remote tiers

**Integration**: AttentionDB implements LMCache's `StorageBackendInterface`. LMCache calls
`put(key, blob)` / `get(key) → blob` / `contains(key) → bool`. AttentionDB manages
memory, disk I/O, eviction, and admission internally. LMCache passes store metadata
(`num_tokens`, `recompute_cost`) at write time so AttentionDB can make eviction and
admission decisions.

---

## 2. Architecture Overview

AttentionDB uses a **hybrid architecture** with local storage on GPU nodes for low-latency hot data
and a dedicated remote cluster for shared data, durability, and cold storage. LMCache sits above
AttentionDB as the orchestration layer, handling token-to-key mapping, prefix matching, and vLLM
integration (see Section 1, "Relationship to LMCache").

```
GPU Node (× N):
  ┌──────────────────────────────────────┐
  │ Inference Engine (vLLM / SGLang)     │
  │         │                            │
  │         ▼                            │
  │ LMCache (orchestration)              │
  │   prefix matching, chunking, serde   │
  │         │                            │
  │         ▼                            │
  │ AttentionDB Storage Engine           │
  │ ┌──────────────────────────────────┐ │
  │ │ T0: GPU HBM  (active decode KV)  │ │  Managed by inference engine
  │ │ T1: CPU DRAM  (16-32 GB)         │ │  Index + hot compressed blobs
  │ │ T2: Local NVMe (1-2 TB)          │ │  Warm blobs (GDS capable)
  │ └──────────────┬───────────────────┘ │
  └────────────────┼─────────────────────┘
                   │ RDMA / NIXL (shared and cold data only)
                   │ (~10-20% of reads under steady state)
  ┌────────────────┼─────────────────────┐
  │ AttentionDB Server Cluster (× M)     │
  │ ┌──────────────▼───────────────────┐ │
  │ │ T3: DRAM (256-512 GB per node)   │ │  Shared prefix KV (system prompts,
  │ │     Index + hot shared cache      │ │  popular RAG contexts)
  │ ├──────────────────────────────────┤ │
  │ │ T4: NVMe (12-48 TB per node)     │ │  Full corpus, historical
  │ │     Log-structured blob store     │ │  conversations, cold data
  │ └──────────────────────────────────┘ │
  │                                      │
  │ Metadata plane (gossip, all nodes)   │
  └──────────────────────────────────────┘
```

### Tier Responsibilities

| Tier | Location | Medium | Contents | Access pattern |
|------|----------|--------|----------|---------------|
| **T0** | GPU node | GPU HBM | Active decode sequence KV | Managed by inference engine, not AttentionDB |
| **T1** | GPU node | CPU DRAM (pinned) | Hash index + hottest compressed KV blocks | Read: every cache lookup. Write: promote from T2/T3 |
| **T2** | GPU node | Local NVMe | Per-session warm KV, recently evicted from T1 | Read: T1 miss, via GDS to GPU. Write: async spill from T1 |
| **T3** | AttentionDB cluster | CPU DRAM | Shared prefix KV (system prompts, popular contexts) | Read: T2 miss for shared data, via RDMA. Write: async replication from GPU nodes |
| **T4** | AttentionDB cluster | NVMe array | Full KV corpus, cold/historical data | Read: T3 miss, promoted to T3 on access. Write: background flush from T3 |

### Data Flow

**Store path** (after prefill completes on GPU):
```
LMCache receives KV tensors from inference engine
  → LMCache compresses (TurboQuant / CacheGen) on GPU
  → LMCache calls AttentionDB.put(key, compressed_blob, metadata)
  → AttentionDB write buffer (pinned CPU DRAM)
  → Background: write buffer → T2 local NVMe (io_uring / GDS write)
  → Background: write buffer → T3 remote DRAM (RDMA, for shared prefixes only)
```

**Retrieve path** (on cache hit):
```
LMCache calls AttentionDB.contains(key) → hit/miss
  → if hit: LMCache calls AttentionDB.get(key)
    → AttentionDB checks T1 DRAM → T2 NVMe → T3 remote → T4 remote internally
    → returns compressed blob to LMCache
  → LMCache decompresses on GPU, returns KV tensors to inference engine
  → if miss at all tiers: LMCache triggers full prefill recompute
```

Note: the internal tier cascade (T1 → T2 → T3 → T4) is transparent to LMCache. From
LMCache's perspective, it calls `get(key)` and either gets a blob or a miss.

---

## 3. Storage Engine

### 3.1 Storage Key

AttentionDB indexes entries by an internal **storage key**. The mapping from inference-level
concepts (tokens, chunks, prefix hashes) to this storage key is performed by the integration
adapter — typically the LMCache `StorageBackendInterface` implementation. AttentionDB
receives the key, it does not compute it from tokens.

```
StorageKey {
    model_id:        u64,     // Routing: identifies the model (for PG placement + eviction tuning)
    tenant_id:       u64,     // Routing: tenant isolation (for PG placement + resource quotas)
    chunk_hash:      u64,     // Identity: hash of this specific chunk (provided by LMCache)
    layer_group_id:  u16,     // Identity: which layer group (for pipeline/tensor parallel)
    chunk_index:     u32,     // Identity: position within the sequence
}
```

**Who computes what**:
- `model_id` and `tenant_id`: The adapter maps LMCache's `CacheEngineKey.model_name` (plus
  tokenizer hash) and deployment context to these u64 values. These fields are used by
  AttentionDB for PG routing (Section 4.2) and per-model eviction tuning.
- `chunk_hash`: Directly from LMCache's `CacheEngineKey.chunk_hash`. LMCache computes this
  via its `ChunkedTokenDatabase` rolling prefix hash. AttentionDB treats it as an opaque
  identity field.
- `layer_group_id` and `chunk_index`: From LMCache's key decomposition.

**Key correctness is LMCache's responsibility**: Token-to-key mapping, prefix hash
computation, tokenizer stability, and chunk boundary detection are all handled by LMCache's
`ChunkedTokenDatabase`. AttentionDB trusts the key it receives. If LMCache computes an
incorrect key (e.g., due to a tokenizer change), AttentionDB will see cache misses — not
errors. This is the correct failure mode for a cache system.

**Key size**: 26 bytes fixed. Fits in a single cache line with padding.

### 3.2 In-Memory Index (T1 / T3)

The index maps `CacheKey → IndexEntry`. Each index entry is **64 bytes** (one CPU cache line),
following the same cache-line alignment principle as Aerospike's `as_index`. The entry packs
storage location, eviction metadata, and access statistics into a single cache-line read:

```
IndexEntry {                        // 64 bytes total, cache-line aligned
    // Storage location (25 bytes)
    tier:           u8,             // T1, T2, T3, T4
    address:        u64,            // DRAM: pointer to blob. NVMe: byte offset in segment
    length:         u32,            // Blob size in bytes
    segment_id:     u32,            // Log segment file ID (NVMe tiers only, 0 for DRAM)
    flags:          u16,            // Compressed, dtype, codec, entry state

    // Eviction metadata (21 bytes)
    recompute_cost: u32,            // num_tokens × cost_per_token (for CW-SLRU decisions)
    last_access:    u48,            // Timestamp of last read (6 bytes, second granularity)
    access_count:   u16,            // Saturating counter for frequency estimation
    slru_segment:   u8,             // 0 = probationary, 1 = protected
    num_tokens:     u32,            // Token count (for cost estimation and metrics)
    created_at:     u48,            // Creation timestamp (6 bytes)

    // Integrity (4 bytes)
    checksum:       u32,            // CRC32C of the blob data

    // Padding (14 bytes → total 64)
    _reserved:      [u8; 14],
}
```

**Why 64-byte entries**: A single `IndexEntry` lookup touches exactly one cache line. No
partial reads, no false sharing between adjacent entries. On modern CPUs with 64-byte L1
cache lines, this is the optimal unit for point lookups.

**Implementation**: Lock-free concurrent hash map (open addressing, Robin Hood hashing),
**partitioned by placement group** (see Section 4.2.5). Each PG on a node gets its own
independent hash map instance.
- Read path: lock-free (atomic load + compare)
- Write path: per-bucket spinlock (writes are rare relative to reads)
- Capacity: Pre-allocated per PG for expected entry count, resized independently in
  background when load factor > 0.7

**Checkpointing**: Each PG's index is independently checkpointed to NVMe (every 30s or on
clean shutdown). Only PGs with changes since the last checkpoint are written (dirty tracking).
On restart, PG checkpoints are loaded in parallel across cores. Checkpoint format: sorted
array of `(CacheKey, IndexEntry)` pairs, binary encoded, ~90 bytes per entry (26 byte key +
64 byte entry). 10M entries = ~900 MB total across all PGs, loads in < 2 seconds from NVMe
with parallel PG loading.

### 3.3 DRAM Cache (T1 / T3)

The DRAM cache stores actual **compressed KV blobs** in memory for lowest-latency access. This
is distinct from the in-memory index (Section 3.2): the index stores 64-byte metadata entries
that *point* to blob data; the DRAM cache stores the blobs themselves (64 KB – 4 MB each).

```
Memory layout on a server node:

Index memory (~640 MB for 10M entries)     DRAM blob cache (user-configured, e.g. 64 GB)
┌───────────────────────┐                  ┌─────────────────────────────────┐
│ IndexEntry            │                  │ Slab: 256 KB class              │
│   tier: T3            │──address──→      │ ┌───────┬───────┬───────┬─── │
│   address: 0x7f...    │                  │ │ blob  │ blob  │ free  │ ...│
│   length: 262144      │                  │ └───────┴───────┴───────┴─── │
│   segment_id: 0       │                  ├─────────────────────────────────┤
│   ...                 │                  │ Slab: 512 KB class              │
└───────────────────────┘                  │ ┌───────────┬───────────┬─── │
                                           │ │ blob      │ free      │ ...│
Always required (lookup                    │ └───────────┴───────────┴─── │
table for all tiers).                      └─────────────────────────────────┘
                                           Optional — can be disabled. If disabled,
                                           reads go directly to NVMe (T2/T4).
```

**The DRAM cache is optional.** When disabled, the index still exists in DRAM (it's small
and always needed), but blob data is served directly from NVMe via GDS or io_uring. Disable
the DRAM cache when:
- DRAM is scarce and needed for model weights or inference engine buffers
- NVMe + GDS latency (~150 μs) is acceptable for the workload
- Cost optimization is more important than minimizing tail latency

**Allocator**: Slab allocator with configurable size classes aligned to common compressed
block sizes. Pre-allocates the configured DRAM budget at startup into large (2 MB) pages
(`mmap` with `MAP_HUGETLB`).

**Why slab, not malloc**: KV cache blocks have a narrow size distribution per model (determined
by chunk_size × hidden_dim × compression ratio). A slab allocator avoids fragmentation and
gives O(1) allocation from a free list. With `malloc`, long-running processes suffer from
heap fragmentation as differently-sized blobs are allocated and freed — the slab approach
guarantees that a freed 256 KB slot can always hold another 256 KB blob.

**Pinned memory**: T1 on GPU nodes uses CUDA pinned memory (`cudaMallocHost`) for zero-copy
transfers to GPU HBM. T3 on AttentionDB servers uses RDMA-registered memory for zero-copy RDMA
sends. Both pinned and RDMA-registered memory are allocated at startup and reused — registration
is expensive and should not be on the hot path.

**Configuration** (DRAM cache):

```yaml
dram_cache:
  enabled: true                          # Set false to skip DRAM blob caching entirely.
                                         # Index always lives in DRAM regardless.
  size: "64GB"                           # Total DRAM budget for blob storage.
                                         # Excludes index memory (index is sized automatically).
                                         # GPU nodes: typically 16-32 GB (shares with model weights).
                                         # Server nodes: typically 128-384 GB.

  allocator:
    type: "slab"                         # "slab" (default) | "buddy" | "system"
                                         # slab: O(1) alloc/free, best for narrow size distributions.
                                         # buddy: good for wide size ranges, more memory efficient.
                                         # system: fall back to malloc (not recommended for production).
    size_classes:                         # Slab size classes in bytes. Auto-configured if omitted.
      - 65536                            # 64 KB
      - 131072                           # 128 KB
      - 262144                           # 256 KB
      - 524288                           # 512 KB
      - 1048576                          # 1 MB
      - 2097152                          # 2 MB
      - 4194304                          # 4 MB
    auto_tune_classes: true              # If true, observe actual blob size distribution during
                                         # the first 60s and adjust class weights. Add/remove classes
                                         # based on which sizes are actually used by the loaded models.

  hugepages:
    enabled: true                        # Use MAP_HUGETLB for slab backing memory.
    page_size: "2MB"                     # "2MB" (default) | "1GB" (if available, fewer TLB misses
                                         # for large caches but requires OS-level hugepage config).

  memory_type: "auto"                    # "auto" | "pinned" | "rdma" | "standard"
                                         # auto: GPU nodes use pinned (cudaMallocHost),
                                         #        server nodes use RDMA-registered.
                                         # pinned: force CUDA pinned memory (GPU nodes only).
                                         # rdma: force RDMA-registered memory (requires RDMA NIC).
                                         # standard: regular mmap (no zero-copy, for testing/debug).

  per_pg_budget: "proportional"          # "proportional" (default) | "equal" | "manual"
                                         # proportional: divide budget across PGs by estimated demand
                                         #   (model size × tenant traffic weight).
                                         # equal: each PG gets size / num_local_pgs.
                                         # manual: use per-PG overrides (see below).
  pg_overrides:                          # Per-PG budget overrides (only with per_pg_budget: manual).
    # - pg_pattern: "llama-70b:*"        # Glob on "model:tenant"
    #   budget: "80GB"
```

### 3.4 NVMe Blob Store (T2 / T4)

The on-disk storage engine is a **log-structured blob store**, not a file-per-entry or
LSM-tree design.

```
Segment file layout (configurable size, default 1 GB per segment):
┌─────────────────────────────────────────────────────────────────┐
│ Segment Header (4 KB, page-aligned)                             │
│   magic: u64                                                    │
│   version: u32                                                  │
│   segment_id: u32                                               │
│   pg_id: u64 (owning placement group)                           │
│   created_at: u64                                               │
│   entry_count: u32                                              │
│   live_count: u32 (decremented on eviction, used by GC)         │
│   checksum: u32                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Entry 0                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Entry Header (64 bytes, cache-line aligned)             │   │
│   │   key: CacheKey (26 bytes)                              │   │
│   │   blob_length: u32                                      │   │
│   │   flags: u16 (compressed, dtype, format)                │   │
│   │   checksum: u32 (CRC32C of blob)                        │   │
│   │   timestamp: u64                                        │   │
│   │   padding to 64 bytes                                   │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │ Blob data (blob_length bytes, aligned with pad)          │   │
│   └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│ Entry 1 ...                                                     │
│ Entry 2 ...                                                     │
│ ...                                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Segment ownership**: Each segment file belongs to a single placement group (`pg_id` in
header). This aligns with PG-scoped indexing — when a PG migrates to another node, its
segment files can be identified and transferred (or discarded) without scanning all segments.

**Write path**:
1. Incoming blobs are appended to the PG's current active segment via `io_uring` (or `pwrite`
   with `O_DIRECT` fallback).
2. Writes are batched: accumulate in a write buffer, flush when buffer is full or on timer.
3. The index is updated only after the write is confirmed durable on NVMe.
4. When a segment reaches the configured size, it is sealed and a new active segment is opened.

**Read path**:
1. Index lookup returns `(segment_id, offset, length)`.
2. Issue `pread` via `io_uring` (or GDS `cuFileRead` for T2 on GPU nodes) directly into the
   target buffer (DRAM for promotion, GPU HBM for GDS).
3. Segment files remain open (file descriptor pool) to avoid open/close overhead.

**Garbage collection**:
- Entries are logically deleted by removing them from the index. The blob remains in the segment.
- Background GC thread scans segments where live-entry ratio falls below the configured
  threshold (tracked by `live_count` in the segment header, decremented on eviction/deletion).
- GC rewrites live entries from fragmented segments into the current active segment, then
  deletes the old segment file.
- GC is rate-limited to the configured fraction of NVMe write bandwidth, preventing I/O
  contention with the serve path.
- GC operates per-PG: a heavily-churning PG triggers GC on its own segments without affecting
  segments belonging to other PGs.

**I/O stack choice**:

| Tier | Primary I/O | Fallback | Reason |
|------|------------|----------|--------|
| T2 (GPU node, read) | GDS (`cuFileRead`) | `io_uring` + `cudaMemcpy` | GDS: NVMe → GPU HBM directly, lowest latency |
| T2 (GPU node, write) | `io_uring` with `O_DIRECT` | `pwrite` with `O_DIRECT` | Async batched writes, no kernel page cache |
| T4 (AttentionDB server, read) | `io_uring` with registered buffers | `pread` with `O_DIRECT` | Batch SSD reads, reduce syscall overhead |
| T4 (AttentionDB server, write) | `io_uring` with `O_DIRECT` | `pwrite` with `O_DIRECT` | Same as T2 |

**Why not LSM-tree / RocksDB**: KV cache blobs are large (256 KB - 4 MB). LSM-tree compaction
would rewrite the entire dataset multiple times (write amplification). The log-structured blob
store has write amplification of ~1.5x (only GC rewrites, and only fragmented segments).

**Why not one-file-per-entry** (LMCache approach): At millions of entries, filesystem metadata
(inodes, directory entries) becomes a bottleneck. `open`/`close` per read adds syscall
overhead. The log-structured approach uses O(100) open segment files, not O(millions) entry files.

**Configuration** (NVMe blob store):

```yaml
nvme_store:
  enabled: true                          # Set false to run DRAM-only (no persistence).
  paths:                                 # NVMe mount points. Multiple drives are striped
    - "/mnt/nvme0/attentiondb"           # for higher aggregate bandwidth.
    - "/mnt/nvme1/attentiondb"
  max_size_per_drive: "7TB"              # Max usage per drive. Reserves space for OS and
                                         # other workloads.

  segments:
    size: "1GB"                          # Segment file size. Larger segments = fewer files,
                                         # less metadata overhead, but coarser GC granularity.
                                         # Range: 256 MB – 4 GB. 1 GB is a good default.
    fd_pool_size: 256                    # Max open file descriptors for segment files.
                                         # More FDs = fewer open/close syscalls on read.

  write_buffer:
    size: "8MB"                          # Per-PG write buffer. Blobs accumulate here before
                                         # flushing to NVMe. Larger = fewer writes, higher
                                         # latency for durability. Range: 1 MB – 64 MB.
    flush_interval_ms: 50                # Flush buffer on this timer even if not full.
                                         # Lower = faster durability, more frequent small writes.
                                         # For KV cache (best-effort durability), 50 ms is fine.

  alignment: "4KB"                       # Blob data alignment on disk. Must match O_DIRECT
                                         # requirements of the filesystem. 4 KB is standard.
                                         # GDS may require different alignment (check cuFile docs).

  io:
    engine: "io_uring"                   # "io_uring" (default) | "posix" | "gds"
                                         # io_uring: async, batched, best for server nodes.
                                         # posix: pread/pwrite with O_DIRECT (fallback if
                                         #   io_uring unavailable, e.g. older kernels).
                                         # gds: GPUDirect Storage for T2 reads on GPU nodes.
                                         #   Falls back to io_uring + cudaMemcpy if GDS
                                         #   unavailable.
    io_uring_queue_depth: 128            # SQ/CQ ring size. Higher = more concurrent I/O ops.
                                         # Match to NVMe device queue depth for best throughput.
    io_uring_registered_buffers: true    # Pre-register read/write buffers with the kernel.
                                         # Reduces per-I/O overhead by avoiding buffer mapping.
    gds_thread_pool_size: 4              # Threads for cuFileRead/cuFileWrite (GDS is sync per call).

  gc:
    enabled: true                        # Disable if storage is append-only and entries have
                                         # TTL (they expire and segments can be bulk-deleted).
    trigger_fragmentation: 0.5           # GC a segment when its live-entry ratio drops below
                                         # this threshold. Lower = less GC, more wasted space.
                                         # Range: 0.2 – 0.8. Default 0.5 (GC when >50% dead).
    max_bandwidth_fraction: 0.2          # Max fraction of NVMe write bandwidth used by GC.
                                         # Prevents GC from competing with the serve path.
                                         # Range: 0.05 – 0.5.
    schedule: "continuous"               # "continuous" (default) | "off_peak"
                                         # continuous: GC runs whenever fragmentation exceeds
                                         #   threshold, rate-limited by max_bandwidth_fraction.
                                         # off_peak: GC runs only during configured low-traffic
                                         #   windows (requires traffic_profile config).
```

### 3.5 Blob Opacity (Compression Is Not AttentionDB's Concern)

AttentionDB stores and serves **opaque blobs**. It does not compress, decompress, or inspect
blob contents. Compression is handled by LMCache (or the inference engine) before `put()` and
after `get()`.

**Design principle**: AttentionDB never looks inside a blob. It stores bytes, returns bytes.
This keeps AttentionDB servers CPU-only and avoids duplicating LMCache's serde layer
(CacheGen, KIVI, TurboQuant, etc.). The `flags` field in `IndexEntry` records a codec hint
(provided by the caller at store time) for debugging and metrics, but AttentionDB does not
act on it.

**Implication for latency budgets**: The GPU decompression step (~50–100 μs) in Section 1's
latency breakdowns is not AttentionDB's latency — it happens after AttentionDB returns the
blob to LMCache. AttentionDB's contribution to read latency ends at "blob delivered to
caller's buffer."

---

## 4. Clustering

### 4.1 Cluster Topology

The cluster has two planes:

**Metadata plane** (all nodes — GPU clients + AttentionDB servers):
- Membership: which nodes exist, their roles, their health
- Placement map: which AttentionDB server holds which data
- Popularity index: access frequency statistics for prefetch/warm-up decisions
- Protocol: gossip (SWIM variant), eventually consistent
- Data size: < 100 MB total, replicated to every node

**Data plane** (AttentionDB servers + GPU nodes for local tiers):
- KV cache blob storage and retrieval
- Protocol: RDMA (NIXL/UCX) for bulk data, ZMQ for control messages. Consider integrating
  Mooncake's Transfer Engine as the RDMA transport layer — it already handles RDMA verbs,
  NVLink, TCP fallback, and buffer registration, avoiding the need to build this from scratch.
- Dual transport: RDMA primary (requires InfiniBand/RoCE), TCP fallback for environments
  without RDMA hardware. The system must perform well on TCP-only clusters (not just degrade).
- Consistency: single-writer per placement group (primary), enforced by epoch-based leases
  (see Section 4.2.6). No distributed locks or full consensus.

### 4.2 Placement Groups

A placement group (PG) is the fundamental unit of data ownership, replication, and isolation
in AttentionDB. The concept is borrowed from Ceph (where PGs serve as an indirection layer
between objects and storage daemons), but adapted with semantic keys for inference workloads.

**Definition**: A placement group is identified by a `(model_id, tenant_id)` pair and maps to
a set of AttentionDB server nodes. All KV cache data for a given model+tenant combination
belongs to exactly one placement group. The PG determines where that data is stored, how it
is replicated, and how it is evicted.

```
PlacementGroup {
    pg_id:              u64,          // hash(model_id, tenant_id)
    model_id:           u64,
    tenant_id:          u64,
    nodes:              Vec<NodeId>,  // Ranked list — [primary, replica-1, replica-2, ...]
    replication_factor: u8,           // Typically 2-3
    entry_count:        u64,          // Current number of cached entries
    dram_bytes_used:    u64,          // DRAM consumed by this PG's data
    nvme_bytes_used:    u64,          // NVMe consumed by this PG's data
    eviction_profile:   EvictionProfile,  // Per-model-class tuning
}
```

**Why semantic keys instead of opaque hashing**: In systems like Cassandra (token ring) or
Ceph (hash mod num_pgs), the mapping from key to partition is opaque — PG-42 might contain
a random mix of unrelated data. AttentionDB uses `(model_id, tenant_id)` as the PG identity
because this is the natural co-location boundary for inference:

- All conversations for `(llama-70b, acme)` share the same system prompts and RAG contexts.
  Co-locating them on the same nodes maximizes prefix KV cache reuse.
- Different models have different KV profiles (chunk size, recompute cost). Per-PG eviction
  tuning is natural when PG = model+tenant.
- Tenant isolation comes for free. Acme's traffic doesn't evict Globex's cached data.

#### 4.2.1 PG-to-Node Mapping via Rendezvous Hashing

AttentionDB uses **rendezvous hashing** (also called Highest Random Weight / HRW, proposed by
Thaler & Ravishankar, 1996) to map each PG to its node set. Unlike Cassandra's token ring or
Ceph's CRUSH maps, rendezvous hashing requires no ring structure, no token assignment, and no
centralized map.

```
Algorithm: For a given PG, compute which nodes own it.

Input:
  pg_id       = hash(model_id, tenant_id)
  cluster     = [node-1, node-2, node-3, node-4, node-5]
  repl_factor = 3

For each node in cluster:
  score[node] = hash(pg_id, node.id)    // deterministic score per (PG, node) pair

Sort nodes by score, descending:
  ranking = [node-3, node-1, node-5, node-4, node-2]
             ^^^^^^  ^^^^^^  ^^^^^^
             rank-1  rank-2  rank-3
            primary  warm-1  warm-2

PG node set = ranking[0..repl_factor] = {node-3, node-1, node-5}
```

Every node in the cluster can independently compute this same ranking with no shared state.

**Comparison with alternative approaches**:

| Property | Token ring (Cassandra) | CRUSH (Ceph) | Rendezvous hash (AttentionDB) |
|----------|----------------------|--------------|-------------------------------|
| Produces full node ranking | No (primary + next N clockwise) | Yes (weighted, topology-aware) | Yes (all nodes ranked) |
| Requires ring/map structure | Token ring + vnodes | CRUSH map (rules + weights) | Nothing — pure function of membership |
| Node add/remove disruption | Splits ranges, needs streaming | ~1/N keys move, rule-dependent | ~1/N keys shift rank, lazy population |
| Zero-coordination failover | No (needs ring walk) | Yes | Yes — rank-2 is already warm |
| Semantic key support | No (opaque partition key) | No (opaque object name) | Yes — PG ID is (model, tenant) |
| Topology awareness | Rack-aware via snitch config | Native (failure domains in rules) | Can be layered (Phase 3 roadmap) |
| Operational overhead | Token repair, cleanup, move | CRUSH map versioning | None — membership changes are sufficient |

**Stability on node add/remove**:
```
Before (5 nodes):
  PG "llama-70b + acme":
    scores: node-1=0.82, node-2=0.91, node-3=0.45, node-4=0.67, node-5=0.38
    ranking: [node-2, node-1, node-4, node-3, node-5]
    owners: {node-2 (primary), node-1 (warm), node-4 (warm)}

Add node-6 (score for this PG = 0.73):
    ranking: [node-2, node-1, node-6, node-4, node-3, node-5]
    owners: {node-2 (primary), node-1 (warm), node-6 (warm)}

Result: Primary unchanged. Rank-2 unchanged. node-4 dropped to rank-4, node-6 took rank-3.
        Only node-6 needs warming for this PG's data. Other PGs may not change at all.
```

**Churn analysis on node add/remove** — adding 1 node to an N-node cluster:

| Cluster size (N) | P(primary changes) = 1/(N+1) | P(any top-K member changes) = K/(N+1) |
|---|---|---|
| 5 nodes | 16.7% of PGs | 50.0% of PGs (K=3) |
| 10 nodes | 9.1% of PGs | 27.3% of PGs |
| 20 nodes | 4.8% of PGs | 14.3% of PGs |
| 50 nodes | 2.0% of PGs | 5.9% of PGs |
| 100 nodes | 1.0% of PGs | 3.0% of PGs |

At 10 nodes, **~9% of PGs get a new primary** — this is not negligible. The shadow join
protocol (Section 5.2) mitigates the impact by warming the new primary before it takes full
write responsibility, and epoch-based leases (Section 4.2.7) ensure the old and new primary
don't both write simultaneously during the transition.

The advantage of rendezvous hashing over token rings is not that churn is zero — it's that
the change is **predictable, minimal, and independently computable**. Every node agrees on
the new ranking without coordination, and only ~1/(N+1) of primary assignments change.

#### 4.2.2 Three-Level Sharding

PGs are the first level of a three-level sharding hierarchy:

```
Level 1 — Placement Group (model + tenant → node set):
  pg = PlacementGroup(model_id, tenant_id)
  pg.nodes = rendezvous_rank(pg.pg_id, cluster_nodes)[0..repl_factor]
  Determines which AttentionDB servers hold data for this model+tenant.

Level 2 — Prefix group (system_prompt_hash → primary within PG):
  primary_node = rendezvous_hash(system_prompt_hash, pg.nodes) → single node
  All conversations sharing a system prompt cluster on the same primary node.
  Maximizes prefix KV cache reuse within the PG.

Level 3 — Session affinity (sequence_id → GPU node):
  preferred_gpu = rendezvous_hash(sequence_id, gpu_nodes_with_model) → single GPU node
  Multi-turn conversations stay on the same GPU node for local T1/T2 cache hits.
  Router uses this as a preference, not a hard constraint (overridden by load balancing).
```

**Request routing example**:
```
Incoming request: model=llama-70b, tenant=acme, system_prompt="You are a helpful assistant..."

1. PG lookup:
   pg_id = hash("llama-70b", "acme") → PG-A
   PG-A nodes = {node-2 (primary), node-1 (warm), node-5 (warm)}

2. Prefix group:
   prompt_hash = hash("You are a helpful assistant...")
   primary_for_prefix = rendezvous_hash(prompt_hash, [node-2, node-1, node-5]) → node-2

3. Client contacts node-2 for prefix KV cache lookup.
   If hit  → RDMA fetch the shared prefix blocks.
   If miss → prefill recompute, then store result on node-2 for future reuse.
```

#### 4.2.3 PG Lifecycle

PGs are created lazily and can be in several states:

```
EMPTY     → First write for this (model, tenant) pair. PG is created, index partition
            allocated, node set computed.

ACTIVE    → Normal operation. Accepting reads and writes on the primary.
            Warm replicas receiving replication.

MIGRATING → Node add/remove changed the PG's node set. New nodes are in
            WARMING → SHADOW → ACTIVE lifecycle (see Section 5.2). Old nodes
            continue serving reads until handoff completes.

DRAINING  → The model is being unloaded or tenant is being decommissioned.
            No new writes. Reads served until TTL expires or explicit eviction.

TOMBSTONE → PG is empty (all entries evicted or expired). The PG metadata is
            retained briefly (1 hour) to avoid re-creation churn, then garbage
            collected from the metadata plane.
```

#### 4.2.4 Per-PG Resource Isolation

Each PG maintains independent resource accounting and eviction:

```
AttentionDB Server Node-2 hosts 3 PGs:

┌─────────────────────────────────────────────┐
│ PG-A: (llama-70b, acme) — PRIMARY           │
│   DRAM budget: 60 GB  │  NVMe budget: 2 TB  │
│   Entries: 5M          │  Hit rate: 94%      │
│   Eviction: CW-SLRU, protection_threshold=low│  (70B model → cache aggressively)
│   Index: independent hash map, 5M entries    │
├─────────────────────────────────────────────┤
│ PG-B: (llama-70b, globex) — WARM REPLICA    │
│   DRAM budget: 30 GB  │  NVMe budget: 1 TB  │
│   Entries: 2M          │  Hit rate: 88%      │
│   Eviction: CW-SLRU, protection_threshold=low│
│   Index: independent hash map, 2M entries    │
├─────────────────────────────────────────────┤
│ PG-C: (mistral-7b, acme) — PRIMARY          │
│   DRAM budget: 20 GB  │  NVMe budget: 500 GB│
│   Entries: 8M          │  Hit rate: 78%      │
│   Eviction: CW-SLRU, protection_threshold=high│ (7B model → evict more aggressively)
│   Index: independent hash map, 8M entries    │
└─────────────────────────────────────────────┘
```

**Budget allocation**: Each node divides its total DRAM and NVMe capacity across the PGs
it hosts, proportional to the PG's expected demand (based on model size and tenant traffic).
Budgets are re-computed when PGs are added/removed and can be adjusted dynamically based
on observed access patterns.

**Isolation guarantees**: A noisy tenant flooding PG-A with writes only triggers eviction
within PG-A. PG-B and PG-C are unaffected. This prevents the common failure mode where one
tenant's traffic pattern degrades cache hit rates for all other tenants.

**Per-PG eviction tuning**: Different models have different KV cache profiles. PG-A
(llama-70b) has large, expensive-to-recompute chunks — protect aggressively. PG-C
(mistral-7b) has small, cheap-to-recompute chunks — evict more freely. This tuning is
automatic based on the model's `cost_per_token` calibration (see Section 6.1).

#### 4.2.5 PG-Scoped Index Partitioning

The in-memory index (Section 3.2) is partitioned by PG rather than using a single global
hash map. Each PG on a node gets its own independent hash map instance:

**Why partitioned (vs. single global map)**:

1. **Independent resize**: When PG-A grows from 5M to 10M entries, only PG-A's hash map
   resizes. PG-B and PG-C are untouched. A global map resize would stall all lookups.

2. **Aligned with data lifecycle**: When a PG migrates to a new node (due to membership
   change), the entire hash map for that PG can be serialized and transferred as a unit.
   No need to scan a global map to filter entries by PG membership.

3. **Eliminated false sharing**: PG-A and PG-B have separate backing arrays in separate
   memory regions. Concurrent writes to PG-A never invalidate CPU cache lines used by
   PG-B lookups — critical on 64+ core servers serving parallel GPU requests.

4. **Scoped checkpointing**: Each PG's index is checkpointed independently. Only dirty
   PGs (those with changes since last checkpoint) are written. Quiet PGs skip entirely.
   Checkpoints can run in parallel across PGs.

5. **Per-PG eviction without secondary bookkeeping**: Eviction within PG-A iterates
   PG-A's hash map directly. No need for a secondary data structure to track which entries
   belong to which PG.

**Lookup path with partitioned index**:
```
1. Compute pg_id = hash(cache_key.model_id, cache_key.tenant_id)    // already known
2. pg_index = node.index_partitions[pg_id]                          // O(1) map lookup
3. location = pg_index.get(cache_key)                               // O(1) hash lookup
```
The PG ID is already computed during request routing, so the extra indirection adds ~1 ns.

#### 4.2.6 PG Scaling Modes (Inline / Compact / Full)

Per-PG hash maps provide excellent isolation but have a fixed overhead per PG (~64 KB minimum
backing array + metadata). In deployments with many low-traffic (model, tenant) pairs, this
overhead can become significant:

```
Problem scenario:
  100 models × 500 tenants = 50,000 PGs
  Most PGs have < 100 entries
  Per-PG overhead: ~64 KB minimum → 50,000 × 64 KB = ~3.2 GB wasted on near-empty maps
```

To handle this, PGs operate in one of three storage modes based on entry count:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ INLINE MODE (< 10 entries)                                              │
│                                                                          │
│ Entries stored directly in the PG metadata struct as a small array.      │
│ No hash map allocated. Linear scan for lookups (fast at < 10 entries).   │
│ Overhead: ~640 bytes per PG (10 × 64-byte IndexEntry).                  │
│                                                                          │
│ Auto-promotes to COMPACT when entry count reaches 10.                   │
├──────────────────────────────────────────────────────────────────────────┤
│ COMPACT MODE (10 – 1,000 entries)                                       │
│                                                                          │
│ Multiple compact PGs share a single hash map, with PG-ID prefix in      │
│ the key to disambiguate. Entries from different compact PGs may share    │
│ the same backing array and segment files. Less isolation, but greatly    │
│ reduced overhead.                                                        │
│                                                                          │
│ Shared compact map: one per node, holds all compact PGs.                │
│ Lookup: hash(pg_id, cache_key) → shared map.                            │
│ Overhead: amortized across all compact PGs on the node.                 │
│                                                                          │
│ Auto-promotes to FULL when entry count reaches 1,000.                   │
│ Auto-demotes from FULL when entry count drops below 500 (hysteresis     │
│ to prevent thrashing at boundary).                                       │
├──────────────────────────────────────────────────────────────────────────┤
│ FULL MODE (> 1,000 entries)                                             │
│                                                                          │
│ Own independent hash map, own slab budget, own segment files.           │
│ Full isolation as described in Section 4.2.4 and 4.2.5.                 │
│                                                                          │
│ Auto-demotes to COMPACT when entry count drops below 500.               │
└──────────────────────────────────────────────────────────────────────────┘
```

**Trade-offs of compact mode** — compact mode explicitly sacrifices two guarantees that
full mode provides. This is acceptable for low-traffic PGs, but the trade-offs must be
understood:

1. **Eviction isolation is weakened**: In full mode, eviction within PG-A never touches
   PG-B's entries. In compact mode, all compact PGs share one hash map and one eviction
   pool. If the shared compact map fills up, evicting an entry from PG-X to make room for
   PG-Y is possible — violating per-PG isolation. This is tolerable for low-traffic PGs
   (where eviction pressure is rare), but if a compact PG begins receiving significant
   traffic, it should be auto-promoted to full mode before contention becomes an issue.
   The auto-promotion threshold (1,000 entries) is designed to trigger well before
   eviction contention is likely.

2. **Checkpointing granularity coarsens**: In full mode, only dirty PGs are checkpointed.
   In compact mode, all compact PGs share a single backing map, so a dirty entry in any
   one compact PG forces the entire shared map to be checkpointed. With 4,500 compact PGs,
   the shared map is effectively always dirty — every checkpoint cycle writes it. In practice
   this is acceptable because compact PGs are small by definition (total entries across all
   compact PGs is bounded by `compact_max_entries × num_compact_pgs`, and the shared map
   size is capped by `compact_shared_map_size`). A 256 MB shared map checkpoints in ~50 ms
   on NVMe — trivial compared to the 30-second checkpoint interval.

3. **Migration granularity increases**: In full mode, migrating a PG means transferring its
   independent hash map. In compact mode, extracting a single PG's entries from the shared
   map requires scanning and filtering. For the rare case where a compact PG is migrated,
   this is an O(shared_map_size) operation. Promoting to full mode before migration avoids
   this cost.

**Configuration**:

```yaml
pg_scaling:
  inline_max_entries: 10                # Below this → inline mode
  compact_max_entries: 1000             # Below this → compact mode
  demotion_hysteresis: 0.5              # Demote at 50% of promotion threshold
  compact_shared_map_size: "256MB"      # Max size of the shared compact hash map
  auto_promote_on_eviction: true        # If a compact PG triggers eviction, promote it to
                                        # full mode immediately (restores isolation).
```

**Memory comparison** for 50,000 PGs (45,000 with < 100 entries, 4,500 with 100-10K, 500 large):

| Approach | Memory overhead |
|----------|----------------|
| All FULL mode | ~3.2 GB (50,000 × 64 KB minimum) |
| Tiered (inline/compact/full) | ~100 MB (45K inline + 4.5K compact + 500 full) |

#### 4.2.7 Primary Ownership Fencing (Epoch Leases)

Gossip-based membership (Section 4.4) is eventually consistent. During a network partition,
two nodes could both compute themselves as primary for the same PG, violating the single-writer
guarantee. While KV cache entries are immutable (no conflicting updates), split-brain writes
cause wasted storage, eviction accounting drift, and potential stale reads after the partition
heals.

**Solution**: Each PG has an **epoch-based lease** that fences primary ownership.

```
PrimaryLease {
    pg_id:        u64,
    epoch:        u64,       // Monotonically increasing. Incremented on every ownership change.
    holder:       NodeId,    // Current primary
    granted_at:   Timestamp,
    expires_at:   Timestamp, // Lease TTL (default: 30 seconds)
    ack_nodes:    Vec<NodeId>, // Nodes that acknowledged this epoch
}
```

**Lease protocol**:

```
1. CLAIM (on ownership change or first write):
   - Candidate primary computes its rendezvous rank for the PG.
   - If rank-1, sends LeaseRequest{pg_id, proposed_epoch = current_epoch + 1} to
     the other nodes in the PG (rank-2, rank-3).
   - Must receive ACK from at least 1 other PG member (quorum of 2 out of 3)
     before accepting writes.

2. GRANT:
   - Acknowledging node checks:
     a) proposed_epoch > its locally known epoch for this PG
     b) no other active lease it has acknowledged for a higher epoch
   - If valid, responds with LeaseAck{pg_id, epoch} and records the grant.

3. RENEW (every lease_renew_interval, default 10s):
   - Primary sends LeaseRenew{pg_id, epoch} to PG members.
   - Must receive at least 1 ACK before current lease expires.
   - If renewal fails (network partition), primary enters FENCED state.

4. FENCED (lease expired, cannot renew):
   - Primary STOPS accepting writes for this PG.
   - Continues serving reads from local cache (stale but safe — entries are immutable).
   - Emits metric: attentiondb_pg_fenced_total.
   - When connectivity restores, must re-acquire lease (may find a higher epoch
     if another node claimed primary during the partition).

5. TAKEOVER (partition heals or node failure):
   - Rank-2 node notices rank-1's lease has expired (via gossip or direct probe).
   - Rank-2 initiates CLAIM with epoch = expired_epoch + 1.
   - Rank-2 becomes new primary only after quorum ACK.
   - Old primary (if still alive but partitioned) is fenced and will discover the
     higher epoch when connectivity restores. It then demotes itself and re-joins
     as a warm replica.
```

**Partition behavior**:

```
Network partition splits cluster:

  Partition A: [node-1 (rank-1), node-2 (rank-2)]
  Partition B: [node-3 (rank-3)]

  Scenario 1 — primary in majority partition:
    node-1 (primary) can reach node-2 → renews lease successfully.
    node-3 cannot reach node-1 or node-2 → cannot claim lease → stays read-only.
    Result: correct single-writer maintained.

  Scenario 2 — primary isolated:
    Partition A: [node-2 (rank-2), node-3 (rank-3)]
    Partition B: [node-1 (rank-1, primary)]

    node-1 cannot renew (no ACK from PG members) → lease expires → FENCED.
    node-2 detects expired lease → claims with epoch+1, gets ACK from node-3.
    node-2 becomes new primary.
    Result: brief write unavailability (lease_ttl duration), then correct single-writer.

  Scenario 3 — all three nodes isolated from each other:
    No node can get quorum → all FENCED → no writes for this PG.
    Reads continue from local cache.
    Result: availability sacrifice for correctness (correct trade-off for a cache system).
```

**Gossip witnesses** — a subtle problem with the basic lease protocol: with
`replication_factor=3`, the primary must reach at least 1 of exactly 2 specific nodes (the
other PG members) to renew. If those 2 nodes happen to be under heavy load, experiencing
NIC congestion, or slow to respond, the primary gets fenced even though the cluster as a whole
is healthy. This is a tighter dependency than gossip failure detection (which contacts random
peers).

To mitigate, lease renewal also accepts **gossip witness ACKs**:

```
Lease renewal quorum can be satisfied by:
  Option A: ACK from at least 1 PG member (standard path, fastest)
  Option B: ACKs from at least 2 non-PG-member nodes that have recently
            received gossip heartbeats from the primary (gossip witnesses)

A gossip witness is any node that:
  1. Has received a direct or indirect gossip heartbeat from the primary within
     the last gossip_interval × 2 (i.e., the primary is definitely alive in gossip)
  2. Can sign a WitnessAttestation{primary_node, pg_id, epoch, witnessed_at}
```

This broadens the quorum base from 2 specific nodes to the entire cluster. The primary is
fenced only if it truly cannot reach anyone — not just because its 2 PG peers are slow.
The trade-off is that gossip witnesses provide a weaker liveness guarantee (they attest "I
saw the primary recently in gossip" not "I am actively replicating data from it"), so
witness-based renewals should trigger a metric (`attentiondb_lease_witness_renewal_total`)
for monitoring.

**Why not full Raft/Paxos**: The lease mechanism only needs to establish "who is the current
primary" — a single leader election, not replicated log consensus. Full Raft would add latency
to every write (log replication round-trip) and operational complexity. Since KV cache entries
are immutable and recomputable, the worst case of a lease expiry is a brief period of write
unavailability — the inference engine falls back to prefill recompute. This is the right
trade-off for a cache system.

**Epoch propagation**: The current epoch for each PG is included in gossip state. Clients
use the epoch to detect stale primaries — if a client receives a response with a lower epoch
than it has seen, it refreshes its placement map and retries with the correct primary.

**Configuration**:

```yaml
ownership:
  lease_ttl_s: 30                        # Lease duration. Longer = more tolerant of transient
                                         # network issues, but slower failover on real failures.
  lease_renew_interval_s: 10             # Renewal frequency. Must be < lease_ttl_s / 2.
  quorum_size: 2                         # Nodes that must ACK a lease (including holder).
                                         # For repl_factor=3, quorum=2. For repl_factor=2, quorum=2
                                         # (both nodes must agree — stricter but safer).
  fenced_read_mode: "local_stale"        # "local_stale" (serve from local cache, may be stale)
                                         # | "reject" (reject all reads for this PG while fenced)
                                         # | "redirect" (redirect to another PG member)
```

### 4.3 Replication

AttentionDB uses **async push replication** from primary to warm replicas:

```
Primary                              Warm Replica
   │                                      │
   │── ReplicateBatch ──────────────────→ │
   │   (keys[], blobs[], priorities[])    │
   │                                      │  Accepts if capacity allows,
   │                                      │  evicts lowest-priority local
   │                                      │  entries to make room.
   │← ReplicateAck ─────────────────────│
   │   (accepted_keys[])                  │
```

**What gets replicated** (priority order):
1. System prompt KV caches (high priority — shared by many conversations, expensive to recompute)
2. Long-context conversation prefixes where `token_count × model_compute_cost > threshold`
3. Frequently accessed conversation KV

**What does NOT get replicated**:
- Short, cheap-to-recompute entries (< 64 tokens on small models)
- Entries with TTL < replication_lag (they'd expire before being useful)
- Per-session decode-phase KV (stored only on the local GPU node's T2)

**Consistency model**: Eventual. A write is visible on the primary immediately. It appears on
warm replicas after replication lag (typically < 1 second). This is acceptable because:
- KV cache entries are immutable (no update conflicts)
- A replica miss just falls through to the primary (correct, slightly slower)
- Loss of unreplicated data means prefill recompute (degraded, not incorrect)

### 4.4 Metadata Plane

The metadata plane uses a SWIM-family gossip protocol for membership and health, augmented with
application-specific state.

**Gossip state per node**:
```
NodeState {
    node_id:           u64,
    node_type:         enum { GpuClient, AttentionDbServer },
    address:           SocketAddr,
    rdma_address:      RdmaAddr,
    status:            enum { Alive, Suspect, Dead, Draining },
    capacity:          CapacityInfo { dram_total, dram_used, nvme_total, nvme_used },
    models_loaded:     Vec<u64>,       // model_id hashes
    incarnation:       u64,            // Monotonic, distinguishes restarts
    last_heartbeat:    Timestamp,
}
```

**Failure detection**:
- Each node pings 3 random peers every 1 second
- Failed ping → ask 3 other nodes to indirect-probe the suspect
- If indirect probes also fail → mark SUSPECT
- After 5 seconds in SUSPECT → mark DEAD
- Dead node's gossip state propagates via piggybacked protocol (O(log N) convergence)

**Popularity index** (for prefetch/warm-up):

Each AttentionDB server maintains a local count-min sketch of access frequencies per `CacheKey`
for building warm-up manifests and predictive prefetch.

**Sketch structure**:
```
PopularitySketch {
    epoch:          u64,         // Time window ID. Rolls over every 30 seconds.
    node_id:        NodeId,      // Which node produced this sketch.
    counters:       [[u32; W]; D], // D hash functions × W counters (e.g., 4 × 8192).
    total_accesses: u64,         // Sum of all increments this epoch (for normalization).
}
```

**Merge protocol**: Sketches are exchanged between peers every 10 seconds via gossip.
Merging uses **additive semantics with deduplication**, not element-wise max:

```
Merge rules:
  Same (node_id, epoch) pair seen again → IGNORE (already incorporated; idempotent)
  New (node_id, epoch) pair             → ADD counters element-wise to aggregate sketch
  Older epoch (epoch < current - 2)     → DISCARD (too stale, window has passed)
```

Why not element-wise max (the naive approach):
- Max undercounts: if node-A sees key K 100 times and node-B sees it 50 times,
  max gives 100 but the true global count is 150.
- Additive merge gives the correct global frequency estimate.
- Deduplication by (node_id, epoch) prevents double-counting when the same gossip
  message propagates through multiple paths — each node's contribution is counted
  exactly once.

**Time windowing**: Sketches decay naturally because each epoch is only valid for
~30 seconds. The aggregate sketch represents "what was popular in the last 1–2 minutes."
This prevents long-ago access patterns from dominating prefetch decisions.

---

## 5. Node Lifecycle

### 5.1 GPU Node Join (Client)

Since GPU nodes use the AttentionDB cluster as a remote tier, joining is lightweight:

```
1. CONNECT (< 1s):
   - GPU node starts AttentionDB client library
   - Loads metadata plane state via gossip seed nodes
   - Opens RDMA connections to AttentionDB servers in its placement groups
   - Registers local T1/T2 storage

2. PREFETCH (1-10s, async):
   - Client requests PrefetchManifest from AttentionDB cluster:
     top-K system prompt KVs for loaded models, ranked by popularity
   - Fetches entries via RDMA into local T1/T2
   - Inference engine can begin serving immediately (misses fall through to remote)

3. ACTIVE:
   - Full read/write operation
   - Local cache warms naturally from traffic + background prefetch completes
```

**GPU cold start is fast** because the AttentionDB cluster already has the data. The GPU node
doesn't need to "build" a cache — it just connects and starts reading.

### 5.2 AttentionDB Server Join

Adding a new AttentionDB server is less frequent but more involved:

```
1. REGISTERING (0-5s):
   - New server announces to metadata plane
   - Gossip propagates membership; hash ring rankings update
   - No traffic routed to this server yet

2. WARMING (5s-2min):
   - Server identifies which keys it should own (from updated rendezvous rankings)
   - Pulls those keys from existing servers via RDMA, prioritized by:
     a) System prompt KVs (highest priority)
     b) Popularity-ranked conversation KVs
   - Progress metric: warmup_coverage = fetched_entries / expected_entries
   - Transitions to SHADOW when coverage > 70% or timeout (60s)

3. SHADOW (1-5min):
   - Server receives 10% of read traffic for its key range
   - On local miss, transparently fetches from the previous primary (not recompute)
   - Stores new writes locally AND forwards to previous primary (dual-write)
   - Observes access patterns to build local popularity model
   - Promoted to ACTIVE when local cache hit rate > 80%

4. ACTIVE:
   - Full weight in hash ring
   - Receives normal read/write traffic
   - Participates in replication as primary or warm replica
```

### 5.3 Planned Scale-Down (Draining)

```
1. DRAINING:
   - Server stops accepting new writes
   - For high-priority entries, pushes to next-ranked server (proactive handoff)
   - Continues serving reads until handoff completes or drain_timeout (60s)

2. DEREGISTER:
   - Server leaves the gossip ring
   - Remaining servers' rankings shift; the next-ranked server becomes primary
   - If the draining server was a warm replica, the primary selects a new warm target
```

### 5.4 Unplanned Failure

```
1. DETECTION (1-6s):
   - Gossip failure detector marks node SUSPECT → DEAD

2. IMMEDIATE FAILOVER:
   - GPU clients with in-flight reads to the dead server time out (~100 ms)
   - Client library retries to rank-2 server (warm replica) — transparent to inference engine
   - Warm replica likely has most high-value entries → most retries are cache hits

3. RECOVERY (background):
   - Metadata plane identifies entries that were only on the dead node (not replicated)
   - These entries are simply lost — they'll be recomputed on next access (graceful degradation)
   - If the node comes back, it re-enters via the REGISTERING → WARMING → SHADOW → ACTIVE flow
   - On restart, it reads its local index checkpoint from NVMe and skips warming for entries
     it still has locally
```

---

## 6. Admission and Eviction

### 6.1 Admission Control

Not all KV cache data should be stored. Admission filtering prevents low-value entries from
consuming storage and causing eviction of high-value ones.

**Admission criteria** (evaluated in order, first match wins):

| Priority | Rule | Rationale |
|----------|------|-----------|
| 1 | Always admit system prompt KV | Shared by many conversations, high reuse probability |
| 2 | Reject if decode-phase single-token append | Tiny, always being generated, would churn storage |
| 3 | Reject if `num_tokens < min_tokens_threshold` (default: 64) | Faster to recompute than to read from storage |
| 4 | Reject if `admission_score < dynamic_threshold` when storage > 70% full | Cost + popularity based filtering (see below) |

**Admission score** — combines recompute cost with estimated reuse probability from the
popularity sketch (Section 4.4):

```
recompute_cost(entry) = num_tokens × cost_per_token(model_id)

reuse_estimate(entry) = popularity_sketch.estimate(entry.prefix_hash)
                        / sketch.total_accesses
                        // Fraction of recent accesses that hit this prefix.
                        // High for shared system prompts, low for unique conversations.

admission_score(entry) = recompute_cost × (1 + reuse_weight × reuse_estimate)
```

Where:
- `cost_per_token` is a calibrated constant per model (measured once during model loading
  via a short benchmark, stored in metadata plane).
- `reuse_estimate` comes from the popularity sketch. A prefix that many conversations share
  (e.g., a common system prompt) has a high reuse estimate. A unique long conversation
  has a low reuse estimate, even if its recompute cost is high.
- `reuse_weight` (default: 5.0) controls how much reuse probability influences admission
  relative to raw recompute cost.

**Why popularity matters for admission**: Without reuse estimation, the admission control
caches every expensive entry equally — including long conversation prefixes that are accessed
once and never reused. Under storage pressure, these one-shot expensive entries evict shared
prefixes that would have been reused hundreds of times. By factoring in the popularity sketch,
the admission control answers "is this entry both expensive AND likely to be reused?" rather
than just "is this entry expensive?"

**Dynamic threshold**: The `dynamic_threshold` rises as storage fills:

```
When storage < 70%:  threshold = 0 (admit everything that passes rules 1-3)
When storage 70-90%: threshold scales linearly from base_threshold to 2× base_threshold
When storage > 90%:  threshold = 2× base_threshold (only high-value, high-reuse entries)
```

This integrates with the write buffer backpressure (Section 7.3) — the client-side
backpressure filters by recompute cost, while the server-side admission filters by
admission_score (cost × reuse). Together, they provide two layers of quality filtering.

### 6.2 Cost-Weighted Segmented LRU (CW-SLRU)

Each DRAM and NVMe tier uses a two-segment eviction policy:

```
┌──────────────────────────────────┐
│        Protected Segment          │  20-30% of tier capacity
│  (high recompute cost entries)    │
│                                   │
│  Admission: recompute_cost >      │
│             protection_threshold  │
│  Eviction: demote to Probationary │
│            (not discard)          │
├──────────────────────────────────┤
│       Probationary Segment        │  70-80% of tier capacity
│  (everything else + demoted)      │
│                                   │
│  Admission: any admitted entry    │
│  Eviction: discard (or demote to  │
│            next tier if available) │
│  Promotion: re-accessed entry     │
│             with high recompute   │
│             cost → Protected      │
└──────────────────────────────────┘
```

**Per-model-class tuning**: Different model classes have different KV profiles:

| Model class | Protection threshold | Min tokens | Notes |
|-------------|---------------------|------------|-------|
| Large dense (70B+) | Low (protect more) | 32 | Expensive prefill, cache aggressively |
| Small dense (7-13B) | High (protect less) | 128 | Cheap prefill, mostly system prompts worth caching |
| MoE (Mixtral-class) | Medium | 64 | Variable per expert |
| MLA (DeepSeek-class) | Medium | 64 | Smaller KV per token, higher density |

**LRU ordering within segments**: When multiple entries are candidates for eviction, evict
suffix chunks before prefix chunks (prefix chunks are shared by more future requests). This
is prefix-cache-friendly ordering, adopted from LMCache's approach.

### 6.3 Cross-Tier Demotion

Eviction doesn't always mean deletion:

| From | To | When |
|------|----|------|
| T1 (local DRAM) | T2 (local NVMe) | T1 full, entry is warm but not hot |
| T3 (remote DRAM) | T4 (remote NVMe) | T3 full, entry is warm but not hot |
| T1 (local DRAM) | Discard | Entry is cold AND exists on T3/T4 (can be re-fetched) |
| T2 (local NVMe) | Discard | Entry is cold AND exists on T3/T4, or below cost threshold |

Demotion is async — the entry is written to the lower tier in the background and removed from
the upper tier only after the write is confirmed.

---

## 7. Storage Engine API

AttentionDB exposes a **storage engine API**, not a cache orchestration API. Cache-level
concerns (prefix matching, chunking, tier routing, compression) are handled by LMCache,
which calls AttentionDB through its `StorageBackendInterface` adapter (see Section 1,
"Relationship to LMCache").

The storage engine runs as an **in-process library** linked into the LMCache/inference
engine process.

**Why in-process**:
- **Latency**: Avoids IPC overhead (~5–20 μs). For T1 DRAM hits (~70 μs), IPC adds 7–28%.
- **Zero-copy paths**: Pinned CPU memory can be shared directly with LMCache's memory pools.
- **Precedent**: NCCL, NIXL, cuBLAS are all in-process libraries.

### 7.1 API Surface

```c
// Lifecycle
attentiondb_status_t attentiondb_open(
    const attentiondb_config_t *config,
    attentiondb_t **handle              // OUT: engine handle
);
attentiondb_status_t attentiondb_close(attentiondb_t *handle);

// Core storage operations — called by LMCache adapter
attentiondb_status_t attentiondb_put(
    attentiondb_t *handle,
    const attentiondb_key_t *key,       // 26-byte StorageKey
    const void *blob,                   // Opaque compressed blob (already compressed by LMCache)
    size_t blob_len,
    const attentiondb_put_opts_t *opts  // num_tokens, recompute_cost, entry_type, ttl
);

attentiondb_status_t attentiondb_get(
    attentiondb_t *handle,
    const attentiondb_key_t *key,
    void *buf,                          // OUT: caller-provided buffer for blob
    size_t buf_len,
    size_t *blob_len_out                // OUT: actual blob size
);

attentiondb_status_t attentiondb_contains(
    attentiondb_t *handle,
    const attentiondb_key_t *keys,
    size_t num_keys,
    bool *results                       // OUT: hit/miss per key
);

attentiondb_status_t attentiondb_delete(
    attentiondb_t *handle,
    const attentiondb_key_t *keys,
    size_t num_keys
);

// Bulk operations
attentiondb_status_t attentiondb_batched_put(
    attentiondb_t *handle,
    const attentiondb_key_t *keys,
    const void **blobs,
    const size_t *blob_lens,
    size_t num_entries,
    const attentiondb_put_opts_t *opts
);

attentiondb_status_t attentiondb_batched_get(
    attentiondb_t *handle,
    const attentiondb_key_t *keys,
    size_t num_keys,
    void **bufs,
    size_t *buf_lens,
    size_t *blob_lens_out
);

// Stats
attentiondb_status_t attentiondb_stats(
    attentiondb_t *handle,
    attentiondb_stats_t *stats          // OUT: hit rates, utilization, eviction counts
);
```

**What this API does NOT include** (handled by LMCache):
- No `lookup` + `retrieve` split (LMCache's `async_lookup_and_prefetch` handles this)
- No `prefetch` (LMCache calls `contains` then `get`; internal T2→T1 promotion is automatic)
- No CUDA stream parameters (AttentionDB returns blobs to CPU buffers; LMCache handles GPU)
- No compression codec selection (blobs arrive pre-compressed)

### 7.2 Store Metadata

The `attentiondb_put_opts_t` struct passes inference-aware metadata that drives eviction
and admission decisions. This metadata is provided by LMCache at store time:

```c
typedef struct {
    uint32_t num_tokens;            // Token count for this chunk (for cost estimation)
    uint32_t recompute_cost;        // num_tokens × cost_per_token (for CW-SLRU)
    uint8_t  entry_type;            // 0=conversation, 1=system_prompt, 2=rag_context
    uint32_t ttl_seconds;           // 0 = no expiry
} attentiondb_put_opts_t;
```

This is how AttentionDB makes intelligent eviction decisions without understanding tokens
or inference semantics — LMCache tells it "this entry cost 5000 to compute and is a system
prompt" and AttentionDB uses that to rank entries for eviction.

### 7.3 Internal Write Path and Backpressure

The storage engine maintains an internal write buffer in pinned CPU DRAM. When LMCache calls
`put()`, the blob is accepted into the write buffer and the call returns immediately. A
background flush thread writes buffered blobs to NVMe (T2/T4) and optionally replicates to
remote servers (T3).

```
LMCache calls attentiondb_put(key, blob, metadata)
  → Admission check (reject if below cost threshold — see Section 6)
  → Copy blob into pinned write buffer
  → Return immediately (non-blocking)
  → Background: write buffer → T2 NVMe (io_uring) and/or T3 remote (RDMA)
```

**Flush policy**: Flush when buffer utilization exceeds 75%, or every `flush_interval_ms`
(default 50 ms), whichever comes first. Flushes are async and never block the caller.

**Adaptive backpressure** — writes are NOT all-or-nothing. Instead, the system applies
**progressive admission filtering** as buffer pressure increases. This ensures that under
load, high-value entries (expensive to recompute) are still cached while low-value entries
are shed first.

```
Buffer utilization:

  0%────────50%────────75%────────90%────────100%
  │         │          │          │           │
  │  NORMAL │ CAUTIOUS │ SELECTIVE│ EMERGENCY │
  └─────────┴──────────┴──────────┴───────────┘

  NORMAL (0–50%):
    All admitted entries are accepted into the write buffer.

  CAUTIOUS (50–75%):
    Raise the admission_cost_threshold dynamically:
      threshold = base_threshold × (utilization / 0.5)
    Low-value entries (short context on small models) start being rejected.
    Metric: attentiondb_write_backpressure_rejections_total{level="cautious"}

  SELECTIVE (75–90%):
    Only accept entries with recompute_cost > high_value_threshold OR
    entries in the "always admit" category (system prompt KV caches).
    Metric: attentiondb_write_backpressure_rejections_total{level="selective"}

  EMERGENCY (90–100%):
    Only accept "always admit" entries (system prompts).
    All other stores are silently dropped.
    Metric: attentiondb_write_backpressure_rejections_total{level="emergency"}

  FULL (100%):
    Drop everything. Never block the caller.
    Metric: attentiondb_write_buffer_drops_total
```

**Why progressive backpressure matters**: Without it, a burst of cheap entries can fill
the buffer and cause expensive entries to be dropped. Progressive filtering ensures the
buffer is always used for the highest-value entries first.

**Configuration**:

```yaml
write_buffer:
  size: "256MB"                          # Total write buffer size in pinned DRAM.
  flush_interval_ms: 50                  # Timer-based flush even if buffer isn't full.
  backpressure:
    enabled: true                        # Set false for simple all-or-nothing drop behavior.
    cautious_threshold: 0.5              # Buffer utilization to start cautious filtering.
    selective_threshold: 0.75            # Buffer utilization to start selective filtering.
    emergency_threshold: 0.9             # Buffer utilization for emergency mode.
    always_admit:                        # Entry types that bypass backpressure until FULL.
      - "system_prompt"
      # - "rag_context"                  # Uncomment if RAG contexts are high-value for your workload.
```

### 7.4 Cluster Routing Cache (Phase 1 only)

When running in `hybrid` or `remote_only` mode, the storage engine maintains a local copy
of the cluster metadata plane:
- Refreshed via gossip (piggyback on heartbeat messages)
- Used for routing: which AttentionDB server to contact for a given key
- Includes PG-to-node mapping, node health, epoch numbers
- Size: < 100 MB (placement maps, node states, popularity sketches)
- Not applicable in `local_only` mode (Phase 0)

---

## 8. Observability

### 8.1 Design Principles

- **Single metric stack**: OpenTelemetry SDK everywhere (no split between prometheus_client
  and OTel). Export via OTLP to any collector, or Prometheus-compatible scrape endpoint.
- **Storage-focused metrics**: AttentionDB reports on storage performance (latency, hit rates,
  eviction, GC). Inference-level metrics (TTFT savings, prefill time avoided, request-level
  hit rates) belong in LMCache, which has the inference context AttentionDB lacks.
- **Per-tenant, per-model labels**: All metrics carry `tenant_id` and `model_id` labels
  (extracted from the `StorageKey`).
- **Histogram over averages**: Tail latency matters for inference SLAs. All latency metrics
  are histograms (p50/p95/p99), never just averages.

### 8.2 Metric Catalog

**Tier performance** (per tier, per node):

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_read_latency_seconds` | Histogram | Read latency by tier (T1/T2/T3/T4), p50/p95/p99 |
| `attentiondb_write_latency_seconds` | Histogram | Write latency by tier |
| `attentiondb_hit_rate` | Gauge | Cache hit rate by tier (rolling window) |
| `attentiondb_hit_rate_by_type` | Gauge | Hit rate split by entry type (system_prompt, conversation, rag_context) |
| `attentiondb_bytes_read_total` | Counter | Bytes read by tier |
| `attentiondb_bytes_written_total` | Counter | Bytes written by tier |
| `attentiondb_eviction_total` | Counter | Evictions by tier, by reason (capacity, TTL, explicit) |
| `attentiondb_storage_utilization` | Gauge | Bytes used / bytes total by tier |

**Admission and eviction**:

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_admission_rejected_total` | Counter | Entries rejected by admission control, by reason |
| `attentiondb_write_backpressure_rejections_total` | Counter | Write buffer rejections by backpressure level |
| `attentiondb_write_buffer_drops_total` | Counter | Entries dropped when write buffer is full |
| `attentiondb_wasted_cache_bytes` | Counter | Bytes cached but evicted before any read (wasted storage) |

**Note**: Inference-correlated metrics (TTFT savings, prefill time avoided, per-request hit
rates) are LMCache's responsibility. LMCache has the inference context to compute these;
AttentionDB only sees storage operations.

**Clustering** (per AttentionDB server):

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_remote_fetch_total` | Counter | Cross-node RDMA fetches (should trend toward zero as caches warm) |
| `attentiondb_remote_fetch_latency_seconds` | Histogram | RDMA fetch latency |
| `attentiondb_routing_efficiency` | Gauge | 1 - (remote_fetches / total_reads). Target > 0.9 |
| `attentiondb_replication_lag_seconds` | Gauge | Time from write on primary to availability on warm replica |
| `attentiondb_node_warmup_progress` | Gauge | 0-1, how warm a joining node is |
| `attentiondb_gossip_convergence_seconds` | Histogram | Time for membership changes to propagate |

**Storage engine** (per AttentionDB server):

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_gc_reclaimed_bytes` | Counter | Bytes freed by garbage collection |
| `attentiondb_gc_bandwidth_fraction` | Gauge | Fraction of NVMe bandwidth used by GC (target < 0.2) |
| `attentiondb_index_entries` | Gauge | Number of entries in the in-memory index |
| `attentiondb_segment_count` | Gauge | Number of active blob segment files |
| `attentiondb_segment_fragmentation` | Gauge | Average dead-entry ratio across segments |
| `attentiondb_ssd_read_latency_seconds` | Histogram | Raw NVMe read latency (detects thermal throttling) |

**Health**:

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_node_status` | Gauge | 1 = healthy, 0 = degraded, -1 = dead |
| `attentiondb_cluster_healthy_nodes` | Gauge | Count of healthy nodes by type |
| `attentiondb_miss_total` | Counter | Total misses across all tiers (caller must recompute) |

### 8.3 Health Checks

Multi-layer health checking:

| Layer | Check | Frequency | Failure action |
|-------|-------|-----------|---------------|
| **NVMe health** | Read latency probe (4 KB read from known offset) | Every 5s | If p99 > 10 ms: mark tier degraded, bypass SSD reads |
| **DRAM allocator** | Free slab count > minimum | Every 1s | If exhausted: increase eviction aggressiveness |
| **RDMA connectivity** | Ping to AttentionDB servers | Every 2s | If unreachable: route to backup server, alert |
| **Index integrity** | Spot-check: read random entry, verify checksum | Every 30s | If corrupt: trigger index rebuild from checkpoint |
| **End-to-end** | Synthetic cache store + retrieve cycle | Every 10s | If latency > threshold: mark node degraded |

### 8.4 Tracing

Distributed tracing (OpenTelemetry) with spans for storage engine operations:
- `attentiondb.get` → `attentiondb.t1_check` → `attentiondb.t2_read` → `attentiondb.t3_fetch` (shows which tier served the request)
- `attentiondb.put` → `attentiondb.admission_check` → `attentiondb.write_buffer` → `attentiondb.replicate`

AttentionDB propagates trace context received from LMCache so that storage spans appear as
children of LMCache's orchestration spans. Compression and decompression spans belong to
LMCache, not AttentionDB.

---

## 9. Graceful Degradation

AttentionDB must never block or crash the inference pipeline. Degradation is predictable and
prioritized:

| Severity | Condition | Behavior |
|----------|-----------|----------|
| **Normal** | All tiers healthy | Full caching, all tiers active |
| **Tier degraded** | SSD latency spike or NIC congestion | Bypass degraded tier, serve from remaining tiers. Accept higher miss rate. |
| **Node degraded** | AttentionDB server unreachable | Failover to warm replica. GPU node uses local T1/T2 only until remote recovers. |
| **Placement group degraded** | Multiple AttentionDB servers fail | Route to any surviving server. Accept remote misses → prefill recompute. |
| **Metadata stale** | Gossip partition | Nodes use stale placement map. Routing may be suboptimal but functional. |
| **Full bypass** | AttentionDB client library failure | Inference engine skips cache entirely. All requests use full prefill. Latency degrades but service is fully available. |

**Implementation**: The client library wraps all AttentionDB operations in a timeout + circuit
breaker. If a tier or server exceeds the latency budget (configurable per tier), it is
temporarily bypassed. The circuit breaker resets after a cooldown period (default 30s).

---

## 10. Configuration

### 10.1 GPU Node (Client) Configuration

```yaml
attentiondb:
  mode: "hybrid"                          # "hybrid" | "local_only" | "remote_only" | "embedded"

  cluster:
    seed_nodes: ["attentiondb-1:7300", "attentiondb-2:7300"]
    rdma_device: "mlx5_0"                 # RDMA NIC device name
    transport: "nixl"                     # "nixl" | "ucx" | "tcp"

  local:
    t1_dram_size: "16GB"                  # Pinned CPU DRAM for hot cache
    t2_nvme_path: "/mnt/nvme0/attentiondb"     # Local NVMe mount point
    t2_nvme_size: "1TB"                   # Max NVMe usage
    t2_use_gds: true                      # Enable GPUDirect Storage
    write_buffer_size: "256MB"            # Write buffer before flush

  # Note: compression config is NOT here — it belongs in LMCache's config.
  # AttentionDB stores opaque blobs; compression is LMCache's concern.

  admission:
    min_recompute_cost: 64                # Don't cache below this cost
    base_threshold: 100                   # Dynamic threshold base for cost-aware filtering

  eviction:
    policy: "cw_slru"                     # "cw_slru" | "lru" | "lfu"
    protected_ratio: 0.25                 # Fraction of capacity for protected segment

  timeouts:
    get_ms: 5                             # Max time for get() before returning miss
    remote_fetch_ms: 100                  # Max time for RDMA fetch
    circuit_breaker_cooldown_s: 30        # Cooldown after repeated timeouts
```

### 10.2 AttentionDB Server Configuration

```yaml
attentiondb_server:
  node_id: "attentiondb-1"
  listen_address: "0.0.0.0:7300"
  rdma_device: "mlx5_0"

  storage:
    t3_dram_size: "384GB"                 # DRAM for shared hot cache
    t4_nvme_paths:                        # NVMe drives for blob store
      - "/mnt/nvme0/attentiondb"
      - "/mnt/nvme1/attentiondb"
      - "/mnt/nvme2/attentiondb"
    t4_nvme_size_per_drive: "7TB"
    segment_size: "1GB"                   # Blob log segment file size
    index_checkpoint_interval_s: 30

  cluster:
    seed_nodes: ["attentiondb-1:7300", "attentiondb-2:7300", "attentiondb-3:7300"]
    replication_factor: 2                 # Copies per entry (including primary)
    gossip_interval_ms: 1000
    suspect_timeout_ms: 5000
    shadow_mode_duration_s: 300
    shadow_traffic_fraction: 0.1
    warmup_min_coverage: 0.7
    warmup_max_duration_s: 60

  gc:
    trigger_fragmentation: 0.5            # GC when segment live ratio < this
    max_bandwidth_fraction: 0.2           # Max NVMe bandwidth for GC

  eviction:
    policy: "cw_slru"
    protected_ratio: 0.25
    t3_watermark: 0.85                    # Start evicting T3 when DRAM > 85%
    t4_watermark: 0.90                    # Start evicting T4 when NVMe > 90%

  observability:
    metrics_port: 9090                    # Prometheus scrape endpoint
    otlp_endpoint: ""                     # OTLP collector (optional)
    enable_tracing: false
    log_level: "info"
```

### 10.3 Deployment Modes

| Mode | Description | Use case |
|------|-------------|----------|
| **`hybrid`** | Local tiers (T1/T2) on GPU node + remote AttentionDB cluster (T3/T4) | Production |
| **`local_only`** | Only T1/T2 on GPU node, no remote cluster | Single-node, testing, air-gapped |
| **`remote_only`** | No local NVMe, T1 DRAM only + remote cluster | When GPU nodes have no NVMe, or need to maximize GPU node resources |
| **`embedded`** | Full AttentionDB server in-process on GPU node | Development, single-node experiments |

---

## 11. Security and Multi-Tenant Hardening

AttentionDB may operate in multi-tenant environments where different customers' KV cache data
coexists on shared infrastructure. While KV cache entries are not as sensitive as raw user
input (they are compressed attention state, not plaintext), they may still contain information
that could be used to reconstruct prompt content. Security must be addressed even within
trusted internal clusters to satisfy compliance requirements.

### 11.1 Threat Model

| Threat | Risk | Mitigation |
|--------|------|------------|
| **Unauthorized cross-tenant read** | Tenant A reads Tenant B's cached KV data | PG-level authz + tenant token enforcement |
| **Network eavesdropping** | Attacker on the network reads KV cache data in transit | mTLS (TCP), encryption for RDMA (see below) |
| **Disk theft / decommission** | Physical NVMe drive contains KV cache data | Encryption at rest |
| **Rogue GPU node** | Compromised inference node reads data from other tenants | Client authentication + server-side PG access control |
| **Metadata leakage** | Access frequency patterns reveal tenant workload characteristics | Tenant-scoped metrics, audit-logged metadata access |
| **Denial of service** | One tenant consumes all resources | Per-PG resource quotas (already designed in Section 4.2.4) |

### 11.2 Authentication

**Node-to-node (server ↔ server, client ↔ server)**:
- **mTLS** for all TCP connections. Each node has a certificate issued by a cluster CA.
  Nodes reject connections from certificates not in the cluster's trust chain.
- **RDMA authentication**: RDMA bypasses the kernel networking stack, so standard TLS
  cannot be applied. Two options:
  - **Physical isolation**: InfiniBand fabrics are typically dedicated networks, not shared.
    If the RDMA fabric is physically isolated to the AttentionDB cluster, network-level
    authentication may be sufficient.
  - **Application-level token**: Include a signed authentication token in every RDMA message
    header. The server verifies the token before processing the request. This adds ~1 μs
    per request (HMAC-SHA256 verification) but provides per-request authentication.

**Client identity**:
- The client library authenticates with a **tenant token** on connection:
  ```
  ClientAuthToken {
      tenant_id:    u64,
      issued_at:    Timestamp,
      expires_at:   Timestamp,
      allowed_models: Vec<u64>,   // model_ids this tenant can access
      signature:    [u8; 32],     // HMAC-SHA256 by cluster secret
  }
  ```
- Tokens are issued by an external identity provider (e.g., Kubernetes service account,
  Vault, internal auth service). AttentionDB does not manage user accounts.

### 11.3 Authorization

**PG-level access control**: The server enforces that a client can only read/write PGs
matching its `(tenant_id, allowed_models)`:

```
On every request:
  1. Extract tenant_id from authenticated client session.
  2. Compute pg_id from the request's (model_id, tenant_id).
  3. Verify: request.tenant_id == session.tenant_id
             AND request.model_id IN session.allowed_models
  4. If not → reject with ACCESS_DENIED.
```

This is a simple, fast check (two comparisons) that runs on every request. Since PGs
are already scoped by (model_id, tenant_id), authorization aligns naturally with the
data model — there is no way for a tenant to accidentally (or intentionally) construct
a CacheKey that maps to another tenant's PG.

### 11.4 Encryption

**In transit**:
- TCP: mTLS provides encryption. No additional design needed.
- RDMA: RDMA bypasses the kernel networking stack, so standard TLS cannot be applied.
  Three options in order of preference:

  **Option 1 — Physical network isolation** (recommended for most deployments):
  InfiniBand fabrics are typically dedicated networks, not shared with untrusted traffic.
  If the RDMA fabric is physically isolated to the AttentionDB cluster, no encryption is
  needed. This preserves zero-copy RDMA semantics and adds zero latency overhead.

  **Option 2 — Application-level encryption** (for shared or untrusted networks):
  Encrypt blob data before RDMA send, decrypt after receive. Use AES-256-GCM with
  per-tenant keys (AES-NI hardware acceleration on modern CPUs).

  **Latency impact of RDMA encryption** — this breaks the zero-copy property that is
  central to the RDMA latency advantage. The data path becomes:
  ```
  Without encryption (zero-copy):
    RDMA-registered buffer → RDMA send → remote RDMA buffer     (~20–200 μs for 256KB–4MB)

  With encryption:
    Plaintext buffer → AES-256-GCM encrypt on CPU → encrypted RDMA buffer
      → RDMA send → remote encrypted RDMA buffer
      → AES-256-GCM decrypt on CPU → plaintext buffer             (+4–40 μs for 256KB–4MB)
  ```
  The encryption overhead is ~2–5 μs per MB (with AES-NI). For a 1 MB blob on 100 Gbps
  InfiniBand, the RDMA transfer itself is ~80 μs — encryption adds ~4 μs (~5% overhead).
  For a 4 MB blob, encryption adds ~16 μs on an ~320 μs transfer (~5%). The percentage
  overhead is modest, but the absolute latency pushes T3 performance from the "RDMA"
  column toward the space between RDMA and TCP. The T3 latency targets in Section 1
  assume unencrypted RDMA; with encryption, expect p50 to increase by ~5–10% and p99
  by more (encrypt/decrypt competes with other CPU work under load).

  **Option 3 — IPsec over RoCEv2** (if available):
  Some network fabrics support hardware-accelerated IPsec for RoCEv2 traffic (e.g.,
  NVIDIA ConnectX-7 with IPsec offload). This provides line-rate encryption without
  CPU involvement, preserving zero-copy semantics. However, this requires specific NIC
  hardware and fabric configuration. Not available on all deployments.

**At rest**:
- **Option A — Filesystem-level**: dm-crypt / LUKS on NVMe drives. Simple, transparent,
  but uses a single key per drive (no per-tenant isolation).
- **Option B — Application-level** (recommended for multi-tenant): Encrypt blob data
  with per-tenant keys before writing to the blob store. Key management via an external
  KMS (Vault, AWS KMS, etc.). The index stores a `key_id` field in the reserved bytes
  of the 64-byte IndexEntry to identify which decryption key to use.
- Compression is applied before encryption (compress → encrypt → write). On read:
  read → decrypt → send to GPU → decompress.

### 11.5 Audit

```yaml
audit:
  enabled: false                         # Enable for compliance-sensitive deployments.
  log_events:
    - "pg_ownership_change"              # Primary lease grants, epoch changes.
    - "node_join_leave"                  # Cluster membership changes.
    - "cross_tenant_access_denied"       # Rejected authorization attempts.
    - "admin_operations"                 # Manual eviction, PG drain, config changes.
  # - "all_reads"                        # WARNING: very high volume. Use only for investigation.
  # - "all_writes"                       # WARNING: very high volume.
  destination: "stdout"                  # "stdout" | "file" | "otlp" (as OTel log events)
```

### 11.6 Phase Planning

Security is designed in layers, with the wire protocol supporting authentication from
Phase 1 so that adding security later does not require breaking protocol changes:

| Phase | Security scope |
|-------|---------------|
| **Phase 1** | Wire protocol includes auth token field (unused but reserved). mTLS optional. No encryption at rest. |
| **Phase 2** | mTLS enforced for TCP. RDMA auth tokens. PG-level authorization. Basic audit logging. |
| **Phase 3** | Per-tenant encryption at rest. KMS integration. Full audit trail. RDMA encryption option. Compliance documentation. |

---

## 12. Competitive Landscape

AttentionDB targets a gap that no existing system fills: a purpose-built, inference-aware storage
engine for KV cache offloading. Below we compare against the four most relevant systems.

### 12.1 Redis

**What it is**: In-memory key-value store; the default "reach for" cache in most stacks. LMCache
uses Redis as a remote backend option.

**Strengths for this workload**:
- Sub-millisecond latency for small values (< 10 KB): ~0.1–0.3 ms p99
- Ubiquitous — every team knows how to operate it
- LRU/LFU eviction policies with per-key TTL
- Cluster mode with 16,384 hash slots

**Critical weaknesses for KV cache offloading**:
- **Single-threaded event loop**: Reading a 10 MB KV cache chunk blocks *all* concurrent
  operations for ~87 ms. Redis officially recommends against values > 1 MB.
- **All-DRAM economics**: Storing 1 TB of KV cache requires ~1 TB of RAM (~$5,000–8,000/mo).
  No hybrid memory / NVMe tier.
- **No RDMA / kernel bypass**: TCP only. For 1–15 MB bulk transfers, TCP adds
  kernel copies and syscall overhead that dominate transfer time.
- **Replication amplifies the problem**: Large-value writes block replica sync, causing
  stale reads on replicas.
- **No inference awareness**: No concept of recompute cost, prefix sharing, or model-aware
  eviction.

**Verdict**: Fundamentally unsuitable for multi-MB KV cache chunks at scale. Acceptable only as a
metadata store or for small-value workloads.

### 12.2 Aerospike

**What it is**: Production-hardened hybrid-memory database (index in DRAM, data on NVMe SSD).
Open source (AGPL), 12+ years in production at PayPal, Snap, Airtel, and others.

**Architecture** (from source analysis of `aerospike-server`):
- **Index**: 64-byte `as_index` entries in red-black trees, organized by 4,096 partitions
  (digest-based sharding). Each entry holds `rblock_id`, `file_id`, TTL, generation.
- **Storage**: SSD engine writes in 8 MiB write-blocks, addressing records in 16-byte
  "rblocks." Records are stored contiguously — no separate blob store.
- **Networking**: Standard TCP/epoll with optional TLS. No RDMA, no kernel bypass, no
  GPUDirect Storage.
- **Eviction**: TTL-based expiration + capacity-driven eviction using TTL histograms
  (evict soonest-expiring records when storage pressure exceeds high-water marks). No
  LRU/LFU or cost-aware policies.
- **Clustering**: 4,096-partition auto-rebalancing with heartbeat-based failure detection
  and fabric-based migration. Battle-tested at scale.

**Strengths for this workload**:
- **Best cost efficiency at scale**: 1 TB dataset needs only ~20–50 GB DRAM (index) plus
  NVMe — 78–87% cheaper than Redis annually.
- **Most mature clustering**: Automatic partition rebalancing, proven at thousands of nodes.
  17–48% lower p99 latency than Redis in benchmarks.
- **Multi-threaded**: Large record reads don't block the entire server.
- **Namespaces + sets**: Closest thing to multi-tenant isolation among the three.

**Critical weaknesses for KV cache offloading**:
- **SSD engine optimized for small records**: 128 KB recommended, 8 MB hard max on SSD.
  KV cache chunks (1–15 MB) either don't fit or cause suboptimal write-block utilization.
- **No RDMA**: TCP-only transport adds latency for bulk data transfer to GPUs.
- **No GPUDirect Storage**: Cannot bypass CPU for NVMe → GPU HBM transfers.
- **TTL-only eviction**: No cost-aware or access-frequency-based eviction. Expensive
  long-context KV caches get evicted just as readily as cheap short-context ones.
- **No inference awareness**: No concept of prefix sharing, model-specific tuning, or
  recompute cost.
- **rblock-based layout**: Contiguous record allocation on SSD causes fragmentation for
  large, variable-size blobs. Log-structured blob store would be more efficient.

**Verdict**: Strong operational foundation and cost model, but the storage engine and network
transport are not designed for the specific I/O patterns of KV cache offloading.

### 12.3 Mooncake Store

**What it is**: Distributed object store for KV cache, part of the Mooncake project (Apache 2.0,
open source at `github.com/kvcache-ai/Mooncake`). Developed by Moonshot AI for their Kimi LLM
serving platform. Comprises two components:

- **Transfer Engine**: High-performance data movement layer supporting RDMA (RoCE/InfiniBand),
  NVLink, TCP, CXL, NVMe-oF. Handles GPU VRAM ↔ CPU DRAM ↔ NVMe transfers across nodes.
- **Mooncake Store**: Distributed object storage with DRAM + SSD tiers, centralized Master for
  metadata, configurable replication.

**Strengths for this workload**:
- **RDMA transport**: Zero-copy, kernel-bypass bulk transfers. For 1–15 MB chunks, RDMA
  delivers ~2–5 μs for DRAM-to-DRAM, orders of magnitude faster than TCP.
- **Designed for KV cache shapes**: Optimized for large, immutable, write-once/read-many blobs.
- **GPU-aware**: Transfer Engine can move data directly between GPU VRAM on different nodes.
- **LMCache integration**: Pluggable as a remote backend via `MooncakeStoreConnector`,
  with zero-copy batched get/put using registered CPU buffers.
- **Can run independently**: Works with or without LMCache — vLLM can integrate directly
  via `MooncakeConnector`.

**Critical weaknesses for KV cache offloading**:
- **Centralized Master**: Single metadata server is a scalability bottleneck and single
  point of failure for large clusters. Contrast with AttentionDB's gossip-based metadata plane.
- **No inference-aware intelligence**: No cost-weighted eviction, no predictive prefetch,
  no model-aware sharding or admission control.
- **Immature operationally**: ~1 year in production (Kimi only). Small open-source community.
  Limited documentation, log-based observability only.
- **Memory allocation overhead**: Internally uses 4 MB pages; reads larger than 4 MB incur
  inefficient memory allocation and copying (reported in GitHub issue #467). Benchmarks show
  retrieval latency of 29s vs 1.77s for local DRAM at chunk_size=256.
- **No native compression**: Compression must be handled externally (by LMCache or the
  inference engine).
- **Requires InfiniBand/RoCE**: The RDMA advantage disappears in environments without
  specialized networking hardware. TCP fallback exists but negates the core value proposition.

**Reuse opportunity**: Mooncake's Transfer Engine is a strong candidate for AttentionDB's data plane
transport layer, rather than building RDMA support from scratch. AttentionDB would provide the
storage engine, metadata plane, and inference-aware intelligence on top.

### 12.4 WEKA (NeuralMesh / Augmented Memory Grid)

**What it is**: Commercial parallel distributed file system optimized for AI workloads.
WEKA's Augmented Memory Grid provides a GDS-integrated storage tier for KV cache offloading.

**Strengths**:
- **GPUDirect Storage native**: Direct DMA from NVMe to GPU HBM, production-proven.
- **Parallel file system**: High aggregate bandwidth across many clients.
- **Enterprise support**: Commercial product with SLA guarantees.

**Weaknesses for KV cache offloading**:
- **File system abstraction**: POSIX overhead for what is fundamentally a key-value workload.
  Inode management, directory traversal, and metadata operations add unnecessary latency.
- **No inference awareness**: Same gap as all others — no cost-aware eviction, prefix sharing,
  or model-specific tuning.
- **Commercial / closed source**: Licensing cost, vendor lock-in.
- **Not purpose-built for KV cache**: A general-purpose parallel file system adapted for
  AI, not designed from scratch for KV offloading patterns.

### 12.5 VDURA (Emerging)

Announced at NVIDIA GTC 2026: RDMA support and "Context-Aware Tiering" for GPU-native AI
infrastructure. Claims direct GPU-to-storage data transfers bypassing CPU, with intelligent
data placement across storage tiers based on workload patterns. Features include DirectFlow
Buffer to local SSD and KVCache Writeback for AI inference persistence. Worth monitoring as
a potential commercial competitor in the purpose-built KV cache storage space.

### 12.6 Relationship with LMCache

LMCache is not a competitor — it is AttentionDB's primary integration target. LMCache is the
**orchestration layer** (token-to-key mapping, prefix matching, vLLM integration, compression).
AttentionDB is the **storage engine** that replaces LMCache's built-in storage backends with
purpose-built C/C++ implementations (see Section 1, "Relationship to LMCache").

**What AttentionDB replaces in LMCache**:

| LMCache component | Current implementation | AttentionDB replacement |
|---|---|---|
| `LocalCPUBackend` | Python dict + `cache_policy` (LRU/LFU) | Lock-free C hash map + slab allocator + CW-SLRU |
| `LocalDiskBackend` / `GdsBackend` | One file per chunk, buffered I/O | Log-structured blob store, io_uring, GDS |
| `RemoteBackend` (Redis, Mooncake) | TCP-only, no inference-aware routing | PG-based routing, RDMA transport, cost-aware admission |

**What LMCache keeps** (AttentionDB does not duplicate):

| LMCache component | Why it stays in LMCache |
|---|---|
| `ChunkedTokenDatabase` | Rolling prefix hash, chunking — tightly coupled to tokenizer/vLLM |
| `StorageManager` tier orchestration | Decides which backend to query — LMCache's cross-backend routing |
| vLLM `KVConnectorBase` adapter | vLLM integration is Python, model-lifecycle aware |
| CacheGen/KIVI/TurboQuant serde | Compression needs GPU context that storage engine shouldn't own |
| `LMCStatsMonitor` / `PrometheusLogger` | Inference-correlated metrics need request-level context |

### 12.7 Summary Scorecard

| Criterion | Redis | Aerospike | Mooncake Store | WEKA | AttentionDB (target) |
|-----------|-------|-----------|----------------|------|----------------|
| Large chunk I/O (1–15 MB) | 2/10 | 5/10 | 7/10 | 6/10 | 9/10 |
| Read latency (sub-ms target) | 7/10 (small) 2/10 (large) | 6/10 | 8/10 (RDMA) 5/10 (TCP) | 7/10 (GDS) | 9/10 |
| Cost at TB scale | 2/10 | 9/10 | 7/10 | 4/10 | 8/10 |
| Transport (RDMA/GDS) | 1/10 | 1/10 | 8/10 | 7/10 | 9/10 |
| Clustering maturity | 5/10 | 9/10 | 4/10 | 8/10 | 7/10 (target) |
| Inference-aware intelligence | 1/10 | 1/10 | 2/10 | 1/10 | 9/10 |
| Operational maturity | 9/10 | 9/10 | 3/10 | 8/10 | 3/10 (initially) |
| Multi-tenant support | 2/10 | 7/10 | 2/10 | 5/10 | 8/10 |
| **Overall fit for KV offloading** | **3/10** | **6/10** | **6/10** | **5/10** | **8/10** |

### 12.8 AttentionDB's Differentiated Position

No existing system combines all three requirements for a production KV cache storage engine:

1. **Storage engine economics** (Aerospike-inspired): Index in DRAM, data on NVMe, but with
   a log-structured blob store designed for 1–15 MB chunks instead of rblock-based layout
   optimized for small records.

2. **High-performance transport** (Mooncake-inspired): RDMA for bulk data transfers with
   zero-copy semantics, plus GDS for local NVMe → GPU HBM. AttentionDB may integrate
   Mooncake's Transfer Engine directly rather than building RDMA support from scratch.

3. **Inference-aware storage intelligence** (unique to AttentionDB): Cost-weighted eviction
   that considers recompute cost, popularity-driven admission control, model/tenant-aware
   sharding via placement groups, and shadow join for cold start mitigation. Combined with
   LMCache's orchestration layer, this delivers end-to-end inference-aware caching that
   neither system could provide alone.

---

## 13. Implementation Roadmap

Each phase has a dedicated specification document with detailed scope, implementation tasks,
configuration, and risk analysis:

- **Phase 0 (MVP)**: [PHASE0-MVP.md](PHASE0-MVP.md) — Local engine on GPU node
- **Phase 1 (Server)**: [PHASE1-SERVER.md](PHASE1-SERVER.md) — Remote cluster + networking

### Phase 0: Local Storage Engine MVP (T1 + T2)

> Full spec: [PHASE0-MVP.md](PHASE0-MVP.md)

- Lock-free concurrent hash map index (single partition, no PGs)
- Slab allocator for DRAM cache (T1) with CUDA pinned memory
- Log-structured blob store for NVMe (T2) with io_uring
- GDS integration for T2 reads
- CW-SLRU eviction with cost-aware admission (without popularity sketch)
- Write buffer with progressive backpressure
- Client library C API + Python bindings
- vLLM integration via LMCache `StorageBackendInterface`
- `local_only` deployment mode
- Structured logging only (no Prometheus, no OTel)

### Phase 1: Remote Cluster (T3 + T4)

> Full spec: [PHASE1-SERVER.md](PHASE1-SERVER.md)

- AttentionDB server daemon
- RDMA data plane — evaluate Mooncake Transfer Engine vs. building on NIXL/UCX directly.
  Transfer Engine provides RDMA verbs, NVLink, TCP fallback, and buffer registration
  out of the box (Apache 2.0 licensed). If integrated, AttentionDB avoids ~3–6 months of
  low-level RDMA development. Fallback: build on NIXL/UCX if Transfer Engine coupling
  is too tight or adds unwanted dependencies.
- TCP transport path for environments without InfiniBand/RoCE (must be a first-class
  path, not an afterthought — many cloud GPU instances lack RDMA)
- Gossip-based metadata plane (SWIM) — explicitly avoiding Mooncake's centralized Master
  design, which is a scalability and availability risk
- Epoch-based primary leases (Section 4.2.7) — must ship with clustering, not bolted on later
- Rendezvous hashing and placement groups with PG scaling modes (inline/compact/full)
- Async replication to warm replicas
- Node join protocol (REGISTERING → WARMING → SHADOW → ACTIVE)
- Draining and planned scale-down
- `hybrid` and `remote_only` deployment modes
- Full observability stack (OTel, Prometheus, tracing, health checks)
- mTLS for TCP. RDMA auth tokens. PG-level authorization.
- Full admission control with popularity sketch integration

### Phase 2: Production Hardening
- Multi-tenant isolation (quotas, per-tenant eviction pools)
- Per-tenant encryption at rest with external KMS
- Topology-aware placement (rack/zone awareness)
- Scheduler integration API (cache-aware routing)
- Grafana dashboard templates
- Kubernetes operator
- Compression codec benchmarking and auto-selection
- Performance benchmarks and regression tests
- Full audit logging and compliance documentation
- Documentation and operational runbooks

---

## 14. Known Risks and Open Questions

This section documents design risks that have been identified but not yet fully resolved,
and open questions that require further investigation or prototyping before implementation.

### 14.1 Risks

**R1. Epoch lease overhead on write latency**
- **Risk**: Every write to a PG requires a valid lease. If the lease renewal is delayed
  (e.g., GC pause, NIC congestion), the PG is temporarily fenced (read-only). This could
  cause intermittent write drops under sustained network pressure.
- **Severity**: Medium. Writes are best-effort in a cache system, but frequent fencing
  degrades hit rate.
- **Mitigation**: Tune `lease_ttl_s` and `lease_renew_interval_s` conservatively. Monitor
  `attentiondb_pg_fenced_total` metric. If fencing rate is high, investigate network health.
- **Status**: Needs production validation under realistic network failure injection.

**R2. Gossip convergence delay during rapid scaling**
- **Risk**: SWIM gossip converges in O(log N) rounds. In a 100-node cluster with 1-second
  ping intervals, a membership change takes ~7 seconds to propagate to all nodes. During this
  window, different nodes have different views of the cluster membership, leading to
  inconsistent rendezvous rankings and potentially misrouted requests.
- **Severity**: Low. Misrouted requests result in a cache miss (the wrong server doesn't
  have the data), not data corruption. The request falls through to prefill recompute.
- **Mitigation**: Clients use server-side epoch validation — if the server's epoch for a PG
  is higher than the client's, the client refreshes its placement map and retries. Also,
  gossip convergence can be accelerated by increasing ping frequency during scaling events.
- **Status**: Acceptable for initial deployment. May need faster convergence for large clusters.

**R3. GPU decompression as latency floor**
- **Risk**: TurboQuant GPU decompression takes ~50–100 μs. This is a hard floor that makes
  T1 DRAM hits only ~2× faster than T2 NVMe hits (not 10× as one might expect). If
  decompression latency increases with future compression codecs, the benefit of the DRAM
  cache tier diminishes.
- **Severity**: Low. This is a physics constraint, not a design flaw. The DRAM cache still
  helps by avoiding NVMe queue contention and providing deterministic latency.
- **Mitigation**: Monitor decompression latency per codec. Consider storing uncompressed
  blobs in T1 for latency-critical models (trades memory for speed). Future work: explore
  decompression-free formats (e.g., pre-dequantized blobs in T1 that skip the GPU kernel).
- **Status**: Understood. No action needed for Phase 1.

**R4. RDMA hardware availability in cloud environments**
- **Risk**: Many cloud GPU instances (e.g., AWS p4d, GCP a2) have InfiniBand/RoCE, but not
  all. Some environments (on-prem, smaller clouds) use only standard Ethernet. If AttentionDB
  is optimized primarily for RDMA, the TCP path may be under-tested and underperforming.
- **Severity**: High for adoption. RDMA is a niche requirement that limits the addressable
  market.
- **Mitigation**: TCP transport must be a first-class path with its own performance testing
  and optimization (io_uring-based TCP, batched transfers, optional Jumbo frames). The design
  should be "great on TCP, exceptional on RDMA" — not "only good on RDMA."
- **Status**: TCP path must be part of Phase 2 acceptance criteria, not deferred.

**R5. PG cardinality explosion in multi-tenant deployments**
- **Risk**: With many (model, tenant) pairs, the number of PGs can grow into tens of
  thousands. Even with inline/compact/full scaling modes, the metadata plane must track
  all PGs, and each PG has lease state, popularity sketches, and replication state.
- **Severity**: Medium. Metadata overhead grows linearly with PG count.
- **Mitigation**: PG scaling modes (Section 4.2.6) bound per-PG overhead. Additionally,
  inactive PGs (TOMBSTONE state) are garbage collected after 1 hour. For extreme cardinality,
  consider PG grouping — multiple (model, tenant) pairs sharing a single PG when traffic
  is below a threshold (sacrificing per-tenant isolation for overhead reduction).
- **Status**: Needs load testing with 10,000+ PGs to validate metadata plane performance.

### 14.2 Open Questions

**Q1. Mooncake Transfer Engine vs. NIXL/UCX for RDMA transport**
- Transfer Engine (Apache 2.0) provides RDMA, NVLink, TCP, and buffer registration.
  Integrating it saves ~3–6 months of RDMA development. However, it may introduce
  coupling with Mooncake's design assumptions and dependency on their build system.
- NIXL/UCX is more general-purpose but lower-level, requiring more development effort.
- **Decision needed before**: Phase 2 implementation starts.
- **Investigation**: Build a prototype with each, benchmark RDMA write throughput and
  latency on a 2-node cluster with ConnectX-7 NICs.

**Q2. Storage engine crash isolation** (RESOLVED — in-process is the default)
- Decision: In-process is the default deployment model. See Section 7 for rationale.
- **Remaining concern**: An in-process crash (segfault in the C/C++ storage engine) takes
  down the LMCache/inference engine process. Mitigations: extensive fuzzing, memory safety
  (Rust preferred), and a `safe_mode` config that disables unsafe operations (raw pointer
  manipulation, O_DIRECT I/O) at the cost of performance.

**Q3. Index checkpoint vs. write-ahead log for crash recovery**
- Current design: periodic checkpoint (every 30s). On crash, up to 30s of index updates
  are lost. These entries still exist on NVMe (in blob segments) but cannot be found
  until the next checkpoint loads or a segment scan runs.
- Alternative: WAL for index updates. Every index mutation is appended to a WAL before
  being applied. On crash, replay the WAL from the last checkpoint. Zero data loss.
- **Trade-off**: WAL adds write latency (~1–5 μs per index update) and NVMe write bandwidth.
  For a cache system where data is recomputable, 30s of lost index entries is acceptable.
  But for production deployments with expensive long-context KV caches, the cost of
  recomputing those entries may justify the WAL overhead.
- **Decision needed before**: Phase 1 completion.

**Q4. How to handle model and tokenizer version changes?**
- This is primarily **LMCache's concern**, since LMCache computes the key from tokens. But
  it affects AttentionDB because `model_id` changes create new PGs and orphan old ones.
- Design decision: `model_id = hash(model_name, weights_version, tokenizer_hash)`. LMCache's
  adapter must ensure any change to weights or tokenizer creates a new `model_id`.
- Should old PGs be eagerly purged? Or allowed to expire via TTL? TTL-based expiry is
  simpler and avoids a "purge storm" during model updates. But a manual
  `attentiondb_delete --model=old_version` command should be available for operators who
  want to reclaim storage immediately.
- **Requires**: Coordination between LMCache adapter and inference engine model loading lifecycle.

**Q5. Integration with inference scheduler for cache-aware routing**
- Maximum benefit requires the inference scheduler to route requests to GPU nodes that already
  have relevant KV cache data (Level 3 session affinity).
- This is primarily **LMCache's concern** — LMCache can query AttentionDB's `contains()` API
  to inform scheduler routing decisions. AttentionDB exposes "which keys are stored locally"
  but does not integrate with the scheduler directly.
- **Requires**: Collaboration between LMCache and vLLM/SGLang scheduler teams.

**Q6. Realistic SLO validation under contention**
- The component latency budgets (Section 1) are engineering estimates based on hardware
  specs and comparable system benchmarks. They have NOT been validated with a prototype
  under realistic production load (32+ concurrent GPU clients, mixed read/write, with
  background GC and replication running).
- **Required before**: Public latency claims or customer commitments.
- **Plan**: Build a load generator that simulates realistic inference traffic patterns
  (bursty, prefix-heavy, mixed model sizes) and measure end-to-end latency distributions
  on target hardware.
