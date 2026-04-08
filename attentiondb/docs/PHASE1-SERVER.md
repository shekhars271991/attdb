# AttentionDB Phase 1 — Remote Cluster (Server)

> This document is a focused extraction of the full [DESIGN.md](DESIGN.md) for Phase 1:
> the AttentionDB server daemon and multi-node cluster. Phase 1 builds on top of the
> Phase 0 local engine ([PHASE0-MVP.md](PHASE0-MVP.md)) by adding remote tiers (T3/T4),
> clustering, replication, security, and full observability.
>
> For the complete architecture reference, see DESIGN.md.

---

## 1. Scope and Goals

### What Phase 1 Adds

Phase 1 introduces the AttentionDB server daemon — a standalone process running on
dedicated (non-GPU) nodes that provides shared, durable, and replicated KV cache storage
to GPU clients. GPU nodes continue running the Phase 0 local engine (T1/T2) and gain
the ability to fetch from and push to the remote cluster (T3/T4).

### Prerequisites

Phase 0 local engine must be functional:
- Lock-free hash map index, slab allocator, log-structured blob store
- CW-SLRU eviction, cost-aware admission
- Client library C API + Python bindings

### Deployment Model

```
GPU Node (× N):
  ┌──────────────────────────────────────┐
  │ Inference Engine (vLLM / SGLang)     │
  │         │                            │
  │         ▼                            │
  │ LMCache (orchestration)              │
  │   prefix matching, chunking, serde   │
  │         │                            │
  │         ▼  StorageBackendInterface    │
  │ AttentionDB Storage Engine           │
  │ ┌──────────────────────────────────┐ │
  │ │ T1: CPU DRAM  (16-32 GB)        │ │  Phase 0 local engine
  │ │ T2: Local NVMe (1-2 TB)         │ │
  │ └──────────────┬───────────────────┘ │
  └────────────────┼─────────────────────┘
                   │ RDMA / TCP (shared and cold data only)
                   │ (~10-20% of reads under steady state)
  ┌────────────────┼─────────────────────┐
  │ AttentionDB Server Cluster (× M)     │
  │ ┌──────────────▼───────────────────┐ │
  │ │ T3: DRAM (256-512 GB per node)   │ │  Shared prefix KV (system prompts,
  │ │     Index + hot shared cache     │ │  popular RAG contexts)
  │ ├──────────────────────────────────┤ │
  │ │ T4: NVMe (12-48 TB per node)     │ │  Full corpus, historical
  │ │     Log-structured blob store    │ │  conversations, cold data
  │ └──────────────────────────────────┘ │
  │                                      │
  │ Metadata plane (gossip, all nodes)   │
  └──────────────────────────────────────┘
```

LMCache continues to own token-to-key mapping, prefix matching, chunking, vLLM
integration, and compression/decompression. AttentionDB provides the storage engine
on GPU nodes (T1/T2) and the remote cluster (T3/T4). See DESIGN.md Section 1
"Relationship to LMCache" for the full boundary definition.

### Latency Targets (Remote Tiers)

| Path | p50 target | p99 target |
|------|-----------|-----------|
| T3 remote DRAM via RDMA | < 500 μs | < 2 ms |
| T3 remote DRAM via TCP | < 2 ms | < 5 ms |
| T4 remote NVMe via RDMA | < 2 ms | < 5 ms |

Component budgets:

```
T3 remote DRAM via RDMA:
  Local index lookup:        ~0.1 μs
  RDMA request to server:    ~2–5 μs
  Server index lookup:       ~0.1 μs
  RDMA data transfer:        ~20–200 μs (100 Gbps InfiniBand, 256KB–4MB)
  NIC queueing/contention:   ~10–100 μs
  PCIe to GPU HBM:           ~10–30 μs
  GPU decompress:            ~50–100 μs
  ─────────────────────────────────────
  Total:                     ~90–435 μs

T3 remote DRAM via TCP:
  Same as RDMA but:
  TCP socket send/recv:      ~100–500 μs (kernel copies, syscalls)
  memcpy from kernel buffer: ~5–20 μs
  ─────────────────────────────────────
  Total:                     ~200–800 μs
```

### Phase 1 Non-Goals

- Topology-aware placement (rack/zone awareness) — Phase 3
- Per-tenant encryption at rest with KMS — Phase 3
- Kubernetes operator — Phase 3
- Scheduler integration API — Phase 3

---

## 2. Transport Layer

### 2.1 Dual Transport Strategy

AttentionDB supports two first-class transport paths. TCP is not a degraded fallback —
it must be tuned and tested as a primary path for environments without RDMA hardware.

| Transport | Hardware | Use case |
|-----------|----------|----------|
| RDMA (InfiniBand/RoCE) | ConnectX-6/7, RDMA fabric | Production clusters with InfiniBand |
| TCP | Standard Ethernet | Cloud GPU instances, on-prem without IB |

### 2.2 RDMA Transport

**Mooncake Transfer Engine evaluation**: Mooncake's Transfer Engine (Apache 2.0) provides
RDMA verbs, NVLink, TCP fallback, and buffer registration out of the box. Integrating it
saves ~3–6 months of RDMA development.

**Decision criteria** (prototype both before committing):
- Transfer Engine: benchmark RDMA write throughput and latency on a 2-node cluster
  with ConnectX-7 NICs. Evaluate dependency footprint, build system coupling.
- NIXL/UCX: benchmark the same workload. Evaluate development effort for buffer
  registration, error handling, connection management.

**RDMA data flow** (server responding to a client read):
```
Client                                    Server
  │                                          │
  │── ReadRequest (RDMA send, ~64 bytes) ──→ │
  │                                          │  Index lookup → find blob
  │← RDMA write directly into client's    ←──│  RDMA write blob data into
  │   pre-registered buffer                  │  client's registered buffer
  │                                          │
  │── ReadAck (confirms receipt) ──────────→ │
```

### 2.3 TCP Transport

For environments without RDMA. Uses `io_uring`-based async TCP with:
- Batched transfers (multiple blobs per TCP round-trip)
- Optional Jumbo frames (9000 MTU) for larger payloads
- `TCP_NODELAY` + `TCP_QUICKACK` to minimize latency
- Connection pooling: pre-established connections per (client, server) pair

---

## 3. Clustering

### 3.1 Cluster Topology

Two planes:

**Metadata plane** (all nodes — GPU clients + AttentionDB servers):
- Membership: which nodes exist, their roles, their health
- Placement map: which server holds which data
- Popularity index: access frequency statistics
- Protocol: gossip (SWIM variant), eventually consistent
- Data size: < 100 MB total, replicated to every node

**Data plane** (servers + GPU nodes for T3/T4 access):
- KV cache blob storage and retrieval
- Protocol: RDMA (primary) or TCP (fallback)
- Consistency: single-writer per PG, enforced by epoch-based leases

### 3.2 Placement Groups

A placement group (PG) is the fundamental unit of data ownership, replication, and
isolation. Identified by a `(model_id, tenant_id)` pair.

```
PlacementGroup {
    pg_id:              u64,          // hash(model_id, tenant_id)
    model_id:           u64,
    tenant_id:          u64,
    nodes:              Vec<NodeId>,  // Ranked list [primary, replica-1, ...]
    replication_factor: u8,           // Typically 2-3
    entry_count:        u64,
    dram_bytes_used:    u64,
    nvme_bytes_used:    u64,
    eviction_profile:   EvictionProfile,
}
```

**Why semantic keys**: `(model_id, tenant_id)` is the natural co-location boundary.
All conversations for a model+tenant share system prompts and RAG contexts. Tenant
isolation comes for free. Per-model eviction tuning is natural.

#### 3.2.1 PG-to-Node Mapping via Rendezvous Hashing

Rendezvous hashing (HRW) maps each PG to its node set. No ring structure, no
centralized map.

```
For each node in cluster:
  score[node] = hash(pg_id, node.id)

Sort nodes by score, descending → full ranking.
PG node set = ranking[0..repl_factor]
```

Every node independently computes the same ranking with no shared state.

**Comparison with alternatives**:

| Property | Token ring (Cassandra) | CRUSH (Ceph) | Rendezvous hash (AttentionDB) |
|----------|----------------------|--------------|-------------------------------|
| Full node ranking | No | Yes | Yes |
| Requires ring/map | Token ring + vnodes | CRUSH rules | Nothing — pure function |
| Node add/remove | Splits ranges | ~1/N keys move | ~1/N keys shift rank |
| Zero-coordination failover | No | Yes | Yes |
| Semantic key support | No | No | Yes |

**Churn on node add/remove** — adding 1 node to an N-node cluster:

| Cluster size (N) | P(primary changes) = 1/(N+1) | P(any top-K changes) = K/(N+1) |
|---|---|---|
| 5 nodes | 16.7% | 50.0% (K=3) |
| 10 nodes | 9.1% | 27.3% |
| 20 nodes | 4.8% | 14.3% |
| 50 nodes | 2.0% | 5.9% |
| 100 nodes | 1.0% | 3.0% |

At 10 nodes, ~9% of PGs get a new primary. Mitigated by shadow join protocol
(Section 4.2) and epoch-based leases (Section 3.2.5).

#### 3.2.2 Three-Level Sharding

```
Level 1 — Placement Group (model + tenant → node set):
  Determines which servers hold data for this model+tenant.

Level 2 — Prefix group (system_prompt_hash → primary within PG):
  Conversations sharing a system prompt cluster on the same primary.
  Maximizes prefix KV cache reuse within the PG.

Level 3 — Session affinity (sequence_id → GPU node):
  Multi-turn conversations stay on the same GPU for T1/T2 hits.
  Preference, not hard constraint (overridden by load balancing).
```

**Routing example**:
```
Request: model=llama-70b, tenant=acme, system_prompt="You are a helpful..."

1. PG lookup:   hash("llama-70b", "acme") → PG-A, nodes = {node-2, node-1, node-5}
2. Prefix group: rendezvous_hash(prompt_hash, PG-A.nodes) → node-2
3. Client contacts node-2 for prefix KV cache lookup.
```

#### 3.2.3 PG Lifecycle

```
EMPTY     → First write. PG created, index partition allocated, node set computed.
ACTIVE    → Normal operation. Reads/writes on primary, replication to warm replicas.
MIGRATING → Node add/remove changed node set. New nodes warming up.
DRAINING  → Model unload or tenant decommission. No new writes.
TOMBSTONE → PG empty. Metadata retained 1 hour, then garbage collected.
```

#### 3.2.4 Per-PG Resource Isolation

Each PG maintains independent budgets and eviction:

```
Server node-2 hosts 3 PGs:

┌─────────────────────────────────────────────┐
│ PG-A: (llama-70b, acme) — PRIMARY           │
│   DRAM: 60 GB  │  NVMe: 2 TB               │
│   Entries: 5M   │  CW-SLRU threshold: low   │
├─────────────────────────────────────────────┤
│ PG-B: (llama-70b, globex) — WARM REPLICA    │
│   DRAM: 30 GB  │  NVMe: 1 TB               │
│   Entries: 2M   │  CW-SLRU threshold: low   │
├─────────────────────────────────────────────┤
│ PG-C: (mistral-7b, acme) — PRIMARY          │
│   DRAM: 20 GB  │  NVMe: 500 GB             │
│   Entries: 8M   │  CW-SLRU threshold: high  │
└─────────────────────────────────────────────┘
```

Noisy tenant flooding PG-A triggers eviction within PG-A only. PG-B and PG-C unaffected.

#### 3.2.5 PG Scaling Modes

Per-PG hash maps have ~64 KB minimum overhead. At 50,000 PGs, this wastes ~3.2 GB.
Three scaling modes address this:

| Mode | Entry count | Implementation | Overhead |
|------|-------------|----------------|----------|
| **Inline** | < 10 | Small array in PG metadata struct. Linear scan. | ~640 bytes |
| **Compact** | 10–1,000 | Multiple compact PGs share a single hash map (pg_id prefix in key). | Amortized |
| **Full** | > 1,000 | Independent hash map, own slab budget, own segments. | ~64 KB+ |

**Promotion/demotion**: Auto-promote to next mode at threshold. Demotion at 50%
of promotion threshold (hysteresis).

**Compact mode trade-offs**:
1. Eviction isolation weakened — shared eviction pool across compact PGs.
2. Checkpointing coarsens — any dirty compact PG forces shared map checkpoint.
3. Migration requires scan+filter from shared map (promote to Full before migrating).

**Configuration**:
```yaml
pg_scaling:
  inline_max_entries: 10
  compact_max_entries: 1000
  demotion_hysteresis: 0.5
  compact_shared_map_size: "256MB"
  auto_promote_on_eviction: true
```

**Memory comparison** (50,000 PGs: 45K small, 4.5K medium, 500 large):

| Approach | Overhead |
|----------|----------|
| All Full mode | ~3.2 GB |
| Tiered (inline/compact/full) | ~100 MB |

#### 3.2.6 PG-Scoped Index Partitioning

The in-memory index is partitioned by PG (not a single global map). Benefits:
1. Independent resize — growing PG-A doesn't stall PG-B lookups.
2. Data lifecycle aligned — PG migration transfers one hash map as a unit.
3. No false sharing — separate backing arrays in separate memory regions.
4. Scoped checkpointing — only dirty PGs are checkpointed.
5. Direct eviction iteration — no secondary bookkeeping for PG membership.

Lookup path:
```
1. pg_id = hash(cache_key.model_id, cache_key.tenant_id)  // already known
2. pg_index = node.index_partitions[pg_id]                 // O(1)
3. location = pg_index.get(cache_key)                      // O(1)
```

#### 3.2.7 Primary Ownership Fencing (Epoch Leases)

Gossip is eventually consistent. During partitions, two nodes could both compute
themselves as primary. Epoch-based leases prevent this.

```
PrimaryLease {
    pg_id:        u64,
    epoch:        u64,       // Monotonically increasing
    holder:       NodeId,
    granted_at:   Timestamp,
    expires_at:   Timestamp, // Default: 30 seconds
    ack_nodes:    Vec<NodeId>,
}
```

**Protocol**:

1. **CLAIM**: Candidate primary (rank-1) sends `LeaseRequest{pg_id, epoch+1}` to
   PG members. Must get ACK from ≥1 other member (quorum of 2/3).

2. **GRANT**: Acknowledging node checks proposed_epoch > locally known epoch.
   Responds with `LeaseAck`.

3. **RENEW** (every 10s): Primary sends `LeaseRenew` to PG members. Must get ≥1 ACK
   before lease expires.

4. **FENCED**: If renewal fails, primary stops writes. Continues serving stale reads.
   Metric: `attentiondb_pg_fenced_total`.

5. **TAKEOVER**: Rank-2 notices expired lease, claims with epoch+1, gets quorum ACK,
   becomes new primary.

**Partition scenarios**:

| Scenario | Behavior | Result |
|----------|----------|--------|
| Primary in majority | Renews successfully | Correct single-writer |
| Primary isolated | Lease expires → FENCED. Rank-2 takes over. | Brief write gap, then correct |
| All nodes isolated | All FENCED. No writes. Reads from local cache. | Availability sacrificed for correctness |

**Gossip witnesses**: Lease renewal also accepts ACKs from non-PG-member nodes that
recently received gossip heartbeats from the primary. This broadens the quorum base
from 2 specific nodes to the entire cluster, preventing false fencing when PG peers
are slow but the primary is actually alive.

**Configuration**:
```yaml
ownership:
  lease_ttl_s: 30
  lease_renew_interval_s: 10
  quorum_size: 2
  fenced_read_mode: "local_stale"   # "local_stale" | "reject" | "redirect"
```

### 3.3 Replication

Async push replication from primary to warm replicas:

```
Primary                              Warm Replica
   │── ReplicateBatch ──────────→ │  (keys[], blobs[], priorities[])
   │← ReplicateAck ──────────── │  (accepted_keys[])
```

**Replicated** (priority order):
1. System prompt KV (high priority, shared by many conversations)
2. Long-context prefixes where `token_count × model_cost > threshold`
3. Frequently accessed conversation KV

**NOT replicated**:
- Short, cheap entries (< 64 tokens on small models)
- Entries with TTL < replication_lag
- Per-session decode-phase KV (local T2 only)

**Consistency**: Eventual. Write visible on primary immediately, on replicas after
~1 second. Replica miss falls through to primary (correct, slightly slower).

### 3.4 Metadata Plane (Gossip)

SWIM-family gossip protocol for membership and health.

**Per-node gossip state**:
```
NodeState {
    node_id:         u64,
    node_type:       enum { GpuClient, AttentionDbServer },
    address:         SocketAddr,
    rdma_address:    RdmaAddr,
    status:          enum { Alive, Suspect, Dead, Draining },
    capacity:        { dram_total, dram_used, nvme_total, nvme_used },
    models_loaded:   Vec<u64>,
    incarnation:     u64,
    last_heartbeat:  Timestamp,
}
```

**Failure detection**:
- Ping 3 random peers every 1 second
- Failed ping → indirect probe via 3 other nodes
- Indirect fail → SUSPECT
- SUSPECT for 5 seconds → DEAD
- O(log N) convergence via piggybacked protocol

### 3.5 Popularity Sketch

Per-server count-min sketch for access frequency estimation. Used for warm-up manifests,
predictive prefetch, and popularity-aware admission.

```
PopularitySketch {
    epoch:          u64,          // Rolls every 30 seconds
    node_id:        NodeId,
    counters:       [[u32; W]; D], // e.g., 4 × 8192
    total_accesses: u64,
}
```

**Merge protocol**: Additive with deduplication (not element-wise max):
```
Same (node_id, epoch) seen again → IGNORE
New (node_id, epoch)             → ADD counters element-wise
Older epoch (< current - 2)     → DISCARD
```

Additive merge gives correct global frequency. Deduplication by `(node_id, epoch)`
prevents double-counting from multi-path gossip. Sketches decay naturally with epoch
rollover (~1–2 minute window).

---

## 4. Node Lifecycle

### 4.1 GPU Node Join (Client)

Lightweight — GPU nodes are clients to the cluster:

```
1. CONNECT (< 1s):
   Load metadata via gossip seeds. Open RDMA/TCP to relevant servers.

2. PREFETCH (1-10s, async):
   Request PrefetchManifest: top-K system prompt KVs ranked by popularity.
   Fetch via RDMA into local T1/T2. Inference starts immediately (misses fall through).

3. ACTIVE:
   Full operation. Local cache warms from traffic + prefetch.
```

### 4.2 AttentionDB Server Join

```
1. REGISTERING (0-5s):
   Announce to metadata plane. Gossip propagates membership.

2. WARMING (5s-2min):
   Pull owned keys from existing servers via RDMA, prioritized by:
   a) System prompt KVs (highest priority)
   b) Popularity-ranked conversation KVs
   Transition to SHADOW at 70% coverage or 60s timeout.

3. SHADOW (1-5min):
   Receives 10% of read traffic. On miss, fetches from previous primary.
   Dual-writes (locally + forwards to previous primary).
   Promoted to ACTIVE at 80% local hit rate.

4. ACTIVE:
   Full weight. Normal read/write. Participates in replication.
```

### 4.3 Planned Scale-Down (Draining)

```
1. DRAINING:
   Stop new writes. Push high-priority entries to next-ranked server.
   Continue reads until handoff completes or 60s timeout.

2. DEREGISTER:
   Leave gossip ring. Next-ranked server becomes primary.
```

### 4.4 Unplanned Failure

```
1. DETECTION (1-6s):
   Gossip: SUSPECT → DEAD.

2. IMMEDIATE FAILOVER:
   Client in-flight reads timeout (~100 ms).
   Retry to rank-2 (warm replica) — transparent to inference engine.

3. RECOVERY (background):
   Unreplicated entries are lost — recomputed on next access.
   If node returns: re-enters REGISTERING → WARMING → SHADOW → ACTIVE.
   Reads local index checkpoint from NVMe, skips warming for existing entries.
```

---

## 5. Admission and Eviction (Full Version)

Phase 1 adds **popularity sketch integration** to the Phase 0 admission control.

### 5.1 Admission Score with Popularity

```
recompute_cost(entry)  = num_tokens × cost_per_token(model_id)

reuse_estimate(entry)  = popularity_sketch.estimate(entry.prefix_hash)
                         / sketch.total_accesses

admission_score(entry) = recompute_cost × (1 + reuse_weight × reuse_estimate)
```

- `reuse_weight` (default: 5.0) controls how much reuse probability influences admission.
- A prefix shared by many conversations has high reuse_estimate. A unique long
  conversation has low reuse_estimate even if expensive.
- Without popularity, admission caches every expensive entry equally — including
  one-shot long conversations that evict shared prefixes. Popularity answers:
  "is this entry both expensive AND likely to be reused?"

### 5.2 Server-Side Admission

On the server, admission filtering uses `admission_score` (cost × reuse) as opposed
to the client-side backpressure which filters by `recompute_cost` alone. Together they
provide two layers of quality filtering.

Dynamic threshold on the server:
```
Storage < 70%:   threshold = 0
Storage 70-90%:  threshold scales linearly
Storage > 90%:   threshold = 2× base (only high-value, high-reuse entries)
```

### 5.3 Eviction

CW-SLRU per tier (T3 DRAM, T4 NVMe), per PG. Same as Phase 0 but with per-PG
isolation and independent tuning.

**Cross-tier demotion** (server-side):

| From | To | When |
|------|----|------|
| T3 (DRAM) | T4 (NVMe) | T3 full, entry warm but not hot |
| T3 (DRAM) | Discard | Entry cold, below cost threshold |
| T4 (NVMe) | Discard | Entry cold, below cost threshold |

---

## 6. Storage Engine Remote Paths

The Phase 0 storage engine gains internal remote fetch capabilities. These are transparent
to LMCache — from LMCache's perspective, `get(key)` returns a blob or a miss. The internal
T1 → T2 → T3 → T4 cascade happens inside AttentionDB.

### 6.1 Extended Get Path

When `attentiondb_get()` misses T1 and T2 locally:
1. Engine looks up PG membership and server ranking from cluster routing cache.
2. Sends `ReadRequest` to the primary server via RDMA or TCP.
3. If primary is unreachable, retries to rank-2 (warm replica).
4. Received blob is promoted to T1 (if DRAM budget allows) or T2.
5. Blob is returned to LMCache's `get_blocking()` call.

### 6.2 Remote Put Path

When `attentiondb_put()` receives a blob with `entry_type=system_prompt` (or similar
high-value type), the engine asynchronously replicates to T3 in addition to local T2:
- Subject to write buffer backpressure (same levels as Phase 0)
- Uses RDMA for low-latency push or TCP as fallback
- LMCache is unaware of this — `put()` returns immediately regardless

### 6.3 Cluster Routing Cache

The engine maintains a local copy of the metadata plane:
- Refreshed via gossip (piggyback on heartbeat messages)
- Used for routing: which server to contact for a given key
- Includes PG-to-node mapping, node health, epoch numbers
- Size: < 100 MB

### 6.4 Connection Pool

Pre-established connections to AttentionDB servers:
- RDMA: Queue Pairs created at startup, one per server
- TCP: Connection pool with configurable max connections per server
- Health-checked: connections probed periodically, dead connections replaced

---

## 7. Server Storage (T3 + T4)

The server reuses the Phase 0 storage engine components with server-specific configuration.

### 7.1 T3 DRAM Cache (Server)

Same slab allocator as Phase 0 T1, but:
- Uses **RDMA-registered memory** instead of CUDA pinned memory
- Larger capacity: 128–384 GB per server (vs. 16–32 GB on GPU nodes)
- Budget divided across PGs proportionally

```yaml
dram_cache:
  enabled: true
  size: "384GB"
  memory_type: "rdma"                    # RDMA-registered for zero-copy sends
  per_pg_budget: "proportional"
```

### 7.2 T4 NVMe Blob Store (Server)

Same log-structured blob store as Phase 0 T2, but:
- Multiple NVMe drives striped for aggregate bandwidth
- Segment files owned by PGs (`pg_id` in header)
- PG migration transfers/discards segments by PG ownership

```yaml
nvme_store:
  paths:
    - "/mnt/nvme0/attentiondb"
    - "/mnt/nvme1/attentiondb"
    - "/mnt/nvme2/attentiondb"
  max_size_per_drive: "7TB"
  segments:
    size: "1GB"
  io:
    engine: "io_uring"                   # No GDS on server (no GPU)
    io_uring_queue_depth: 128
```

---

## 8. Observability (Full Stack)

Phase 1 adds Prometheus, OpenTelemetry, and distributed tracing on top of Phase 0's
structured logging.

### 8.1 Design Principles

- **Single metric stack**: OpenTelemetry SDK everywhere. Export via OTLP or
  Prometheus-compatible scrape endpoint.
- **Storage-focused metrics**: AttentionDB reports on storage performance. Inference-level
  metrics (TTFT savings, prefill time avoided) belong in LMCache.
- **Per-tenant, per-model labels**: All metrics carry `tenant_id` and `model_id`.
- **Histograms over averages**: Tail latency matters for SLAs.

### 8.2 Metric Catalog

**Tier performance** (per tier, per node):

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_read_latency_seconds` | Histogram | By tier, p50/p95/p99 |
| `attentiondb_write_latency_seconds` | Histogram | By tier |
| `attentiondb_hit_rate` | Gauge | By tier (rolling window) |
| `attentiondb_hit_rate_by_type` | Gauge | By entry type (system_prompt, conversation, rag_context) |
| `attentiondb_bytes_read_total` | Counter | By tier |
| `attentiondb_bytes_written_total` | Counter | By tier |
| `attentiondb_eviction_total` | Counter | By tier, by reason |
| `attentiondb_storage_utilization` | Gauge | By tier |

**Admission and eviction**:

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_admission_rejected_total` | Counter | Entries rejected by admission control |
| `attentiondb_write_backpressure_rejections_total` | Counter | Write buffer rejections by level |
| `attentiondb_write_buffer_drops_total` | Counter | Entries dropped at full buffer |
| `attentiondb_wasted_cache_bytes` | Counter | Bytes evicted before any read |

**Note**: Inference-correlated metrics (TTFT, prefill time saved) belong in LMCache.

**Clustering**:

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_remote_fetch_total` | Counter | Cross-node fetches |
| `attentiondb_remote_fetch_latency_seconds` | Histogram | Fetch latency |
| `attentiondb_routing_efficiency` | Gauge | 1 - (remote / total). Target > 0.9 |
| `attentiondb_replication_lag_seconds` | Gauge | Primary → replica delay |
| `attentiondb_node_warmup_progress` | Gauge | 0–1, joining node warmth |
| `attentiondb_gossip_convergence_seconds` | Histogram | Membership propagation time |
| `attentiondb_pg_fenced_total` | Counter | PGs that entered FENCED state |
| `attentiondb_lease_witness_renewal_total` | Counter | Lease renewals via gossip witnesses |

**Storage engine**:

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_gc_reclaimed_bytes` | Counter | Freed by GC |
| `attentiondb_gc_bandwidth_fraction` | Gauge | NVMe bandwidth used by GC |
| `attentiondb_index_entries` | Gauge | Index entry count |
| `attentiondb_segment_count` | Gauge | Active segment files |
| `attentiondb_segment_fragmentation` | Gauge | Average dead-entry ratio |
| `attentiondb_ssd_read_latency_seconds` | Histogram | Raw NVMe latency |

**Health**:

| Metric | Type | Description |
|--------|------|-------------|
| `attentiondb_node_status` | Gauge | 1=healthy, 0=degraded, -1=dead |
| `attentiondb_cluster_healthy_nodes` | Gauge | By node type |
| `attentiondb_miss_total` | Counter | Total misses across all tiers |

### 8.3 Health Checks

| Layer | Check | Frequency | Failure action |
|-------|-------|-----------|---------------|
| NVMe | 4 KB latency probe | 5s | Mark tier degraded, bypass SSD |
| DRAM | Free slab count | 1s | Increase eviction aggressiveness |
| RDMA | Ping to servers | 2s | Route to backup, alert |
| Index | Random entry checksum | 30s | Rebuild from checkpoint |
| End-to-end | Synthetic store+retrieve | 10s | Mark node degraded |

### 8.4 Distributed Tracing

OpenTelemetry spans for storage engine operations:
- `attentiondb.get` → `attentiondb.t1_check` → `attentiondb.t2_read` → `attentiondb.t3_fetch`
- `attentiondb.put` → `attentiondb.admission_check` → `attentiondb.write_buffer` → `attentiondb.replicate`

AttentionDB propagates trace context from LMCache so storage spans appear as children
of LMCache's orchestration spans. Compression/decompression spans belong to LMCache.

---

## 9. Graceful Degradation

| Severity | Condition | Behavior |
|----------|-----------|----------|
| Normal | All tiers healthy | Full caching |
| Tier degraded | SSD spike or NIC congestion | Bypass degraded tier |
| Node degraded | Server unreachable | Failover to warm replica. Local T1/T2 only. |
| PG degraded | Multiple servers fail | Route to surviving server. Accept misses. |
| Metadata stale | Gossip partition | Stale placement map. Suboptimal but functional. |
| Full bypass | Storage engine failure | LMCache skips AttentionDB, falls back to prefill. |

Timeout + circuit breaker per tier. Cooldown default: 30 seconds.

---

## 10. Security and Multi-Tenant Hardening

### 10.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Unauthorized cross-tenant read | PG-level authz + tenant token |
| Network eavesdropping | mTLS (TCP), RDMA options below |
| Disk theft | Encryption at rest |
| Rogue GPU node | Client auth + server-side PG access control |
| Metadata leakage | Tenant-scoped metrics, audit logging |
| Denial of service | Per-PG resource quotas |

### 10.2 Authentication

**Node-to-node**: mTLS for all TCP. Cluster CA issues certificates.

**RDMA authentication**: Two options:
1. Physical isolation (InfiniBand is typically a dedicated fabric)
2. Application-level HMAC-SHA256 token in every RDMA message header (~1 μs overhead)

**Client identity**: Tenant token on connection:
```
ClientAuthToken {
    tenant_id:      u64,
    issued_at:      Timestamp,
    expires_at:     Timestamp,
    allowed_models: Vec<u64>,
    signature:      [u8; 32],  // HMAC-SHA256
}
```
Issued by external identity provider (Kubernetes SA, Vault, etc.).

### 10.3 Authorization

PG-level access control on every request:
```
1. Extract tenant_id from authenticated session.
2. Compute pg_id from request's (model_id, tenant_id).
3. Verify: request.tenant_id == session.tenant_id
           AND request.model_id IN session.allowed_models
4. If not → ACCESS_DENIED.
```

### 10.4 Encryption

**In transit**:
- TCP: mTLS
- RDMA (in preference order):
  1. Physical network isolation (zero overhead, recommended)
  2. Application-level AES-256-GCM encryption (~5% overhead)
  3. IPsec over RoCEv2 with NIC offload (ConnectX-7, line-rate)

**At rest** (Phase 1 ships Option A; Option B deferred to Phase 3):
- Option A: dm-crypt / LUKS on NVMe drives (single key per drive)
- Option B: Per-tenant keys via external KMS (Vault, AWS KMS)

### 10.5 Audit

```yaml
audit:
  enabled: false
  log_events:
    - "pg_ownership_change"
    - "node_join_leave"
    - "cross_tenant_access_denied"
    - "admin_operations"
  destination: "stdout"                  # "stdout" | "file" | "otlp"
```

### 10.6 Security Phases

| Phase | Scope |
|-------|-------|
| Phase 1 | Wire protocol includes auth token field (reserved). mTLS optional. |
| Phase 2 | mTLS enforced. RDMA auth tokens. PG-level authz. Basic audit. |
| Phase 3 | Per-tenant encryption at rest. KMS. Full audit. RDMA encryption. |

---

## 11. Configuration

### 11.1 GPU Node (Client) — Hybrid Mode

```yaml
attentiondb:
  mode: "hybrid"

  cluster:
    seed_nodes: ["attentiondb-1:7300", "attentiondb-2:7300"]
    rdma_device: "mlx5_0"
    transport: "nixl"                    # "nixl" | "ucx" | "tcp"

  local:
    t1_dram_size: "16GB"
    t2_nvme_path: "/mnt/nvme0/attentiondb"
    t2_nvme_size: "1TB"
    t2_use_gds: true
    write_buffer_size: "256MB"

  # Compression config belongs in LMCache, not here.
  # AttentionDB stores opaque blobs.

  admission:
    min_recompute_cost: 64
    base_threshold: 100

  eviction:
    policy: "cw_slru"
    protected_ratio: 0.25

  timeouts:
    get_ms: 5
    remote_fetch_ms: 100
    circuit_breaker_cooldown_s: 30
```

### 11.2 AttentionDB Server

```yaml
attentiondb_server:
  node_id: "attentiondb-1"
  listen_address: "0.0.0.0:7300"
  rdma_device: "mlx5_0"

  storage:
    t3_dram_size: "384GB"
    t4_nvme_paths:
      - "/mnt/nvme0/attentiondb"
      - "/mnt/nvme1/attentiondb"
      - "/mnt/nvme2/attentiondb"
    t4_nvme_size_per_drive: "7TB"
    segment_size: "1GB"
    index_checkpoint_interval_s: 30

  cluster:
    seed_nodes: ["attentiondb-1:7300", "attentiondb-2:7300", "attentiondb-3:7300"]
    replication_factor: 2
    gossip_interval_ms: 1000
    suspect_timeout_ms: 5000
    shadow_mode_duration_s: 300
    shadow_traffic_fraction: 0.1
    warmup_min_coverage: 0.7
    warmup_max_duration_s: 60

  gc:
    trigger_fragmentation: 0.5
    max_bandwidth_fraction: 0.2

  eviction:
    policy: "cw_slru"
    protected_ratio: 0.25
    t3_watermark: 0.85
    t4_watermark: 0.90

  observability:
    metrics_port: 9090
    otlp_endpoint: ""
    enable_tracing: false
    log_level: "info"
```

### 11.3 Deployment Modes

| Mode | Description | Use case |
|------|-------------|----------|
| `hybrid` | Local T1/T2 + remote T3/T4 | Production |
| `local_only` | T1/T2 only, no cluster | Single-node (Phase 0) |
| `remote_only` | T1 DRAM only + remote cluster | GPU nodes without NVMe |
| `embedded` | Full server in-process on GPU node | Development |

---

## 12. Implementation Tasks

| # | Task | Dependency | Estimated effort |
|---|------|------------|-----------------|
| 1 | AttentionDB server daemon (process scaffold, config, lifecycle) | Phase 0 complete | 2 weeks |
| 2 | Transport layer — RDMA (Transfer Engine or NIXL/UCX prototype) | — | 4-6 weeks |
| 3 | Transport layer — TCP (io_uring-based async) | — | 2 weeks |
| 4 | Gossip metadata plane (SWIM) | #1 | 3 weeks |
| 5 | Rendezvous hashing + placement groups | #4 | 2 weeks |
| 6 | PG scaling modes (inline/compact/full) | #5 | 2 weeks |
| 7 | Epoch-based primary leases | #5 | 2 weeks |
| 8 | Server-side index (PG-partitioned hash map) | Phase 0 hash map, #5 | 2 weeks |
| 9 | Server-side T3 DRAM cache (RDMA-registered slab allocator) | Phase 0 slab, #2 | 1 week |
| 10 | Server-side T4 NVMe blob store (PG-scoped segments) | Phase 0 blob store, #5 | 1 week |
| 11 | Async replication (primary → warm) | #7, #8, #2 | 2 weeks |
| 12 | Node lifecycle (join/warming/shadow/active/draining) | #4, #5, #11 | 3 weeks |
| 13 | Popularity sketch + merge protocol | #4 | 1 week |
| 14 | Full admission control (cost × popularity) | Phase 0 admission, #13 | 1 week |
| 15 | Client library — remote fetch paths + metadata cache | Phase 0 client, #2/#3, #4 | 3 weeks |
| 16 | Full observability (OTel, Prometheus, tracing, health checks) | #1 | 3 weeks |
| 17 | Security — mTLS, RDMA auth tokens, PG-level authz | #2, #3, #5 | 3 weeks |
| 18 | Integration test: multi-node cluster + GPU clients | #1–#17 | 2 weeks |
| 19 | Benchmark: hybrid mode vs. local-only, RDMA vs. TCP | #18 | 2 weeks |

**Critical path**: Phase 0 → #1 → #4 → #5 → #7 → #8 → #11 → #12 → #18

**Parallelizable**:
- #2 and #3 (transport) can run in parallel with #4–#7 (clustering)
- #16 (observability) can run in parallel with #8–#12 (storage + lifecycle)
- #13 (popularity) can run in parallel with #8–#10

---

## 13. Known Risks (Phase 1 Scope)

**R1. Epoch lease overhead on write latency**
If lease renewal is delayed (GC pause, NIC congestion), the PG is temporarily fenced.
Medium severity. Monitor `attentiondb_pg_fenced_total`. Needs production validation
under network failure injection.

**R2. Gossip convergence during rapid scaling**
O(log N) rounds (~7s in 100-node cluster). Misrouted requests → cache miss, not corruption.
Low severity. Mitigated by epoch validation + accelerated gossip during scaling.

**R3. RDMA hardware availability**
Many cloud instances lack InfiniBand. TCP path must be first-class, not an afterthought.
High severity for adoption. TCP must be part of Phase 1 acceptance criteria.

**R4. PG cardinality explosion**
With many (model, tenant) pairs, PG count can reach tens of thousands.
Mitigated by PG scaling modes. Needs load testing with 10,000+ PGs.

**R5. Mooncake Transfer Engine coupling**
Integrating saves months but may introduce unwanted dependencies.
Decision needed before Phase 1 starts: prototype both Transfer Engine and NIXL/UCX.

**R6. Realistic SLO validation**
Latency budgets are engineering estimates, not validated under production load.
Load generator required before public latency claims.
