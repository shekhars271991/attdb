#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# KV CACHE OFFLOADING BENCHMARK — ATTENTIONDB
#
# OBJECTIVE:
#   Prove that in-process KV cache offloading with AttentionDB eliminates
#   the network hop required by Aerospike and delivers lower TTFT, higher
#   throughput, and 25% lower infrastructure cost on the SAME workload.
#
# THE CORE PROBLEM (identical to the Aerospike benchmark):
#   An L4 GPU (24 GB) can cache KV for ~8 unique 8K-token prefixes.
#   A real multi-tenant platform has 40+ unique prefixes.
#   Without external cache: evicted prefixes recompute from scratch (~3s).
#   With Aerospike: evicted prefixes fetched via network (~1000ms).
#   With AttentionDB: evicted prefixes fetched from LOCAL NVMe (~300ms)
#   or LOCAL DRAM (~100us) — no network hop, no second instance.
#
# WHY ATTENTIONDB OVER AEROSPIKE:
#
#   Aerospike:
#     - Separate i3en.xlarge ($0.452/hr), network round-trip per fetch
#     - 2 instances to manage, VPC/AZ co-location required
#     - All data on remote NVMe via TCP (~5 Gbps, ~1000ms per 1.14 GB)
#
#   AttentionDB:
#     - In-process on the GPU node, zero network latency
#     - T1 DRAM cache (16 GB cudaMallocHost): ~14/40 prefixes in RAM
#     - T2 local NVMe (io_uring): remaining prefixes at ~300us
#     - Single instance, 25% cheaper ($1.32 vs $1.77/hr)
#
# CACHE HIERARCHY:
#   L0 — GPU KV cache:        ~10 GiB  → ~8 prefixes   (0ms)
#   L1 — AttentionDB T1 DRAM: 16 GiB   → ~14 prefixes  (~100us)
#   L2 — AttentionDB T2 NVMe: 500+ GiB → ALL 40 prfxs  (~300us)
#   L3 — GPU recompute:       unlimited → 3000ms (what we avoid)
#
#   vs Aerospike hierarchy:
#   L0 — GPU KV cache:   ~10 GiB  → ~8 prefixes   (0ms)
#   L1 — CPU RAM (L1):   5 GiB    → ~4 prefixes   (~100ms)
#   L2 — Aerospike NVMe: 50+ GiB  → ALL 40 prfxs  (~1000ms via TCP)
#   L3 — GPU recompute:  unlimited → 3000ms
#
# ARCHITECTURE:
#
#   ┌───────────────────────────────────────────────────────┐
#   │  GPU Node — g6.4xlarge (single instance)              │
#   │  L4 24GB, 16 vCPUs, 64 GB RAM, local NVMe ~600 GB    │
#   │                                                       │
#   │  vLLM + LMCache                                       │
#   │    │                                                  │
#   │    └─ AttentionDB (in-process, pybind11)              │
#   │        ├─ T1: 16 GB pinned DRAM → ~14 prefixes       │
#   │        └─ T2: local NVMe (io_uring) → ALL 40 prefixes│
#   │                                                       │
#   │  GPU KV cache: ~8 prefixes                            │
#   │  T1 DRAM:      ~14 prefixes (most frequent)           │
#   │  T2 NVMe:      ALL 40 prefixes (persistent)           │
#   └───────────────────────────────────────────────────────┘
#
# WHAT WE PROVE (single GPU node, restart between phases):
#   Phase 1 — Baseline: 40 prefixes with APC only.
#             GPU cache overflows, constant re-computation, high TTFT.
#   Phase 2 — Warm AttentionDB: stores all 40 prefixes (~45.6 GB).
#             T1 holds 14 hot prefixes, T2 holds all 40.
#   Phase 3 — Cold restart: T1 DRAM lost, but T2 NVMe + checkpoint
#             persist. Checkpoint reloads index. Every prefix served
#             from local NVMe (~300us) vs recomputed (~3s) or fetched
#             over network (~1000ms with Aerospike).
#
# PREREQUISITES:
#   - 1 EC2 instance: g6.4xlarge (or similar with L4 + local NVMe)
#   - Docker + NVIDIA container toolkit
#   - HuggingFace token with access to Ministral-8B-Instruct-2410
#
# COST ESTIMATE:
#   1× g6.4xlarge: $1.32/hr (~$2.64 for a 2-hour session)
#   vs Aerospike:  $1.77/hr (g6.4xlarge + i3en.xlarge)
#   Savings:       25%
#
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# VARIABLES — paste in EVERY SSH session
# ──────────────────────────────────────────────────────────────────────


export MODEL="mistralai/Ministral-8B-Instruct-2410"
export VLLM_PORT=8010
export HF_CACHE="/opt/dlami/nvme/huggingface"
export GPU_MEM_UTIL=0.92
export MAX_MODEL_LEN=16384
export VLLM_IMAGE="lmcache/vllm-openai:v0.4.2"
export ADB_VLLM_IMAGE="vllm-attentiondb:latest"

# Benchmark workload — 40 prefixes to overflow GPU cache
export PREFIX_LEN=8192
export SUFFIX_LEN=128
export OUTPUT_LEN=256
export NUM_PREFIXES=40
export NUM_PROMPTS_COLD=80       # 2 per prefix
export NUM_PROMPTS_WARM=120      # 3 per prefix — steady state
export REQUEST_RATE=1.0

# Paths on the host (g6.4xlarge NVMe mount)
export ADB_DATA_DIR="/opt/dlami/nvme/attentiondb"
export ADB_CONFIG_DIR="/tmp/attentiondb_config"
export LMCACHE_CONFIG_DIR="/tmp/lmcache_attentiondb_config"

# Per-prefix KV size: 8192 tokens × 144 KB/token = 1.14 GB
# (Ministral-8B: 36 layers × 8 KV heads × 128 dim × 2 bytes × 2 [K+V])
# Total for 40 prefixes: 40 × 1.14 GB = 45.6 GB
# GPU KV cache budget: ~10 GiB → fits ~8 prefixes
# T1 DRAM: 16 GiB → fits ~14 prefixes
# T2 NVMe: unlimited → ALL 40 prefixes


# ══════════════════════════════════════════════════════════════════════
# STEP 0: BUILD CUSTOM vLLM + ATTENTIONDB IMAGE
# ══════════════════════════════════════════════════════════════════════

# 0a. Pull the base image
docker pull ${VLLM_IMAGE}

# 0b. Build the AttentionDB-enabled vLLM image
# Context is the attentiondb-server directory (parent of bench/)
cd "${SCRIPT_DIR}/.."
docker build -f Dockerfile.benchmark -t ${ADB_VLLM_IMAGE} .

# 0c. Verify the build
docker run --rm ${ADB_VLLM_IMAGE} python -c \
  "import importlib; mod = importlib.import_module('_attentiondb'); print('AttentionDB plugin OK:', [x for x in dir(mod) if not x.startswith('_')])"


# ══════════════════════════════════════════════════════════════════════
# STEP 1: PREPARE CONFIGS AND DIRECTORIES
# ══════════════════════════════════════════════════════════════════════
#
# SSH into the GPU instance:
#   ssh -i "your-key.pem" ubuntu@<gpu-node-public-ip>

# 1a. One-time NVMe + Docker setup (skip if already done)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sleep 2

sudo mkdir -p /opt/dlami/nvme/docker
sudo python3 -c "
import json, os
path = '/etc/docker/daemon.json'
d = json.load(open(path)) if os.path.exists(path) else {}
d['data-root'] = '/opt/dlami/nvme/docker'
with open(path, 'w') as f: json.dump(d, f, indent=2)
"
sudo systemctl restart docker
sleep 3

mkdir -p ${HF_CACHE}

# 1b. Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Expected: NVIDIA L4, 23034 MiB

# 1c. Create AttentionDB data directory on NVMe
sudo mkdir -p ${ADB_DATA_DIR}
sudo chown $(id -u):$(id -g) ${ADB_DATA_DIR}

# 1d. Write AttentionDB engine config
mkdir -p ${ADB_CONFIG_DIR}
cp "${SCRIPT_DIR}/config/attentiondb.yaml" ${ADB_CONFIG_DIR}/attentiondb.yaml
echo "AttentionDB config at ${ADB_CONFIG_DIR}/attentiondb.yaml"

# 1e. Write LMCache config pointing to AttentionDB
mkdir -p ${LMCACHE_CONFIG_DIR}
cp "${SCRIPT_DIR}/config/lmcache.yaml" ${LMCACHE_CONFIG_DIR}/config.yaml
echo "LMCache config at ${LMCACHE_CONFIG_DIR}/config.yaml"

cat ${LMCACHE_CONFIG_DIR}/config.yaml


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: BASELINE — 40 prefixes, NO external cache (APC only)
# ══════════════════════════════════════════════════════════════════════
#
# Purpose: Show what happens when the prefix working set exceeds GPU
# KV cache. With 40 prefixes and ~8 GPU cache slots, roughly 80% of
# distinct-prefix requests hit evicted entries and pay full recompute.
#
# This phase is IDENTICAL to the Aerospike baseline — same image, same
# workload. The only change is in Phase 2/3 where we use AttentionDB.
# ──────────────────────────────────────────────────────────────────────

# --- 1a. Start clean vLLM (no LMCache, no AttentionDB) ---
docker rm -f ministral-brain 2>/dev/null

docker run -d --name ministral-brain \
  --runtime=nvidia --gpus all --ipc=host \
  --network host \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -e HF_TOKEN=${HF_TOKEN} \
  -e TMPDIR=/tmp \
  ${VLLM_IMAGE} \
  --model ${MODEL} \
  --port ${VLLM_PORT} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --dtype bfloat16 \
  --enforce-eager \
  --trust-remote-code \
  --disable-hybrid-kv-cache-manager

docker logs -f ministral-brain
# Wait for "Application startup complete", then Ctrl+C

curl -sf http://localhost:${VLLM_PORT}/health && echo " READY" || echo " NOT READY"

# --- 1b. Baseline COLD: 40 prefixes × 8K prefix, 80 prompts ---
echo "=== PHASE 1b: BASELINE COLD — 40 prefixes, APC only ==="
docker exec ministral-brain vllm bench serve \
  --model ${MODEL} \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --port ${VLLM_PORT} \
  --temperature 0 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len ${PREFIX_LEN} \
  --prefix-repetition-suffix-len ${SUFFIX_LEN} \
  --prefix-repetition-num-prefixes ${NUM_PREFIXES} \
  --prefix-repetition-output-len ${OUTPUT_LEN} \
  --num-prompts ${NUM_PROMPTS_COLD} \
  --request-rate ${REQUEST_RATE}

# RECORD: TTFT Mean, P99
# EXPECTED: High TTFT — many prefixes computed from scratch, some evicted
#   and recomputed. Comparable to Aerospike Phase 1b (~116,866 ms mean).

# --- 1c. Baseline WARM: APC has some prefixes, but NOT all 40 ---
echo "=== PHASE 1c: BASELINE WARM — 40 prefixes, APC only ==="
docker exec ministral-brain vllm bench serve \
  --model ${MODEL} \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --port ${VLLM_PORT} \
  --temperature 0 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len ${PREFIX_LEN} \
  --prefix-repetition-suffix-len ${SUFFIX_LEN} \
  --prefix-repetition-num-prefixes ${NUM_PREFIXES} \
  --prefix-repetition-output-len ${OUTPUT_LEN} \
  --num-prompts ${NUM_PROMPTS_WARM} \
  --request-rate ${REQUEST_RATE}

# RECORD: TTFT Mean, P99
# EXPECTED: Still high for evicted prefixes. APC only holds ~8 of 40.

docker rm -f ministral-brain 2>/dev/null


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: WARM ATTENTIONDB — Compute + store all 40 prefixes
# ══════════════════════════════════════════════════════════════════════
#
# Purpose: Process all 40 prefixes and store their KV cache through
# LMCache into AttentionDB (~45.6 GB total). After this phase:
#   - T1 DRAM holds ~14 most-recently-accessed prefixes (16 GB)
#   - T2 NVMe holds ALL 40 prefixes (~45.6 GB)
#   - Checkpoint on NVMe records the full index
#
# Unlike Aerospike, there is NO network hop. AttentionDB runs
# in-process via pybind11 — put() and get() are function calls.
# ──────────────────────────────────────────────────────────────────────

# --- Clean any previous AttentionDB data ---
rm -rf ${ADB_DATA_DIR}/*
echo "Cleaned AttentionDB data directory: ${ADB_DATA_DIR}"

# --- 2a. Start vLLM + LMCache + AttentionDB ---
docker rm -f ministral-brain 2>/dev/null

docker run -d --name ministral-brain \
  --runtime=nvidia --gpus all --ipc host \
  --network host \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -v ${ADB_CONFIG_DIR}/attentiondb.yaml:/etc/attentiondb/attentiondb.yaml:ro \
  -v ${LMCACHE_CONFIG_DIR}:/etc/lmcache:ro \
  -v ${ADB_DATA_DIR}:/opt/dlami/nvme/attentiondb \
  -e HF_TOKEN=${HF_TOKEN} \
  -e LMCACHE_CONFIG_FILE=/etc/lmcache/config.yaml \
  -e LMCACHE_USE_EXPERIMENTAL=True \
  -e PYTHONHASHSEED=0 \
  -e TMPDIR=/tmp \
  ${ADB_VLLM_IMAGE} \
  ${MODEL} \
  --port ${VLLM_PORT} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --dtype bfloat16 \
  --enforce-eager \
  --trust-remote-code \
  --disable-hybrid-kv-cache-manager \
  --disable-log-stats \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

docker logs -f ministral-brain
# Wait for "Application startup complete", then Ctrl+C
# Look for: "AttentionDB adapter initialised" in logs

curl -sf http://localhost:${VLLM_PORT}/health && echo " READY" || echo " NOT READY"

# --- 2b. Warmup: compute 40 prefixes, store KV to AttentionDB ---
echo "=== PHASE 2b: WARMING ATTENTIONDB — 40 prefixes ==="
docker exec ministral-brain vllm bench serve \
  --model ${MODEL} \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --port ${VLLM_PORT} \
  --temperature 0 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len ${PREFIX_LEN} \
  --prefix-repetition-suffix-len ${SUFFIX_LEN} \
  --prefix-repetition-num-prefixes ${NUM_PREFIXES} \
  --prefix-repetition-output-len ${OUTPUT_LEN} \
  --num-prompts ${NUM_PROMPTS_COLD} \
  --request-rate ${REQUEST_RATE}

# RECORD: TTFT Mean, P99

# --- 2c. Verify AttentionDB is populated ---
echo "=== Checking AttentionDB stats ==="
docker exec ministral-brain python3 -c "
import _attentiondb
e = _attentiondb.Engine('/etc/attentiondb/attentiondb.yaml')
s = e.stats()
print(f'  index_entries:        {s.index_entries}')
print(f'  t1_used_bytes:        {s.t1_used_bytes / (1024**3):.1f} GiB')
print(f'  t2_total_bytes_disk:  {s.t2_total_bytes_on_disk / (1024**3):.1f} GiB')
print(f'  t2_num_segments:      {s.t2_num_segments}')
print(f'  wb_submitted:         {s.wb_submitted}')
print(f'  wb_flushed:           {s.wb_flushed}')
print(f'  admission_rejected:   {s.admission_rejected}')
e.close()
"
# Expected:
#   index_entries ~41,000 (40 prefixes × ~1024 chunks each)
#   t1_used_bytes ~16 GiB (capacity limited)
#   t2_total_bytes_disk ~30-46 GiB
#   admission_rejected = 0 (min_recompute_cost=0 admits all)

# Also check NVMe usage on the host
echo "=== NVMe data on host ==="
du -sh ${ADB_DATA_DIR}
ls -la ${ADB_DATA_DIR}/


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: COLD RESTART WITH WARM ATTENTIONDB — The key test
# ══════════════════════════════════════════════════════════════════════
#
# Purpose: Restart vLLM — GPU KV cache and T1 DRAM are wiped.
# But T2 NVMe blobs + checkpoint persist on the host's NVMe volume.
#
# On restart, AttentionDB reloads the checkpoint and recovers the
# full index. Every prefix is served from local NVMe (~300us per
# chunk) instead of:
#   - Recomputed from scratch (~3s per prefix)  [Phase 1]
#   - Fetched over TCP from Aerospike (~1000ms) [Aerospike Phase 3]
#
# THIS IS THE KEY COMPARISON:
#   Phase 1b (no cache):          ~116,866 ms mean TTFT (recompute)
#   Aerospike Phase 3b (TCP):     ~73,613 ms mean TTFT (network fetch)
#   Phase 3b (AttentionDB, NVMe): target significantly lower (local I/O)
# ──────────────────────────────────────────────────────────────────────

# --- 3a. Kill and restart (T2 data persists via volume mount) ---
docker rm -f ministral-brain 2>/dev/null

echo "=== NVMe data persisted ==="
du -sh ${ADB_DATA_DIR}
ls -la ${ADB_DATA_DIR}/

docker run -d --name ministral-brain \
  --runtime=nvidia --gpus all --ipc host \
  --network host \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -v ${ADB_CONFIG_DIR}/attentiondb.yaml:/etc/attentiondb/attentiondb.yaml:ro \
  -v ${LMCACHE_CONFIG_DIR}:/etc/lmcache:ro \
  -v ${ADB_DATA_DIR}:/opt/dlami/nvme/attentiondb \
  -e HF_TOKEN=${HF_TOKEN} \
  -e LMCACHE_CONFIG_FILE=/etc/lmcache/config.yaml \
  -e LMCACHE_USE_EXPERIMENTAL=True \
  -e PYTHONHASHSEED=0 \
  -e TMPDIR=/tmp \
  ${ADB_VLLM_IMAGE} \
  ${MODEL} \
  --port ${VLLM_PORT} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --dtype bfloat16 \
  --enforce-eager \
  --trust-remote-code \
  --disable-hybrid-kv-cache-manager \
  --disable-log-stats \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

docker logs -f ministral-brain
# Wait for "Application startup complete", then Ctrl+C
# Look for: "AttentionDB adapter initialised" + checkpoint recovery messages

curl -sf http://localhost:${VLLM_PORT}/health && echo " READY" || echo " NOT READY"

# --- 3b. Verify checkpoint recovery ---
echo "=== Post-restart AttentionDB stats ==="
docker exec ministral-brain python3 -c "
import _attentiondb
e = _attentiondb.Engine('/etc/attentiondb/attentiondb.yaml')
s = e.stats()
print(f'  index_entries:        {s.index_entries}')
print(f'  t1_used_bytes:        {s.t1_used_bytes / (1024**3):.1f} GiB (cold, rebuilding)')
print(f'  t2_total_bytes_disk:  {s.t2_total_bytes_on_disk / (1024**3):.1f} GiB (persisted)')
print(f'  t2_num_segments:      {s.t2_num_segments}')
e.close()
"
# Expected:
#   index_entries ~41,000 (recovered from checkpoint)
#   t1_used_bytes ~0 GiB (cold DRAM, will rebuild on access)
#   t2_total_bytes_disk ~30-46 GiB (persisted from Phase 2)

# --- 3c. THE KEY TEST: cold GPU, 40 prefixes from local NVMe ---
echo "=== PHASE 3c: COLD RESTART — 40 prefixes from AttentionDB ==="
docker exec ministral-brain vllm bench serve \
  --model ${MODEL} \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --port ${VLLM_PORT} \
  --temperature 0 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len ${PREFIX_LEN} \
  --prefix-repetition-suffix-len ${SUFFIX_LEN} \
  --prefix-repetition-num-prefixes ${NUM_PREFIXES} \
  --prefix-repetition-output-len ${OUTPUT_LEN} \
  --num-prompts ${NUM_PROMPTS_COLD} \
  --request-rate ${REQUEST_RATE}

# EXPECTED RESULTS (Phase 3c vs Aerospike Phase 3b vs Baseline Phase 1b):
#
#   Metric          Baseline (1b)   Aerospike (3b)   AttentionDB (3c)   Why
#   ──────────────  ──────────────  ──────────────   ────────────────   ───
#   Mean TTFT       ~116,866 ms     ~73,613 ms       << 73,613 ms       Local NVMe (~300us/chunk)
#                                                                        vs TCP (~1ms/chunk).
#                                                                        1040 chunks: ~300ms vs ~1000ms
#
#   Throughput      ~0.23 req/s     ~0.31 req/s      > 0.31 req/s       Fetch ~3x faster, no TCP contention
#
#   P99 TPOT        ~610 ms         ~237 ms          < 237 ms           No network jitter
#
#   Infra cost      $1.32/hr        $1.77/hr         $1.32/hr           No Aerospike node
#
# KEY METRICS FROM LOGS:
#   docker logs ministral-brain 2>&1 | grep -E "Retrieved|attentiondb|checkpoint"

# --- 3d. Steady state — second round after T1 warms up ---
echo "=== PHASE 3d: STEADY STATE — T1 hot, T2 backing ==="
docker exec ministral-brain vllm bench serve \
  --model ${MODEL} \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --port ${VLLM_PORT} \
  --temperature 0 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len ${PREFIX_LEN} \
  --prefix-repetition-suffix-len ${SUFFIX_LEN} \
  --prefix-repetition-num-prefixes ${NUM_PREFIXES} \
  --prefix-repetition-output-len ${OUTPUT_LEN} \
  --num-prompts ${NUM_PROMPTS_WARM} \
  --request-rate ${REQUEST_RATE}

# EXPECTED: Even lower TTFT — T1 DRAM now holds ~14 most-accessed
# prefixes (~100us), T2 serves the rest (~300us). No recomputes,
# no network. This is AttentionDB's steady-state advantage.


# ══════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════

docker rm -f ministral-brain 2>/dev/null

echo "=== Final AttentionDB data on NVMe ==="
du -sh ${ADB_DATA_DIR}

# To completely clean up:
#   rm -rf ${ADB_DATA_DIR} ${ADB_CONFIG_DIR} ${LMCACHE_CONFIG_DIR}
#   docker rmi ${ADB_VLLM_IMAGE}
# Terminate EC2 instance when done to stop charges.


# ══════════════════════════════════════════════════════════════════════
# REAL-WORLD MAPPING — LegalAI Deployment with AttentionDB
# ══════════════════════════════════════════════════════════════════════
#
# Same "LegalAI" scenario: 40 law firms, 8K-token prefix each.
#
# WITH ATTENTIONDB (vs Aerospike):
#   - No separate storage node (saves $0.452/hr per GPU node)
#   - T1 DRAM: top-14 firms always in memory (~100us, zero-copy DMA)
#   - T2 NVMe: all 40 firms persisted locally (~300us via io_uring)
#   - Node restart: checkpoint recovery, immediate serving
#   - No VPC networking concerns, no cross-AZ latency
#
# SCALING:
#   Tenants   AttentionDB (in-process)        Aerospike (remote)      Delta
#   ──────────────────────────────────────────────────────────────────────
#   40        g6.4xlarge $1.32/hr              + i3en.xlarge $0.452    -25%
#   100       g6.4xlarge $1.32/hr (local NVMe  + i3en.xlarge $0.452   -25%
#             fits 100 × 1.14 GB = 114 GB)
#   500       g6.4xlarge $1.32/hr (570 GB fits  + i3en.xlarge $0.452  -25%
#             on 600 GB local NVMe)
#
#   AttentionDB eliminates the Aerospike node entirely. The local
#   NVMe on g6.4xlarge (~600 GB) handles up to 500 tenants. Beyond
#   that, a larger NVMe instance (g6.12xlarge) still avoids the
#   second node.
