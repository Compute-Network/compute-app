# Pipeline & Shard System — Planning Document

Draft: 2026-04-04 | For discussion — not finalized

## Current State

Single-node inference only. The orchestrator forms "pipelines" but currently each pipeline is just one node running the full model. The water-filling layer allocation algorithm exists but isn't exercised in production yet.

## Goal

Split large models across multiple nodes via pipeline parallelism so the network can run models no single machine could handle (e.g., DeepSeek R1 671B across 8 nodes with 24GB GPUs each).

---

## 1. Shard Distribution Algorithm

### Pre-computed Shards
Each model is pre-split into GGUF shards at defined layer boundaries. Shards are hosted on HuggingFace and cached at `~/.compute/models/<model_id>/`.

### Layer Allocation (Water-Filling)
Already implemented in `scheduler.ts:allocateLayers()`. Assigns layers proportional to each node's TFLOPS, capped by VRAM.

**Improvement needed:** The current algorithm doesn't account for:
- **Network bandwidth** between nodes — high-bandwidth pairs should be adjacent stages
- **Geographic proximity** — co-located nodes should be preferred for adjacent stages
- **Existing shard cache** — prefer nodes that already have the right shard downloaded

### Proposed Enhanced Allocation

```
Score(node, stage) = 
    w1 * (tflops_normalized) +
    w2 * (has_shard_cached ? 1 : 0) +
    w3 * (bandwidth_to_adjacent / max_bandwidth) +
    w4 * (1 - latency_to_adjacent / max_latency)

Weights: w1=0.3, w2=0.4, w3=0.2, w4=0.1
```

The shard cache bonus (w2) is heavily weighted because downloading a 5GB shard on pipeline formation would add minutes of latency. Nodes that already have the shard are strongly preferred.

---

## 2. Trustworthy Node Selection

### Problem
Nodes can:
- Drop out mid-inference (connection lost, machine sleeps, user quits)
- Return garbage/corrupted results
- Be deliberately malicious (return wrong completions)

### Trust Score Algorithm

```
TrustScore(node) = 
    base_score +
    uptime_bonus +
    completion_bonus -
    failure_penalty -
    recency_decay

Where:
    base_score = 50 (new nodes start neutral)
    uptime_bonus = min(30, consecutive_hours_online * 0.5)
    completion_bonus = min(20, successful_requests * 0.01)
    failure_penalty = failed_requests * 5 (decays over 24h)
    recency_decay = hours_since_last_heartbeat * 2
```

**Score ranges:**
- 0-30: Untrusted — only used for non-critical, redundant stages
- 30-60: Standard — can serve any stage
- 60-90: Reliable — preferred for critical first/last stages
- 90-100: Veteran — gets priority assignment and higher reward multiplier

### Trust Score Storage
Add `trust_score: number` to the `nodes` table. Updated on:
- Successful request completion (+0.01)
- Failed/timed-out request (-5)
- Heartbeat received (decay prevention)
- Pipeline drop-out (-10)

### Pipeline Stage Priority
- **First stage** (receives user prompt): Requires trust ≥ 50
- **Last stage** (produces final output): Requires trust ≥ 50
- **Middle stages**: Trust ≥ 30 acceptable
- **Redundant stages** (if available): Trust ≥ 0, results cross-validated

---

## 3. Shard Drop-out & Recovery

### Problem
When a node in a multi-stage pipeline disconnects:
1. Active inference requests on that pipeline fail
2. The pipeline is incomplete — remaining stages can't process without the dropped shard
3. Users experience errors until recovery

### Detection
- WebSocket disconnect event (immediate)
- Heartbeat timeout (120s fallback)
- Request timeout (120s)

### Recovery Strategy

**Tier 1: Hot Spare (< 5 seconds)**
Keep a pool of "standby" nodes that have popular shards pre-cached. On drop-out:
1. Immediately assign standby node to the vacated stage
2. Update pipeline routing
3. Retry the failed request

**Cost:** Standby nodes earn reduced rewards (20% of active rate) to incentivize staying available.

**Tier 2: Fast Reassignment (5-30 seconds)**
Find an available node that already has the shard cached:
1. Query `nodes` table for `downloaded_models LIKE '%model_id%'` AND `pipeline_id IS NULL`
2. Assign to pipeline
3. Node loads shard from disk (fast — already cached)
4. Resume pipeline

**Tier 3: Cold Reassignment (30s - 5min)**
No cached node available:
1. Find best available node by VRAM/TFLOPS
2. Node downloads the specific shard from HuggingFace
3. Once loaded, assign to pipeline
4. Queue failed requests for retry

**Tier 4: Pipeline Reformation (> 5min)**
If no single node can fill the gap:
1. Terminate the broken pipeline
2. Call `formPipeline()` fresh with all available nodes
3. New pipeline may have different stage allocation

### Client-Side Handling
- compute-code should retry on 502/503 with exponential backoff (1s, 2s, 4s)
- Show "Reconnecting to network..." rather than hard error
- Max 3 retries before surfacing error to user

---

## 4. Bandwidth & Location-Aware Routing

### Problem
Pipeline parallelism sends intermediate activations between stages. For a 70B model:
- Activation tensor per token: ~16KB (hidden_dim=8192, fp16)
- With context: up to 16KB * seq_len per forward pass
- At 30 tok/s: ~500KB/s sustained between stages

This is manageable on broadband but problematic on:
- Slow connections (< 5 Mbps upload)
- High-latency links (> 100ms RTT)
- Nodes on opposite sides of the globe

### Measurement
Add a `bandwidth_test` step during node registration:
1. Orchestrator sends a 1MB test payload to the node via WebSocket
2. Measure round-trip time
3. Calculate effective bandwidth: `1MB / (rtt/2)`
4. Store as `measured_bandwidth_mbps` in nodes table

### Geographic Clustering
Prefer nodes in the same cloud region or geographic area for adjacent pipeline stages:

```
Affinity(nodeA, nodeB) = 
    1.0 if same_region
    0.7 if same_continent
    0.3 if cross_continent_low_latency (< 50ms)
    0.1 if cross_continent_high_latency
```

### Pipeline Formation with Location Awareness
After water-filling determines how many layers each node gets, sort the assignment order by geographic proximity:

```
1. Pick first stage node (highest TFLOPS with sufficient VRAM)
2. For each subsequent stage:
   a. Score = TFLOPS_weight * tflops + Affinity_weight * affinity_to_previous_stage
   b. Pick highest scoring available node
```

---

## 5. Anti-Gaming Measures

### Sybil Detection
- Same IP address running multiple "nodes" with same wallet → counted as one
- GPU fingerprinting: compare reported GPU model/VRAM across nodes from same wallet
- Rate limit registrations: max 5 nodes per wallet

### Result Validation
For high-value requests, run the same prompt through 2 different pipeline paths and compare:
- Token-level comparison (should be identical for temperature=0)
- If divergence > threshold → flag the pipeline with lower trust scores
- Only do this for ~1% of requests (sampling-based)

### Idle Node Farming Prevention
Nodes that are "online" but never serve requests don't earn rewards:
- Rewards only accrue on actual token generation
- Uptime alone doesn't generate revenue
- This naturally prevents parking idle nodes for rewards

---

## 6. Implementation Phases

### Phase 1: Model-Aware Routing (DONE)
- [x] Nodes report `downloaded_models` in heartbeat
- [x] Orchestrator filters by model when forming pipelines
- [x] "auto" model selection based on network availability
- [x] compute-code model switcher

### Phase 1.5: Error Handling & Resilience (DONE)
- [x] Circuit breaker for failing nodes (3 failures in 5min → skip for 60s)
- [x] Retry with pipeline reformation (up to 3 attempts on non-streaming)
- [x] Pipeline health check before routing (verify first-stage node still connected)
- [x] Reject pending requests immediately on node disconnect (was waiting 120s)
- [x] GGUF integrity verification (magic header + size check) on download and load
- [x] Stale .tmp file cleanup from interrupted downloads
- [x] API key validation caching (60s TTL, graceful Supabase outage handling)
- [x] Pipeline formation deduplication (concurrent requests share one formation)
- [x] Idle pipeline reaping (10min timeout → free nodes)
- [x] Post-process parallelization (rewards + usage run concurrently)
- [x] Download size verification before rename
- [x] Streaming error recovery (terminate broken pipeline, inform client to retry)

### Phase 2: Multi-Node Pipeline (Next)
- [ ] Pre-compute shard definitions for each model (layer ranges, download URLs)
- [ ] Modify daemon to load specific shard range (not full model)
- [ ] Implement inter-stage communication (node-to-node activation passing)
- [ ] Test with 2-node pipeline on LAN

### Phase 3: Trust & Recovery
- [ ] Add `trust_score` to nodes table
- [ ] Implement trust score updates on success/failure
- [ ] Hot spare pool for popular models
- [ ] Automatic pipeline recovery on node dropout

### Phase 4: Bandwidth Optimization
- [ ] Bandwidth measurement during registration
- [ ] Geographic clustering for pipeline formation
- [ ] Activation tensor compression (quantize intermediates to int8)
- [ ] Batched activation transfer (send multiple tokens' activations together)

### Phase 5: Scale & Economics
- [ ] Dynamic reward rates based on model demand
- [ ] Shard caching incentives (bonus for keeping rare shards)
- [ ] Pipeline preformation (keep popular model pipelines warm)
- [ ] Load balancing across multiple active pipelines for same model

---

## Open Questions

1. **Shard granularity:** Should each shard be exactly N layers, or variable based on VRAM tiers (4GB, 8GB, 16GB, 24GB)?

2. **Activation format:** Raw fp16 tensors or compressed? Compression adds CPU overhead but reduces bandwidth 4-8x.

3. **Failure attribution:** When a multi-stage pipeline produces wrong output, which stage caused it? Need a way to verify intermediate activations.

4. **Economic model for shards:** Should nodes that cache rare/large shards earn a storage bonus? How much?

5. **Pipeline warmup:** First request to a cold pipeline is slow (model loading). Should we keep N pipelines warm per model? Cost: N nodes idle but loaded.
