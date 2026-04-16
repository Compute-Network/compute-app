# Network Architecture Strategy — April 2026

## Overview

This document captures research, performance analysis, and implementation direction
for the Compute Network distributed inference architecture. It reflects a full
strategic review of pipeline parallelism vs MoE routing, the role of dedicated nodes,
and how to build toward large-model pipeline inference without blocking current
production progress.

The conclusion: **dedicated nodes with the llama.cpp backend are the production path
today**. The pipeline engine is built in parallel as a staged capability, revealed
publicly once enough nodes exist to make it viable.

---

## 1. The Problem Space

Compute Network's goal is a DePIN inference network where contributors earn $COMPUTE
tokens by serving inference workloads across consumer hardware (primarily Apple Silicon
Macs). The key architectural question is how to distribute inference across those nodes.

Three candidate approaches were evaluated:

1. **Dedicated single-node inference** — each node runs a complete model
2. **Pipeline parallelism** — model layers split sequentially across N nodes
3. **Mixture-of-Experts (MoE) routing** — tokens routed to expert-holding nodes per layer

---

## 2. Performance Analysis

### Network Assumptions

All analysis uses:
- Upload: 30 Mbps (home) or 100 Mbps (upgraded)
- Download: 100 Mbps
- Same-country RTT: 25ms
- Model: GLM-5.1 class, ~130B parameters, Q4 quantized ≈ 65GB
- Hidden state: hidden dim 12,288, f16 = **24KB per token transfer**

### 2.1 Pipeline Parallelism

Each node holds a contiguous range of layers. Hidden state passes between stages at
layer boundaries only.

**5 Macs, 30Mbps, optimal config (3 stages × ~13 layers):**

```
Per hop:  6.6ms upload + 25ms RTT = 31.6ms
2 hops:   63ms network overhead
Compute:  ~500ms (sequential across stages)

Single user:   ~563ms/token → 1.8 tok/s
5 users:       ~878ms/token cycle → 1.1 tok/s per user
```

**20 Macs, 30Mbps, optimal config (6 parallel pipelines × 3 Macs):**

| Concurrent Users | Tok/s per User |
|---|---|
| 5 | 1.8 |
| 20 | 1.3 |
| 60 | 1.0 |
| 100 | 0.7 |

**Key property:** adding more Macs means more parallel pipelines, not longer ones.
Optimal pipeline depth is the minimum number of nodes required to fit the model.
For a 65GB Q4 model on 32GB Macs: 3 nodes minimum (21.7GB per node).

**Effect of 100Mbps upgrade:**

Minimal. Bandwidth was not the bottleneck — 25ms RTT is. Single-user timing barely
changes. High-concurrency batching improves modestly.

```
Single user: 554ms (100Mbps) vs 563ms (30Mbps)  — negligible
60 users:    598ms (100Mbps) vs 680ms (30Mbps)   — modest improvement
```

### 2.2 MoE Routing

Tokens dispatched to expert-holding nodes at every MoE layer. Assumes 20 MoE layers,
2-of-8 experts active per token.

**The hard latency floor:**

```
Pipeline minimum:  2 hops  × 25ms RTT = 50ms   irreducible
MoE minimum:      20 hops  × 25ms RTT = 500ms  irreducible
```

Even with infinite bandwidth, MoE on 25ms internet RTT has 10× more irreducible
latency overhead per token than an optimally-staged pipeline. Each MoE layer requires
a full round trip because each layer depends on the previous.

**5 Macs, MoE vs Pipeline:**

| Users | Pipeline (tok/s) | MoE (tok/s) |
|---|---|---|
| 1 | 1.6 | 0.9 |
| 5 | 1.1 | 0.9 |
| 20 | — | 1.0 |
| 40 | 0.9 | 1.0 (crossover) |
| 100 | 0.5* | 0.8 |

*Pipeline hard queues at 6 slots with 20-Mac config

**20 Macs, MoE vs Pipeline:**

Crossover moves to ~40 concurrent users. Below that, pipeline wins. Above that,
MoE degrades more gracefully because it has no hard slot limit.

### 2.3 Space Invaders Benchmark (~400 output tokens)

Model: GLM-5.1 class, 20 Macs, 5 concurrent users:

| Configuration | Time |
|---|---|
| Single Mac (if model fit — it doesn't) | ~3.3 min |
| 5 Macs, pipeline (5 stages) | ~6 min |
| 20 Macs, pipeline (6 parallel × 3) | **~3.7 min** |
| 20 Macs, MoE | ~5 min |

### 2.4 Conclusion: MoE is a Datacenter Architecture

MoE is designed for datacenter switch latency (1–5ms), not consumer internet (25ms).

| Network | RTT | MoE floor | Pipeline floor |
|---|---|---|---|
| Home internet (same country) | 25ms | ~600ms/token | ~550ms/token |
| Home internet (cross-country) | 80ms | ~1700ms/token | ~660ms/token |
| Datacenter (same region) | 1ms | ~120ms/token | ~502ms/token |

At 1ms datacenter RTT, MoE inverts and wins decisively (8+ tok/s vs 2 tok/s) because
its expert batching efficiency dominates. On consumer internet, pipeline wins at all
practical concurrency levels.

**Pipeline parallelism is the correct distributed inference strategy for a DePIN
network of consumer Macs on residential internet.**

---

## 3. The Cold Start Problem and Why Dedicated Nodes Come First

A pure pipeline-only network has a paradox:

- Pipeline inference performs best with many concurrent users (to batch across stages)
- You cannot attract many concurrent users before the network performs well
- Performance is poor before critical mass

This is resolved by separating the product into two layers:

**Layer 1 — Works at any scale (ship now):**
Dedicated nodes each run a complete model. Different nodes serve different users.
The orchestrator routes requests to whichever node has capacity and the right model
loaded. Value is proportional to node count from day one. No batching requirement.
No critical mass required.

**Layer 2 — Unlocks large models (build now, reveal later):**
Pipeline groups enable models too large for any single node (70B, 130B+). Built
and tested in parallel. Activated in production when node density crosses the
threshold where pipeline groups can be reliably assembled.

This means:
- Production path is working and earning today
- Pipeline capability is developed without blocking revenue
- Public reveal of large-model inference is a milestone event, not a scramble

---

## 4. Hybrid Architecture Design

### 4.1 Node Types

The orchestrator node registry is extended with a capability field:

```
Dedicated node:
  - Has sufficient memory for a complete model
  - Runs llama-server, serves full inference requests
  - Zero network hops, lowest latency
  - Earns rewards per completion

Pipeline node:
  - Has sufficient memory for one stage shard (~22GB of a 130B model)
  - Runs stage backend, participates in pipeline groups
  - Groups assembled by orchestrator
  - Earns rewards per stage contribution
```

A single Mac can be both, depending on memory:
- 32GB Mac: can hold Gemma 4B (2.5GB) + one GLM pipeline stage (22GB) simultaneously
- 24GB Mac: dedicated only for large models, both for small models

### 4.2 Routing Logic

```
Incoming request for Model X
  ↓
Is there a dedicated node with Model X loaded and capacity?
  YES → route directly, 1-2 hop latency, fastest path
  NO  ↓
Is there a pipeline group available for Model X?
  YES → route through stages, 2-hop latency, ~1.1-1.8 tok/s
  NO  ↓
Queue request, assemble pipeline group from available stage nodes
```

Dedicated nodes are the premium tier. Pipeline groups expand what is *possible*
(models larger than any single machine), not just what is *fast*.

### 4.3 Dual Backend on a Single Node

A node running both dedicated and pipeline backends simultaneously is managed by
the compute-daemon:

**Memory layout (32GB Mac):**
```
Gemma 4B Q4 (dedicated):         ~2.5GB  (Metal buffers, unified memory)
GLM pipeline stage (1/3 of 65GB): ~22GB  (Metal buffers, unified memory)
OS + daemon:                       ~3GB
Total:                            ~27.5GB  ✓
```

Both models sit in unified memory simultaneously. No eviction on switch.

**Process suspension for GPU resource management:**

```rust
// Unix (macOS, Linux)
#[cfg(unix)]
fn suspend_backend(pid: u32) {
    kill(Pid::from_raw(pid as i32), Signal::SIGSTOP);
}

#[cfg(unix)]
fn resume_backend(pid: u32) {
    kill(Pid::from_raw(pid as i32), Signal::SIGCONT);
}

// Windows
#[cfg(windows)]
fn suspend_backend(handle: HANDLE) {
    NtSuspendProcess(handle);
}

#[cfg(windows)]
fn resume_backend(handle: HANDLE) {
    NtResumeProcess(handle);
}
```

SIGSTOP freezes the process instantly. Model weights remain in unified memory.
KV cache is preserved. GPU drains its current command buffer then goes idle.
SIGCONT resumes with zero reload cost. Existing conversations continue uninterrupted.

**Daemon state machine:**

```
DEDICATED_ACTIVE
  llama-server running, accepting requests
  pipeline stage backend loaded but idle
  ↓ pipeline request arrives

DRAINING
  llama-server finishes current in-flight request
  new dedicated requests queued
  ↓ current request complete

PIPELINE_ACTIVE
  llama-server SIGSTOP'd
  stage backend has full GPU
  processes pipeline stage
  ↓ complete

SIGCONT → llama-server resumes
DEDICATED_ACTIVE (queued requests served immediately)
```

**Initial implementation:** run both backends concurrently, let Metal/CUDA scheduler
time-share GPU. Implement SIGSTOP/SIGCONT when GPU contention becomes measurable
under real load. For a startup-scale network, concurrent mode is sufficient.

---

## 5. Cross-Platform Considerations

### 5.1 Process Suspension

| Platform | Primitive | Notes |
|---|---|---|
| macOS | SIGSTOP / SIGCONT | Standard, works via `nix` crate |
| Linux | SIGSTOP / SIGCONT | Standard, works via `nix` crate |
| Windows | NtSuspendProcess / NtResumeProcess | `windows` crate, functionally equivalent |

The daemon already has `#[cfg]` branches for platform-specific behaviour. This is
a small addition to the existing process management module.

### 5.2 Memory Architecture

| Platform | Memory model | Dual-load viable |
|---|---|---|
| Apple Silicon 32GB+ | Unified (CPU+GPU share pool) | Yes, cleanly |
| Apple Silicon 24GB | Unified | Tight, works for most configs |
| Linux NVIDIA 48GB+ VRAM | Discrete | Yes |
| Linux NVIDIA 24GB VRAM | Discrete | CPU offload required for pipeline stage |
| Linux NVIDIA <16GB VRAM | Discrete | Dedicated only (small models) |
| Linux / Windows CPU-only | RAM | Yes, RAM is usually sufficient |
| Windows NVIDIA | Discrete | Same as Linux NVIDIA |

Apple Silicon is uniquely suited to the dual-backend approach because unified memory
eliminates VRAM as a separate constraint. This is a genuine hardware advantage, not
just a performance one.

### 5.3 Node Capability Registration

Nodes report memory at registration time. The orchestrator tags accordingly:

```
Apple Silicon 32GB:
  dedicated_capable: true   (any model up to ~28GB)
  pipeline_capable:  true   (stage shards up to ~22GB, dual-load)

Linux RTX 4090 (24GB VRAM):
  dedicated_capable: true   (models up to ~20GB VRAM)
  pipeline_capable:  true   (with CPU offload, degraded speed)

Linux RTX 3080 (10GB VRAM):
  dedicated_capable: true   (small models only, up to ~8GB)
  pipeline_capable:  false
```

Routing respects capabilities. The network degrades gracefully rather than breaking.

---

## 6. Why Not llama.cpp RPC

llama.cpp includes a `llama-rpc-server` that appears to enable distributed inference.
Research confirmed it is **not** suitable for the pipeline approach:

- The coordinator node must load the full computation graph (requires full model
  visibility), defeating the purpose for models larger than any single node
- RPC workers act as tensor compute offload, not true pipeline stages — the coordinator
  orchestrates every individual operation, not just stage boundaries
- Intermediate activations route through or are coordinated by the head node rather
  than flowing directly between pipeline stages
- Not designed for the hidden-state-at-boundary protocol needed for true pipeline
  parallelism where no single node needs the full model

The custom stage-forward engine (compute-backend) is therefore necessary. llama.cpp
RPC does not provide an equivalent shortcut.

---

## 7. Why Not MoE (Summary)

MoE routing has a structural incompatibility with residential internet latency:

- 20+ sequential network round trips per token (one per MoE layer)
- 25ms RTT × 20 = 500ms irreducible latency floor per token
- This floor exists regardless of bandwidth — even with 10Gbps symmetric, MoE
  is slower than pipeline on 25ms internet for single requests
- MoE only becomes competitive at datacenter latency (1–5ms) where the floor
  drops to 20–100ms and expert batching efficiency dominates
- For a consumer DePIN network, pipeline wins at all practical concurrency levels
  on residential internet

---

## 8. Implementation Plan

### Phase 1 — Production (Current)

**Status: Deployed**

- Dedicated nodes running llama-server (llama.cpp backend)
- Orchestrator relay routing requests to nodes via WebSocket
- Node registry, billing, rewards, API keys
- Single-node inference: stable, tested end-to-end

No changes needed. This runs in production.

### Phase 2 — Pipeline Engine (Build Now, Silent)

**Goal:** Build the pipeline stage backend alongside the existing llama-server path.
Not exposed publicly. Tested internally and on testnet.

**Daemon changes:**
- Add second backend process slot for stage engine
- Memory availability check at startup (confirm both models fit)
- Request type discrimination (dedicated vs pipeline stage)
- SIGSTOP/SIGCONT integration (or concurrent mode initially)
- Cross-platform process suspension abstraction

**Orchestrator changes:**
- Extend node registry: `pipeline_capable`, `stage_index`, `stage_model_id`
- Pipeline group assembly logic: match N nodes holding consecutive stages of the
  same model, verify adjacency, form a group
- Stage routing: relay hidden state between stages rather than completion text
- Group health monitoring: detect stage dropout, reassemble or failover

**Stage engine:**
- Bind compute-backend stage-forward execution to a real llama.cpp layer range
  OR use llama.cpp with `--override-tensor` / layer slice loading
- Expose the hidden-state-in, activation-out interface per the stage-forward contract
- Validate against the `StageBoundaryPlan` / `StageResumeRequest` handshake already
  defined in compute-backend

**Testing milestones:**
1. Two Macs, single pipeline split, small model (Gemma 2B): end-to-end hidden state
   transfer produces matching output to single-node reference
2. Three Macs, GLM-5.1 stage split, confirm timing matches analysis (~1.8 tok/s)
3. Daemon dual-backend: Gemma dedicated serving simultaneously with GLM pipeline stage
4. Failover: drop a pipeline node mid-generation, confirm graceful degradation

### Phase 3 — Public Reveal

**Trigger condition:** sufficient pipeline-capable nodes registered that the
orchestrator can reliably assemble pipeline groups on demand without queuing.

Estimated threshold: ~15 pipeline nodes per model (allows 5 concurrent 3-node
pipeline groups with redundancy).

**Reveal:**
- Enable pipeline routing in orchestrator (feature flag flip)
- Expose large model list in UI (models that could not previously be served)
- Announce $COMPUTE rewards for pipeline node contribution
- Pricing tier: large-model pipeline requests at higher rate than dedicated

---

## 9. Key Numbers Reference

| Scenario | Tok/s per user |
|---|---|
| Single Mac, model fits, dedicated | 5–8 (small) / 1.5–3 (70B) |
| 5 Macs, pipeline, 5 users | 1.1 |
| 20 Macs, 6 parallel pipelines, 5 users | 1.8 |
| 20 Macs, 6 parallel pipelines, 60 users | 1.67 |
| 20 Macs, MoE, 5 users | 1.1 |
| 20 Macs, MoE, 60 users | 0.93 |
| Space Invaders (400 tok), 20 Mac pipeline, 5 users | ~3.7 min |

**Upload bandwidth impact:** minimal. 30Mbps vs 100Mbps changes single-user timing
by <10ms per token. Latency (25ms RTT) dominates, not bandwidth.

**Optimal pipeline depth:** minimum nodes to fit the model. For 65GB Q4 on 32GB
Macs: 3 nodes. Do not add stages beyond what memory requires — every extra hop
adds 25ms+ RTT with no compute benefit.

---

## 10. Open Questions for Codex

The following implementation decisions are unresolved and should be investigated:

1. **Stage engine binding** — does the stage-forward-lab execution engine bind
   cleanly to a real llama.cpp layer range, or is a separate llama.cpp process
   per stage (with `--override-tensor` or GGUF shard) the better path? The latter
   reuses the existing llama-server process management code.

2. **KV cache across stages** — in autoregressive generation, each stage holds KV
   cache for its layers. On a resume (next token), the stage receives the hidden
   state and needs its own KV cache to continue. How is KV cache lifecycle managed
   when a pipeline group reassembles between tokens?

3. **Stage dropout handling** — if a pipeline node disconnects mid-generation, can
   the orchestrator substitute a different node holding the same stage? This requires
   KV cache portability or cold resume from the carry state.

4. **Carry state necessity** — the compute-backend carry path (lane-indexed sparse
   substate) adds protocol complexity. For clean layer-boundary splits, the hidden
   state alone is the complete sufficient statistic. Carry should be treated as a
   future optimisation (mid-layer splits, speculative pre-warming) not a Phase 2
   requirement.

5. **Concurrent vs SIGSTOP mode** — determine empirically at what GPU utilisation
   Metal/CUDA time-sharing degrades enough to warrant SIGSTOP switching. Start
   concurrent, instrument, decide.

---

*Document authored: April 2026*
*Authors: Claude (Sonnet 4.6) + Compute-Network founder*
*Based on: full distributed inference architecture review session*
