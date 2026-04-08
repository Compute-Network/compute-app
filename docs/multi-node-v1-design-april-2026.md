# Multi-Node V1 Design — April 2026

## Decision

Compute should pursue **true stage-based pipeline parallelism** as the multi-node architecture.

Do **not** treat the current llama.cpp RPC head/worker path as the long-term foundation for internet-distributed inference.

## Why

The repo currently contains two different multi-node ideas:

1. **Stage-based pipeline parallelism**
   - Nodes own contiguous layer ranges
   - Nodes exchange activations between stages
   - Shard caching, placement, and recovery all make architectural sense
   - Matches the product story in `docs/technical-overview.md`

2. **llama.cpp RPC head/worker**
   - One head runs `llama-server --rpc`
   - Worker nodes run `rpc-server`
   - Good for prototyping remote offload
   - Does not cleanly match the shard marketplace / cached-stage / public-node vision

The current docs, economics, and scheduling ideas all assume the first model, while the current runtime prototype is closer to the second.

That split should end now. Multi-node v1 should be built around **stage ownership and activation passing**, not RPC offload.

## Architecture Choice

### Chosen path

**Stage-based pipeline parallelism**

Each stage:
- owns a fixed contiguous layer range
- loads its own shard locally
- receives activations from the previous stage
- runs its local forward pass
- sends activations to the next stage
- final stage samples and returns tokens upstream

### Deprioritized path

**llama.cpp RPC head/worker**

Keep the current RPC code only as:
- a short-term experiment
- a LAN comparison tool
- a reference for process orchestration

Do not keep building product strategy around it.

## Scope For V1

Multi-node v1 should be intentionally narrow:

- 2 nodes only
- same model only
- fixed shard boundaries
- LAN only
- one prompt/completion path
- no dynamic recovery
- no trust scoring
- no economic bonuses
- no public internet deployment yet

If 2-node LAN does not work well, broader public-node pipeline work should stop there until the bottleneck is understood.

## What Reuses Well From Today

These pieces remain useful:

- orchestrator pipeline records and stage metadata
- current layer-allocation concepts in `scheduler.ts`
- node assignment plumbing over orchestrator WS
- node auth/session model
- QUIC transport ideas in `crates/compute-network`
- existing single-node control-plane hardening

## What Must Change

### 1. Pick one execution engine

The v1 execution path should be:

- `compute-network` stage transport
- explicit stage runner lifecycle
- explicit shard loading by layer range

The v1 execution path should **not** depend on:

- head node owning the whole inference graph
- llama.cpp RPC worker semantics

### 2. Make shard definitions first-class

For each model used in multi-node testing, define:

- total layer count
- shard boundaries
- shard artifact URLs
- shard checksums
- per-shard size and VRAM expectations

These should be explicit metadata, not inferred ad hoc at runtime.

### 3. Treat scheduler inputs differently

The current allocator is mostly:

- TFLOPS
- VRAM

For real multi-node formation, the first important inputs are:

- shard availability on node
- RTT to adjacent candidates
- measured stage-to-stage throughput
- warm shard already loaded

Raw TFLOPS matters, but it is not enough.

### 4. Define decode performance targets before public-node work

Pipeline parallelism over consumer internet is plausible.
Great user experience is not guaranteed.

V1 must explicitly measure:

- TTFT
- per-token latency
- tokens/sec
- activation transfer latency
- stage imbalance
- failure behavior when one node disappears

## Suggested Milestones

### Milestone 0: Architecture Freeze

Deliverable:
- one short ADR stating that multi-node v1 is stage-based pipeline parallelism
- RPC path marked experimental

Success:
- no ambiguity in docs or code reviews about the chosen direction

### Milestone 1: 2-Node LAN Prototype

Deliverable:
- one model
- two fixed shards
- two local/LAN machines
- explicit activation passing between stages
- successful completion end to end

Success:
- deterministic completions at temperature 0
- stable repeated runs
- measured TTFT and decode latency

### Milestone 2: 2-Node LAN Benchmarking

Measure:
- single-node baseline
- 2-node LAN TTFT
- 2-node LAN tok/s
- stage utilization
- activation payload size
- effect of micro-batching

Success:
- clear understanding of where latency goes
- no guesswork about bottlenecks

### Milestone 3: 3-Node LAN Prototype

Deliverable:
- three fixed stages
- same instrumentation
- same model family

Success:
- predictable scaling behavior
- known breakpoints for stage imbalance and transfer latency

### Milestone 4: Controlled WAN Pilot

Only after LAN is solid.

Deliverable:
- 2 nodes across real internet links
- same region or low-latency regions first
- no public network yet

Success:
- known TTFT increase vs LAN
- known per-token penalty vs LAN
- concrete thresholds for “usable” vs “not worth it”

### Milestone 5: Public Multi-Node Pilot

Only after:
- shard loading is reliable
- basic stage verification exists
- routing has latency-aware placement
- recovery semantics are defined

## What Not To Build Yet

Do not make these part of v1:

- hot spare pools
- standby economics
- shard caching bonuses
- public trust scores
- geographic optimization beyond simple same-region preference
- TOPLOC-grade verification everywhere
- cross-pipeline redundancy
- advanced WAN recovery

Those all depend on first proving the basic runtime is viable.

## Risks

### 1. Decode latency may dominate

This is the biggest technical risk.
Even if bandwidth is acceptable, per-token cross-stage latency may make WAN decode too slow for interactive use.

### 2. Scheduler quality may matter more than model runtime

Bad placement can ruin otherwise viable multi-node performance.
This makes routing instrumentation essential.

### 3. Shard lifecycle is a product in itself

Download, verify, cache, warm, reload, evict, and reassign is already substantial systems work before any economics are layered on top.

### 4. Verification remains unsolved for early production

Activation hashing and blame assignment are promising, but not mature enough to block the first prototype.

## Recommendation

Proceed with multi-node work only under these rules:

1. Freeze the architecture on stage-based pipeline parallelism.
2. Treat llama.cpp RPC as experimental and non-strategic.
3. Prove 2-node LAN first.
4. Instrument everything before optimizing.
5. Do not add economics or trust complexity until the runtime path is proven.

## Bottom Line

The multi-node opportunity is real, and the strategy is directionally correct.

But the repo must stop mixing:

- a **true activation-passing pipeline design**
- with a **remote-offload RPC prototype**

Pick the pipeline design, prove it on LAN, then expand carefully.
