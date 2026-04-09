# Multi-Node V1 Implementation Plan — April 2026

## Goal

Build a 2-node LAN prototype for stage-based pipeline parallelism without dragging in public-network complexity too early.

## V1 Boundaries

Hard constraints:

- 2 nodes only
- same LAN
- one model family
- fixed shard boundaries
- deterministic inference path
- no automatic failover
- no trust scoring
- no rewards/economic changes
- no public-node scheduling yet

## Architecture Slices

### Slice 1: Shard Metadata

Add explicit shard manifests for the first test model.

Deliverables:

- per-model shard manifest format
- shard IDs
- layer ranges
- download/local artifact locations
- checksums
- expected shard size

Likely files/modules:

- new manifest definitions under `crates/compute-network` or `crates/compute-daemon`
- model metadata extension from current `scheduler.ts` model definitions

### Slice 2: Local Shard Loader

The daemon must be able to load a shard by layer range rather than assuming a full model.

Deliverables:

- `load_shard(model_id, shard_id)` lifecycle
- local cache location
- checksum verification
- shard state reporting in runtime state / heartbeat

Likely files/modules:

- `crates/compute-daemon/src/inference/manager.rs`
- new shard loader module in `crates/compute-daemon/src/inference/`

### Slice 3: Stage Runtime

Promote the stage runner from dormant abstraction to real execution path for the prototype.

Deliverables:

- head stage runtime
- tail stage runtime
- activation receive / process / forward loop
- token return path

Likely files/modules:

- `crates/compute-network/src/stage.rs`
- `crates/compute-network/src/transport/*`
- daemon integration in `crates/compute-daemon/src/runtime.rs`

Prototype enablement:

- gate stage-mode behind `experimental.stage_mode_enabled = true` in `~/.compute/config.toml`
- default must remain `false` until the stage runtime is real

### Slice 4: Assignment Model

Assignments must become shard/stage-native.

Deliverables:

- assignment includes shard ID, stage role, peer addresses
- node reports shard inventory and loaded shard
- orchestrator can form a fixed 2-stage pipeline

Likely files/modules:

- `orchestrator/src/services/scheduler.ts`
- `orchestrator/src/services/relay.ts`
- `orchestrator/src/types/pipeline.ts`
- `crates/compute-daemon/src/relay.rs`

### Slice 5: 2-Node Scheduler

Implement the narrowest possible scheduler for v1.

Rules:

- exactly 2 stages
- fixed model
- fixed shard split
- only choose nodes that already qualify for the assigned shard
- prefer LAN / low-latency peers

Prototype enablement:

- orchestrator side gated behind `EXPERIMENTAL_STAGE_MODE=true`
- daemon side gated behind `experimental.stage_mode_enabled = true`
- both must be enabled before a stage-based Gemma assignment is acted on

Likely files/modules:

- `orchestrator/src/services/scheduler.ts`

### Slice 6: Instrumentation

Metrics are mandatory before optimization.

Add:

- pipeline formation time
- shard load time
- stage RTT
- activation send/receive latency
- TTFT
- inter-token latency
- per-stage processing time

Likely files/modules:

- `crates/compute-daemon/src/runtime.rs`
- `crates/compute-network/src/stage.rs`
- `orchestrator/src/routes/completions.ts`

## First Concrete Model Choice

Use **`gemma-4-e4b-q4`** for the first prototype.

Why:

- already the main testing model in the current stack
- fast enough to iterate on repeatedly
- small enough to keep shard handling simple
- deterministic enough for correctness checks
- avoids introducing a second unstable variable while building the runtime

For LAN v1, split it into 2 fixed shards:

- shard A: layers `0-13`
- shard B: layers `14-27`

This is not the final target for network scale. It is the first runtime proof target.

## Milestone Plan

### Milestone A: Repo Alignment

Deliverables:

- ADR accepted
- implementation plan accepted
- RPC path explicitly marked experimental in code comments/docs

### Milestone B: Fixed Shard Prototype

Deliverables:

- first shard manifest
- daemon can load shard A or shard B
- stage runtime can start in head or tail mode

### Milestone B.5: Backend Feasibility Spike

Deliverables:

- one real stage execution backend selected for the prototype
- deterministic 2-stage LAN completion for `gemma-4-e4b-q4`
- single-node vs 2-stage output comparison at temperature 0
- explicit decision on whether `llama.cpp` remains viable for stage execution

Success:

- stage transport is connected to real stage math, not placeholder pass-through
- performance measurements are meaningful enough to guide further work

### Milestone C: 2-Node End-to-End LAN Inference

Deliverables:

- head stage sends activations to tail stage
- tail stage returns tokens
- end-to-end prompt succeeds

### Milestone D: Measurement Pass

Deliverables:

- baseline single-node numbers
- 2-node LAN numbers
- clear TTFT and per-token cost breakdown

### Milestone E: Go / No-Go

Proceed only if:

- correctness is stable
- latency is understandable
- there is a plausible path to usable WAN performance

## First Code Changes

These are the first changes I would make in the codebase:

1. Add a shard manifest type and one concrete manifest.
   First manifest: `gemma-4-e4b-q4`, 2 shards, fixed LAN v1 split.
2. Add a prototype stage-mode runtime path in the daemon behind a feature flag or explicit experimental mode.
3. Extend assignment messages to carry shard identity and peer topology for 2-node mode.
4. Add prototype-only pipeline formation for a fixed 2-stage model.
5. Instrument end-to-end timings before optimizing anything.

## What To Avoid Right Now

Do not do these first:

- dynamic shard sizing
- full water-filling for heterogeneous public nodes
- WAN optimization
- standby node systems
- economic incentives
- verification systems beyond basic correctness checks

## Success Criteria

The prototype is successful if it proves all of the following:

- deterministic correctness at temperature 0
- repeatable end-to-end inference over 2 nodes
- measured overhead by stage and transport segment
- a clear answer on whether WAN investigation is justified

## Current Status

- Assignment schema carries `assignment_mode`, `shard_id`, `start_layer`, `end_layer`, `upstream_addr`, and `downstream_addr`.
- `gemma-4-e4b-q4` is the fixed prototype model with a `0-13` / `14-27` split.
- The orchestrator has a fixed 2-node Gemma prototype branch behind `EXPERIMENTAL_STAGE_MODE=true`.
- The daemon has an explicit `experimental.stage_mode_enabled` gate and a distinct stage prototype runtime.
- The daemon also supports `experimental.stage_backend`, currently defaulting to `prototype` for real deterministic stage execution without depending on llama.cpp hidden-state support.
- A local 2-node head/tail QUIC roundtrip harness passes.
- The daemon relay can now send experimental non-streaming requests into the head-stage prototype path.
- Streaming remains unsupported for stage mode and should fail explicitly until token streaming is implemented through the transport path.
- Stage execution is now isolated behind a dedicated backend module so the prototype can swap away from the current llama.cpp placeholder path without rewriting transport/runtime code.
- The stage payload contract now distinguishes request ingress (`PromptV1`) from inter-stage traffic (`HiddenStatesV1`), so the head stage no longer forwards prompt-shaped JSON as if it were a hidden-state tensor.
- The current hidden-state contract is still a stubbed envelope, not true model activations, but it gives the runtime a real stage-forward boundary to build on.
