# ADR-0001: Multi-Node V1 Uses Stage-Based Pipeline Parallelism

## Status

Accepted

## Date

2026-04-08

## Context

Compute has two different multi-node directions present in the repo:

1. **Stage-based pipeline parallelism**
   - Nodes own contiguous layer ranges
   - Nodes load their own shard locally
   - Activations move explicitly between stages
   - This matches the long-term product story, scheduler goals, and economics

2. **llama.cpp RPC head/worker**
   - One head process runs `llama-server --rpc`
   - Worker nodes run `rpc-server`
   - This is useful for experiments, but it centralizes too much execution logic in the head

The repo's documentation and product direction assume the first architecture, while parts of the current runtime prototype implement the second.

This ambiguity is now a liability.

## Decision

Compute multi-node v1 will use **stage-based pipeline parallelism** as the primary architecture.

This means:

- each node owns a fixed contiguous layer range
- each node loads its own shard locally
- each stage performs a local forward pass on activations it receives
- activations move directly between stages
- the tail stage produces tokens
- the head stage handles orchestration for the request and streams results back upstream

The current llama.cpp RPC head/worker path will be treated as:

- experimental
- useful for LAN benchmarking and comparison
- not the strategic long-term architecture for public-internet multi-node inference

## Consequences

### Positive

- Aligns implementation with the product and documentation
- Makes shard caching and shard placement meaningful
- Matches public-internet heterogeneous-node research more closely
- Preserves freedom to change local execution backend later
- Makes future stage-level verification possible

### Negative

- Requires more custom runtime work than relying on llama.cpp RPC
- Requires explicit shard manifests and shard loading lifecycle
- Requires explicit activation transport and stage protocol design
- Delays public multi-node rollout until the stage runtime is proven

### Neutral

- The existing llama.cpp RPC code can still be kept for experiments
- The current scheduler ideas remain useful, but only after re-centering them around stage placement rather than RPC offload

## Non-Goals For V1

These are not part of multi-node v1:

- public internet rollout
- trust-score driven routing
- standby economics
- shard caching bonuses
- hot spare pools
- full automatic recovery on stage drop
- advanced verification / TOPLOC everywhere

## First Milestone

The first milestone is a **2-node LAN prototype**:

- one model
- fixed shard boundaries
- explicit activation passing
- deterministic completion correctness
- TTFT / per-token latency instrumentation

If this milestone does not perform acceptably, broader multi-node work should pause until the bottleneck is understood.

## Related Documents

- [multi-node-v1-design-april-2026.md](/Users/macintosh/Documents/projects/Compute/compute-app/docs/multi-node-v1-design-april-2026.md)
- [stage-backend-feasibility-april-2026.md](/Users/macintosh/Documents/projects/Compute/compute-app/docs/stage-backend-feasibility-april-2026.md)
- [pipeline-shard-planning.md](/Users/macintosh/Documents/projects/Compute/compute-app/docs/pipeline-shard-planning.md)
- [technical-overview.md](/Users/macintosh/Documents/projects/Compute/compute-app/docs/technical-overview.md)
