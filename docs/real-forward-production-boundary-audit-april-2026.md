# Real-Forward Production Boundary Audit - April 2026

## Purpose

This document is the boundary between:

- code that is shaping the production staged inference path
- code that remains useful as lab or prototype scaffolding

The immediate reason for writing it is simple: the repo has two different
histories inside it now.

- `crates/stage-forward-lab/src/real_forward.rs` is the real Gemma path.
- `crates/stage-forward-lab/src/lib.rs` still contains a large amount of toy,
  sketch, and typed carry/contract machinery.

Those are not the same track anymore. Future work should not treat them as if
they are.

For the next backend step beyond the CPU reference implementation, also read
`docs/stage-acceleration-architecture-april-2026.md`. That document defines
how acceleration should be introduced without changing the defended
`real_forward` stage contract.

The first concrete part of that plan is now in place: the daemon runtime loads
`real_forward` through a stage-local provider boundary, with `cpu-ref`
wrapping the current Rust implementation.

## Current Facts

### 1. The production candidate is `real_forward`

The only backend that currently produces the defended real Gemma behavior is:

- `crates/stage-forward-lab/src/real_forward.rs`

The daemon's `real_forward` runtime path goes through:

- `crates/compute-daemon/src/inference/stage_backend.rs`
- `crates/compute-daemon/src/stage_runtime.rs`

This path now has:

- real Gemma tokenization
- Gemma instruct prompt formatting
- real hidden-state forwarding
- real per-stage decode session reuse
- real prompt-prefill reuse
- real local and transport-path prompt-suite parity

For the current delivery target, this means:

- Gemma E4B is treated as a **2-stage production shape**
- real `N > 2` work on Gemma is not the active milestone
- any further `N > 2` work should be justified by a model that actually needs
  it or by a concrete production consumer

### 2. The real-forward payload contract is narrow

The shared tensor type still carries:

- `continuation`
- `transient`
- `carry`

in [StageTensor](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/lib.rs#L65).

But the real-forward path does not materially use that richer surface.

On the real path:

- head prompt ingress returns `StageTensor` with `continuation = None`,
  `transient = None`, `carry = None` in
  [real_forward.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/real_forward.rs#L1005)
- downstream forwarding does the same in
  [real_forward.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/real_forward.rs#L2188)
- the daemon reconstructs real-forward inputs with those fields unset in
  [stage_backend.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/compute-daemon/src/inference/stage_backend.rs#L643)

The real transport contract is effectively:

- hidden-state bytes
- optional aux bytes packed into `StageTensor.bytes`
- stage trace
- prompt text only where needed for ingress/debug
- max token count

Not:

- typed carry lanes
- typed transient checkpoints
- resume contract negotiation

### 3. The Gemma PLE / per-layer projection path is implemented

The current branch does load and use:

- `per_layer_model_proj.weight`
- `per_layer_proj_norm.weight`
- `per_layer_token_embd.weight`

The active combination path is in
[real_forward.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/real_forward.rs#L1097).

That means the older diagnosis that Gemma coherence is blocked because the
per-layer auxiliary path is missing is no longer accurate for this branch.

### 4. The sketch/carry machinery is real code, but not a real-forward dependency

The heavy typed carry surface in `lib.rs` is still there:

- `StageCarryState`
- `StageTransientState`
- `StageCarryPolicy`
- transfer frames
- resume contracts
- provenance / stale-limit logic
- `PackedResidencySketchBackend`

That machinery is exercised by the sketch backend in:

- [lib.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/lib.rs#L6624)

It is not on the defended real-forward path.

## Production Contract From Here

The production staged contract should be treated as:

1. Head stage tokenizes prompt ingress.
2. Head stage produces full hidden-state tensors plus compact aux bytes.
3. Intermediate stages consume hidden-state tensors plus aux bytes and produce
   the same shape.
4. Tail stage owns logits and sampling.
5. Each stage persists its own per-request decode state locally across decode
   steps.

That last point matters. For autoregressive generation, the stage boundary does
not eliminate the need for downstream KV or equivalent cached attention state.
The current 2-stage real path already has local decode session storage in
[real_forward.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/real_forward.rs#L140)
and tail-side continuation reuse in
[real_forward.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/real_forward.rs#L1888).

So the real question is not whether stage-local decode state exists. It does.
The question is whether the same invariant is preserved cleanly once real
`N > 2` artifacts exist.

## Keep, Quarantine, Retire

### Keep as shared production-facing primitives

These are still worth treating as real shared infrastructure:

- `StageForwardBackend` trait in
  [lib.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/lib.rs#L4214)
- packed stage artifact/index/store types and tensor classification in
  `crates/stage-forward-lab/src/lib.rs`
- `StageLayout`
- byte framing helpers for hidden-state plus aux payloads
- prompt suite definitions in
  [prompt_suite.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/stage-forward-lab/src/prompt_suite.rs)
- the real-forward chain compare harness in
  [real_stage_chain_compare.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/compute-daemon/src/bin/real_stage_chain_compare.rs)

### Quarantine as research/lab-only for now

These should not shape production contract decisions unless the real-forward
path starts consuming them explicitly:

- `DeterministicStubBackend`
- `ToyLinearBackend`
- `ArtifactBackedToyBackend`
- `PackedResidencySketchBackend`
- typed carry / transient / resume-contract machinery in
  `crates/stage-forward-lab/src/lib.rs`

This code can stay in-tree for now. It is still useful for experiments and
isolated tests. But it should be treated as research-side code, not as the
default template for extending `real_forward`.

### Retire later if no real-forward consumer appears

If the real-forward path still does not consume typed carry/transient state
after real `3+` stage validation and decode-session audits, the likely next
cleanup is:

1. move sketch-specific transfer/carry machinery behind a clearly named module
2. stop exporting it as if it were part of the production contract
3. delete unused pieces once references are gone

Not today. But that is the likely direction.

## Rules For New Work

From this point on:

1. No new production feature should depend on `StageCarryState`,
   `StageTransientState`, or resume-contract machinery unless there is an
   explicit real-forward consumer.
2. No new `N > 2` architectural claim should be made without real artifacts and
   real-forward parity.
3. No new benchmark should be treated as meaningful if it only exercises sketch
   backends.
4. Tail-only ownership of logits and sampling remains a hard interface rule.
5. Prompt tokenization stays at ingress/head; downstream stages do not
   re-tokenize prompts as part of the production contract.

## Immediate Checklist

This is the one-page checklist for "what must be true before coherent real
`N > 2` matters."

- [x] Real 2-stage local prompt-suite parity
- [x] Real 2-stage transport-path parity
- [x] Shared prompt suite for local and networked validation
- [x] Generic real-forward chain compare harness
- [x] Real `3+` stage Gemma artifacts exist
- [ ] Real `3+` stage prompt-suite parity passes
- [ ] Per-stage decode-session / KV persistence is audited for real `N > 2`
- [ ] Cold and warm generation metrics are re-run on the validated `N > 2` path
- [ ] Sketch-side carry/contract exports are either justified by a real-forward
      consumer or explicitly quarantined

## Recommended Next Actions

In order:

1. Freeze sketch-side contract expansion.
2. Treat current prototype chain tooling as "enough."
3. Keep Gemma E4B focused on the defended 2-stage path until there is a real
   need to reopen `N > 2`.
4. Harden 2-stage generation quality, cache behavior, and performance on the
   validated prompt suite.
5. Only reopen real `N > 2` work when there is:
   - a model that actually requires it
   - or an explicit plan for cross-stage state such as shared-KV handoff
6. Only then decide whether any typed carry surface belongs in the production
   contract.

## Newly Confirmed `N > 2` Blocker

Real `3`-stage artifacts now exist for:

- `0-13`
- `14-27`
- `28-41`

The current `real_stage_chain_compare` result is important:

- the checked `0-13 / 14-27 / 28-41` split is now rejected at layout load
- the explicit failure is:
  - `layer 28 requires shared KV from layer 22 outside this stage`
  - `current contract keeps shared-KV caches stage-local`

That means:

- the compare harness is doing its job
- transport parity is not the blocker
- the remaining blocker is an explicit real-forward split constraint for this
  model shape

The underlying cause is Gemma E4B's shared-KV attention pattern in
`real_forward`, where later layers can depend on source-layer caches rooted at
layers `22` and `23`. A boundary that places those source layers in one stage
and layers `24+` in a later stage is not a valid production split under the
current stage-local cache contract.

## Bottom Line

The repo is in a better place than the older worklog implies on real Gemma
correctness, and in a worse place than the worklog implies on contract
clarity.

The right move now is not to extend the sketch contract surface. It is to force
all new staged-inference decisions through the real-forward path until the code
either proves a need for shared carry machinery or makes it safe to retire.
