# Stage Backend Feasibility — April 2026

## Summary

The current multi-node stage prototype is **transport-real but math-fake**.

That is now the main constraint.

The orchestrator assignment schema, daemon stage bootstrap, QUIC stage transport, and
2-node LAN roundtrip harness are all good enough to keep. The current `llama.cpp`
backend path is not yet a real stage execution backend because it does not expose a
hidden-state forward API that matches Compute's stage-based design.

## What Is Working

- shard-aware assignment schema
- explicit `stage` assignment mode
- fixed 2-stage Gemma prototype scheduling
- daemon stage runtime bootstrap
- QUIC transport between head and tail
- local 2-node roundtrip harness
- experimental non-streaming relay entry into the head-stage prototype path

## What Is Not Real Yet

The current stage execution path does **not** perform a true forward pass on hidden
states between stage boundaries.

Local code evidence:

- [`crates/compute-daemon/src/inference/llamacpp.rs`](/Users/macintosh/Documents/projects/Compute/compute-app/crates/compute-daemon/src/inference/llamacpp.rs)
  explicitly says the activation path is placeholder and currently just passes data
  through in `get_activations()`.
- [`crates/compute-daemon/src/stage_runtime.rs`](/Users/macintosh/Documents/projects/Compute/compute-app/crates/compute-daemon/src/stage_runtime.rs)
  currently uses the full local GGUF as a shard stand-in and relies on the
  placeholder `forward()` behavior from the engine.

So today:

- transport behavior is useful to test
- stage correctness is not
- performance numbers from this path would be misleading

As of the latest runtime refactor, the stage wire contract now explicitly separates:

- `PromptV1` for request ingress into the head stage
- `HiddenStatesV1` for inter-stage payloads

That is an important architectural improvement because the runtime is no longer
pretending that prompt JSON is itself a hidden-state tensor. It is still not real
model-stage math, but it is now the correct boundary for a future true-forward backend.

## Why The Current llama.cpp Path Is Not Enough

Official `llama-server` documentation currently advertises:

- chat/completions
- embeddings
- reranking
- tokenization helpers
- monitoring

But there is no documented hidden-state `/forward`-style endpoint for feeding one
stage's activations into the next stage's layers.

Sources:

- `llama-server` feature and endpoint documentation:
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- The server README documents `/embedding`, `/embeddings`, `/completion`,
  `/v1/chat/completions`, `/tokenize`, `/detokenize`, `/slots`, and related endpoints,
  but not a hidden-state forward endpoint.
- Official README also frames `llama-server` and RPC as serving/offload tools rather
  than an explicit stage-boundary activation API:
  https://github.com/ggml-org/llama.cpp

There is also evidence that the RPC path itself is still performance-sensitive and not
the architecture Compute wants long term:

- `llama.cpp` RPC utilization issue:
  https://github.com/ggml-org/llama.cpp/issues/15463
- RPC large-model instability issue:
  https://github.com/ggml-org/llama.cpp/issues/15055

These do not make RPC unusable, but they reinforce the same conclusion:
RPC is a useful comparison/prototype tool, not a clean foundation for Compute's
stage-owned public-node pipeline design.

## Decision

Do **not** keep expanding the current stage prototype as if the backend question is
already solved.

The next milestone should be a **backend feasibility spike**.

## Recommended Next Step

### Goal

Produce one **real** deterministic 2-stage LAN completion for `gemma-4-e4b-q4`.

### Required outcome

The stage backend must support:

- fixed contiguous layer ownership
- explicit local shard load
- forward pass from prior-stage hidden states
- token sampling on the tail stage
- deterministic output comparison against single-node baseline

### Options

#### Option A: Patch or extend llama.cpp for a real stage-forward endpoint

Pros:

- keeps Apple Silicon-friendly stack
- reuses current local serving/runtime work
- preserves GGUF pipeline

Cons:

- highest implementation risk
- requires custom backend/server work
- likely diverges from upstream and becomes a maintenance burden

#### Option B: Use a different backend for the stage prototype

Pros:

- fastest route to proving whether stage-based architecture is fundamentally viable
- separates architecture validation from `llama.cpp` limitations

Cons:

- adds backend complexity early
- may not match eventual deployment backend

#### Option C: Keep current transport prototype and stop before real stage math

Pros:

- no immediate backend work

Cons:

- does not answer the core technical question
- risks spending time on control-plane work without proving execution viability

## Recommendation

Use **Option A or B immediately**, and do not keep growing placeholder execution.

If the goal is fastest truth discovery:

- prefer the backend path that can produce one real 2-stage LAN completion soonest
- keep the existing transport/control-plane skeleton
- treat current `llama.cpp` placeholder forwarding as scaffolding only

## Practical Repo Impact

Short term:

- keep `stage_runtime.rs` and the transport harness
- keep scheduler/schema work already done
- do not benchmark stage performance yet

Next implementation slice:

1. isolate the stage execution backend behind a cleaner abstraction
2. spike a real stage forward path
3. compare single-node vs 2-stage deterministic output
4. only then wire the orchestrator path further

## Bottom Line

The multi-node prototype is at the right stopping point for plumbing.

The next step is no longer "more scheduler work" or "more transport work."
It is a backend reality check.
