# Stage Acceleration Architecture - April 2026

## Purpose

The current `real_forward` backend is the defended correctness path, but its
warm continuation rate on Apple Silicon is still far below `llama.cpp` because
it runs the hot stage math on the CPU. This document defines the acceleration
direction without turning the project into a Metal-only fork.

## Rules

1. `real_forward` remains the production contract.
2. CPU `real_forward` remains the reference implementation and parity oracle.
3. Any accelerated backend must preserve the existing stage boundary:
   - prompt/token ingress on the head stage
   - hidden-state bytes between stages
   - per-stage decode-session / KV persistence across continuation steps
   - tail-only logits and sampling
4. No platform-specific backend may become the public contract. Metal, CUDA,
   Vulkan, and DirectML are execution targets behind one stage-local runtime
   contract.

## Execution Model

The project should converge on a two-layer design:

- Layer 1: stage runtime / transport / prompt-suite validation
- Layer 2: stage-local execution provider

Layer 2 is where acceleration belongs. The stage runtime should be able to
load the same packed stage artifact into one of these provider families:

- `cpu-ref`
  - current Rust `real_forward`
  - correctness oracle
  - fallback on every platform
- future accelerated provider
  - same stage contract
  - platform-specific kernel target underneath
  - intended implementation path: `ggml`-family kernels / device backends

That provider boundary now exists in the daemon runtime:

- `crates/compute-daemon/src/inference/stage_backend.rs`
- `crates/compute-daemon/src/inference/real_forward_artifact.rs`
- `crates/compute-daemon/src/inference/real_forward_provider.rs`
- `crates/compute-daemon/src/inference/real_forward_provider_ggml.rs`
- `crates/compute-daemon/src/bin/real_forward_provider_probe.rs`
- `crates/compute-daemon/src/bin/real_forward_provider_compare.rs`

`RealForwardEngine` no longer has to own `RealGemmaBackend` directly forever.
It now talks to a stage-local provider interface, with the current
`cpu-ref` provider wrapping the defended Rust `real_forward` path.

## Cross-Platform Target Matrix

- macOS Apple Silicon
  - preferred target: `metal`
- Linux NVIDIA
  - preferred target: `cuda`
- Windows NVIDIA
  - preferred target: `cuda`
- Windows / Linux non-NVIDIA GPU
  - preferred target: `vulkan`
- Windows-specific fallback path if Vulkan is not viable
  - `directml`

The important constraint is that these are targets, not separate public APIs.

## What Is In Scope Now

The codebase now carries an explicit stage-acceleration plan in the daemon
config and backend startup path:

- `experimental.stage_acceleration = "auto"` by default
- `experimental.stage_acceleration_provider = "auto"` by default
- the runtime resolves a desired acceleration target from hardware or explicit
  config
- the runtime resolves a desired provider family separately from the device
  target:
  - `cpu-ref`
  - future `ggml`
- the runtime loads `real_forward` through a stage-local provider boundary
- in `auto` mode, the active provider remains `cpu-ref` until an accelerated
  provider lands
- explicit non-CPU acceleration requests now fail cleanly instead of silently
  pretending CPU reference execution satisfies them

This is deliberate. It gives the project one execution-planning surface before
the first non-CPU provider is added.

It also gives the project two provider-facing tools before acceleration lands:

- `real_forward_provider_probe`
  - confirms provider selection / load behavior for one stage
- `real_forward_provider_compare`
  - compares a candidate provider against `cpu-ref` at the stage boundary
  - uses the same packed artifacts and prompt-suite cases
  - is the intended parity gate for the first accelerated provider

The load path is also now single-sourced through
`real_forward_artifact.rs`, which resolves packed stage directories, index
paths, tokenizer files, and stage layout into one provider-facing load spec.

The `ggml` provider family also now has its own module boundary in
`real_forward_provider_ggml.rs`. It is still unavailable by design, but it is
no longer just a string in the planner.

The important new constraint is now explicit in that provider module:

- the current real-forward packed stage artifacts are **not** directly
  consumable by stock `llama-server`
- stock `llama-server` is a runtime substrate, not yet a stage-local provider
  for this project
- a real `ggml` provider will need a custom stage worker contract with:
  - split-boundary hidden-state egress on non-tail stages
  - hidden-state ingress on downstream stages
  - stage-local KV/session persistence across continuation steps
  - tail-only sampling/logits behavior

That means the next `ggml` milestone is not "spawn llama-server against the
packed stage dir." It is "bridge a custom stage-local worker onto a `ggml`
runtime substrate without breaking the existing real-forward contract."

That worker contract is now also codified in
`crates/compute-daemon/src/inference/ggml_stage_worker.rs`, so the provider can
report the exact stage-role capabilities and operations the future worker must
implement.

That module now also builds a worker init spec from the resolved stage load
spec and runtime substrate plan. So the `ggml` provider can report:

- what the future worker must do
- and the exact stage/runtime/tokenizer inputs it would be initialized with

There is now also a dedicated bootstrap host target at
`crates/compute-daemon/src/bin/ggml_stage_worker_host.rs`. It is still
bootstrap-only, but it establishes the process boundary the future `ggml`
provider should launch instead of pretending stock `llama-server` is the
stage worker.

The worker bootstrap payload is now serialized with stable `snake_case` enum
names, so the future process boundary does not depend on Rust variant casing.

The first real end-to-end worker slice is now live for metadata-only
operations:

- `tokenize_text`
- `tokenize_generation_prompt`
- `decode_token_ids`
- `eos_token_id`

The dedicated validation path for that slice is:

- `crates/compute-daemon/src/bin/ggml_stage_worker_probe.rs`
- `crates/compute-daemon/src/bin/real_forward_provider_metadata_compare.rs`

There is now also a real execution-side worker gate:

- `crates/compute-daemon/src/bin/ggml_stage_worker_forward_compare.rs`
  - fresh-request head-stage `begin_token_ids` parity
- `crates/compute-daemon/src/bin/ggml_stage_worker_tail_compare.rs`
  - fresh-request tail-stage `continue_forward` parity

The next gate up from the raw worker bins is now live too:

- `crates/compute-daemon/src/bin/real_forward_provider_bootstrap_compare.rs`
  - validates the partial-capability `ggml-bootstrap` provider through the
    actual `RealForwardStageProvider` interface
  - uses fresh-vs-fresh reference execution per case
  - checks head `begin_token_ids`, tail `continue_forward`, tail `sample_tail`,
    and stage-shaped capability reporting on the defended 2-stage Gemma path

The `ggml` bootstrap worker wire path now also uses an explicit tensor response
envelope instead of inlining raw tensor bytes as giant JSON number arrays. That
keeps the public provider contract unchanged while making the worker transport
shape more realistic for large hidden-state payloads.

The bootstrap provider now also has a real persistent worker lifecycle:

- the provider lazily starts a long-lived localhost worker process on first use
- stage-local requests are sent over a framed TCP localhost channel
- the worker keeps a single `RealGemmaBackend` loaded for the life of that
  process
- `clear_decode_session` is now a real worker request, not just declared in the
  contract

That means `ggml-bootstrap` now owns per-stage decode-session state across
continuation steps, even though the worker is still using the CPU reference
backend internally.

There is now a dedicated continuation gate for that capability:

- `crates/compute-daemon/src/bin/real_forward_provider_bootstrap_continuation_compare.rs`
  - keeps the same request ID alive across repeated head/tail steps
  - validates exact sequence parity against `cpu-ref`
  - proves stage-local decode-session ownership through the provider seam

The worker execution core now also sits behind its own internal executor
boundary:

- `crates/compute-daemon/src/inference/ggml_stage_executor.rs`

Current behavior:

- the provider serializes the requested worker executor into the worker init
  payload
- the worker host builds its executor from that payload, not from process env
- the bootstrap provider now uses a mixed route:
  - metadata path -> `ggml-worker`
  - head execution path -> `ggml-worker`
  - downstream hidden-state forward -> `cpu-ref-worker`
  - tail sampling -> `ggml-worker`
- the `ggml-worker` branch is no longer metadata-only:
  - metadata ops are live
  - tail `sample_tail` is live
  - head `begin_token_ids` is live through the defended full-head `debug_layer_cap=21`
    route

That matters because the first real `ggml` executor can now be activated
through the same provider/worker contract that already passes metadata,
first-token, and continuation parity. The next actual backend swap is no
longer hidden behind `COMPUTE_GGML_STAGE_EXECUTOR`; it has a typed init field
and a dedicated worker-side executor seam.

There is now also a daemon-local validated stage binding manifest for the
future `ggml` executor:

- `crates/compute-daemon/src/inference/ggml_stage_manifest.rs`

It is built from `StageTensorStore::model_view()` in `stage-forward-lab`, so
the daemon is no longer guessing tensor names independently. The manifest now
records, and validates against the packed stage index:

- head-stage prompt/token ingress tensors
- shared positional / per-layer auxiliary tensors
- per-layer attention / FFN / projection tensors
- tail logits / output norm tensors
- unknown or extra tensor counts

The `ggml-bootstrap` provider now attaches that manifest summary to its load
error, which means the first real `ggml` executor can be implemented against a
single validated binding surface instead of scattered string lookups.

There is now also a typed operator plan derived from that manifest:

- `crates/compute-daemon/src/inference/ggml_stage_plan.rs`

It converts the validated packed-stage surface into a stage-role-aware operator
graph:

- shared ingress / positional / per-layer auxiliary bindings
- typed per-layer attention / FFN / projection bindings
- tail logits / output bindings

That means the first real `ggml-worker` execution op no longer needs to start
from raw tensor-name discovery. It can target a pre-validated operator plan.

That plan is now also exposed at the first two real execution boundaries:

- head `begin_token_ids`
  - emits a concrete `begin_plan=...` summary on the live `ggml-worker` error path
- downstream/tail `continue_forward`
  - emits a concrete `continue_plan=...` summary on the live `ggml-worker` error path

So both sides of the defended 2-stage split now fail against typed
execution-slice plans instead of only a generic unsupported-operation error.

There is now also a bound execution recipe on the worker path:

- it binds the existing `stage-forward-lab` execution-program order against the
  packed stage store
- it carries exact tensor names, GGML types, and dimensions per op
- the live `ggml-worker` error path now includes `recipe=...` alongside the
  op-specific plan

That means the first real `ggml-worker` compute op no longer needs to invent
its own layer-op ordering. It can reuse the validated execution-program order
already derived from the packed stage artifact.

There is now also a materialized execution recipe on the live worker path:

- it reads the bound packed-stage tensors from the `.pack` file
- it records stable byte hashes per tensor
- it reports unique tensor count and total bound tensor bytes for the staged
  execution slice

So the first real `ggml-worker` compute op now has:

- a validated op-specific plan
- a validated op order
- exact bound tensor entries
- verified byte-backed tensors on the live error path

There is now also one real executor-owned compute op behind `ggml-worker`:

- tail `sample_tail` no longer bails inside the bootstrap executor
- it runs from the typed `sample_tail` plan plus the packed-stage store
- it applies output RMS norm and logits projection directly from the bound
  stage tensors
- it matches `cpu-ref` on the defended Gemma 2-stage core suite both:
  - directly at the worker boundary
  - and through the `ggml-bootstrap` provider seam

There is now also one real head-side execution slice behind `ggml-worker`:

- head `begin_token_ids` now executes for `debug_layer_cap=0`
- that slice covers:
  - token embedding lookup/scaling
  - prompt-aux / PLE payload materialization
  - hidden-state payload framing
- it matches `cpu-ref` on the defended Gemma 2-stage core suite at the worker
  boundary, including:
  - hidden-state bytes/hash
  - prompt-aux bytes/hash
- tightening the compare gate to include aux bytes/hash caught a real bug:
  the first attempt emitted only stage-local PLE aux, while the reference head
  path emits full-model PLE aux coverage
- that worker-boundary parity now also holds for:
  - `debug_layer_cap=1`
  - `debug_layer_cap=21` on the defended `0-20` head stage

That is enough to prove fresh-request full-head parity at the worker boundary.
The missing piece was repeated-request continuation parity through the provider
seam.

So the defended `ggml-worker` boundary is now:

- metadata ops
- head `begin_token_ids` parity through full head depth
- downstream `continue_forward` parity through repeated continuation
- tail `sample_tail`

And the defended `ggml-bootstrap` provider routing is now:

- metadata -> `ggml-worker`
- head `begin_token_ids` -> `ggml-worker`
- downstream hidden-state forward -> `ggml-worker`
- tail `sample_tail` -> `ggml-worker`

That does not mean every op is already accelerated. The current worker split is:

- metadata ops run on the `ggml-worker` path
- head `begin_token_ids` now uses a real persistent `ggml` head-stack runtime
  for the full defended `0-20` head stage after token embedding / prompt-aux
  framing on the worker side
- tail `sample_tail` now has a real persistent `ggml` graph runtime and no
  longer relies on bootstrap CPU math when that runtime initializes
- with `debug_layer_cap=1`, the first tail layer in downstream
  `continue_forward` now has a real persistent `ggml` runtime path instead of
  bootstrap CPU matmuls
- with `debug_layer_cap=2`, that same real `ggml` path holds exact parity
  through the first two tail layers
- with `debug_layer_cap=4`, the same runtime now holds exact parity through the
  first four tail layers
- with no debug cap on the tail stage, the full tail `continue_forward` path
  now uses the same real persistent `ggml` tail-stack runtime by default
- with no debug cap on the head stage, the full head `begin_token_ids` path
  now also uses the real persistent `ggml` head-stack runtime by default

On Apple Silicon the live worker banner now reports a real Metal-backed sample
runtime, for example:

- `ggml-graph-sample backend=MTL0 hidden_dim=2560 vocab_size=262144`

The remaining blocker is no longer provider routing, session ownership, or
stage-layer hidden-state execution. It is the remaining CPU-side head ingress
work around the real graph path. Token embedding lookup, per-layer token
embedding lookup, and the PLE model projection now have defended `ggml`
runtime paths too; the remaining head-ingress gap is the CPU-side scaling,
normalization, payload framing, and surrounding worker/runtime overhead.

There is now also a dedicated head-ingress benchmark for that exact slice:

- `crates/compute-daemon/src/bin/ggml_stage_worker_head_ingress_bench.rs`
  - builds `cpu-ref-worker` and `ggml-worker` directly from the same worker
    init/load path with `debug_layer_cap=0`
  - measures head `begin_token_ids` ingress only, not host transport
  - reports per-case timing for:
    - token embedding gather
    - PLE token embedding gather
    - PLE model projection
    - PLE normalize/combine
    - prompt-aux encoding
    - hidden-state encoding
    - payload framing

Current measured result on the defended core suite (`metal`, `5` iterations):

- before batched PLE projection:
  - `cpu-ref-worker` head ingress average: about `2.6 ms`
  - `ggml-worker` head ingress average: about `23.1 ms`
  - hottest `ggml-worker` bucket: `ple_model_proj` at about `19.2 ms`
- after moving `per_layer_model_proj` onto a batched prompt-length `ggml`
  runtime:
  - `cpu-ref-worker` head ingress average: about `2.6 ms`
  - `ggml-worker` head ingress average: about `6.0 ms`
  - hottest `ggml-worker` bucket: `ple_model_proj` at about `2.1 ms`
- after flattening prompt-aux storage/encoding and rerunning the same bench in
  isolation:
  - `cpu-ref-worker` head ingress average: about `2.9 ms`
  - `ggml-worker` head ingress average: about `3.9 ms`
  - hottest `ggml-worker` bucket: `ple_model_proj` at about `2.3 ms`
- after fusing batched `per_layer_model_proj`, proj RMS norm, and token-embed
  combine into one persistent `ggml` PLE ingress graph:
  - `cpu-ref-worker` head ingress average: about `2.6 ms`
  - `ggml-worker` head ingress average: about `3.2 ms`
  - hottest `ggml-worker` bucket: fused `ple_model_proj` at about `2.5 ms`

That means the next optimization target is now evidence-based:

- first: the remaining fused PLE ingress graph cost
- then: prompt-aux normalize/combine work
- then: hidden-state encoding / payload framing / worker overhead

There is now also a live staged-runtime timing readout in the compare harness:

- `crates/compute-daemon/src/stage_runtime.rs` now returns `ttft_ms` and
  `total_ms` in `StagePrototypeResponse`
- `crates/compute-daemon/src/bin/real_stage_chain_compare.rs` now prints
  runtime TTFT, total time, continuation tok/s, and total tok/s for the actual
  `real_forward + metal + ggml` path

Current live signal on the defended 2-stage Gemma core suite is still much
slower than the local `cpu-ref` path, even after the head-ingress cuts:

- cold one-token case (`france_one_word`)
  - local: `ttft=2284ms total=2677ms`
  - staged runtime: `ttft=16545ms total=19719ms`
- warmed multi-token cases inside the same process:
  - `sky_blue_sentence`: `ttft=12099ms total=18925ms cont_tok_s=0.73`
  - `sky_red_sentence`: `ttft=11512ms total=18347ms cont_tok_s=0.73`
  - `cache_reason_sentence`: `ttft=17861ms total=27505ms cont_tok_s=0.73`

So the head-ingress microbench is no longer the whole story. The next
performance pass needs worker/runtime-level profiling across the full staged
request path, not just more cap-0 ingress micro-optimizations.

That worker/runtime profiling now exists at the head runtime boundary. On the
same `cache_reason_sentence` live staged case:

- before eager worker-session prewarm:
  - `tokenize=1771ms`
  - `head=23525ms`
  - `down_wait=9510ms`
  - `tail_engine=9074ms`
  - first step:
    - `head=17211ms`
    - `down_wait=6158ms`
    - `tail_engine=5789ms`
- after eager metadata/execution/sample session prewarm:
  - `tokenize=1ms`
  - `head=21799ms`
  - `down_wait=5882ms`
  - `tail_engine=5422ms`
  - first step:
    - `head=15478ms`
    - `down_wait=2587ms`
    - `tail_engine=2197ms`

So eager prewarm is worth keeping because it removes obvious lazy worker
startup from the first request, but it does not solve the main runtime
problem. The dominant remaining cost is still real head-stage compute on the
first prompt step, followed by real tail-stage compute.

There is now also a dedicated head-stage execution profile bin:

- `crates/compute-daemon/src/bin/ggml_stage_worker_head_execution_profile.rs`

On the defended head stage (`0-20`) for `cache_reason_sentence`, after a warm
executor bring-up, the real `ggml` head path first measured:

- total head execution: about `2.28 s`
- ingress before the stack: about `3.6 ms`
- payload encode after the stack: about `0.09 ms`
- per-layer times are flat, roughly `103-117 ms` each across all `21` layers

After adding executor-side caching for repeated F32 norm vectors and layer
output scales, the same warm profile dropped to about `2.05 s`, with ingress
still only about `4.0 ms` and payload encode about `0.25 ms`.

The profiled per-layer split on that warmed run is now explicit:

- attention CPU glue: about `6.9-12.9 ms` per layer
- attention ggml matmuls: about `19.2-24.6 ms` per layer
- FFN CPU glue: about `2.5-4.3 ms` per layer
- FFN ggml matmuls: about `45.7-51.2 ms` per layer
- PLE branch: about `16.0-16.3 ms` per layer

That matters because it rules out two simple bad hypotheses:

- the head stack is not blocked by one pathological layer
- the main remaining head cost is not prompt ingress or payload framing

It also changes the next optimization target. The dominant warm head cost is
now clearly the real `ggml` matmul path itself, especially FFN matmuls, and
the current prompt-prefill executor is still issuing those runtimes one token
at a time inside each layer.

The live chain numbers still barely move after that cache cut, so the dominant
remaining gap is broader per-layer stage compute, not just worker bring-up or
small tensor rereads.

There is now also a dedicated worker-boundary gate for that exact problem:

- `crates/compute-daemon/src/bin/ggml_stage_worker_head_continuation_compare.rs`
  - keeps one persistent head worker session alive across repeated
    `begin_token_ids` calls under the same request ID
  - drives next-token continuation from the local tail reference path
  - compares:
    - `cpu-ref` incremental head output
    - `ggml-worker` repeated-request head output
    - fresh full-history recompute for the same accumulated token history

Current finding from that gate:

- for repeated head continuation, `ggml-worker` now matches `cpu-ref`
  incremental hidden-state bytes
- `cpu-ref` incremental still differs from fresh full-history recompute on some
  repeated steps, which means the old failure mode was reference-side
  incremental-vs-fresh behavior, not a remaining `ggml-worker` continuation bug
- this holds at:
  - `debug_layer_cap=1`
  - `debug_layer_cap=21` on the defended `0-20` head stage

There is now also a provider-side continuation result that matters:

- `real_forward_provider_bootstrap_continuation_compare` now passes on the
  defended Gemma 2-stage core suite with:
  - fresh `cpu-ref` providers rebuilt per case on the reference side
  - persistent `ggml-worker` head execution across repeated steps
  - persistent `ggml-worker` downstream `continue_forward` execution across
    repeated steps

There is now also a layer-scoped tail gate for the next runtime swap:

- `ggml_stage_worker_tail_compare` now accepts `debug_layer_cap`
- `ggml_stage_worker_tail_continuation_compare` now accepts `debug_layer_cap`
- on the defended Gemma 2-stage path, `debug_layer_cap=1`, `2`, and `4` all
  pass for:
  - raw tail forward parity across the core suite
  - repeated tail continuation parity on the same worker session
- those passing capped gates are now running through a real persistent `ggml`
  tail-stack runtime on the worker side, not the old bootstrap CPU matmul path
- the default uncapped tail provider path now also passes through:
  - `real_forward_provider_bootstrap_compare`
  - `real_forward_provider_bootstrap_continuation_compare`
- the default uncapped head provider path now also passes through:
  - `real_forward_provider_bootstrap_compare`
  - `real_forward_provider_bootstrap_continuation_compare`

That means the full defended 2-stage Gemma provider seam now runs on the real
`ggml-worker` execution route for:

- head `begin_token_ids`
- downstream `continue_forward`
- tail `sample_tail`

The remaining acceleration gap is now the worker-side CPU ingress path around
the head graph, not the stage-layer hidden-state runtime itself.

## Planning Model

The acceleration planner now treats these as separate inputs:

- target preference
  - `auto`, `cpu`, `metal`, `cuda`, `vulkan`, `directml`
- provider preference
  - `auto`, `cpu-ref`, `ggml`

That split matters because Metal/CUDA/Vulkan/DirectML describe where execution
should land, while `cpu-ref` and future `ggml` describe how stage-local
execution is implemented.

## Immediate Next Milestones

1. Keep the current 2-stage Gemma `real_forward` path as the correctness gate.
2. Prototype a stage-local accelerated provider behind the same contract.
3. Swap the worker executor from `cpu-ref-worker` to the first real `ggml`
   executor while keeping the existing bootstrap compare gates green.
4. Extend the now-real head cap-`0` slice to:
   - head cap `1`
   - full head `begin_token_ids`
   - then tail `continue_forward`
5. Start that swap from the validated stage binding manifest rather than
   re-deriving tensor names inside the executor.
6. Keep aux parity in the worker gates; hidden-only parity is insufficient at
   the split boundary because downstream continuation depends on prompt-aux
   framing too.
7. Bring it up on one platform first, but do it through the shared stage
   acceleration plan rather than a Metal-only code path.
8. Reuse the existing prompt-suite and two-stage parity harnesses to validate
   the accelerated backend against the CPU reference path.
