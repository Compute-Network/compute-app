# Stage Real-Forward Breakthroughs And Dense Porting Guide

## Scope

Before extending multi-stage architecture work, also read
`docs/real-forward-production-boundary-audit-april-2026.md`. It defines the
current production boundary between `real_forward` and the sketch/carry
machinery in `lib.rs`.

For the post-CPU acceleration direction, also read
`docs/stage-acceleration-architecture-april-2026.md`. That document defines
the cross-platform execution target matrix and the rule that platform-specific
acceleration stays behind the same stage-local runtime contract.

For the current Gemma E4B effort, read this guide as a **2-stage production
guide**. `3+` stage work is future-model preparation, not the active delivery
goal for this model.

This document captures two things:

1. What materially worked in the Gemma 4 E4B two-stage real-forward effort.
2. How future agents should apply the same approach to other dense decoder-only models such as Qwen, LLaMA-family, Mistral-family, and similar non-MoE models.

This is not a generic transformer tutorial. It is a practical bring-up and optimization guide for this codebase.

Relevant implementation and bench files:

- `crates/stage-forward-lab/src/real_forward.rs`
- `crates/stage-forward-lab/src/real_math.rs`
- `crates/stage-forward-lab/src/quants.rs`
- `crates/stage-forward-lab/src/bin/real_two_stage_probe.rs`
- `crates/stage-forward-lab/src/bin/real_two_stage_bench.rs`
- `crates/stage-forward-lab/src/bin/real_projection_bench.rs`
- `crates/stage-forward-lab/src/bin/real_ffn_bench.rs`
- `crates/stage-forward-lab/src/bin/real_two_stage_generate.rs`
- `crates/stage-forward-lab/src/bin/real_two_stage_generate_bench.rs`
- `crates/stage-forward-lab/src/bin/real_two_stage_generate_sweep.rs`
- `crates/stage-forward-lab/src/bin/real_two_stage_generate_prefix_bench.rs`
- `crates/compute-daemon/src/inference/stage_backend.rs`
- `docs/stage-real-forward-handoff-april-2026.md`

## Current Defended State

Current defended behavior on the local split Gemma E4B artifacts:

- Prompt: `"The capital of France is"`
- Full uncapped two-stage output: `"Paris"`
- Selected token: `9079`
- Warm reused-backend `real_two_stage_bench`:
  - head: `min=136ms median=139ms avg=140ms max=148ms`
  - tail: `min=133ms median=138ms avg=138ms max=145ms`
  - sample: `min=14ms median=15ms avg=14ms max=15ms`
  - total: `min=283ms median=292ms avg=292ms max=307ms`
  - deterministic: `PASS`
- Warm profile average:
  - head: `attn=28ms`, `ffn=98ms (gate+up=52ms down=36ms)`, `ple=13ms`
  - tail: `attn=24ms`, `ffn=100ms (gate+up=52ms down=37ms)`, `ple=13ms`

Current isolated FFN benchmark on real stage-1 layer 0, six inputs:

- norm: `~19us avg`
- gate+up: `~2590us avg`
- activation: `~492us avg`
- down: `~2161us avg`
- total: `~5429us avg`

Current generation status:

- the real staged path now has a first correct greedy autoregressive decode loop
- that loop uses exact token-id prompt ingress on the head stage
- first token seeds per-stage local decode caches, and continuation reuses them
- KV/cache reuse is now present in the local staged path, though generation is
  still greedy-only and should be benchmarked separately from next-token TTFT
- decode sessions are explicitly cleared after each generation run instead of
  only relying on the bounded in-memory session cap
- stop strings now work end-to-end in the daemon path and in the direct
  packed-artifact generators, with `finish_reason` reported as `stop` or
  `length`

Example local generation run:

- prompt: `"The capital of France is"`
- `max_tokens=3`
- generated text: `"Paris,a"`
- TTFT: `~1015ms`
- total: `~1807ms`
- continuation: `~2.53 tok/s`

Warm generation bench on the same prompt with `max_tokens=3`:

- TTFT: `min=301ms median=316ms avg=315ms max=330ms`
- total: `min=1014ms median=1047ms avg=1045ms max=1075ms`
- continuation: `min=2.68 tok/s median=2.68 tok/s avg=2.74 tok/s max=2.87 tok/s`

Warm stop-aware generation bench on the same prompt with `max_tokens=8` and
stop sequence `","`:

- TTFT: `min=286ms median=288ms avg=287ms max=288ms`
- total: `min=628ms median=628ms avg=629ms max=631ms`
- continuation: `min=2.92 tok/s median=2.92 tok/s avg=2.93 tok/s max=2.94 tok/s`
- finish reason: `stop`
- output: `"Paris"`

Prompt-length generation sweep on the same loaded backends with `max_tokens=4`
and no stop sequences:

- short, 6 prompt tokens:
  - TTFT: `min=17ms median=19ms avg=18ms max=19ms`
  - total: `min=390ms median=398ms avg=398ms max=408ms`
  - continuation: `0 tok/s` because this case stops after the first generated token
- medium, 15 prompt tokens:
  - TTFT: `min=18ms median=18ms avg=18ms max=19ms`
  - total: `min=1156ms median=1157ms avg=1158ms max=1162ms`
  - continuation: `min=2.62 tok/s median=2.63 tok/s avg=2.63 tok/s max=2.64 tok/s`
- long, 47 prompt tokens:
  - TTFT: `min=26ms median=26ms avg=26ms max=27ms`
  - total: `min=1196ms median=1211ms avg=1207ms max=1216ms`
  - continuation: `min=2.52 tok/s median=2.53 tok/s avg=2.54 tok/s max=2.57 tok/s`

This is the current generation profile:

- head-stage exact-prefix prefill reuse now cuts warm TTFT materially on repeated prompts
- tail-stage exact full-prompt prefill reuse now removes the remaining warm prompt-length slope
- head-stage longest-shared-token-prefix reuse now gives a real keep for non-identical prompts
  that share a token prefix
- tail-stage opaque prefix-hint reuse now complements that on the downstream side
- on repeated exact prompts, TTFT is now close to a single continuation-step cost
- cold / first-seen prompts are still expensive and still grow with prompt length
- continuation throughput is materially flatter because the new decode caches
  avoid replaying the full prompt on every generated token
- the latest 2-stage cold-path keep extended the `>6 input` quantized chunk fast
  path to the nested-output batched matmul path in `real_math.rs`, which reduced
  first-token TTFT materially on the defended Gemma instruct sweep
- for repeated exact prompts, continuation is the main remaining cost
- for first-seen unrelated prompts, broader prefill reduction is still the next meaningful target

## Prompting And Generation Quality

The previous generation quality issues on medium/long prompts were mostly a
prompt-formatting problem.

The current defended generation path now uses an explicit Gemma instruct wrapper
through `crates/stage-forward-lab/src/prompting.rs`, and the daemon
`real_forward` generation path uses the same formatting by default through
`tokenize_generation_prompt`.

Important supporting fix:

- `crates/stage-forward-lab/src/tokenizer.rs` now prefers `<turn|>` as EOS when
  it exists, instead of always using `<eos>`

This changed local real packed-artifact behavior materially. A fixed prompt
sanity harness now exists in:

- `crates/stage-forward-lab/src/bin/real_prompt_sanity.rs`

Current real local sanity result in `gemma_instruct` mode:

- `france_one_word`
  - first token: `"Paris"`
  - continuation: `"Paris"`
- `france_exact_output`
  - first token: `"Paris"`
  - continuation: `"Paris"`
- `sky_blue_sentence`
  - first token: `"As"`
  - continuation: `"AsRayleigh scattering,called"`
- `sky_red_sentence`
  - first token: `"As"`
  - continuation: `"As when the sun is low"`
- `cache_reason_sentence`
  - first token: `"Because"`
  - continuation: `"Because the initial token generation requires a full"`
- overall: `PASS`

The practical lesson for future agents is straightforward:

- do not treat raw prompt text as the default quality path for instruct-tuned
  Gemma-family models
- lock a model-family-specific prompt mode first
- then evaluate first-token and short continuation behavior on the same fixed
  suite before resuming performance work

There is now a second validation layer for that suite:

- `crates/compute-daemon/src/bin/real_two_node_prompt_compare.rs`

This harness starts real head and tail stage runtimes on loopback using the
actual transport path, points them at the same packed artifacts through a
temporary stage root, and compares the two-node result directly against the
local packed-artifact reference.

Current real loopback two-node compare result:

- `france_one_word`
  - local: `"Paris"` / `[50429]`
  - two-node: `"Paris"` / `[50429]`
- `france_exact_output`
  - local: `"Paris"` / `[50429]`
  - two-node: `"Paris"` / `[50429]`
- `sky_blue_sentence`
  - local: `"AsRayleigh scattering,called"`
  - two-node: `"AsRayleigh scattering,called"`
- `sky_red_sentence`
  - local: `"As when the sun is low"`
  - two-node: `"As when the sun is low"`
- `cache_reason_sentence`
  - local: `"Because the initial token generation requires a full"`
  - two-node: `"Because the initial token generation requires a full"`
- overall: `PASS`

The suite is now tiered in `crates/stage-forward-lab/src/prompt_suite.rs`:

- `core`
  - `france_one_word`
  - `france_exact_output`
  - `sky_blue_sentence`
  - `sky_red_sentence`
  - `cache_reason_sentence`
- `all`
  - core plus:
    - `yes_exact_output`
    - `no_exact_output`
    - `kv_cache_exact_output`
    - `paris_comma_stop`

Current `all`-mode loopback result also passes:

- `yes_exact_output`
  - local: `"Yes"` / `[10784]`
  - two-node: `"Yes"` / `[10784]`
- `no_exact_output`
  - local: `"No"` / `[3771]`
  - two-node: `"No"` / `[3771]`
- `kv_cache_exact_output`
  - local: `"KV cache"` / `[57137, 15612]`
  - two-node: `"KV cache"` / `[57137, 15612]`
- `paris_comma_stop`
  - local: `"Paris"` / `[50429]`
  - two-node: `"Paris"` / `[50429]`
- overall: `PASS`

There is now also a standalone stage host in:

- `crates/compute-daemon/src/bin/real_stage_runtime_host.rs`

And `real_two_node_prompt_compare` can target that hosted tail through an
explicit downstream address instead of starting its own in-process tail. That
deployment shape also passes on the full `all` suite.

This matters for future agents because it gives a clean escalation path:

1. validate locally with packed artifacts
2. validate in-process loopback head/tail parity
3. validate cross-process remote-tail parity
4. only then move to true non-loopback two-machine validation

That means the prompt-quality validation stack now has two concrete layers:

1. local packed-artifact generation sanity
2. live loopback two-node transport-path parity against the same prompt suite

And practically, three operating modes:

1. `core` for fast routine validation
2. `all` for stronger exact/stop-sensitive validation
3. ad hoc probes only when bringing up a genuinely new behavior

Future agents should keep that structure. Do not resume performance work on the
staged path until both layers agree on the prompt family being tuned.

## Planning For More Than Two Stages

The current defended path is 2-stage because that is the smallest real split
that exercises:

- prompt ingress at the head
- hidden-state transport
- downstream continuation
- token return to the head

That is the correct first milestone, but it is not the likely final deployment
shape for larger models. Future agents should assume that some models will need
`3+` stages.

What already generalizes reasonably well:

- stage specs use `stage_index`, `total_stages`, and layer ranges
- stage hosts are not inherently tied to only one specific layer span
- the transport protocol already moves activations/tokens rather than assuming a
  monolithic model server

What does **not** yet generalize cleanly:

- compare tooling still assumes one head and one downstream tail
- downstream connectivity is modeled as a single peer in the current prototype
- validation output is split as head vs tail, not as an ordered stage chain
- cache/prefill reuse policy has only been defended on the current 2-stage split

So future work should follow this order:

1. keep 2-stage parity green as the baseline
2. add an ordered `Vec<StagePrototypeSpec>` / address-list style orchestration
   path for validation tooling
3. add chain-aware prompt-suite comparison for `3+` stages
4. then optimize N-stage routing / cache reuse / timing only after parity is
   established

The main engineering rule here is simple:

- do not let the current 2-stage validation success turn into accidental
  head/tail-only interfaces

Treat head/tail as the defended base case, not the architectural limit.

There is now a defended prototype relay step in that direction:

- intermediate stages in the prototype runtime can now forward activations
  downstream and relay the final token response back upstream
- this is covered in
  `crates/compute-daemon/tests/stage_prototype_integration.rs`

Current covered prototype chain shapes:

- 3-stage roundtrip:
  - `0-9 -> 10-19 -> 20-27`
- 4-stage roundtrip on catalog-derived Gemma 4 E4B ranges:
  - `0-10 -> 11-21 -> 22-31 -> 32-41`

Future agents should treat that as a runtime-shape proof, not as full real-model
multi-stage readiness. It proves relay and ordered stage chaining in the
prototype path, but it does not yet prove real-forward packed-artifact parity or
performance for `N > 2`.

There is now reusable prototype-chain tooling as well:

- `start_stage_prototype_chain(...)` in
  `crates/compute-daemon/src/stage_runtime.rs`
- `crates/compute-daemon/src/bin/prototype_stage_chain_roundtrip.rs`
- `crates/compute-daemon/src/bin/prototype_stage_runtime_host.rs`
- `crates/compute-daemon/src/bin/prototype_stage_chain_client.rs`

Use those before adding more one-off chain setup in tests. The intended order is:

1. derive stage ranges from the model catalog
2. prove a prototype `N`-stage chain with the roundtrip bin
3. prove an externally hosted chain with the client/host bins
4. only then start porting the same chain shape to real-forward artifacts

Current defended external prototype-chain proof:

- 3 hosted stage processes:
  - head `0-9`
  - middle `10-19`
  - tail `20-27`
- external client connected to the hosted head over the transport path
- prompt: `"Hello"`
- `max_tokens=8`
- result:
  - finish reason: `length`
  - token ids: `[80, 114, 111, 116, 111, 116, 121, 112]`
  - content: `"Prototyp"`

That is still prototype behavior, not real-forward quality. But it is the
right next-level proof that chain execution is not confined to in-process test
harnesses.

There is now also a proper external parity harness for that shape:

- `crates/compute-daemon/src/bin/prototype_stage_chain_compare.rs`

It compares:

- a local in-process prototype chain as the reference
- an externally hosted prototype chain as the target
- the shared prompt suite from `stage-forward-lab`

Current defended result on the shared `core` suite for a hosted 3-stage chain:

- `france_one_word` -> PASS
- `france_exact_output` -> PASS
- `sky_blue_sentence` -> PASS
- `sky_red_sentence` -> PASS
- `cache_reason_sentence` -> PASS
- overall -> PASS

Future agents should use this before claiming that an externally hosted
prototype chain is equivalent to the local reference. Do not skip straight from
ad hoc client calls to real-forward multi-stage work.

There is now a matching real-forward-side preparation harness:

- `crates/compute-daemon/src/bin/real_stage_chain_compare.rs`

It takes explicit stage specs in `path@start-end` form, then compares:

- a local direct real-forward chain as the reference
- a runtime real-forward chain started from the same stage specs
- the shared prompt suite and prompt mode

Current defended result on the checked-in 2-stage Gemma artifacts:

- `packed-stage-1/stage-1-required.index.json@0-20`
- `packed-stage-2/stage-2-required.index.json@21-41`
- prompt mode: `gemma_instruct`
- suite: `core`
- all 5 core cases: `PASS`

That does not mean we have real-forward `N > 2` parity yet. We do not, because
the currently available real packed artifacts are still only the 2-stage split.
What it does mean is that the real-forward validation tooling is now shaped for
arbitrary ordered stage specs, so the next real `3+` stage artifact set can be
plugged into the same harness instead of needing another one-off compare path.

Updated real packed-artifact sweep in `gemma_instruct` mode with `max_tokens=4`:

- short / 25 prompt tokens
  - cold: `ttft 2633ms`, `total 3037ms`
  - warm: `ttft min=18ms median=19ms avg=18ms max=19ms`
  - warm total: `min=415ms median=432ms avg=428ms max=437ms`
  - output: `"Paris"`
- medium / 27 prompt tokens
  - cold: `ttft 1773ms`, `total 2962ms`
  - warm: `ttft min=19ms median=20ms avg=20ms max=21ms`
  - warm total: `min=1204ms median=1211ms avg=1222ms max=1251ms`
  - output: `"AsRayleigh scattering"`
- long / 59 prompt tokens
  - cold: `ttft 4556ms`, `total 5785ms`
  - warm: `ttft min=25ms median=26ms avg=25ms max=26ms`
  - warm total: `min=1230ms median=1257ms avg=1270ms max=1323ms`
  - output: `"Because reusing the KV"`

That means the current branch is no longer in the state where only the narrow
France probe works. The local packed-artifact generator now produces sane,
deterministic outputs on a small fixed suite under the right prompt mode.

The generation sweep now also prints a cold TTFT split and first-token stage
profiles. Current real-artifact signal:

- short / 6 prompt tokens:
  - TTFT `872ms = head 427ms + tail 383ms + sample 62ms`
  - head cold profile: `attn 56ms, ffn 223ms, ple 131ms`
  - tail cold profile: `attn 42ms, ffn 210ms, ple 126ms`
- medium / 15 prompt tokens:
  - TTFT `1234ms = head 626ms + tail 593ms + sample 15ms`
  - head cold profile: `attn 181ms, ffn 395ms, ple 28ms`
  - tail cold profile: `attn 161ms, ffn 400ms, ple 29ms`
- long / 47 prompt tokens:
  - TTFT `3748ms = head 1920ms + tail 1810ms + sample 18ms`
  - head cold profile: `attn 767ms, ffn 1007ms, ple 83ms`
  - tail cold profile: `attn 695ms, ffn 1027ms, ple 82ms`

The cold-path keep behind that change is in `real_math.rs`:

- large token-major quantized batched matmuls now split large input-token
  batches into `6+6+...+remainder` chunks when the workload is large enough
- that reuses the exact six-input `Q4_K` / `Q6_K` fast paths on medium/long
  prompt prefill instead of falling back to the generic `>6 input` path
- focused `>6 input` correctness tests compare the new path against stitched
  single-input outputs

That changed the practical conclusion:

- on first-seen prompts, sample time is still noise
- the time is still roughly split between head and tail
- FFN is still the largest single cold bucket
- but attention is now much closer on long prompts
- so the next real cold-path target is prompt-prefill compute on both stages,
  not logits and not PLE micro-optimizations

Shared-prefix warmup now has its own real judge in `real_two_stage_generate_prefix_bench`.
Current defended signal on the packed artifacts with `max_tokens=1`:

- short shared-prefix probe: `846ms -> 451ms`
- medium shared-prefix probe: `1113ms -> 827ms`
- long shared-prefix probe: `4641ms -> 507ms`

The important lesson is that this keep works because the tail now receives an
explicit opaque token-prefix identity and reuses cached full-prompt KV/state by
truncating to the shared prefix, while the head no longer requires the entire
cached prompt token sequence to be a prefix before it can reuse work. The
earlier hidden-state-equality matcher was mixed and was correctly reverted.

This means:

- correctness is in good shape on the defended uncapped path
- PLE is no longer a meaningful warm bottleneck
- logits/sampling are cheap
- attention is smaller than FFN
- paired `Q4_K gate+up` remains the largest steady-state kernel

## Main Advantages And Breakthroughs

### 1. The pipeline is now correct on real local artifacts

This is the biggest functional breakthrough.

The branch moved from nonsense outputs and shape/race bugs to deterministic correct next-token behavior on a real split model. The important point is not just that the code runs. It now runs correctly on the actual packed-stage artifacts, locally, without needing two separate machines.

Key correctness breakthroughs:

- full prompt hidden-state sequence is forwarded across the stage boundary, not only the last token
- tail-side layer execution uses prompt token positions correctly
- Gemma 4 hybrid attention behavior is modeled explicitly instead of treated as a generic transformer
- head dimension and attention layout are inferred from real tensor shapes and norms rather than guessed
- shared-KV later-layer behavior is handled
- the uncapped real path is the correctness judge, not capped smoke modes

Practical result:

- `"The capital of France is"` -> `"Paris"`
- additional real prompt-set checks also pass locally on one machine

### 2. The branch now has reliable single-machine real-artifact tests

This is a major operational advantage for future work.

The codebase now has ignored real-artifact regressions that:

- load the real local stage-1 and stage-2 packed artifacts
- run the two-stage path on one machine
- assert deterministic real outputs

This matters because it removes dependency on two systems for core regression testing.

Important tests:

- `local_real_e4b_two_stage_output_matches_paris_if_artifacts_present`
- `local_real_e4b_two_stage_outputs_match_small_prompt_set_if_artifacts_present`

These are the right place to catch correctness drift after optimization work.

### 3. We now have trustworthy measurement layers instead of noisy guessing

This was a major methodology breakthrough.

The work is no longer driven by vague whole-model timing alone. It now has three concrete measurement layers:

1. `real_two_stage_probe`
   - fresh-ish two-stage end-to-end signal

2. `real_two_stage_bench`
   - warm reused-backend steady-state signal
   - this is the best judge for user-facing repeated inference latency

3. `real_projection_bench` and `real_ffn_bench`
   - isolated packed-tensor kernel signal
   - this is the best judge for quantized projection work

This changed the optimization process materially. Many plausible changes were rejected because the isolated bench or warm whole-model bench showed they were noise or regressions.

### 4. Quantized projection work is now much more efficient

This is the main runtime breakthrough.

The codebase no longer pays the old cost model of repeatedly dequantizing large matrices just to use them once. The current path relies on streamed, batched, row-dot-style quantized execution.

Major kept wins included:

- streamed quantized layer projections instead of full matrix dequantization
- streamed logits scoring
- paired `Q4_K gate+up`
- multi-row `Q4_K` and `Q6_K` row-dot helpers
- unchecked indexing after up-front validation in the hot loops
- explicit `mul_add` in the hot accumulation paths
- targeted `#[inline(always)]` on the hot helper layers
- pointer-read hot loops for the six-input `Q4_K` path

This is what drove warm reused-backend behavior down into the current defended `~568ms` median band.

### 5. PLE cold-path cost was isolated and overlapped away

Another important systems breakthrough:

- PLE setup was split into concrete buckets: lookup / project / combine / materialize
- shard load now prewarms the PLE tensors that matter for first-request latency
- the code lazily resolves the large PLE tensors instead of eagerly decoding them at load

The result is better cold behavior without polluting the warm path.

### 6. Privacy on the stage transport is better than it was

This is not full node privacy, but it is still a real improvement.

Kept transport/privacy hardening:

- stage-to-stage hidden-state payloads no longer forward plaintext prompt text on the real path
- stage token transport no longer forwards plaintext completion text by default
- prototype hidden-state payloads no longer derive synthetic bytes directly from raw prompt bytes
- prototype tail completions no longer echo prompt text

This does not solve the harder problem that the executing head/tail node still sees the data it must compute on. But it removed avoidable plaintext leaks from transport and prototype behavior.

### 7. The branch is disciplined about rejecting fake wins

This is one of the most important process improvements for future agents.

Many ideas looked reasonable and were rejected:

- naive `mmap` path
- dense GEMM replacement for PLE
- chunked FFN contract
- several slab/layout rewrites
- several loop unroll and pointer-shape rewrites
- various wrapper and fill changes

That matters because the current branch is not carrying optimization folklore. It is carrying changes that survived the real local probes and benches.

## What Actually Moved The Needle

The following categories produced real, repeatable wins:

### Correctness-first model bring-up

Nothing downstream mattered until the uncapped real path produced correct output.

Useful order:

1. tokenization/detokenization works
2. one-layer math is sane
3. full-sequence stage handoff is correct
4. full uncapped next-token output is correct
5. only then optimize

### Model-specific architecture handling

Generic transformer assumptions were not enough.

What had to be modeled explicitly:

- hybrid attention patterns
- head-dimension derivation
- RoPE details
- per-head norm behavior
- shared-KV behavior
- logits / output head selection

### Focus on quantized row-dot kernels, not generic refactors

The real warm bottleneck ended up in quantized FFN projection kernels, especially paired `Q4_K gate+up`.

That means:

- big wins came from kernel math and hot-path load behavior
- small wins did not come from broad refactors
- many innocent-looking structural changes regressed the branch

### Use isolated benches as gates

The fastest way to waste time here is to optimize on whole-model noise.

Use:

- `real_ffn_bench` to judge FFN kernel work
- `real_projection_bench` to judge projection path work
- `real_two_stage_bench` only after an isolated win survives

## What To Reuse For Future Dense Model Ports

If the future target is a dense non-MoE decoder-only model, reuse this approach.

Good candidates:

- Qwen dense variants
- LLaMA-family dense variants
- Mistral-family dense variants
- similar RoPE/RMSNorm/GQA-style decoder-only models

Bad first targets:

- MoE models
- recurrent/state-space hybrids
- models whose main speed/value depends on expert sparsity

## Porting Guide For Future Agents

### 1. Start by deciding whether the model is in-scope

A model is a good candidate if it is:

- dense
- decoder-only
- causal attention
- tensor layout can be packed into the current stage format
- no expert-routing runtime

A model is not a good next target if it needs:

- expert routing
- sparse expert paging as a core runtime feature
- fundamentally different state handling

### 2. Create a model-specific backend, but keep the generic pieces generic

Use the Gemma work as the template:

- create `Real<Model>Backend` analogous to `RealGemmaBackend`
- keep shared math in `real_math.rs` and `quants.rs`
- keep architecture-specific policy in the backend and layer config path

The model backend should own:

- tokenizer integration
- model config loading
- tensor name mapping
- family-specific attention rules
- logits head selection
- stage payload encode/decode rules

Do not fork the math stack unless the model truly requires it.

### 3. Build a minimal architecture checklist before writing fast code

For any dense target, answer these first:

- hidden dim
- FFN dim
- number of heads
- number of KV heads
- actual head dim
- RMSNorm vs LayerNorm
- exact FFN activation
- RoPE base theta
- RoPE scaling variant
- rotary dimension
- whether packed rope frequencies should be used
- any sliding-window attention
- any alternating full/sliding pattern
- any shared-KV or grouped-layer reuse
- attention softcaps or logits softcaps
- output projection tensor name and shape
- tokenizer type and vocab handling

Future agents should not assume Qwen or any other family matches Gemma.
They should verify the model's real config and tensor layout first.

### 4. Bring the model up in the smallest possible correctness steps

Recommended sequence:

1. load the packed tensors
2. verify tokenizer round-trip
3. verify logits tensor selection
4. verify one matrix multiply and one norm path
5. verify a single-layer step with a debug probe
6. verify head stage output payload shape
7. verify tail consumes full sequence context
8. verify full uncapped next-token output on one prompt
9. add at least one ignored real-artifact regression
10. only then optimize

Good supporting tools already in the repo:

- `real_forward_probe`
- `forward_debug_probe`
- `layer_step_probe`
- `real_two_stage_probe`

### 5. Treat stage-boundary semantics as part of correctness

The stage boundary is not a transport detail. It is part of model correctness.

Future ports should preserve the current lessons:

- forward full prompt hidden-state sequence when the tail needs sequence context
- preserve aux bytes and hidden-state framing correctly
- do not forward plaintext prompt text downstream unless there is no alternative
- do not rely on stage-local prompt reconstruction unless the path really requires it

### 6. Add the right tests immediately

For a new dense model, add:

1. architecture tests
   - layer config derivation
   - attention config behavior
   - any sliding/full/shared-KV pattern

2. stage payload tests
   - hidden-state transfer shape
   - prompt text elision on transport if applicable

3. local real-artifact ignored tests
   - at least one single-prompt correctness test
   - ideally a small prompt set

4. kernel equivalence tests
   - any new optimized kernel must match the scalar/reference path

Do not trust capped smoke modes as correctness proof.

### 7. Reuse the existing benchmark ladder

For any future dense port, bring these up early:

- `real_two_stage_bench`
- `real_projection_bench`
- `real_ffn_bench`

This is the right optimization ladder:

1. isolated projection/FFN bench
2. warm reused-backend whole-model bench
3. fresh two-stage probe

Keep changes only if they survive that order.

### 8. Optimization order for future dense ports

Do not start by micro-tuning random loops.

Use this order:

1. correctness on the uncapped real path
2. streamed quantized projections instead of full decode
3. paired projection paths where the model naturally has them
   - for example gate+up if both are same quant type and same shape
4. multi-row quantized row-dot helpers
5. hot-loop arithmetic cleanup
   - bounds checks
   - validated unchecked indexing
   - `mul_add`
   - helper inlining
6. logits scorer reuse
7. cold-path prewarm and lazy decode
8. only then smaller loop/address tweaks

This order worked on Gemma because it attacked the real cost centers first.

### 9. What future agents should expect for Qwen-like dense models

For dense Qwen-style models, expect the port to be closer to Gemma than to MoE systems, but do not assume exact compatibility.

Working assumptions for future agents:

- decoder-only transformer is plausible
- RoPE is likely still central
- RMSNorm-like behavior is likely
- GQA/MQA variants are possible
- FFN activation and tensor naming must be verified per model version
- attention and logits softcaps may or may not exist

What to verify instead of assuming:

- exact tensor names in the packed artifacts
- whether the logits head is tied or separate
- whether Q/K norms exist and how they are shaped
- whether rotary width is full or partial
- whether sliding attention exists
- whether KV heads differ from attention heads

The practical message: a dense Qwen-family port is likely a backend port, not a whole new runtime class.

### 10. Performance traps to avoid

These are patterns that looked promising and often were not:

- generic dense GEMM replacements for already-fast hot paths
- naive `mmap` assumptions
- chunking that increases total row-dot work
- loop unrolling without isolated bench proof
- output-layout churn that adds copies elsewhere
- removing initialization when downstream code still depends on overwritten assumptions

Future agents should assume that many small "obvious" tweaks will fail unless proven by the isolated benches.

### 11. Acceptance criteria for a future dense-model port

A future dense port should not be considered done until it has all of the following:

1. full uncapped local real-artifact prompt produces the expected next token
2. at least one ignored local real-artifact regression passes
3. stage payloads are shape-correct and do not carry avoidable plaintext
4. `real_two_stage_bench` is deterministic
5. `real_ffn_bench` or equivalent identifies the real warm bottleneck
6. all optimization keeps are justified by isolated bench plus whole-model bench

### 12. What future agents should document while porting

Future agents should maintain a rolling handoff like the current Gemma one and keep it factual:

- what changed
- what moved in the isolated bench
- what moved in the warm whole-model bench
- what was rejected
- what correctness risks remain

Do not let the branch become a pile of unverifiable folklore.

## Recommended Commands

### Core validation

```bash
cargo test -p stage-forward-lab --quiet
cargo check -p stage-forward-lab --bins --quiet
cargo check -p compute-daemon --quiet
```

### Real warm whole-model benchmark

```bash
cargo run --release -p stage-forward-lab --bin real_two_stage_bench -- \
  ../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json \
  ../compute-backend/out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json \
  "The capital of France is" \
  ../compute-backend/out/gemma-e4b-2stage/vocab.json \
  5
```

### Real isolated FFN benchmark

```bash
cargo run --release -p stage-forward-lab --bin real_ffn_bench -- \
  ../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json \
  0 \
  6 \
  8 \
  256
```

### Real ignored correctness regressions

```bash
caffeinate -is cargo test -p stage-forward-lab \
  local_real_e4b_two_stage_output_matches_paris_if_artifacts_present \
  -- --ignored --quiet

caffeinate -is cargo test -p stage-forward-lab \
  local_real_e4b_two_stage_outputs_match_small_prompt_set_if_artifacts_present \
  -- --ignored --quiet
```

## Bottom Line For Future Agents

The most important lessons are:

- get the uncapped real path correct first
- build the right benches before optimizing
- keep model-family behavior explicit
- focus on the real hot kernel, not nearby folklore
- reject changes that do not survive the isolated bench and the warm whole-model bench

For dense non-MoE models, this codebase is now in a good place. The Gemma work is not just a one-off. It is a reusable bring-up and optimization pattern.
