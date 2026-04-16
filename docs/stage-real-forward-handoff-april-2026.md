# Stage Real-Forward Pipeline Handoff - April 2026

## Current Signal

Before more `N > 2` work, use
`docs/real-forward-production-boundary-audit-april-2026.md` as the contract
checklist for what is production-facing and what is still research-side
scaffolding.

For the acceleration direction beyond the current CPU reference backend, use
`docs/stage-acceleration-architecture-april-2026.md`. It defines the
cross-platform backend target matrix and keeps Metal/CUDA/Vulkan/DirectML
behind one stage-local runtime contract.

That runtime contract now exists in code: `RealForwardEngine` loads through a
stage-local provider boundary, with the defended Rust `real_forward` path
plugged in as the current `cpu-ref` provider.

There is now also a stage-local provider parity gate in
`crates/compute-daemon/src/bin/real_forward_provider_compare.rs`. Any future
accelerated provider should clear that stage-boundary compare against
`cpu-ref` before it is judged through the full runtime chain.

The stage runtime now also reports live completion timing:

- `crates/compute-daemon/src/stage_runtime.rs` returns `ttft_ms` and
  `total_ms` in `StagePrototypeResponse`
- `crates/compute-daemon/src/bin/real_stage_chain_compare.rs` now prints
  runtime TTFT, total time, continuation tok/s, and total tok/s for the live
  staged path

That surfaced the current end-to-end reality for `real_forward + metal + ggml`
on the defended 2-stage Gemma path:

- cold `france_one_word`: `ttft=16545ms total=19719ms`
- warmed multi-token continuation cases inside the same process:
  - `sky_blue_sentence`: `ttft=12099ms total=18925ms cont_tok_s=0.73`
  - `sky_red_sentence`: `ttft=11512ms total=18347ms cont_tok_s=0.73`
  - `cache_reason_sentence`: `ttft=17861ms total=27505ms cont_tok_s=0.73`

So the recent head-ingress improvements are real, but they have not yet turned
into acceptable end-to-end staged throughput. The remaining bottleneck is now
somewhere in the broader worker/runtime path, not just cap-0 ingress.

The next profiling cut made that more specific. The head runtime now reports
bucketed request timing, and on `cache_reason_sentence` it shows:

- before eager worker-session prewarm:
  - `tokenize=1771ms`
  - `head=23525ms`
  - `down_wait=9510ms`
  - `tail_engine=9074ms`
- after eager worker-session prewarm:
  - `tokenize=1ms`
  - `head=21799ms`
  - `down_wait=5882ms`
  - `tail_engine=5422ms`

The main conclusion is stable now:

- eager worker/session startup belongs in `load_shard()`
- but the remaining blocker is not lazy startup anymore
- the first prompt step is dominated by real head-stage compute, and the tail
  stage is still the next-largest live cost

The new head execution profile bin makes that sharper:

- `crates/compute-daemon/src/bin/ggml_stage_worker_head_execution_profile.rs`

On the defended head stage for `cache_reason_sentence`, after warm executor
bring-up:

- initial measured total head execution was about `2.28 s`
- after caching repeated F32 norm vectors and layer scales inside the executor,
  that dropped to about `2.05 s`
- ingress before the stack stayed tiny, around `3.6-4.0 ms`
- payload encode after the stack stayed tiny, around `0.09-0.25 ms`
- layer timings are flat across the whole head stack:
  roughly `74-111 ms` per layer for layers `0..20`
- the warmed per-layer split is now explicit:
  - attention CPU glue: about `6.9-12.9 ms`
  - attention ggml matmuls: about `19.2-24.6 ms`
  - FFN CPU glue: about `2.5-4.3 ms`
  - FFN ggml matmuls: about `45.7-51.2 ms`
  - PLE branch: about `16.0-16.3 ms`

So the next target should not be “find the one bad head layer.” The data says
the head stack is broadly uniform, and the live staged request path barely
moves even after this cache cut. The remaining first-step gap is more likely
prompt-length-specific stage compute overhead around the real head path. More
concretely: the current prompt-prefill executor is still paying the FFN and
attention `ggml` runtimes token-by-token inside each layer, so the next real
cut is batched head-layer prefill execution rather than another small CPU-side
ingress tweak.

Provider load resolution is now also single-sourced in
`crates/compute-daemon/src/inference/real_forward_artifact.rs`, so future
providers can consume one resolved stage load spec instead of duplicating
artifact/path/layout logic.

The acceleration planner now also separates:

- target preference: `metal/cuda/vulkan/directml/cpu`
- provider family: `cpu-ref/ggml`

That is intentional. The next non-CPU provider should land as a provider
family behind the same stage contract, not as a Metal-only fork.

The `ggml` family now also exists as its own module in
`crates/compute-daemon/src/inference/real_forward_provider_ggml.rs`. It is
still unavailable, but the first real implementation now has a dedicated
provider boundary instead of growing inside the generic factory.

That provider now also makes the main blocker explicit: the current packed
real-forward stage artifacts are not stock `llama-server` artifacts. A future
`ggml` provider can use `llama.cpp` / `ggml` as runtime substrate, but it still
needs a custom stage worker that honors the real-forward split contract:

- hidden-state egress at non-tail boundaries
- hidden-state ingress on downstream stages
- stage-local decode-session / KV persistence
- tail-only sampling

That worker surface is now codified in
`crates/compute-daemon/src/inference/ggml_stage_worker.rs` so the provider
error path reports both:

- why stock `llama-server` is insufficient
- what the future stage worker must actually implement
- what the worker would be initialized with once it exists

There is also now a bootstrap-only host bin at
`crates/compute-daemon/src/bin/ggml_stage_worker_host.rs`. It does not execute
the stage yet, but it gives the future `ggml` provider a real process target
for worker bring-up instead of another abstract planning layer.

That worker path now has a real metadata slice end to end:

- the host handles `tokenize_text`
- `tokenize_generation_prompt`
- `decode_token_ids`
- `eos_token_id`

and there is now a dedicated metadata parity gate at
`crates/compute-daemon/src/bin/real_forward_provider_metadata_compare.rs`.

The worker path now also has execution-side fresh-request gates:

- `crates/compute-daemon/src/bin/ggml_stage_worker_forward_compare.rs`
  for head-stage `begin_token_ids`
- `crates/compute-daemon/src/bin/ggml_stage_worker_tail_compare.rs`
  for tail-stage `continue_forward`

For the current milestone, Gemma E4B should be treated as a defended
**2-stage** target:

- head: `0-20`
- tail: `21-41`

`3+` stage Gemma work is now future-facing architecture prep, not the active
delivery target. The active objective is to keep the real 2-stage path correct,
usable, and measurably faster.

The next hard result is now in:

- real `3`-stage Gemma artifacts were generated successfully at
  `../compute-backend/out/gemma-e4b-3stage`
- required packed stage dirs now exist for:
  - `0-13`
  - `14-27`
  - `28-41`
- the stricter `real_stage_chain_compare` gate shows that this `3`-stage chain
  is not a valid real-forward split for Gemma E4B
- the runtime now rejects it at load time with:
  - `layer 28 requires shared KV from layer 22 outside this stage`
  - `current contract keeps shared-KV caches stage-local`
- this means the next real blocker is no longer ambiguous harness behavior; it
  is an explicit model/split constraint around Gemma's shared-KV dependency
  pattern at layers `22/23 -> 24+`

## Latest Update

Prompt-formatting and generation quality got a real keep.

Another real 2-stage keep landed on the cold prompt path:

- in `crates/stage-forward-lab/src/real_math.rs`, the quantized batched
  nested-output matmul path now uses the same `6+6+...+remainder` input-chunk
  fast path that the token-major FFN path already used
- this matters on first-seen prompt prefill because attention-side batched
  projections were still taking the generic `>6 input` path while FFN had
  already been cut over
- correctness is covered by new chunked-input tests for both `Q4_K` and `Q6_K`
  nested-output matmuls

- The real staged generation path now defaults to an explicit Gemma instruct
  prompt wrapper instead of feeding raw user text directly into the model.
- The wrapper is in `crates/stage-forward-lab/src/prompting.rs` and is now
  used by:
  - the local packed-artifact generators
  - the local packed-artifact benches
  - the daemon head-stage `real_forward` generation path
- Gemma EOS handling is now corrected in
  `crates/stage-forward-lab/src/tokenizer.rs`: when present, `<turn|>` is used
  as EOS ahead of `<eos>`.
- There is now a fixed real packed-artifact quality harness in
  `crates/stage-forward-lab/src/bin/real_prompt_sanity.rs`.

The current quality harness runs the real local packed stages in
`gemma_instruct` mode and checks both:

- first-token behavior with `max_tokens=1`
- short deterministic greedy continuation on the same prompt

Current real local sanity result on the packed Gemma artifacts:

```text
france_one_word:
  first token  -> "Paris"
  continuation -> "Paris"

france_exact_output:
  first token  -> "Paris"
  continuation -> "Paris"

sky_blue_sentence:
  first token  -> "As"
  continuation -> "AsRayleigh scattering,called"

sky_red_sentence:
  first token  -> "As"
  continuation -> "As when the sun is low"

cache_reason_sentence:
  first token  -> "Because"
  continuation -> "Because the initial token generation requires a full"

overall: PASS
```

That changes the quality picture materially:

- the earlier medium/long garbage continuations were mostly a prompt-formatting
  problem, not only a decode-loop problem
- under Gemma instruct formatting, the medium/long prompts are now producing
  sane deterministic continuations on the real local packed-artifact path
- the direct raw prompt `"The capital of France is"` is no longer the right
  general generation validation prompt under this mode; it tends to echo rather
  than answer

Updated real packed-artifact generation sweep (`real_two_stage_generate_sweep`,
`max_tokens=4`, `prompt_mode=gemma_instruct`) now looks like this:

```text
short / 25 prompt toks:
  cold ttft 2263ms = head 1165ms + tail 1046ms + sample 52ms
  cold total 2661ms
  warm ttft min=18ms median=19ms avg=18ms max=19ms
  warm total min=390ms median=398ms avg=398ms max=408ms
  output "Paris"

medium / 27 prompt toks:
  cold ttft 1509ms = head 766ms + tail 728ms + sample 15ms
  cold total 2667ms
  warm ttft min=18ms median=18ms avg=18ms max=19ms
  warm total min=1156ms median=1157ms avg=1158ms max=1162ms
  output "AsRayleigh scattering"

long / 59 prompt toks:
  cold ttft 3705ms = head 1890ms + tail 1798ms + sample 17ms
  cold total 4890ms
  warm ttft min=26ms median=26ms avg=26ms max=27ms
  warm total min=1196ms median=1211ms avg=1207ms max=1216ms
  output "Because reusing the KV"
```

Current useful conclusions:

- the local staged generation path is now mechanically correct enough to run a
  real deterministic prompt suite with sane medium/long outputs
- warm TTFT is now dominated by cached prompt reuse and sampling overhead
- cold TTFT is still split roughly between head and tail prompt prefill
- the latest cold-path keep reduces first-token TTFT materially on all three
  prompt classes without changing the defended outputs
- FFN is still large on first-seen prompts, but attention is substantial too on
  the longer prompt
- the next high-value work is quality-preserving prompt/prefill improvement and
  2-stage generation hardening, not more Gemma `N > 2` expansion
  then live two-node validation on the same prompt suite

Everything below this section remains useful history, but older raw-prompt
generation outputs should now be treated as superseded by the `gemma_instruct`
validation path above.

There is now also a real loopback two-node comparison harness in
`crates/compute-daemon/src/bin/real_two_node_prompt_compare.rs`.

There is also now a standalone stage runtime host at
`crates/compute-daemon/src/bin/real_stage_runtime_host.rs`.

That bin can host a real-forward stage runtime as a separate process from a
direct packed-stage artifact path, which makes the compare harness usable in a
cross-process or cross-machine setup instead of only the in-process loopback
mode.

The prompt suite is now centralized in
`crates/stage-forward-lab/src/prompt_suite.rs` and has two modes:

- `core`
  - the 5 stable prompt cases that are fast enough for routine validation
- `all`
  - the core suite plus 4 harder exact/stop-sensitive cases

Supporting runtime changes:

- `start_stage_prototype_with_bind_addr(...)` in
  `crates/compute-daemon/src/stage_runtime.rs` allows head and tail stage
  runtimes to bind on distinct loopback ports instead of both trying to use the
  fixed default port.
- `COMPUTE_STAGE_ROOT` is now honored by the real-forward packed-stage resolver
  in `crates/compute-daemon/src/inference/stage_backend.rs`, which makes it
  possible to point loopback stage runtimes at a temporary mirrored packed-stage
  root without mutating the real `~/.compute/stages` tree.
- stage prototype responses now include completion token IDs internally so the
  comparison harness can validate token-level agreement, not only text.

Current real loopback two-node compare result against the same packed artifacts
and the same Gemma instruct prompt suite:

```text
france_one_word:
  local    -> "Paris" / [50429]
  two-node -> "Paris" / [50429]

france_exact_output:
  local    -> "Paris" / [50429]
  two-node -> "Paris" / [50429]

sky_blue_sentence:
  local    -> "AsRayleigh scattering,called"
  two-node -> "AsRayleigh scattering,called"

sky_red_sentence:
  local    -> "As when the sun is low"
  two-node -> "As when the sun is low"

cache_reason_sentence:
  local    -> "Because the initial token generation requires a full"
  two-node -> "Because the initial token generation requires a full"

overall: PASS
```

Current extended `all` suite additions:

```text
yes_exact_output:
  local    -> "Yes" / [10784]
  two-node -> "Yes" / [10784]

no_exact_output:
  local    -> "No" / [3771]
  two-node -> "No" / [3771]

kv_cache_exact_output:
  local    -> "KV cache" / [57137, 15612]
  two-node -> "KV cache" / [57137, 15612]

paris_comma_stop:
  local    -> "Paris" / [50429]
  two-node -> "Paris" / [50429]

overall: PASS
```

That gives the staged path three useful validation layers now:

1. local packed-artifact sanity on the shared suite
2. live loopback two-node parity on the same suite
3. optional `all` mode with exact-output and stop-sensitive probes

There is now also a fourth deployment-oriented validation mode:

4. standalone remote-tail parity, where:
   - `real_stage_runtime_host` runs the tail stage as its own process
   - `real_two_node_prompt_compare` runs only the local head and points
     `downstream_addr` at that hosted tail

Current remote-tail process result on `suite_mode=all` also passes:

```text
france_one_word      -> PASS
france_exact_output  -> PASS
sky_blue_sentence    -> PASS
sky_red_sentence     -> PASS
cache_reason_sentence-> PASS
yes_exact_output     -> PASS
no_exact_output      -> PASS
kv_cache_exact_output-> PASS
paris_comma_stop     -> PASS
overall              -> PASS
```

That is the first validated cross-process, remote-capable prompt-suite compare
path for real-forward generation. It is still on one machine here, but it uses
the same process split and downstream-address plumbing we would use for a real
non-loopback tail node.

That is the strongest current cross-node statement we can make:

- the live staged transport path now matches the local packed-artifact path on a
  fixed real prompt suite
- prompt token counts, finish reasons, output text, and completion token IDs all
  agree on that suite
- the remaining quality risk is no longer “does transport-path generation match
  local?” for these cases; it is broadening the prompt suite and then measuring
  the real networked path under more varied conditions

## Multi-Stage Note

The current validation and tooling are intentionally strongest on the 2-stage
shape because that is the simplest real split worth defending. That should not
be mistaken for the final architecture target.

For larger models, the likely deployment target is `N > 2` stages, not only:

- one head node
- one tail node

The current runtime already has some of the right general shape:

- `StagePrototypeSpec` carries `stage_index` and `total_stages`
- stage assignment/control messages already talk about stage ranges, not only a
  hardcoded head/tail role
- the standalone stage host can already represent arbitrary stage spans

But the current validation tooling is still effectively 2-stage-specific:

- one head
- one downstream tail
- one compare harness that assumes a single downstream address

The next planning constraint is therefore not just “networked validation,” it is
“networked validation that stays compatible with an eventual N-stage chain.”

Practical implications for the next phase:

1. Do not harden new APIs around a single downstream peer.
2. Keep stage-local prompt/decode correctness independent from chain length.
3. Treat head/tail validation as the first defended slice of a more general
   stage graph, not the final product shape.
4. When adding future validation tooling, prefer inputs like ordered stage specs
   and stage-address lists over special-purpose head/tail flags.

What will need dedicated work before claiming true multi-stage readiness:

- stage-chain orchestration for `3+` stages
- end-to-end prompt-suite parity across an arbitrary ordered chain
- session lifecycle / release / cleanup across multiple downstream nodes
- latency accounting split by stage, not only head vs tail
- prefix/prefill reuse rules when intermediate stages are remote and chain-local
  caches may diverge

So the right reading of the current milestone is:

- 2-stage generation is now defended locally and across the transport path
- this is the base case
- the next architecture planning target is generalized N-stage execution and
  validation, because that is the shape larger models will likely require

There is now also a defended prototype-runtime relay step toward that:

- intermediate stages can forward activations downstream and relay final token
  responses back upstream
- the relay path is covered in
  `crates/compute-daemon/tests/stage_prototype_integration.rs`

Current prototype chain coverage:

- 3-stage runtime chain roundtrip:
  - `0-9 -> 10-19 -> 20-27`
- 4-stage runtime chain roundtrip using catalog-derived Gemma 4 E4B splits:
  - `0-10 -> 11-21 -> 22-31 -> 32-41`

What this proves:

- ordered stage chains work beyond a single head/tail hop in the prototype path
- the runtime can relay a final downstream token response through one or more
  intermediate stages

What it does not prove yet:

- real-forward packed-artifact `N > 2` execution
- prompt-suite parity through a real `N > 2` hidden-state chain
- multi-stage performance characteristics

There is now also reusable runtime tooling for the prototype path:

- `start_stage_prototype_chain(...)` in
  `crates/compute-daemon/src/stage_runtime.rs`
- `crates/compute-daemon/src/bin/prototype_stage_chain_roundtrip.rs`
- `crates/compute-daemon/src/bin/prototype_stage_runtime_host.rs`
- `crates/compute-daemon/src/bin/prototype_stage_chain_client.rs`

That means `N > 2` chain bring-up is no longer represented only by integration
tests. There is now:

- a reusable ordered-chain startup helper
- a standalone local roundtrip bin for catalog-derived stage splits
- a standalone per-stage host process for external chain assembly
- a standalone external client that can drive a hosted head stage over the
  transport path

Current external hosted-chain proof:

- hosted 3-stage prototype chain:
  - head `0-9` at `127.0.0.1:9401`
  - middle `10-19` at `127.0.0.1:9402`
  - tail `20-27` at `127.0.0.1:9403`
- external client run:
  - prompt: `"Hello"`
  - `max_tokens=8`
  - elapsed: `3ms`
  - token ids: `[80, 114, 111, 116, 111, 116, 121, 112]`
  - content: `"Prototyp"`
  - finish reason: `length`

That is the first defended external multi-process chain call through a hosted
head stage. It is still prototype-only, but it proves the transport path can be
driven without relying on in-process startup helpers.

There is now also a chain-aware external compare harness:

- `crates/compute-daemon/src/bin/prototype_stage_chain_compare.rs`

Current defended external compare result on the shared `core` prompt suite:

- hosted 3-stage prototype chain:
  - head `0-13` at `127.0.0.1:9411`
  - middle `14-27` at `127.0.0.1:9412`
  - tail `28-41` at `127.0.0.1:9413`
- compare mode:
  - local in-process prototype chain as reference
  - external hosted chain as target
  - shared `core` prompt suite
- result:
  - `france_one_word` -> PASS
  - `france_exact_output` -> PASS
  - `sky_blue_sentence` -> PASS
  - `sky_red_sentence` -> PASS
  - `cache_reason_sentence` -> PASS
  - overall -> PASS

That is the first defended external-chain parity harness for `N > 2` stages on
the prototype path.

There is now also a generic real-forward chain compare harness:

- `crates/compute-daemon/src/bin/real_stage_chain_compare.rs`

What it does:

- takes explicit stage specs in `path@start-end` form
- loads a local direct real-forward chain as the reference
- starts a runtime chain from the same stage specs
- compares text, finish reason, token IDs, and prompt token counts on the shared
  prompt suite

Current defended result on the existing 2-stage Gemma artifacts:

- stage specs:
  - `packed-stage-1/stage-1-required.index.json@0-20`
  - `packed-stage-2/stage-2-required.index.json@21-41`
- prompt mode: `gemma_instruct`
- suite: `core`
- result:
  - `france_one_word` -> PASS
  - `france_exact_output` -> PASS
  - `sky_blue_sentence` -> PASS
  - `sky_red_sentence` -> PASS
  - `cache_reason_sentence` -> PASS
  - overall -> PASS

This matters because the real-forward validation tooling is now in the right
shape for future `N > 2` work, even though the current checked-in real artifacts
are still only the defended 2-stage split.

Large prompt-prefill FFN got a real keep.

- Token-major quantized batched matmuls now split large input-token batches into
  `6+6+...+remainder` chunks when the workload is large enough.
- That lets medium/long prompt prefill reuse the exact six-input `Q4_K` / `Q6_K`
  fast paths that were previously only hit on small token counts.
- The keep is in `crates/stage-forward-lab/src/real_math.rs` and is covered by
  focused `>6 input` correctness tests against stitched single-input outputs.

Real packed-artifact cold generation sweep (`real_two_stage_generate_sweep`,
`max_tokens=4`) moved like this:

```text
short  /  6 toks:  839ms ->  872ms  (noise / no keep signal here)
medium / 15 toks: 1848ms -> 1234ms
long   / 47 toks: 6961ms -> 3748ms
```

Current cold split after the keep:

```text
short / 6 toks:
  ttft 872ms = head 427ms + tail 383ms + sample 62ms
  head cold: attn 56ms,  ffn 223ms, ple 131ms
  tail cold: attn 42ms,  ffn 210ms, ple 126ms

medium / 15 toks:
  ttft 1234ms = head 626ms + tail 593ms + sample 15ms
  head cold: attn 181ms, ffn 395ms, ple 28ms
  tail cold: attn 161ms, ffn 400ms, ple 29ms

long / 47 toks:
  ttft 3748ms = head 1920ms + tail 1810ms + sample 18ms
  head cold: attn 767ms, ffn 1007ms, ple 83ms
  tail cold: attn 695ms, ffn 1027ms, ple 82ms
```

So the cold-path picture changed in a useful way:

- logits/sample time is still noise
- PLE is still not the next thing to grind
- FFN is still the largest single bucket on first-seen prompts
- but after this keep, attention is now much closer, especially on long prompts

Shared-prefix prompt reuse now has a stronger keep on both stages.

- The staged aux payload now carries an opaque token-prefix hash chain alongside
  the existing prompt aux bytes.
- Head-side prefill reuse now uses the longest shared token prefix, not only
  exact cached prompt prefixes.
- Tail-side prefill reuse now uses that hint to find the longest matching cached
  prompt prefix by identity, then truncates the cached KV/state to that prefix
  instead of guessing from hidden-state equality.
- Exact repeated-prompt caches still stay in front of this path; this only
  improves non-identical prompts that share a real token prefix.

Real packed-artifact prefix bench (`real_two_stage_generate_prefix_bench`,
`max_tokens=1`) now moves cleanly in the right direction:

```text
short  shared-prefix probe: 846ms -> 451ms
medium shared-prefix probe: 1113ms -> 827ms
long   shared-prefix probe: 4641ms -> 507ms
```

That is the first defensible two-sided shared-prefix keep. The older
hidden-state matcher is still out; this kept path uses explicit token-prefix
identity on the tail and real token-prefix matching on the head.

The real staged Gemma path is no longer next-token-only.

- `RealGemmaBackend` now has explicit token-id prompt ingress on the head stage.
- The stage runtime head path now uses a greedy autoregressive loop for the
  `real_forward` backend, requesting one downstream token at a time without
  reintroducing plaintext prompt transport between stages.
- There is also a direct local generator probe at
  `crates/stage-forward-lab/src/bin/real_two_stage_generate.rs`.

Current state of that generator:

- correctness: real multi-token generation now works
- strategy: first token seeds per-stage local decode caches, continuation steps
  reuse those caches and run one token at a time
- quality: no known regression on the uncapped real path
- performance: continuation is now materially faster, though the decode path is
  still greedy-only and not yet fully benchmarked as a separate steady-state mode
- lifecycle: decode sessions are now explicitly cleared after generation instead
  of only aging out through the bounded cache

Example real local run on `"The capital of France is"` with `max_tokens=3`:

```text
step 1 -> Paris
step 2 -> ,
step 3 -> a
generated text: "Paris,a"
ttft: ~1015ms
total: ~1807ms
continuation tok/s: ~2.53
```

Warm generation bench on the same prompt with `max_tokens=3`, reusing loaded
backends:

```text
ttft: min=301ms median=316ms avg=315ms max=330ms
total: min=1014ms median=1047ms avg=1045ms max=1075ms
continuation: min=2.68 tok/s median=2.68 tok/s avg=2.74 tok/s max=2.87 tok/s
output: "Paris,a"
deterministic: PASS
```

Stop-string handling is now wired through both the daemon path and the direct
packed-artifact generators.

Real local stop-aware generation run on `"The capital of France is"` with
`max_tokens=8` and stop sequence `","`:

```text
step 1 -> Paris
step 2 -> ,
generated text: "Paris"
finish reason: stop
ttft: ~2828ms
total: ~3770ms
continuation tok/s: ~1.06
```

Warm stop-aware generation bench on the same prompt with `max_tokens=8`,
reusing loaded backends:

```text
ttft: min=286ms median=288ms avg=287ms max=288ms
total: min=628ms median=628ms avg=629ms max=631ms
continuation: min=2.92 tok/s median=2.92 tok/s avg=2.93 tok/s max=2.94 tok/s
finish reason: stop
output: "Paris"
deterministic: PASS
```

Prompt-length generation sweep on the same loaded backends with `max_tokens=4`
and no stop sequences:

```text
short  (6 prompt toks):   ttft ~15ms, total ~1012-1027ms, cont ~2.96-3.01 tok/s
medium (15 prompt toks):  ttft ~16ms, total ~1017-1029ms, cont ~2.96-3.00 tok/s
long   (47 prompt toks):  ttft ~20-21ms, total ~1056-1071ms, cont ~2.85-2.90 tok/s
```

The current prompt-length signal is straightforward:

- head-stage prompt prefill now reuses exact cached prefixes across repeated
  prompt runs.
- tail-stage prompt prefill now reuses exact cached full-prompt ingress payloads
  across repeated prompt runs.
- On repeated exact prompts, warm TTFT is now close to continuation-step cost
  instead of growing with prompt length.
- Cold / first-seen prompts are still prompt-length-sensitive; this keep only
  removes repeated-prompt prefill work.
- Continuation throughput is much flatter because the decode caches now avoid
  replaying the full prompt on every subsequent token.
- That means the remaining generation work splits in two:
  - continuation-side throughput if repeated exact prompts are the main case
  - or broader first-seen prompt prefill work if cross-request prefix locality
    is not enough

The new `real_two_stage_generate_sweep` profiler also gives a cleaner cold TTFT
split. On first-seen prompts, sample time is negligible and the cost is roughly
split between head and tail, with FFN dominating both sides:

```text
short / 6 toks:
  ttft 839ms = head 404ms + tail 381ms + sample 54ms
  head cold: attn 53ms,  ffn 210ms, ple 126ms
  tail cold: attn 41ms,  ffn 208ms, ple 125ms

medium / 15 toks:
  ttft 1848ms = head 951ms + tail 882ms + sample 15ms
  head cold: attn 172ms, ffn 732ms, ple 29ms
  tail cold: attn 147ms, ffn 705ms, ple 27ms

long / 47 toks:
  ttft 6961ms = head 3538ms + tail 3406ms + sample 17ms
  head cold: attn 724ms, ffn 2680ms, ple 76ms
  tail cold: attn 650ms, ffn 2673ms, ple 78ms
```

So the next cold-path target is not logits or PLE micro-work. It is prompt-prefill
FFN cost on both stages.

The first two-node real-forward stage pipeline has completed over the network.

- Request: `req-1776018105680-1`
- Model: `gemma-4-e4b-q4`
- API response included `prototype_stage_mode: true`
- Head node: `compute-vm`, stage `0/2`
- Tail node: `compute-macintosh`, stage `1/2`
- Transport: node-to-node activation payload over the stage QUIC transport
- Returned content: `できてできてでき`

This proves the app-level pipeline plumbing is now real enough to:

- form a two-node pipeline assignment
- run the head stage request path
- connect the head node to the downstream tail node
- send a hidden-state activation payload
- run the tail stage and return tokens upstream
- return an OpenAI-compatible non-streaming response through the relay

## What Was Addressed After The Run

The head log showed:

```text
[stage] Ignoring unexpected downstream response while handling req-1776018105680-1
```

That was caused by the downstream tail replying `Ready` to the head's `AssignLayers`
control message while the head was already waiting for final tokens.

The stage runtime now explicitly waits briefly for downstream `Ready` after sending
`AssignLayers`, before sending the activation payload. Late `Ready` messages during
the token wait are also handled as readiness noise instead of generic unexpected
responses.

The response text/token handling was also narrowed. If the tail reports byte-counted
token IDs for Unicode text, the head now avoids blindly treating every character as
one model token unless the text shape makes that safe.

The real-forward app adapter now uses `RealGemmaBackend` tokenization/detokenization
when that backend is loaded. The observed `prompt_tokens: 24` from the successful
run was byte-counting the prompt in the adapter, not the real tokenizer path.

The tail stage also no longer converts decoded completion text into Unicode scalar
IDs for upstream transport. `StageSample` now carries explicit sampled token IDs;
the real-forward sampler fills that field from the generated IDs, and the daemon
uses those IDs directly when building upstream token payloads. Toy and sketch lab
backends fill the same field deterministically from text for test/reference parity.
The daemon conversion is isolated in a tested helper so the current non-streaming
response path preserves sampled IDs while still carrying the full decoded text in
the final generated-token text field for `TokenPayload.text` assembly.

The real-forward tail sampler now prefers an explicit `output.weight` logits head
when the tail artifact carries one, falling back to tied `token_embd.weight` only
when no explicit output head is present. Logits projection shape validation uses
the incoming hidden-state payload dimension, not a fallback config dimension. The
logits path now decodes the selected head as a matrix and scores through the same
`real_math::matmul` path as the layer projections, instead of manually dequantizing
rows in the sampler loop.

`RealGemmaBackend::trace_tail_logits` now exposes a deterministic tail-logits trace:
projection tensor name, hidden dimension, vocab size, selected token ID/score,
top-k logits, and normalized tail-state RMS. The `real_two_stage_probe` and
`real_single_node_probe` binaries print this trace so the next real-model pass can
compare tail logits against a known-good reference without involving app transport.
The trace is `serde`-serializable and both probes print a `trace json` line for
machine-readable capture/diffing.

Suggested real-artifact capture command:

```bash
cargo run -p stage-forward-lab --bin real_two_stage_probe -- \
  out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json \
  out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json \
  'reply exactly: TRANSIENT PROBE' \
  out/gemma-e4b-2stage/vocab.json
```

To compare two captured traces, save either raw JSON or copied `trace json` probe
lines to files, then run:

```bash
cargo run -p stage-forward-lab --bin real_trace_compare -- trace-a.log trace-b.log 0.0001
```

The probe also accepts optional debug caps after `vocab.json`:

```bash
cargo run -p stage-forward-lab --bin real_two_stage_probe -- \
  <stage-1.index.json> <stage-2.index.json> '<prompt>' <vocab.json> <layer_cap> <vocab_cap>
```

These caps are smoke-test tools only. A vocab cap changes the selected token because
it limits the logits search space.

Local artifact observations from the `compute-backend` packs:

- Uncapped full-stage run was stopped after more than a minute; it was CPU-bound in
  the head forward path, not hung.
- Capped run with `layer_cap=1` and full vocab completed. It produced finite head
  and tail states, used `token_embd.weight` as the logits tensor, selected token
  `102634`, decoded text `"શા"`, and took roughly `16.2s` total, with `7.3s` in
  full-vocab sampling.
- Capped run with `layer_cap=1` and `vocab_cap=8192` selected token `3690` /
  text `"ber"` from the capped vocab slice, so treat that as performance/sanity
  evidence only. After making the probe reuse one logits pass for trace and
  sampling, avoiding a full-vocab sort for top-k trace reporting, and decoding
  only the capped logits rows, then narrowing PLE projection work to the active
  stage layer window, the same smoke path took roughly `6.7s`: `5.2s` head,
  `1.1s` tail, and `0.4s` combined trace+sampling.
- `RealGemmaBackend` now caches decoded layer matrices and norm vectors behind
  `Arc<RwLock<...>>` caches so repeated forwards on the same loaded backend avoid
  re-dequantizing operator weights. A capped single-node two-pass probe with the
  same real artifacts stayed deterministic and moved from roughly `9.3s` on the
  first pass to `8.2s` on the second pass.
- Large matrix-vector projections now use a row-parallel matvec path for both the
  main forward projections and the PLE row-range projection path. On the real
  uncapped `"The capital of France is"` probe, that cut the release-path total
  from roughly `20.98s` down to `7.59s` while preserving the `"Paris"` output.
- Quantized layer projections now stream row-wise Q4_K / Q5_K / Q6_K matvecs from
  cached raw tensor bytes instead of fully dequantizing each matrix before a
  single use. On the same uncapped `"The capital of France is"` probe, that cut
  the release-path total again from roughly `7.59s` down to `5.31s` while
  preserving the `"Paris"` output.
- The forward path now batches layer projections across the prompt sequence so
  each streamed quantized row is decoded once per layer per prompt instead of
  once per token. This applies to Q/K/V, output projection, FFN projections, and
  the PLE model projection/gates. On the same uncapped `"The capital of France
  is"` probe, that cut the release-path total again from roughly `5.31s` down to
  `3.73s` while preserving the `"Paris"` output.
- The tail logits path now scores the vocab projection by streaming raw tensor
  rows directly into argmax/top-k accumulation instead of materializing a dense
  vocab matrix. On the same uncapped `"The capital of France is"` probe, that
  cut `trace+sampling` from roughly `554ms` down to `137ms`, and total release
  time from roughly `3.73s` down to `3.10s`, while preserving the `"Paris"`
  output and top-k trace.
- The streamed quantized matvec path now uses fused Q4_K dot products instead of
  dequantizing each Q4_K row into a temporary float buffer before every dot. Q4_K
  is the dominant live format in the real stage packs, so this reduced both
  forward projections and streamed logits scoring. On the same uncapped `"The
  capital of France is"` probe, that cut release time again from roughly
  `3.10s` down to `2.44s` (`1.20s` head, `1.15s` tail, `95ms` trace+sampling)
  while preserving the `"Paris"` output and top-k trace.
- The same streamed quantized path now also uses fused Q6_K dot products for the
  subset of live `attn_v.weight` and `ffn_down.weight` tensors packed in Q6_K.
  On warm uncapped `--release` reruns of the same `"The capital of France is"`
  probe, that moved total time again from about `2.44s` down to roughly
  `2.39s` (`1.17s` head, `1.13s` tail, `99ms` trace+sampling) while preserving
  the `"Paris"` output and top-k trace.
- Decoded token rows are now cached on the loaded backend for both
  `token_embd.weight` and the `Q5_K` `per_layer_token_embd.weight` PLE path.
  These caches are bounded and clear on overflow instead of growing without
  bound. They are aimed at the daemon-style reused-backend path, not one cold
  probe. On the ignored local three-prompt capital-city regression that reuses
  one loaded head/tail backend across prompts, runtime moved from the last
  recorded `153.7s` down to about `86.6s` while preserving the expected `"Rome"`,
  `"Tokyo"`, and `"Berlin"` outputs.
- The real two-stage probe now prints aggregate per-stage forward timings for
  embed/auxiliary prep/attention/FFN/PLE. On the current warm uncapped
  `"The capital of France is"` probe, head and tail each spend about
  `150-180ms` in attention, about `130ms` in PLE, and about `840-860ms` in FFN.
  That makes FFN the next cold-path target rather than attention or prompt-aux
  decode.
- FFN now reuses the `gate_all` buffer in place when applying GeLU and
  multiplying by `up_all`, instead of allocating a separate activated hidden
  buffer before the down projection. On warm uncapped `--release` reruns of the
  same `"The capital of France is"` probe, that moved total time from about
  `2.43s` down to roughly `2.36s`, with FFN time dropping to about
  `811-837ms` on head and `820-838ms` on tail while preserving `"Paris"`.
- The quantized batched matmul hot path now precomputes input slice refs once
  and uses offset-based `Q4_K` / `Q6_K` batched dot kernels, so FFN no longer
  rebuilds tiny block-slice vectors inside every quantized output row. On warm
  uncapped `--release` reruns of the same `"The capital of France is"` probe,
  that moved total time again from about `2.36s` down to roughly `2.22s`, with
  FFN time dropping to about `768-790ms` on head and `769-805ms` on tail while
  preserving `"Paris"`.
- The same batched quantized path now uses one contiguous row-major scratch slab
  instead of allocating one `Vec<f32>` per output row, and `Q5_K` row-dot
  coverage is now fused as well instead of falling back to full row
  dequantization. On steady warm uncapped `--release` reruns of the same
  `"The capital of France is"` probe, totals now land around `2220-2235ms`,
  with head FFN at about `752-770ms` and tail FFN at about `748-751ms`, while
  still returning `"Paris"`.
- The attention, FFN, and PLE epilogues now consume their projection buffers in
  place instead of allocating fresh `rms_norm(...)` and `vec_add(...)` result
  vectors for every token of every layer. On warm uncapped `--release` reruns
  of the same `"The capital of France is"` probe, steady totals now land around
  `2187-2210ms`, with head FFN around `746-758ms`, tail FFN around `743-756ms`,
  and the same `"Paris"` output.
- FFN `gate` and `up` projections now run through one paired `Q4_K` batched path
  when both tensors share the same `Q4_K` shape, so the hot loop only walks the
  shared input slices once for both projections. On warm uncapped `--release`
  reruns of the same `"The capital of France is"` probe, totals now land around
  `1895-1960ms`, with head FFN around `584-599ms`, tail FFN around `588-630ms`,
  paired `gate+up` around `332-360ms`, down around `236-259ms`, and the same
  `"Paris"` output.
- The general single-matrix `Q4_K` batched path now processes two rows at once,
  reusing the same input slices for both rows before moving on. That reduces the
  remaining single-matrix `Q4_K` work across FFN down and the attention-side
  `Q4_K` projections. On warm uncapped `--release` reruns of the same
  `"The capital of France is"` probe, totals now land around `1729-1748ms`,
  with head FFN around `538-550ms`, tail FFN around `554-570ms`, head attention
  around `121-123ms`, tail attention around `102-104ms`, and the same
  `"Paris"` output.
- The same two-row treatment now applies to the general single-matrix `Q6_K`
  batched path, which covers the remaining `ffn_down` / `attn_v` work packed in
  `Q6_K`. On warm uncapped `--release` reruns of the same
  `"The capital of France is"` probe, totals now land around `1691-1700ms`,
  with head FFN around `525-534ms`, tail FFN around `535-543ms`, head
  attention around `121-122ms`, tail attention around `102-109ms`, and the same
  `"Paris"` output.
- The dense batched `matmul_many_range` path now also processes two rows at
  once, which matters for the remaining `F32` PLE `inp_gate` / `proj` work. On
  warm uncapped `--release` reruns of the same `"The capital of France is"`
  probe, totals now land around `1661-1673ms`, with head PLE around
  `125-128ms`, tail PLE around `126-131ms`, and per-stage PLE split at roughly
  `63-66ms` gate plus `61-64ms` proj while still returning `"Paris"`.

## Output-Quality Status

The real two-stage E4B path now has a working single-machine correctness hit.

On the local packed artifacts under `../compute-backend/out/gemma-e4b-2stage`, the
full uncapped two-stage probe for:

```text
The capital of France is
```

now returns:

```text
Paris
```

Treat this as the first real correctness pass for the local artifact split path,
not just an integration pass.

What is now fixed:

- The stage boundary carries the full prompt hidden-state sequence, not just the
  final token state. Tail layers now see prompt context for their own attention.
- Downstream real-forward hidden-state payloads no longer carry plaintext
  `prompt_text`. The head stage now frames prompt-derived PLE aux data into the
  hidden-state bytes, and the daemon forwards `prompt: null` while preserving the
  true hidden-state byte length in the envelope metadata.
- Tail RoPE positions now come from prompt token positions instead of incorrectly
  using the stage/layer index as a position offset.
- Gemma 4 E4B hybrid attention is partially modeled:
  - sliding layers use `rope_theta=10000` and `sliding_window=512`
  - full layers use `rope_theta=1000000`
  - full layers use partial rotary width instead of rotating the whole head
  - full layers now use Gemma 4 proportional RoPE semantics instead of treating
    partial rotary as a smaller standalone head
- Full-attention head geometry now trusts `attn_q_norm` / `attn_k_norm` width,
  which fixes the first global layer from the wrong `16x256` interpretation to the
  actual `8x512` attention layout.
- The last `num_kv_shared_layers=18` layers now reuse K/V caches from the last
  non-shared layer of the same attention type instead of incorrectly recomputing
  fresh K/V for every layer.
- Shared-KV attention now preserves causal query limits when reusing full-sequence
  K/V caches. Earlier shared layers could let earlier prompt tokens attend beyond
  their causal prefix.
- Attention now applies the weightless per-head `v_norm` path before storing
  value states into the attention cache. Earlier probes were feeding raw V into
  Gemma 4 attention, which materially distorted token selection.
- Gemma 4 E4B now ignores the packed `rope_freqs.weight` tensor and uses
  config-derived RoPE directly. The packed tensor carries pathological values for
  this pack (for example `1e30` in the later entries) and was corrupting sliding
  attention when treated as a raw frequency multiplier.
- Weighted RMSNorm paths now use the decoded GGUF weight directly instead of
  `1 + weight`. The packed norm tensors are real scales, not offset deltas.
- FFN gate activation now matches the published Gemma 4 text config
  (`gelu_pytorch_tanh`) instead of `silu`.
- `real_two_stage_probe` now resolves `vocab_scores.json` relative to the passed
  `vocab.json` path instead of a hard-coded local default.

Current artifact probe signal:

- capped `layer_cap=6`, `vocab_cap=8192` remains a smoke-only tool and should not
  be treated as correctness evidence; it now returns `"roid"` on the current tree
- capped `layer_cap=6`, full vocab returns `"seawater"` with PLE on and
  `"citrus"` with `disable_ple`, which is still wrong and confirms partial-layer
  caps are not a correctness proxy
- full uncapped prompt now produces `"Paris"` on the real two-stage split path

Most likely remaining model deltas:

- compare a few real-layer traces against a known-good reference around the first
  full-attention block (`blk.5`) to make sure the now-correct next token is not
  still relying on a compensating error elsewhere
- verify Gemma 4 PLE/per-layer auxiliary flow numerically against a reference
  implementation now that downstream prompt re-tokenization is gone
- validate a few more known prompts so output quality is not overfit to the one
  Paris regression case

Likely next debug targets:

- compare tail logits against a known-good reference for the same final hidden state
- compare the stage-1 hidden state against a single-node reference at the split layer
- verify Gemma 4 PLE/per-layer auxiliary flow is numerically aligned with the reference
- compare final logit projection orientation against a known-good reference
- verify prompt tokenization is identical between the stage path and reference path
- add a deterministic `real_forward` trace fixture that records per-layer RMS/range
- expand the local real-artifact regression set with 2-3 more short prompts now
  that a first single-machine correctness test is in place

## Database / Supabase Handoff

No database changes were made in this pass.

When a Supabase-capable agent picks this up, ask it to verify the deployed schema and
RLS/policies support these fields and transitions:

- `nodes.pipeline_capable`
- `nodes.stage_backend_kind`
- `nodes.pipeline_id`
- `nodes.pipeline_stage`
- `nodes.pipeline_total_stages`
- any fields used to pass or derive `upstream_addr` / `downstream_addr`
- assignment push payload fields for `assignment_mode=stage`
- node heartbeat/update path preserving stage capability while a pipeline is active

Known repo migration to compare against:

- `supabase/migrations/20260412000001_add_placement_fields_to_nodes.sql`

If the local context is compacted, preserve this exact instruction for the Supabase
handoff:

```text
Do not infer database state from the code alone. Use Supabase MCP to inspect the
deployed schema and policies. Confirm the stage pipeline assignment fields exist,
are writable by the orchestrator service role, are readable by daemon assignment
flows, and are not blocked by RLS. Report any missing columns/policies/migrations;
do not apply changes without review.
```

## Verification

Local checks after the runtime patch:

- `cargo test -p compute-daemon stage_runtime --quiet` passed: 3 tests
- `cargo test -p compute-daemon stage_backend --quiet` passed: 7 tests
- `cargo test -p stage-forward-lab --quiet` passed: 91 tests
- `cargo test -p stage-forward-lab --bin real_trace_compare --quiet` passed: 3 tests
- `cargo run -p stage-forward-lab --bin real_trace_compare --quiet -- <matching-trace-a> <matching-trace-b> 0.0001` passed locally with generated trace files
- `cargo run -p stage-forward-lab --bin real_trace_compare --quiet -- <trace-a> <divergent-trace-c> 0.0001` failed locally as expected on selected-token/top-k divergence
- `cargo test -p stage-forward-lab real_two_stage_roundtrip_produces_finite_output --quiet` passed with a direct token-ID assertion
- `cargo test -p stage-forward-lab real_tail_sampling_prefers_output_weight_over_tied_embeddings --quiet` passed
- `cargo test -p stage-forward-lab two_stage_prompt_transfer_preserves_sequence_context_for_tail_attention --quiet` passed
- `cargo test -p stage-forward-lab hidden_state_payload_frames_prompt_aux_without_forwarding_prompt_text --quiet` passed
- `cargo test -p stage-forward-lab layer_config_preserves_model_level_rope_and_logit_settings --quiet` passed
- `cargo test -p stage-forward-lab layer_config_uses_qk_norm_width_for_full_attention_layers --quiet` passed
- `cargo test -p stage-forward-lab layer_attention_config_matches_gemma4_e4b_hybrid_pattern --quiet` passed
- `cargo test -p stage-forward-lab gemma4_e4b_ignores_packed_rope_freq_tensor --quiet` passed
- `cargo test -p stage-forward-lab gqa_attention_seq_with_limit_preserves_causality_for_shared_cache --quiet` passed
- `cargo test -p stage-forward-lab gqa_attention_seq_with_window_uses_recent_tokens_only --quiet` passed
- `cargo test -p stage-forward-lab q4_k_dot_matches_constructed_row --quiet` passed
- `cargo test -p stage-forward-lab q6_k_dot_matches_constructed_row --quiet` passed
- `cargo test -p stage-forward-lab q5_k_row_into_matches_allocating_decode --quiet` passed
- `cargo test -p stage-forward-lab matmul_quantized_range_matches_constructed_q4_k_row --quiet` passed
- `cargo test -p stage-forward-lab matmul_quantized_range_matches_constructed_q6_k_row --quiet` passed
- `cargo test -p stage-forward-lab matmul_quantized_many_range_matches_repeated_single_input --quiet` passed
- `cargo test -p stage-forward-lab matmul_raw_top_k_tracks_argmax_and_top_scores --quiet` passed
- `cargo test -p stage-forward-lab gqa_attention_seq_matches_manual_softmax_weighting --quiet` passed
- `cargo test -p stage-forward-lab gelu_pytorch_tanh_mul_inplace_matches_separate_ops --quiet` passed
- `cargo test -p stage-forward-lab q4_k_dot_matches_constructed_row --quiet` passed with the new `dot_many_row_refs_into` path
- `cargo test -p stage-forward-lab proportional_rope_zero_pads_non_rotated_dims_across_full_head --quiet` passed
- `cargo test -p compute-daemon real_forward_begin_prompt_omits_plaintext_prompt_from_forwarded_payload --quiet` passed
- `cargo test -p stage-forward-lab local_real_e4b_two_stage_capped_output_is_deterministic_if_artifacts_present -- --ignored --quiet` passed locally against `../compute-backend/out/gemma-e4b-2stage`, but it takes roughly 3 minutes in the default debug test profile so it is intentionally ignored by default
- `cargo test -p stage-forward-lab local_real_e4b_two_stage_output_matches_paris_if_artifacts_present -- --ignored --quiet` passed locally against `../compute-backend/out/gemma-e4b-2stage`; after the bounded token-row caches it still takes about `31.6s` in the default debug test profile. On macOS, run it under `caffeinate` if the machine may sleep.
- `cargo test -p stage-forward-lab local_real_e4b_two_stage_outputs_match_small_prompt_set_if_artifacts_present -- --ignored --quiet` passed locally against `../compute-backend/out/gemma-e4b-2stage`; it asserts `"Rome"`, `"Tokyo"`, and `"Berlin"` on the real two-stage path and now takes about `86.6s` in the default debug test profile. On macOS, run it under `caffeinate` if the machine may sleep.
- `cargo check -p stage-forward-lab --bins --quiet` passed
- `cargo check -p compute-daemon --quiet` passed
- Capped real-artifact smoke probe passed:
  `cargo run -p stage-forward-lab --bin real_two_stage_probe --quiet -- ../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json ../compute-backend/out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json 'reply exactly: TRANSIENT PROBE' ../compute-backend/out/gemma-e4b-2stage/vocab.json 1 8192`
  selected token `3690`, trace/sample selected-token match `true`, total timed
  path `6744ms`
- Updated capped real-artifact probe on `"The capital of France is"` now returns
  `"roid"` at `layer_cap=6`, `vocab_cap=8192`
- Updated capped real-artifact probe with `disable_ple` on the same prompt now
  returns `"citrus"` when run against the full vocab with `disable_ple`
- Updated full uncapped real-artifact probe on `"The capital of France is"` now
  returns `"Paris"`
- The same full uncapped real-artifact probe now takes roughly `2391ms` total in
  warm `--release` reruns (`1167ms` head, `1125ms` tail, `99ms` trace+sampling)
- The same probe now also reports phase timings: head `embed=0ms aux=15ms attn=176ms ffn=842ms ple=133ms`,
  tail `embed=0ms aux=0ms attn=149ms ffn=859ms ple=131ms`
- After the in-place FFN gate fusion, warm `--release` reruns now land around
  `2344-2389ms`; a representative run was `2357ms` with head
  `embed=0ms aux=16ms attn=168ms ffn=829ms ple=129ms` and tail
  `embed=0ms aux=0ms attn=150ms ffn=824ms ple=130ms`
- After the ref-based quantized batched-dot follow-up, warm `--release` reruns
  now land around `2221-2285ms`; a representative run was `2221ms` with head
  `embed=0ms aux=15ms attn=160ms ffn=770ms ple=130ms` and tail
  `embed=0ms aux=0ms attn=136ms ffn=769ms ple=133ms`
- After flattening the batched quantized row scratch into a single slab, steady
  warm `--release` reruns now land around `2220-2235ms`; representative runs
  were `2220ms` and `2225ms` with head
  `embed=0ms aux=16ms attn=166-167ms ffn=752-764ms ple=136-138ms` and tail
  `embed=0ms aux=0ms attn=140-142ms ffn=748-752ms ple=137-138ms`
- After switching the attention/FFN/PLE epilogues to in-place buffer reuse,
  steady warm `--release` reruns now land around `2187-2210ms`; representative
  runs were `2187ms`, `2202ms`, and `2210ms` with head
  `embed=0ms aux=16-17ms attn=161-165ms ffn=746-758ms ple=133-135ms` and tail
  `embed=0ms aux=0ms attn=136-143ms ffn=743-756ms ple=133-135ms`
- After adding the paired `Q4_K` batched path for FFN `gate` + `up`, warm
  `--release` reruns now land around `1895-1960ms`; representative runs were
  `1895ms`, `1897ms`, `1924ms`, and `1960ms` with head
  `embed=0ms aux=15ms attn=169-175ms ffn=584-599ms (gate+up=332-342ms down=236-246ms) ple=138-144ms`
  and tail
  `embed=0ms aux=0ms attn=141-144ms ffn=588-630ms (gate+up=338-360ms down=240-259ms) ple=139-141ms`
- After switching the general `Q4_K` batched path to process two rows at once,
  warm `--release` reruns now land around `1729-1748ms`; representative runs
  were `1729ms`, `1734ms`, `1738ms`, and `1748ms` with head
  `embed=0ms aux=15-16ms attn=121-123ms ffn=538-550ms (gate+up=329-338ms down=198-204ms) ple=138-141ms`
  and tail
  `embed=0ms aux=0ms attn=102-104ms ffn=554-570ms (gate+up=337-343ms down=204-219ms) ple=136-140ms`
- After switching the general `Q6_K` batched path to process two rows at once,
  warm `--release` reruns now land around `1691-1700ms`; representative runs
  were `1691ms`, `1700ms`, and `1700ms` with head
  `embed=0ms aux=15-16ms attn=121-122ms ffn=525-534ms (gate+up=331-339ms down=182-183ms) ple=134-141ms`
  and tail
  `embed=0ms aux=0ms attn=102-109ms ffn=535-543ms (gate+up=337-343ms down=187-189ms) ple=139-142ms`
- After switching dense `matmul_many_range` to process two rows at once, warm
  `--release` reruns now land around `1661-1673ms`; representative runs were
  `1661ms`, `1671ms`, `1672ms`, and `1673ms` with head
  `embed=0ms aux=8ms attn=120-125ms ffn=525-535ms (gate+up=330-339ms down=186-189ms) ple=125-128ms (gate=63-65ms proj=61-63ms)`
  and tail
  `embed=0ms aux=0ms attn=102-103ms ffn=530-540ms (gate+up=333-341ms down=186-188ms) ple=126-131ms (gate=65-66ms proj=63-64ms)`
- Attention timing is now split into projection/core/output sub-phases in the
  real probe. On the same warm uncapped `--release` probe, the attention core
  itself is only about `3ms` per stage; the rest is projection work. A quick
  `K` + `V` paired-`Q4_K` experiment was not a stable win on the real probe, so
  it was reverted. The branch keeps the split profiling only. Current warm
  `--release` reruns still land around `1663-1671ms`; representative runs were
  `1663ms`, `1663ms`, and `1671ms` with head
  `embed=0ms aux=8ms attn=121-130ms (qkv=73-81ms core=3ms out=44ms) ffn=525-531ms (gate+up=330-335ms down=184-185ms) ple=129-130ms (gate=65ms proj=63ms)`
  and tail
  `embed=0ms aux=0ms attn=101-103ms (qkv=49-51ms core=3ms out=46-47ms) ffn=532-536ms (gate+up=334-336ms down=185-189ms) ple=128-130ms (gate=64-65ms proj=63-64ms)`
- The general single-matrix `Q4_K` batched path now also processes four rows at
  once before falling back to the existing two-row/single-row remainder paths.
  That hits the largest remaining single-matrix `Q4_K` work, especially
  attention `Q` / output and the remaining FFN projections. On warm uncapped
  `--release` reruns of the same `"The capital of France is"` probe, totals now
  land around `1590-1597ms`; representative runs were `1590ms`, `1593ms`, and
  `1597ms` with head
  `embed=0ms aux=7-8ms attn=105-107ms (qkv=62-64ms core=3ms out=37-38ms) ffn=507-514ms (gate+up=332-339ms down=164ms) ple=124-129ms (gate=62-64ms proj=60-63ms)`
  and tail
  `embed=0ms aux=0ms attn=87-90ms (qkv=41-44ms core=3ms out=39-41ms) ffn=515-520ms (gate+up=337-340ms down=167-169ms) ple=124-126ms (gate=63-64ms proj=60-61ms)`
- The paired `Q4_K` gate/up path now processes two row-pairs at once by reusing
  the same 4-row helper across `gate0`, `gate1`, `up0`, and `up1` before
  falling back to the existing single-row-pair remainder path. That directly
  targets the dominant FFN `gate+up` work. On warm uncapped `--release` reruns
  of the same `"The capital of France is"` probe, totals now land around
  `1475-1489ms`; representative runs were `1475ms`, `1483ms`, and `1489ms`
  with head
  `embed=0ms aux=7-8ms attn=103-106ms (qkv=62-64ms core=3ms out=36-37ms) ffn=452-461ms (gate+up=275-284ms down=165-166ms) ple=126-127ms (gate=63ms proj=61-62ms)`
  and tail
  `embed=0ms aux=0ms attn=86-88ms (qkv=42-43ms core=3ms out=38-39ms) ffn=460-462ms (gate+up=279-281ms down=170-171ms) ple=126-128ms (gate=63-64ms proj=61-62ms)`
- The row-major-to-token-major output rebuild now has a small-input specialized
  transpose helper, which matters because the real prompt-side path is running
  with tiny `seq_len` and large row counts. This applies to dense batched
  matmul, general quantized batched matmul, and the paired `Q4_K` gate/up path.
  On warm uncapped `--release` reruns of the same `"The capital of France is"`
  probe, totals now land around `1464-1482ms`; representative runs were
  `1464ms`, `1466ms`, and `1482ms` with head
  `embed=0ms aux=8ms attn=103-104ms (qkv=61-62ms core=3ms out=36-37ms) ffn=448-465ms (gate+up=271-275ms down=166-178ms) ple=125-127ms (gate=63-64ms proj=60-62ms)`
  and tail
  `embed=0ms aux=0ms attn=84-86ms (qkv=41-42ms core=3ms out=37-38ms) ffn=453-458ms (gate+up=273-276ms down=168-171ms) ple=126ms (gate=63-64ms proj=61ms)`
- The hot `Q4_K` two-row and four-row batched dot routines now also have a
  dedicated six-input fast path. That cuts the short-prompt projection path
  without changing the public matmul API and is now covered explicitly by
  six-input equivalence tests. On warm uncapped `--release` reruns of the same
  `"The capital of France is"` probe, totals now land around `1439-1500ms`;
  representative runs were `1439ms`, `1500ms`, and `1546ms` with head
  `embed=0ms aux=7-8ms attn=100-110ms (qkv=60-66ms core=3ms out=35-39ms) ffn=442-477ms (gate+up=268-282ms down=164-184ms) ple=125-132ms (gate=63-66ms proj=61-65ms)`
  and tail
  `embed=0ms aux=0ms attn=82-91ms (qkv=40-44ms core=3ms out=36-42ms) ffn=441-466ms (gate+up=265-280ms down=165-175ms) ple=126-128ms (gate=64-65ms proj=61-62ms)`
- The hot `Q6_K` paired-row and single-row batched dot routines now also have a
  dedicated six-input fast path. That targets the remaining short-prompt
  `ffn_down` and other `Q6_K` projection work without changing the external
  matmul API. On warm uncapped `--release` reruns of the same
  `"The capital of France is"` probe, totals now land around `1430ms`; the two
  steady reruns after the rebuild both came in at `1430ms` with head
  `embed=0ms aux=7-8ms attn=97-98ms (qkv=57-58ms core=3ms out=34ms) ffn=432-435ms (gate+up=269-270ms down=153-154ms) ple=127ms (gate=64ms proj=62ms)`
  and tail
  `embed=0ms aux=0ms attn=83ms (qkv=40-41ms core=3ms out=37ms) ffn=441-442ms (gate+up=269-273ms down=158-162ms) ple=125-127ms (gate=63-64ms proj=61-62ms)`
- The layer loop no longer clones the full hidden-state sequence just to keep a
  residual copy for attention/FFN/PLE epilogues. It still clones only the
  normalized projection inputs where needed, but the residual add now reads from
  `states[t]` immediately before overwrite. That removes three full-sequence
  clones per layer on the common path. On warm uncapped `--release` reruns of
  the same `"The capital of France is"` probe, totals now land around
  `1420-1423ms`; representative runs were `1420ms` and `1423ms` with head
  `embed=0ms aux=7-8ms attn=98-99ms (qkv=57-59ms core=3ms out=35ms) ffn=429-436ms (gate+up=266-273ms down=152-153ms) ple=126ms (gate=63-64ms proj=61ms)`
  and tail
  `embed=0ms aux=0ms attn=82-83ms (qkv=40ms core=3ms out=37ms) ffn=436-439ms (gate+up=270-272ms down=155-156ms) ple=126ms (gate=63-64ms proj=61ms)`
- Real-forward shard load no longer eagerly decodes the large
  `per_layer_model_proj.weight` matrix or `per_layer_proj_norm.weight`. Those
  now resolve lazily through the existing backend caches on first use, which
  cuts shard startup time by roughly `100ms` per side on the real split E4B
  packs. On rebuilt uncapped `--release` probe runs of
  `"The capital of France is"`, head load moved from roughly `370ms` down to
  about `263-267ms` and tail load moved from roughly `378-391ms` down to about
  `270-271ms`, while preserving the correct `"Paris"` output.
- PLE prompt preparation now has a bounded per-token combined cache keyed by
  token ID. The full per-layer combined PLE vector is token-specific, so
  repeated tokens across requests no longer pay the projection/norm/combine
  setup cost every time. On the uncapped reused-backend single-node probe for
  `"The capital of France is"`, total time moved from about `1892ms` on the
  first in-process pass and `970ms` on the second pass down to about `1594ms`
  on the first pass and `911ms` on the second pass, while preserving
  deterministic `"Paris"` output and identical hidden states.
- PLE aux profiling now breaks the prompt-side `aux` bucket into
  `lookup/project/combine/materialize`. On the lazy-decode path, the first
  uncapped fresh head pass showed `aux=112ms` with essentially all of that in
  `project`, not `combine` or `materialize`. Based on that, shard load now
  starts a background PLE prewarm thread for `per_layer_model_proj.weight` and
  `per_layer_proj_norm.weight`, and the first prompt only joins it if needed.
  On rebuilt uncapped `--release` runs of `"The capital of France is"`, fresh
  two-stage head `aux` dropped from about `112ms` to about `10ms`, head forward
  moved from about `815ms` to about `719ms`, and fresh total moved from about
  `1602ms` to about `1533ms`, while keeping shard load around `272-275ms`.
  Reused-backend single-node totals landed at about `1511ms` on the first pass
  and `907ms` on the second pass with the same `"Paris"` output.
- FFN batched projections now stay in a contiguous token-major slab across the
  `gate+up -> down` boundary instead of materializing nested `Vec<Vec<f32>>`
  outputs and then rebuilding input refs for the down projection. This keeps
  the hot FFN path flatter without changing outputs. On rebuilt uncapped
  `--release` real two-stage runs of `"The capital of France is"`, fresh totals
  moved from about `1536ms` down to about `1449-1464ms`, with head
  `ffn=439-459ms (gate+up=273-294ms down=154-155ms)` and tail
  `ffn=439-441ms (gate+up=273-275ms down=155ms)`. On the sequential reused
  single-node probe, totals landed at about `1435-1447ms` on the first pass
  and `899-943ms` on the second pass, with deterministic hidden states and the
  same `"Paris"` output.
- The full-vocab logits scorer now uses the existing multi-row quantized kernels
  in `matmul_raw_top_k` instead of calling `dot_row` one row at a time for
  `Q4_K` and `Q6_K` tensors. That keeps token selection identical while cutting
  the isolated sampling bucket. On warm uncapped `--release` reruns of the same
  `"The capital of France is"` probe, `trace+sampling` moved from about
  `100-107ms` down to about `88ms`, and total moved to about `1420ms` with head
  `ffn=436ms`, tail `ffn=442ms`, and the same selected token `9079` / text
  `"Paris"`.
- The common RMSNorm paths now skip the per-element modulo when the decoded
  weight length already matches the vector width. That applies to the standard
  RMSNorm helpers and the per-head Q/K norm path, which is the common case on
  the real Gemma 4 E4B artifacts. Warm uncapped `--release` reruns of the same
  `"The capital of France is"` probe stayed in the same good band at about
  `1424-1425ms`, with `trace+sampling` holding around `85-86ms` and the same
  `"Paris"` output. This is not a headline win, but it removes avoidable work
  from a very hot math path without changing behavior.
- Added `real_two_stage_bench` and `real_projection_bench` so the next kernel
  passes can be judged on reused-backend warm runs and isolated packed-tensor
  projections instead of noisier one-shot probes. The projection bench now
  measures the real stage tensors directly and can split quantized row-dot work
  from token-major rebuild. On the live E4B tensors with six inputs, the hot
  `blk.0.ffn_gate.weight + blk.0.ffn_up.weight` pair lands around
  `~10.0ms avg`, while `blk.0.ffn_down.weight` lands around `~6.5ms avg`.
- Those isolated projection timings showed the token-major rebuild is not the
  bottleneck: transpose is only about `19us` on the paired `Q4_K gate+up` path
  and about `4-5us` on the `Q6_K ffn_down` path. That means the remaining real
  work is inside the row-dot kernels themselves, not the output reshaping.
- Based on that bench, the general single-matrix `Q6_K` path now has a four-row
  six-input fast path, which is the real `ffn_down` shape on the warm
  reused-backend Gemma E4B path. On the isolated real stage-1
  `blk.0.ffn_down.weight` projection bench, average time moved from about
  `6.46ms` down to about `5.54ms` for six inputs. On rebuilt warm uncapped
  `real_two_stage_bench` runs of `"The capital of France is"`, totals landed at
  `min=878ms median=897ms avg=892ms max=910ms`, with head/tail
  `ffn=333ms` and `down=111-112ms`, while keeping deterministic `"Paris"`
  output.
- The six-input `Q4_K` fast paths now use unchecked indexing internally after
  validating all block and input bounds up front. That removes a large number of
  repeated bounds checks from the hot paired `gate+up` kernel without changing
  the external API. Those same six-input `Q4_K` kernels now also use explicit
  `mul_add` in the hot accumulation loops. On the isolated real stage-1
  `blk.0.ffn_gate.weight + blk.0.ffn_up.weight` projection bench, average time
  moved from about `10.27ms` down to about `6.96ms` for six inputs, while the
  transpose step stayed around `17-18us`. Exact floating-point intermediates can
  differ slightly because of the fused accumulation order, but the selected
  token stayed stable on the real path.
  On repeated rebuilt warm uncapped `real_two_stage_bench` runs of
  `"The capital of France is"`, totals landed at
  `min=652ms median=661ms avg=661ms max=672ms`, with head
  `ffn=238ms (gate+up=136ms down=92ms)` and tail
  `ffn=244ms (gate+up=138ms down=96ms)`, while keeping deterministic `"Paris"`
  output.
- Added `real_ffn_bench` to measure a full packed-layer FFN step directly:
  `ffn_norm -> gate+up -> gelu*up -> down` on real stage tensors, without the
  rest of the model in the way. On rebuilt `--release` runs against
  `blk.0` in stage 1 with six inputs, the split is now:
  `norm ~19us`, `gate+up ~6.6ms avg`, `activation ~0.5ms avg`,
  `down ~5.3ms avg`, `total ~12.4ms avg`. That confirms the next real
  target remains the quantized projection kernels themselves, especially paired
  `Q4_K gate+up`, not RMSNorm, activation, or the token-major rebuild.
- Added a partial-input quantized accumulate helper plus a chunked mode in
  `real_ffn_bench` to test the larger FFN contract directly: compute
  `gate+up -> activation` in `256`-wide chunks and accumulate `ffn_down`
  against those chunks instead of materializing the full activated `10240`
  slab. On the same rebuilt `--release` layer-0 bench, that prototype is
  clearly slower: full FFN path lands around `12.5ms avg`, while the chunked
  prototype lands around `19.1ms avg` (`chunk-g+u ~9.1ms`, `chunk-down ~9.5ms`).
  So the naive chunked contract is not a keep for the runtime in its current
  form.
- The six-input `Q6_K` fast paths now use the same style as the kept `Q4_K`
  work: validated unchecked indexing plus fused `mul_add` accumulation in the
  hot inner loops for the single-row, paired-row, and four-row helpers. On the
  rebuilt `--release` `real_ffn_bench` for real stage-1 `blk.0` with six
  inputs, `ffn_down` moved from about `5.48ms avg` down to about `4.47ms avg`,
  and full FFN total moved from about `12.53ms avg` down to about `11.88ms avg`
  (`gate+up ~6.90ms`, `activation ~0.49ms`, `down ~4.47ms`). On rebuilt warm
  uncapped `real_two_stage_bench` runs of `"The capital of France is"`, totals
  landed at `min=624ms median=636ms avg=656ms max=733ms`, with head
  `ffn=236ms (gate+up=139ms down=87ms)` and tail
  `ffn=233ms (gate+up=138ms down=85ms)`, while keeping deterministic `"Paris"`
  output.
- For the hot paired `Q4_K gate+up` path, forcing `#[inline(always)]` on
  `get_scale_min_k4`, `fp16_to_f32`, and the six-input `Q4_K` paired/four-row
  helpers produced another clean win. On rebuilt `--release`
  `real_ffn_bench` runs against real stage-1 `blk.0` with six inputs,
  `gate+up` moved from about `6.90ms avg` down to about `6.10ms avg`, and full
  FFN total moved from about `11.88ms avg` down to about `11.00ms avg`
  (`down ~4.36ms`, `activation ~0.49ms`). On rebuilt warm uncapped
  `real_two_stage_bench` runs of `"The capital of France is"`, totals landed at
  `min=595ms median=599ms avg=609ms max=658ms`, with head
  `ffn=220ms (gate+up=129ms down=80ms)` and tail
  `ffn=221ms (gate+up=129ms down=81ms)`, while keeping deterministic `"Paris"`
  output.
- A smaller follow-up keep was forcing `#[inline(always)]` on the `Q4_K`
  two-row and four-row wrapper entry points themselves. That is not a major
  kernel change, just a wrapper-overhead cut, but it stayed on the right side
  of the numbers: rebuilt `real_ffn_bench` moved from about `11.00ms avg` down
  to about `10.95ms avg` total (`gate+up ~6.04ms avg`, `down ~4.41ms avg`),
  and rebuilt warm uncapped `real_two_stage_bench` runs of
  `"The capital of France is"` landed at
  `min=594ms median=599ms avg=607ms max=633ms`, still deterministic `"Paris"`.
- Another clean keep was switching the six-input `Q4_K` hot loops to pointer
  reads for the input slices and packed `qs` bytes while keeping the same
  unchecked-indexing + `mul_add` arithmetic. On rebuilt `--release`
  `real_ffn_bench` runs against real stage-1 `blk.0` with six inputs,
  `gate+up` moved from about `6.04ms avg` down to about `5.70ms avg`, and full
  FFN total moved from about `10.95ms avg` down to about `10.62ms avg`
  (`down ~4.40ms avg`, `activation ~0.49ms avg`). On rebuilt warm uncapped
  `real_two_stage_bench` runs of `"The capital of France is"`, totals landed at
  `min=562ms median=568ms avg=577ms max=607ms`, with head
  `ffn=208ms (gate+up=120ms down=77ms)` and tail
  `ffn=206ms (gate+up=118ms down=78ms)`, still deterministic `"Paris"`.
- The next real keep was an Apple-Silicon-only `aarch64` fast path for the
  hot six-input four-row `Q4_K` helper in `quants.rs`. This does not touch the
  stage protocol or mixed-node compatibility; it only changes the local block
  kernel on `aarch64`, while non-Apple platforms keep the scalar path. The
  first NEON attempts were rejected, but a direct four-row design with vector
  accumulators across the full 32-position inner loop was worth keeping. On
  rebuilt `real_ffn_bench` runs against real stage-1 `blk.0` with six inputs,
  `gate+up` moved from about `5.70ms avg` down to about `3.39ms avg`, while
  full FFN total moved from about `10.62ms avg` down to about `8.28ms avg`
  (`down ~4.38ms avg`, `activation ~0.48ms avg`). On rebuilt warm uncapped
  `real_two_stage_bench` runs of `"The capital of France is"` with five warm
  passes, totals landed at `min=419ms median=421ms avg=431ms max=473ms`, with
  head `ffn=147ms (gate+up=72ms down=64ms)` and tail
  `ffn=149ms (gate+up=72ms down=67ms)`, while keeping deterministic `"Paris"`
  output.
- The next keep was the matching Apple-Silicon-only `aarch64` fast path for
  the hot six-input four-row `Q6_K` helper used by `ffn_down`. This again only
  changes the local block kernel on `aarch64`; stage payloads and mixed-node
  interoperability are unchanged. On rebuilt `real_ffn_bench` runs against
  real stage-1 `blk.0` with six inputs, `down` moved from about `4.38ms avg`
  down to about `2.12ms avg`, while full FFN total moved from about
  `8.28ms avg` down to about `6.05ms avg` (`gate+up ~3.43ms avg`,
  `activation ~0.48ms avg`). On rebuilt warm uncapped `real_two_stage_bench`
  runs of `"The capital of France is"` with five warm passes, totals landed at
  `min=375ms median=381ms avg=389ms max=424ms`, with head
  `ffn=131ms (gate+up=77ms down=44ms)` and tail
  `ffn=123ms (gate+up=72ms down=41ms)`, while keeping deterministic `"Paris"`
  output.
- The next keep was a deeper Apple-Silicon-only `aarch64` rewrite of the hot
  six-input four-row `Q4_K` helper to decode packed q-bytes with NEON integer
  ops eight at a time instead of doing scalar nibble extraction into temporary
  coefficient arrays. This still only changes the local block kernel on
  `aarch64`; stage payloads and mixed-node interoperability remain unchanged.
  On rebuilt `real_ffn_bench` runs against real stage-1 `blk.0` with six
  inputs, `gate+up` moved from about `3.43ms avg` down to about `2.47ms avg`,
  while full FFN total moved from about `6.05ms avg` down to about
  `5.27ms avg` (`down ~2.27ms avg`, `activation ~0.50ms avg`). On rebuilt warm
  uncapped `real_two_stage_bench` runs of `"The capital of France is"` with
  five warm passes, totals landed at `min=312ms median=319ms avg=321ms
  max=334ms`, with head `ffn=98ms (gate+up=52ms down=35ms)` and tail
  `ffn=102ms (gate+up=53ms down=39ms)`, while keeping deterministic `"Paris"`
  output.
- The next keep specialized the warm tail logits scorer for the real
  `token_embd.weight` path. `matmul_raw_top_k` now uses a dedicated single-input
  four-row `Q4_K` path instead of routing the 262k-row vocab scan through the
  generic batched helper with a one-element input slice. On rebuilt warm
  uncapped `real_two_stage_bench` runs of `"The capital of France is"` with
  five warm passes, the sample bucket dropped from about `41ms avg` down to
  about `14ms avg`, and total warm latency moved from about
  `min=312ms median=319ms avg=321ms max=334ms` down to
  `min=287ms median=295ms avg=295ms max=312ms`, while keeping deterministic
  `"Paris"` output / token `9079`. This only changes local logits math; stage
  payloads and mixed-node interoperability remain unchanged.
- Stage-to-stage token transport no longer forwards plaintext completion text by
  default. The daemon now emits `TokenPayload.text = null` on the upstream stage
  hop and reconstructs text at the head from token IDs when needed. This
  removes one avoidable plaintext output leak from the stage QUIC transport,
  though the executing stage still necessarily sees the prompt/output it is
  computing on.
- Capped single-node repeat-forward probe passed:
  `cargo run -p stage-forward-lab --bin real_single_node_probe --quiet -- ../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json ../compute-backend/out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json 'reply exactly: TRANSIENT PROBE' ../compute-backend/out/gemma-e4b-2stage/vocab.json 1 8192`
  first pass `9326ms`, second pass `8196ms`, deterministic states/text
- `rustfmt --check crates/compute-daemon/src/stage_runtime.rs crates/compute-daemon/src/inference/stage_backend.rs crates/stage-forward-lab/src/lib.rs crates/stage-forward-lab/src/real_forward.rs crates/stage-forward-lab/src/bin/real_two_stage_probe.rs crates/stage-forward-lab/src/bin/real_single_node_probe.rs crates/stage-forward-lab/src/bin/real_trace_compare.rs` passed

Broader `cargo fmt -p compute-daemon --check` still reports formatting drift in
other daemon files that predated this handoff. That should be cleaned separately to
avoid mixing unrelated formatting churn with stage-runtime fixes.

## Latest GGML Bootstrap State

- `ggml-bootstrap` is now the explicit accelerated-provider boundary for
  `real_forward`; it is not production-loadable yet, but it is no longer
  "fully unavailable".
- Metadata parity on the defended 2-stage Gemma path passes through the custom
  worker process:
  - `target/release/real_forward_provider_metadata_compare`
  - `overall: PASS`
- Provider-level execution parity now also passes on the same path through the
  actual `RealForwardStageProvider` interface:
  - `target/release/real_forward_provider_bootstrap_compare`
  - `overall: PASS`
- Provider-level continuation parity now also passes through the same seam:
  - `target/release/real_forward_provider_bootstrap_continuation_compare`
  - `overall: PASS`
- That bootstrap compare checks:
  - head-stage `tokenize_generation_prompt`
  - head-stage `begin_token_ids`
  - tail-stage `continue_forward`
  - tail-stage `sample_tail`
  - stage-shaped capability reporting with `per_stage_decode_sessions=true`
- The continuation compare keeps the same request ID alive across repeated
  head/tail steps and validates exact sequence parity against `cpu-ref`.
- The earlier hidden-hash mismatch in the bootstrap compare was a judge bug:
  the local reference side was reusing provider/backend state across cases,
  while the worker side was fresh per request. The compare now rebuilds the
  local reference path per case, matching the worker execution shape.
- The worker transport for full tensor responses now uses an explicit tensor
  wire envelope rather than giant inline JSON number arrays for raw bytes.
- `ggml-bootstrap` now uses a persistent localhost worker process behind the
  provider seam, so stage-local decode sessions are actually owned by the
  worker and can be cleared via a real `clear_decode_session` request.
- The worker execution core is now split behind
  `crates/compute-daemon/src/inference/ggml_stage_executor.rs`.
  - the bootstrap provider now routes:
    - metadata ops through `ggml-worker`
    - execution ops through `cpu-ref-worker`
  - executor choice is now serialized in the worker init payload
  - the worker host builds the executor from that init payload instead of
    reading `COMPUTE_GGML_STAGE_EXECUTOR`
  - there is now also a real `ggml-worker` bootstrap branch
  - today that `ggml-worker` branch is metadata-only:
    - `tokenize_text`
    - `tokenize_generation_prompt`
    - `decode_token_ids`
    - `eos_token_id`
  - forward/sample ops on `ggml-worker` fail explicitly against the validated
    stage binding manifest
  - this is the seam the first real `ggml` execution op should replace without
    touching provider or worker transport code again
- There is now also a validated binding manifest for the packed stage artifact
  at `crates/compute-daemon/src/inference/ggml_stage_manifest.rs`.
  - it is built from `StageTensorStore::model_view()`, not a second ad hoc
    classifier in the daemon
  - it validates the core per-layer attention/FFN contract before the worker
    launches
  - it records the real head/shared/tail tensor surface of the packed stage
  - the bootstrap provider now prints that binding summary in its load error
    for both head and tail stages
  - on the defended Gemma 2-stage packs, the binding summary is clean:
    no unknown layer tensors, full norm/projection coverage, and expected
    logits/output-norm availability by stage role
- There is now also a typed operator plan in
  `crates/compute-daemon/src/inference/ggml_stage_plan.rs`.
  - it is derived from the validated binding manifest
  - it records explicit shared and per-layer operator bindings for the future
    `ggml-worker` executor
  - the provider and worker error surfaces now include that operator summary
  - head `begin_token_ids` now also emits a concrete `begin_plan=...`
  - downstream/tail `continue_forward` now also emits a concrete
    `continue_plan=...`
  - so the first real `ggml` execution slices on both sides of the 2-stage
    split can target validated execution plans instead of raw packed tensor
    names
- There is now also a bound execution recipe on the worker path.
  - it reuses the existing `stage-forward-lab` execution-program order
  - it binds each op to exact packed-stage tensor entries, GGML types, and
    dimensions
  - the live `ggml-worker` error path now includes `recipe=...` for the staged
    execution slice
  - so the first real `ggml-worker` compute implementation can target:
    - a validated op-specific plan
    - a validated op order
    - exact bound packed tensors
- There is now also a materialized execution recipe on the live worker path.
  - it reads the bound tensors from the packed stage store
  - it records stable byte hashes for the staged execution slice
  - it reports unique tensor count and total tensor bytes
  - the live `ggml-worker` error path now includes `materialized=...`
  - so the first real `ggml-worker` compute implementation now has verified
    byte-backed tensors available at the same boundary where the op-specific
    plan and bound op order are already exposed
- `ggml-worker` now owns one real compute op itself:
  - tail `sample_tail`
  - it runs from the typed `sample_tail` plan plus the packed stage store
  - it applies output RMS norm and logits projection directly from the bound
    tensors instead of bouncing through `RealGemmaBackend`
  - it passes on the defended Gemma 2-stage core suite both:
    - directly at the worker boundary via
      `ggml_stage_worker_sample_compare`
    - and through the provider seam via
      `real_forward_provider_bootstrap_compare`
- `ggml-worker` now also owns the first real head execution slice:
  - head `begin_token_ids` for `debug_layer_cap=0`
  - that slice covers:
    - token embedding lookup/scaling
    - prompt-aux / PLE materialization
    - hidden-state payload framing
  - it passes on the defended Gemma 2-stage core suite directly at the worker
    boundary via `ggml_stage_worker_forward_compare ... ggml 0`
  - hidden-state bytes and prompt-aux bytes both match `cpu-ref`
- that fresh-request worker-boundary parity now also extends to:
  - `debug_layer_cap=1`
  - `debug_layer_cap=4`
  - `debug_layer_cap=21` for the full defended head stage
- The head compare gate is now stricter:
  - `GgmlStageWorkerTensorSummary::hidden_contract_matches(...)` now also
    requires aux byte length/hash parity
  - that caught a real bug in the first head cap-`0` attempt: it emitted only
    stage-local PLE aux, while the reference head path emits full-model PLE aux
    coverage in the split-boundary payload
- Current defended `ggml-bootstrap` routing is now:
  - metadata -> `ggml-worker`
  - head `begin_token_ids` -> `ggml-worker`
  - downstream hidden-state forward -> `ggml-worker`
  - tail `sample_tail` -> `ggml-worker`
- The uncapped provider path now also clears:
  - `real_forward_provider_bootstrap_compare`
  - `real_forward_provider_bootstrap_continuation_compare`
- There is now also a dedicated head-ingress benchmark:
  - `ggml_stage_worker_head_ingress_bench`
  - it builds `cpu-ref-worker` and `ggml-worker` from the same stage-worker
    init path with `debug_layer_cap=0`
  - it isolates head `begin_token_ids` ingress instead of benchmarking worker
    transport or full head-layer execution
  - initial core-suite result on Metal (`5` iterations):
    - `cpu-ref-worker` average: about `2.6 ms`
    - `ggml-worker` average: about `23.1 ms`
    - hottest `ggml-worker` bucket: `ple_model_proj` at about `19.2 ms`
  - after batching the `per_layer_model_proj` `ggml` path by prompt length:
    - `cpu-ref-worker` average: about `2.6 ms`
    - `ggml-worker` average: about `6.0 ms`
    - hottest `ggml-worker` bucket: `ple_model_proj` at about `2.1 ms`
  - after flattening prompt-aux storage/encoding and rerunning the same bench in
    isolation:
    - `cpu-ref-worker` average: about `2.9 ms`
    - `ggml-worker` average: about `3.9 ms`
    - hottest `ggml-worker` bucket: `ple_model_proj` at about `2.3 ms`
  - after fusing batched `per_layer_model_proj`, proj RMS norm, and token-embed
    combine into one persistent `ggml` PLE ingress graph:
    - `cpu-ref-worker` average: about `2.6 ms`
    - `ggml-worker` average: about `3.2 ms`
    - hottest `ggml-worker` bucket: fused `ple_model_proj` at about `2.5 ms`
  - that narrows the next optimization target to the remaining fused PLE
    ingress graph cost first, then payload-encoding / worker-overhead work

There is now also a dedicated head continuation debug gate:
- `ggml_stage_worker_head_continuation_compare`
- it keeps one persistent head worker session alive across repeated
  `begin_token_ids` calls under the same request ID
- it compares three things on each repeated head step:
  - `cpu-ref` incremental head output
  - `ggml-worker` repeated-request head output
  - fresh full-history recompute for the accumulated token history

Current result from that gate:
- repeated head continuation in `ggml-worker` now matches `cpu-ref`
  incremental hidden-state bytes
- `cpu-ref` incremental still differs from fresh full-history recompute on some
  repeated steps, which means the old failure mode was reference-side
  incremental-vs-fresh behavior, not a remaining `ggml-worker` continuation bug
- that is true at:
  - `debug_layer_cap=1`
  - `debug_layer_cap=21` for the defended `0-20` head stage

There is now also a provider-side continuation result that matters:
- `real_forward_provider_bootstrap_continuation_compare` passes on the defended
  Gemma 2-stage core suite
- that required two fixes:
  - rebuilding the `cpu-ref` reference providers per case in the compare
    harness, so prompt-cache pollution does not masquerade as `ggml` drift
  - keeping the provider's persistent `ggml-worker` tail execution session
    alive across repeated continuation steps

So the next execution milestone is narrower than it looked:
- not fresh head forward parity
- not head continuation semantics
- specifically: replace the bootstrap worker's CPU-side hidden-state execution
  with real accelerated `ggml` kernels

There is now also one real accelerated worker-side graph runtime in the live
path:
- `ggml-worker` tail `sample_tail` builds a persistent `ggml` graph runtime
  from the validated stage plan and packed stage store
- on Apple Silicon that runtime now comes up on Metal and reports a live backend
  label like:
  - `ggml-graph-sample backend=MTL0 hidden_dim=2560 vocab_size=262144`
- direct worker sample parity still passes through
  `ggml_stage_worker_sample_compare`
- provider seam parity still passes through
  `real_forward_provider_bootstrap_compare`
- because graph/runtime bring-up is slower than the old bootstrap path, the
  persistent worker ready timeout was raised to avoid false startup failures

There is now also a defended tail-layer gate:
- `ggml_stage_worker_tail_compare` accepts `debug_layer_cap`
- `ggml_stage_worker_tail_continuation_compare` accepts `debug_layer_cap`
- with `debug_layer_cap=1`, `2`, and `4`, both the raw tail forward gate and
  the repeated tail continuation gate pass on the defended Gemma 2-stage core
  path
- those capped paths now run through a real persistent `ggml` tail-stack
  runtime inside the worker instead of bootstrap CPU matmuls
- the uncapped default tail provider path now also passes through:
  - `real_forward_provider_bootstrap_compare`
  - `real_forward_provider_bootstrap_continuation_compare`

That means the next runtime swap can keep growing the same real tail runtime
layer by layer instead of jumping straight from layer 1 to the full tail stage.

Current boundary:
- `cpu-ref` remains the production correctness oracle
- `ggml-bootstrap` now owns the full provider seam, including continuation-state
  ownership
- provider-side head execution is defended on `ggml-worker` and now uses a
  real persistent `ggml` head-stack runtime by default
- downstream hidden-state forward is defended on `ggml-worker` and uses a real
  persistent `ggml` tail-stack runtime by default
- tail `sample_tail` now has a real accelerated `ggml` graph runtime
- the remaining acceleration gap is now the worker-side CPU ingress path
  around the head graph; token embedding lookup, per-layer token embedding
  lookup, and the PLE model projection now also have defended `ggml` runtime
  paths, so what remains is the CPU-side scaling, normalization, payload
  framing, and worker/runtime overhead
