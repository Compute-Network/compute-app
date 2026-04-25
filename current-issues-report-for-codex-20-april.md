# Current Issues Report for Codex — 20 April 2026

## Scope

This report is for the **llama-stage-backend split inference path** in `compute-app`, focused on:
- staged **head/tail** inference for **Gemma-4-E4B-Q4**
- `RemoteStageGateway` / managed stage nodes
- speculative decoding with **Gemma-3-270M** as the draft model
- current production target: `api.computenetwork.sh`

Relevant code:
- `crates/llama-stage-backend/src/lib.rs`
- `crates/llama-stage-backend/src/bin/spec_gateway_probe.rs`
- `crates/compute-daemon/src/inference/llama_stage_gateway.rs`

## Current backend focus

Backend currently under active investigation is:
- **`llama-stage-backend` split inference / staged gateway path**

This is the path that runs a model in two parts:
- **head stage**
- **tail stage**
- optional **speculative decoding** on top of the split pipeline

## Latest shipped change

Shipped in **v0.3.6**:
- stage tensor JSON bytes now serialize as **base64** instead of serde_json's default **array of numbers**
- this change was made in `crates/stage-forward-lab/src/lib.rs`
- goal: reduce wire size and parse overhead on the split inference transport path

Commit pushed:
- `5868675` — *Shrink stage tensor JSON payloads with base64 encoding.*

## Current measured results

### 1. Local split-inference probe on one Mac simulating two devices

Latest successful end-to-end run after the transport patch:
- **baseline (spec OFF): 2.24 tps**
- **spec ON: 4.69 tps**
- **speedup: 2.09x**
- **ttft: 880 ms**
- **prefix agreement: 13/13 (100%)**

This was run with:
- head shard: `head-0-20.gguf`
- tail shard: `tail-21-41.gguf`
- draft model: `gemma-3-270m-it-Q4_K_M.gguf`
- probe: `crates/llama-stage-backend/src/bin/spec_gateway_probe.rs`

### 2. Earlier local loopback validation on symmetric hardware

Previously validated result on a single M-series Mac:
- **baseline (spec OFF): 2.84 tps**
- **spec ON: 6.58 tps**
- **speedup: 2.32x**
- **80/80 tokens identical**

Interpretation:
- speculative decoding itself is working
- the algorithm is capable of a real speedup locally
- production underperformance is not explained by spec correctness

### 3. Production / cross-machine result

Observed production behavior (VM + Mac topology):
- around **1.4–1.6 tps**
- increasing K from **4/4** to **6/12** increased average proposal depth but reduced acceptance and did **not** materially improve throughput

Interpretation:
- production bottleneck appears to be **topology / hardware asymmetry / slow-node compute**, not speculative-decoding logic alone

## Main current issues

### Issue 1 — Split inference is still far slower than expected versus monolithic local inference

Current split numbers are still much lower than expected relative to a full local monolithic run.

User expectation / working assumption:
- full local **Gemma-4 small** monolithic model likely runs at roughly **60–70 tps** on the same machine
- this number has **not yet been re-verified in this session** because the full monolithic GGUF was not found locally at test time

Meaning:
- even with transport improvements, staged split inference is still dramatically below single-node local throughput
- main gap remains the key engineering problem

### Issue 2 — Production performance is topology-limited

Strong evidence suggests production speed is dominated by one or more of:
- slower VM node
- asymmetric hardware between peers
- per-token compute dominating RTT savings
- speculative decode amortization not large enough to overcome slow-stage execution

Current conclusion:
- **do not assume K tuning will solve the production bottleneck**
- K=6/12 did not materially fix throughput and reduced acceptance

### Issue 3 — Local transport overhead was real, but not the whole story

The previous wire format serialized `Vec<u8>` tensor payloads as JSON number arrays, which caused:
- larger payloads
- higher CPU parse cost
- unnecessary overhead on each stage handoff

This is now improved via base64 JSON encoding in `v0.3.6`, but latest split result is still only:
- **2.24 tps baseline**
- **4.69 tps spec**

So transport was a real tax, but it is **not the only bottleneck**.

### Issue 4 — We still need a clean apples-to-apples single-node benchmark

Missing measurement right now:
- exact local **single-node same-model same-prompt** TPS on the monolithic GGUF

This is important because it would quantify the true penalty of:
- head/tail split execution
- gateway orchestration
- inter-stage transport
- duplicate/runtime overhead versus monolithic execution

## Most likely bottleneck ranking right now

1. **Per-stage compute cost is still too high**
   - split path likely pays much more than expected in stage execution itself
   - especially harmful if one side is materially slower

2. **Cross-stage orchestration overhead is still substantial even after base64**
   - transport improved, but split path still has request/response sequencing and serialization overhead

3. **Spec decode helps, but only multiplies an already slow base path**
   - spec is a win on top of baseline
   - but if baseline staged execution is too slow, final TPS remains low

4. **Production topology asymmetry**
   - likely the reason production remains around ~1.5 tps even though local spec proved much better

## What has already been validated

Validated:
- speculative decoding is functionally correct on the split path
- local spec path can outperform local split baseline by about **2.1–2.3x**
- base64 wire patch builds and its JSON roundtrip test passes
- v0.3.6 has been committed, pushed, and tagged

Also fixed during this work:
- locally copied sidecar binaries on macOS can hang silently until re-signed
- required workaround when copying rebuilt sidecars into `~/.compute/bin/`:

```bash
codesign --force --sign - ~/.compute/bin/llama_stage_tcp_node
codesign --force --sign - ~/.compute/bin/llama_stage_gateway_tcp_node
```

## Highest-priority next checks

### A. Re-measure true monolithic local baseline

Need exact TPS for:
- full single-node Gemma-4-E4B-Q4 (same prompt, same machine)

Goal:
- confirm or reject the current working belief that monolithic local speed is around **60–70 tps**

### B. Compare split baseline against monolithic directly

Need a clean ratio:
- **monolithic TPS** vs **split baseline TPS** vs **split speculative TPS**

That ratio will show where the real penalty is.

### C. Profile where time is spent in staged execution after v0.3.6

Need a fresh timing breakdown across:
- head compute
- tail compute
- gateway overhead
- stage handoff / serialization
- per-token control flow

### D. Re-test production on two machines after v0.3.6 deploy

Need to verify whether the base64 transport patch moves the real networked result above the previous ~1.4–1.6 tps band.

## Working summary for Codex

If Codex is investigating the split inference bottleneck, the current working picture is:

- **Spec decode is not the primary problem** — it is working and gives about **2x+** lift locally.
- **Production remains slow mainly because the base split path is slow and topology is asymmetric.**
- **JSON tensor transport was a real bug/tax and has now been reduced in v0.3.6**, but current split TPS is still far below expected monolithic local performance.
- The biggest unanswered question is still:
  - **Why is staged split execution so much slower than a full local monolithic run on the same class of hardware?**

## Useful commands

### Local split probe

```bash
HEAD_MODEL=~/.compute/stages/gemma-4-e4b-q4/head-0-20.gguf \
TAIL_MODEL=~/.compute/stages/gemma-4-e4b-q4/tail-21-41.gguf \
HEAD_START=0 HEAD_END=20 TAIL_START=0 TAIL_END=20 \
./target/release/spec_gateway_probe \
  ~/.compute/models/gemma-3-270m-it-Q4_K_M.gguf \
  ~/.compute/stages/gemma-4-e4b-q4/head-0-20.gguf \
  "The capital of France is" \
  40
```

### Single-node baseline probe (needs full monolithic GGUF path)

```bash
./target/release/llama_single_node_baseline_probe \
  /path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  "The capital of France is"
```
