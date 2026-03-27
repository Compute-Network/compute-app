# Tech Research — Distributed Pipeline-Parallel Inference

Research on open-source projects we can build on instead of starting from scratch.

---

## Prime Intellect (PRIME-VLLM)

### What it is

A thin Python wrapper (~250 lines) around stock vLLM that adds pipeline-parallel inference over the internet. **Not a vLLM fork** — it uses forward hooks to intercept activations between layers and ship them between nodes.

- Repo: `github.com/PrimeIntellect-ai/prime-vllm` (no license file, ~40 stars)
- Transport lib: `github.com/PrimeIntellect-ai/prime-iroh` (**MIT licensed**, Rust + Python via PyO3)

### How it works

1. Model is pre-sharded via `shard.py` — slices HuggingFace model layers into contiguous chunks, saves each as a standalone HF model
2. Each node loads its shard into stock vLLM with `enforce_eager=True` and `VLLM_USE_V1=0` (legacy engine, no CUDA graphs)
3. Forward hooks on first/last layers intercept hidden states:
   - First layer: receives activations from previous pipeline stage
   - Last layer: sends activations to next pipeline stage
   - Sampler hook: last rank sends sampled tokens backward through a ring back to rank 0
4. Serialization: plain `pickle.dumps()` on CPU tensors (no compression — optimization opportunity)

### Transport: QUIC via Iroh

The P2P layer (`prime-iroh`) wraps **Iroh** (n0-computer, 8100+ stars, Apache 2.0) — a Rust networking stack built on QUIC.

- NAT traversal: tries direct UDP first, QUIC hole punching, falls back to public relay servers
- QUIC multipath: relay and direct paths are first-class QUIC paths, congestion controller chooses optimally
- Discovery: uses n0's discovery service. Nodes identified by Ed25519 public keys, not IPs
- Wire format: 4-byte LE length prefix + raw bytes. Multiple named streams per connection for concurrent micro-batch comms
- Supports artificial latency injection for benchmarking

### Critical findings from their research

1. **Micro-batching does NOT help inference throughput** — unlike training, decode is memory-bandwidth-bound, not compute-bound. Processing 1 or 2 sequences takes roughly the same time. No compute to overlap with network transfer.
2. **Hard ceiling**: at 100ms latency with 2 pipeline stages, max throughput is ~5 tok/s regardless of GPU speed. Fundamental physics constraint.
3. **Synchronous pipeline schedules (as in vLLM) are a strong baseline** for inference over internet.
4. Future directions they identify: increase compute density during network waits, reduce memory footprint (lighter KV cache), enable asynchronous execution that tolerates variable latency/availability.

### Scale demonstrated

- SYNTHETIC-2 campaign: 1,253 GPUs (49x 8xH200, 43x 8xH100, plus consumer 3090s and 4090s), ran DeepSeek-R1-0528 (671B MoE), generated 4M verified reasoning traces in 3 days
- No published per-pipeline throughput numbers

### Fault tolerance

- **None within a pipeline** — if a node dies, everything stops. No reconnection, checkpoint, or retry.
- Orchestrator-level fault tolerance exists in their separate `protocol` repo (task rescheduling when nodes fail)
- TOPLOC (verifiable inference) can pinpoint faulty workers and slash them

### What's reusable for Compute

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| `prime-iroh` (MIT) | **Yes — best candidate** | Clean Rust/Python QUIC P2P with NAT traversal. `pip install prime-iroh` |
| Hook-based vLLM integration | Study, don't copy | Clever but fragile — depends on vLLM internals that change between versions |
| Model sharding approach | Yes | Simple layer slicing, saves as standalone HF models |
| `prime-pipeline` repo | Study | Better abstractions (comm, serialization, offload) but uses GPT-Fast, not vLLM |
| `protocol` repo (orchestration) | Study | Rust, actively maintained, but tightly coupled to their platform |

### What we'd need to add

1. **Fault tolerance within a pipeline**: reconnection, KV cache checkpointing, node replacement mid-inference
2. **Dynamic/uneven sharding**: assign more layers to faster GPUs (currently requires equal layer counts)
3. **Activation compression**: replace pickle with FP8/INT8 quantized transfer (~4x bandwidth savings)
4. **Continuous batching**: prime-vllm uses offline `generate()`, not serving engine. Need AsyncLLMEngine or OpenAI-compatible server
5. **Speculative execution**: multiple pipeline groups process speculatively to hide latency

---

## Gradient Network (PARALLAX)

### What it is

A distributed model serving framework that turns heterogeneous, geographically-dispersed hardware into a unified inference cluster. **First-class Apple Silicon support with custom Metal shaders.**

- Repo: `github.com/GradientHQ/parallax` (**Apache 2.0**, ~1,177 stars, actively maintained — last push 3 days ago)
- Paper: arXiv 2509.26182
- Language: Python 77.5%, TypeScript 12.3%, Metal 6.3%, C++ 1.3%

### How it works

**Pipeline parallelism only** (same conclusion as Prime Intellect — tensor parallelism too bandwidth-hungry for internet).

**Two-phase scheduler:**

- **Phase 1 (Offline — model allocation):** Decides which node gets which layers. Uses dynamic programming with water-filling heuristic. Distributes layers proportionally to node compute power (TFLOPS FP16) and memory bandwidth. Enforces contiguous layer intervals per GPU.
- **Phase 2 (Per-request — route selection):** Treats Phase 1 output as a DAG, uses DP to find minimum-latency path through available servers. Nodes publish latency metrics every 1-2 seconds to a DHT. Stale data expires in 60 seconds.

**Three routing strategies:**
1. DP routing (optimal latency)
2. Randomized selection over dynamically discovered pipelines (load balancing)
3. Round-robin over fixed pre-registered pipelines (simple, stable)

**Rebalancing triggers:** When no full pipeline covers all layers, or load coefficient of variation exceeds 0.25.

### Transport/networking

- **Lattica** — Gradient's own P2P framework built on **libp2p**
- **gRPC + Protocol Buffers** for RPC (forward pass requests via streaming gRPC)
- **ZeroMQ** for internal IPC
- **Hivemind's DHT** for decentralized peer discovery (60-second TTL announcements)
- **NAT traversal** via libp2p circuit relays and DCUtR (hole punching)
- Model weights distributed via block storage with checksum validation

### Apple Silicon support (key differentiator)

- Uses **MLX** as the Mac inference backend
- **Custom hand-written Metal shaders** for paged attention — including model-specific variants for DeepSeek V3.2 and GPT-OSS
- **Paged Flash Attention in Metal** — first in the MLX ecosystem for distributed inference
- Paged KV Cache manager with block-based virtual-memory-inspired design
- Continuous batching on macOS
- For NVIDIA: uses SGLang and vLLM backends

### Performance numbers (from paper)

Tested on 5x RTX 5090 + 2x RTX 4090, public internet, ~10ms inter-node:
- **vs HexGen:** Up to 3.6x higher throughput, up to 3.2x lower latency
- **vs Petals:** 3.1x lower end-to-end latency, 5.3x lower inter-token latency, 3.1x higher throughput
- Scheduler overhead: Phase 1 = 0.10ms (4 GPUs) to 8.55ms (256 GPUs); Phase 2 = 0.014ms/req to 6.6ms/req

### Models supported (40+)

DeepSeek V3.2/R1, MiniMax M2/M2.1 (230B sparse MoE), Qwen 3/3-Next/2.5, Llama 3/3.1/3.2/3.3 (8B-70B), GLM 4.7, Kimi-K2, GPT-OSS (20B, 120B).

### Fault tolerance

- Pipeline rejection + abort broadcasting when a node fails
- No KV cache recovery (Petals does this better)

### What's reusable for Compute

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| Scheduling algorithms | **Yes — study closely** | Two-phase DP scheduler is well-documented in the paper. Water-filling for heterogeneous hardware is exactly what we need |
| Metal kernels | **Yes (Apache 2.0)** | Hand-written paged attention for Apple Silicon. Could port or reference |
| Layer allocation logic | **Yes** | `src/scheduling/layer_allocation.py` — modular, clean |
| Routing logic | **Yes** | `src/scheduling/request_routing.py` — DP/Random/RoundRobin |
| Full framework | Maybe | Large dependency surface (Lattica, Hivemind, gRPC, ZMQ). Could use as a reference rather than dependency |

---

## Petals (BigScience)

### What it is

BitTorrent-style collaborative LLM inference and fine-tuning. Servers self-select which transformer blocks to host. Clients chain servers together.

- Repo: `github.com/bigscience-workshop/petals` (**MIT**, ~10,027 stars)
- Paper: arXiv 2209.01188 (ACL 2023)
- **Semi-dormant** — last commit Sept 2024, 112 open issues

### How it works

- Pipeline parallelism. Each server hosts consecutive transformer blocks.
- **Servers self-select blocks** — examine DHT to find which blocks have worst throughput, serve those. No central coordinator.
- **Clients route via beam search** — ping nearby servers, find optimal-latency chain.
- Transport: **Hivemind library** (libp2p underneath), Kademlia DHT for discovery.
- **Dynamic blockwise quantization** for activation compression — halves bandwidth with no noticeable quality loss.

### Fault tolerance (best of the three)

- Clients cache all intermediate activations sent to each server
- If a server dies, client reroutes and replays cached activations to replacement server to restore KV cache state
- Graceful degradation without coordinator

### Apple Silicon

Limited — PyTorch MPS backend only, no MLX, no custom Metal kernels. Fundamentally designed around CUDA.

### Performance

- BLOOM-176B on 14 mixed servers: 0.83 steps/second
- ~1 token/sec for 176B model, ~4-6 tokens/sec for 70B
- 8 concurrent clients: ~20% slowdown per client
- Min requirements: 8GB GPU VRAM, 100 Mbit/s bandwidth

### What's reusable for Compute

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| Activation caching for fault tolerance | **Study closely** | Best fault tolerance design of the three |
| Self-selection algorithm | Study | Elegant decentralized approach, but we have an orchestrator |
| Fine-tuning architecture | Future reference | Frozen weights + local LoRA adapters. Multiple users fine-tune simultaneously |
| Blockwise activation quantization | **Yes — implement this** | 2x bandwidth savings with no quality loss |

---

## Head-to-Head Summary

| Dimension | Prime Intellect | Parallax | Petals |
|-----------|----------------|----------|--------|
| **Transport** | QUIC (Iroh) | libp2p + gRPC | libp2p (Hivemind) |
| **Apple Silicon** | No | First-class (MLX + Metal) | Basic (MPS) |
| **Scheduling** | None (manual) | Two-phase DP (best) | Decentralized self-selection |
| **Fault tolerance** | None | Pipeline abort | Activation cache + reroute (best) |
| **Activation compression** | None | None documented | Blockwise quantization (best) |
| **License** | No license (iroh=MIT) | Apache 2.0 | MIT |
| **Maintenance** | Minimal | Active | Dormant |
| **Performance** | Theoretical analysis only | 3.1x faster than Petals | Baseline |
| **Fine-tuning** | No | No | Yes (LoRA, soft prompts) |

---

## Recommended approach for Compute

**Don't pick one — combine the best ideas from all three:**

1. **Transport: Iroh (from Prime Intellect)** — MIT-licensed Rust library with QUIC, NAT traversal, hole punching. Clean, minimal, fits our Rust stack. `prime-iroh` wraps it with tensor-aware send/recv.

2. **Scheduling: Parallax's two-phase DP** — well-documented algorithm for heterogeneous hardware. Water-filling for layer allocation + per-request DP routing. Implement in Rust in our orchestrator crate.

3. **Apple Silicon inference: Reference Parallax's Metal kernels** — Apache 2.0 licensed paged attention shaders. Study their MLX integration for our native Metal path.

4. **Fault tolerance: Petals' activation caching** — clients/coordinators cache intermediate activations. On node failure, replay to replacement node to restore KV cache. Adapt for our orchestrator-coordinated model.

5. **Activation compression: Petals' blockwise quantization** — 2x bandwidth savings for free. Critical for consumer internet connections.

6. **Model sharding: Prime Intellect's approach** — simple layer slicing, save as standalone HF models. Pre-compute shards for common configurations.

**Key constraint to design around:** At 100ms inter-node latency with N stages, max throughput is ~(1000 / (N * 100ms)) tokens/sec. With 5 nodes = ~2 tok/s per pipeline. Compensate with many parallel pipelines and batch workloads (not real-time chat).
