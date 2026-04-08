# Research Findings — April 2026

Compiled from 6 investigation rounds across codebase audits, industry research, competitor analysis, and Twitter/X sourcing.

---

## 1. llama.cpp Developments (March-April 2026)

### Speculative Decoding / Multi-Token Prediction
- Self-speculative MTP landed: model predicts multiple tokens at once, no separate draft model needed
- With separate draft model (e.g., Llama-3.2-1B drafting for 70B): **4-5x speedup** (30 → 160 t/s on RTX 3090)
- Enable with `--speculative` flags
- **Action:** Enable for CUDA nodes. MTP works automatically on newer model architectures

### RPC Distributed Backend (GGML_RPC)
- Native llama.cpp layer splitting across machines over TCP
- Layers distributed proportional to each device's VRAM via `--tensor-split`
- `mesh-llm` reference implementation shows coordination with llama-server
- Local disk caching avoids re-transferring large tensors on reconnect
- **Action:** Prototype as foundation for pipeline parallelism (implemented in this sprint)

### Quantization Updates
- Q4_K_M remains the 2026 sweet spot
- TurboQuant (TQ3_0): 3.5-bit KV cache quantization, still landing (CPU + Vulkan)
- AWQ scales integration into GGUF pipeline in progress
- New modular quantization recipe system (PR #21070)

### Known Issues
- Threading regression (Issue #21042): 50% slowdown with >16 threads on some quantized models
- Keep `-t 6` for M3 Pro (6 P-cores)

---

## 2. Apple Silicon Inference (2026)

### vllm-mlx — Major New Player
| Framework | Single-User | 32-User Throughput | TTFT |
|-----------|------------|-------------------|------|
| vllm-mlx | 42 t/s | **1,150 t/s** | ~120ms |
| Ollama | **58 t/s** | 720 t/s | **~45ms** |
| llama.cpp | 52 t/s | 890 t/s | ~85ms |

- PagedAttention on MLX, continuous batching, prefix caching (5.8x TTFT speedup)
- OpenAI-compatible API
- Docker Model Runner now supports vLLM on macOS Metal (Docker Desktop 4.62+)
- **Trade-off:** Better concurrency than llama.cpp, but less portable and narrower model support

### MLX + Apple Silicon Trajectory
- Apple published M5 Neural Accelerator paper (Jan 2026): 4x TTFT speedup vs M4
- Apple explicitly optimizing silicon for MLX compute patterns
- Long-term Apple commitment to MLX confirmed
- Memory bandwidth (not compute) is the bottleneck for token generation

### Evaluate Later
- vllm-mlx as optional Apple Silicon backend (major throughput gain)
- Rapid-MLX: 2-4x faster than llama.cpp, DeltaNet state snapshots for multi-turn TTFT <200ms

---

## 3. KV Cache Innovations

### llm-d Prefix-Cache Aware Scheduling
- vLLM pods emit "KVEvents" signaling cache block creation/eviction
- Global KV-Block Index maps block-hashes to pod locations
- Scheduler computes "cache affinity score" + load-aware metrics
- Results: **57x faster TTFT**, 25% throughput increase, 170x vs cache-blind routing
- Metadata overhead: 339KB to track a 365GB cache pool
- **Action:** Implemented prefix-affinity routing in this sprint

### Disaggregated Prefill/Decode
- Splitting prefill (compute-bound) and decode (memory-bound) onto separate GPU pools
- NVIDIA Dynamo: 30x for DeepSeek-R1 671B
- Relevant for future multi-node pipeline where stages specialize

### FP8 KV Cache
- Now in vLLM, ~50% memory reduction
- llama.cpp has q8_0/q4_0 KV cache (already enabled in our config)

---

## 4. Distributed Inference

### Prime Intellect (Most Technically Similar)
- Pipeline parallelism over public internet with heterogeneous consumer HW
- TOPLOC verification: LSH of activations + blame assignment
- Key finding: **async micro-batch schedules don't help inference** (memory-bandwidth-bound, not compute-bound)
- Variable "thinking budgets" per node based on VRAM
- Open-sourced: PRIME-IROH, PRIME-VLLM, PRIME-PIPELINE
- No economic/token layer (research only)
- **Action:** TOPLOC verification implemented in this sprint

### Exo Labs
- Now has RDMA over Thunderbolt 5 (99% latency reduction for LAN)
- Tensor parallelism (within-layer split) vs our pipeline parallelism (across-layer split)
- Good for LAN clusters, not internet-distributed nodes
- No economic layer, no fault tolerance, no multi-user support

### Key Architecture Insight
- Activation transfer between pipeline stages: ~32KB per layer (fp16, 16384 hidden dims)
- Latency matters more than bandwidth for distributed inference
- Pipeline parallelism works over internet; tensor parallelism requires LAN/RDMA

---

## 5. Solana DePIN Patterns

### Official Recommendations (Solana DePIN Quickstart)
- **cNFTs for node identity**: 2M nodes mintable for ~1 SOL (Helium proved at 991K scale)
- **Claim-based rewards**: Users claim via Merkle proofs — most gas-efficient
- **ZK Compression**: 10K distributions for ~1.83 SOL vs ~20+ SOL traditional
- Keep data off-chain, only proofs/roots on-chain

### Recommended Stack
- Merkle Distributor SDK (Jupiter reference) for claim-based rewards
- Helius AirShip for ZK-compressed airdrops
- Tuktuk (Helium) for automated oracle distribution
- **Action:** Detailed analysis provided separately (evaluate before implementing)

---

## 6. Competitor Analysis

### Strategic Position
**Pipeline parallelism in production DePIN is unoccupied.**

| Competitor | Pipeline Parallel? | Consumer HW? | Token? | Production? |
|---|---|---|---|---|
| Exo | Local only | Yes | No | Yes (local) |
| Prime Intellect | Research | Yes | No | No |
| io.net | No | No (datacenter) | Yes | Yes |
| Akash / AkashML | No | No (datacenter) | Yes | Yes |
| Nosana | No | Yes | Yes (Solana) | Yes |
| Bittensor | No | Mixed | Yes | Yes |
| Hyperbolic | No | No | Yes | Yes |
| **Compute** | **Yes** | **Yes** | **Yes (Solana)** | **Building** |

### Key Competitive Insights
- **Nosana** adding Apple Silicon support (H1 2026) — direct Solana competitor
- **Prime Intellect** biggest technical threat if they add an economic layer
- **Akash's AkashML** setting UX bar for managed inference (OpenAI-compatible)
- **Bittensor** gaming vulnerabilities highlight need for verification > staking games

### What to Adopt
- Hyperbolic's **Proof of Sampling**: dynamic verification rates based on node reputation
- Prime Intellect's **TOPLOC**: activation hashing + blame assignment at pipeline boundaries
- Akash's **OpenAI-compatible managed inference** API pattern
- Exo's **zero-config device discovery** UX

### Biggest Risks
1. Prime Intellect adding a token/incentive layer
2. Memory-bandwidth bottleneck means pipeline parallelism doesn't scale linearly for decode
3. Nosana competing on same chain (Solana) for same provider base

---

## 7. Twitter/X Findings

### @dealignai (Jinho Jang) — HIGH relevance
- JANG mixed-precision quantization: 8-bit attention, 4-bit MLP
- Fits 31B models in 18GB on Apple Silicon (Gemma-4-31B-JANG_4M-CRACK)
- MLX-native safetensors format
- **Action:** Consider adding JANG-quantized models to catalog

### @Av1dlive — HIGH relevance
- ClawRouter: multi-dimensional LLM request scoring across 55+ models
- x402 micropayment protocol on Solana for agent-native payments
- 92% cost reduction via smart routing
- **Action:** Study for scheduler scoring improvements

### @doodlestein (Jeffrey Emanuel) — MEDIUM relevance
- DCG: Rust destructive command guard for AI agents (SIMD-accelerated)
- FrankenTUI: ultra-optimized Rust TUI with SIMD rendering
- **Action:** Study DCG patterns for daemon safety

### @open_founder (Tim Hafner) — MEDIUM relevance
- BRAID: large model creates reasoning graph, small model executes (70x cost reduction)
- OpenServ multi-agent orchestration
- **Action:** Conceptually interesting for pipeline optimization

---

## 8. Code Audit Findings (Remaining)

### Critical Bugs
1. `dirs::home_dir()` fallback creates literal `~` directory (config.rs, service.rs)
2. Non-atomic config writes (corruption on crash)
3. Orchestrator intervals have no jitter (thundering herd)
4. Rate limiting per-process only (doesn't work across replicas)
5. `claimRateLimits` Map never GC'd (memory leak)
6. Globe projection depth threshold off (`rz > -0.1` should be `> 0.0`)

### Performance
- `sysinfo::System::refresh_all()` called when only CPU/memory needed
- Config cloned 7 times in main.rs
- No network/disk I/O metrics in heartbeat (needed for pipeline routing)

### Missing for Production
- No distributed rate limiting (Redis needed for multi-replica)
- No Prometheus metrics from llama-server `--metrics` endpoint
- No query timeouts on Supabase calls in middleware
- Wallet registration needs signature proof (currently accepts any wallet)
