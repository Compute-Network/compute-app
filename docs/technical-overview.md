# Compute — Technical Documentation

Reference documentation for the Compute network. Intended for website docs, contributor guides, and internal reference.

---

## How Pipeline Parallelism Works

Compute uses **pipeline parallelism** to run AI models too large for any single machine. Instead of splitting each layer across GPUs (tensor parallelism, which requires datacenter-grade bandwidth), we split the model's layers into sequential stages and chain consumer devices together.

### Example: Running a 250B Parameter Model

A 250B model has ~120 transformer layers requiring ~125GB of VRAM. No single consumer GPU has that much memory. Compute splits the work:

```
User request: "What is quantum computing?"
        │
        ▼
┌─────────────────────────────────────┐
│  Node A — RTX 4090 (24GB)          │
│  Layers 0-39                        │
│  Tokenizes input → runs 40 layers   │
│  Sends activation tensor (8-16KB)   │
│  ──── QUIC over internet ────▶      │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Node B — M3 Max (36GB)            │
│  Layers 40-79                       │
│  Receives activation → runs layers  │
│  ──── QUIC over internet ────▶      │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Node C — RTX 3090 (24GB)          │
│  Layers 80-119                      │
│  Runs final layers → samples token  │
│  Sends token back to Node A         │
└─────────────────────────────────────┘
        │
        ▼
  "Quantum..." → repeat for each token
```

### Why Pipeline Over Tensor Parallelism?

| Approach | Data transferred per layer | Works over internet? |
|----------|---------------------------|---------------------|
| **Tensor parallelism** | All-reduce after every operation (~GB/s needed) | No — requires NVLink/InfiniBand |
| **Pipeline parallelism** | One activation tensor per stage (~8-16KB per token) | Yes — works on regular broadband |

Pipeline parallelism only sends one small activation between stages, making it viable over consumer internet connections. The tradeoff is higher per-token latency (each token must pass through all stages), which we mitigate with micro-batching.

### Micro-Batching

While Node C processes token 1 of request A, Node A can start processing token 1 of request B. This keeps all GPUs busy and increases overall throughput even though individual request latency is higher.

### Layer Allocation

The pipeline scheduler uses a **water-filling algorithm** inspired by Parallax's two-phase dynamic programming approach. Layers are assigned proportional to each node's compute power (TFLOPS), with VRAM constraints respected:

- A node with 2x the TFLOPS gets ~2x the layers
- No node is assigned more layers than its VRAM can hold
- The last node in the pipeline gets any remaining layers

---

## Model Sharding & Downloads

### Nodes Only Download What They Need

When assigned to a pipeline, a node downloads only its **shard** — the subset of model layers it's responsible for. A node running layers 40-79 of a 120-layer model downloads ~33% of the total model size.

- **First assignment**: download shard (~10-40GB depending on model and quantization)
- **Subsequent requests with same model**: shard is cached locally
- **Reassignment to different layers**: download new shard, old one stays cached
- **Integrity**: each shard has a SHA256 checksum verified before use

### Supported Models

| Model | Parameters | Quantization | Total VRAM | Min Nodes |
|-------|-----------|-------------|-----------|-----------|
| Mistral 7B | 7B | Q4 | 4GB | 1 (any laptop) |
| Llama 3.1 8B | 8B | Q4 | 5GB | 1 |
| Llama 3.1 70B | 70B | Q4 | 35GB | 2-3 |
| Llama 3.1 70B | 70B | FP16 | 140GB | 6+ |
| Qwen 2.5 72B | 72B | Q4 | 37GB | 2-3 |
| DeepSeek R1 | 671B | Q4 (MoE) | 335GB | 5-10 |

Small models (7B-13B) can run entirely on a single laptop, earning 100% of request rewards. Larger models require multi-node pipelines.

---

## GPU & Platform Support

### Execution Backends

| Platform | Backend | How it runs |
|----------|---------|------------|
| **NVIDIA (Linux/Windows)** | CUDA | llama.cpp in Docker container with GPU passthrough |
| **Apple Silicon (macOS)** | Metal | llama.cpp native binary — no Docker needed |
| **CPU-only (all platforms)** | CPU | llama.cpp CPU mode — slower but works anywhere |

### Why Native on Apple Silicon?

Docker on macOS cannot access the Metal GPU. Apple Silicon nodes run llama.cpp as a native process with Metal acceleration, achieving competitive inference speeds without containerization.

---

## Idle Detection & Resource Priority

### Your Machine Comes First

Compute never interferes with your normal computer usage. The daemon monitors system activity every 2 seconds and adjusts its resource usage:

| System State | Detection Method | Compute Behavior |
|-------------|-----------------|-----------------|
| **Idle** | No user input for 5+ min, low CPU | Full allocation — GPU and CPU available for inference |
| **Light Use** | Normal browsing, typing | Reduced allocation — inference continues at lower priority |
| **Heavy Use** | Gaming, video editing, high GPU/CPU | **Paused** — inference stops immediately |
| **On Battery** | Battery power detected | **Paused** — preserves battery life |

### How Pausing Works

1. User starts a game → GPU usage spikes
2. Idle detector classifies state as "Heavy Use" within 2 seconds
3. Current inference token completes (no mid-computation corruption)
4. Node sends status update to orchestrator
5. Orchestrator reassigns layers to another available node (failover)
6. User closes game → after idle threshold → node rejoins the network

### Configuration

All thresholds are user-configurable in `~/.compute/config.toml`:

```toml
[node]
max_gpu_usage = 90          # Max % of GPU to use when idle
max_cpu_usage = 50          # Max % of CPU to use when idle
idle_threshold_minutes = 5  # Minutes of inactivity before "idle"
pause_on_battery = true     # Stop inference on battery power
pause_on_fullscreen = true  # Stop when fullscreen app detected
```

---

## Rewards System

### Core Model: Layers x Tokens

Nodes are rewarded based on actual work performed:

```
node_reward = (layers_served / total_layers) x request_reward
```

A node running 40 of 120 layers earns 33% of the request's reward. This naturally rewards more powerful hardware — the scheduler assigns more layers to faster GPUs.

### Why This is Fair

- **Fast GPUs earn more** because they handle more layers (more work, more energy cost)
- **Small GPUs still earn** because pipelines need all stages filled — every node is essential
- **Solo small models** (7B/13B) run entirely on a single laptop, earning 100% of those rewards
- **No reward for idling** — you earn by processing tokens, not by being online

### Small Node Protection

Large models require diverse hardware to fill all pipeline stages. A pipeline waiting for one more 8GB node to complete will prioritize any available small GPU. Scarcity creates demand.

Additionally, the majority of real-world API traffic is for smaller models (7B-13B) that fit on a single consumer GPU. Small nodes get their own workload tier.

### Configurable Parameters

All reward parameters are controlled server-side by the orchestrator and can be adjusted without client updates:

- **Base rate**: reward per token per layer
- **Floor rate**: minimum per-token reward for any participant (taperable over time)
- **Busy hour bonuses**: higher rates during peak demand
- **Scarcity multiplier**: bonus when specific hardware is needed to complete a pipeline
- **Uptime multiplier**: reliability bonus for consistent nodes
- **Anti-whale protection**: diminishing returns for operators running many nodes from one wallet

This allows generous early incentives to build the network, with the flexibility to taper to sustainable rates as real API revenue grows.

---

## Network Communication

### Transport: QUIC

All node-to-node communication uses **QUIC** (via the Quinn library), providing:

- **Encrypted by default** — TLS 1.3 built into the protocol
- **Low latency** — 0-RTT connection establishment
- **Multiplexed streams** — activation forwarding and control messages share one connection
- **NAT-friendly** — UDP-based, works through most firewalls

### Message Types

| Message | Direction | Purpose |
|---------|-----------|---------|
| **Activations** | Forward (stage N → N+1) | Hidden state tensors between pipeline stages |
| **Tokens** | Backward (last stage → first) | Generated tokens flowing back to the requester |
| **Control** | Any direction | Layer assignments, ready signals, teardown |
| **Ping/Pong** | Between peers | Latency measurement for pipeline optimization |

### Wire Protocol

Messages use a simple length-prefixed format: 4-byte little-endian length + JSON payload. Future optimization: FP8 quantized activation transfer to reduce bandwidth.

---

## Authentication & Account Linking

### How Accounts Work

1. **CLI registration**: user enters their Solana public address → node registered in database
2. **Website sign-in**: user connects wallet (Phantom, Solflare, etc.) → Supabase Web3 auth verifies ownership via signature
3. **Auto-linking**: database trigger matches the wallet address and links the node to the authenticated account
4. **Claiming rewards**: user presses `[c]` in the terminal dashboard → opens the claim page in browser → connects wallet → signs claim transaction on Solana

No private keys ever touch the CLI. The terminal only stores the public address. Ownership is proven on the website through standard wallet signature verification (EIP-4361 / Sign In With Solana).

---

## CLI Commands Reference

| Command | Description |
|---------|------------|
| `compute` | Launch TUI (onboarding → splash → dashboard) |
| `compute start` | Start the daemon |
| `compute stop` | Stop the daemon |
| `compute status` | One-line status with wallet and node info |
| `compute dashboard` | Full TUI dashboard |
| `compute init` | First-time setup wizard (CLI mode) |
| `compute benchmark` | Run GPU/CPU/network benchmarks |
| `compute hardware` | Show detected hardware as JSON |
| `compute doctor` | Diagnose issues with actionable recommendations |
| `compute nodes` | List online nodes in the network |
| `compute earnings` | Show earnings summary |
| `compute wallet set <addr>` | Set Solana wallet address |
| `compute config show` | Show current configuration |
| `compute update` | Self-update to latest release |
| `compute service install` | Auto-start on login (launchd/systemd) |
| `compute logs -f` | Tail daemon logs |

---

## System Requirements

### Minimum

- 8GB RAM
- 10GB free disk space
- Broadband internet connection
- macOS, Linux, or Windows

### Recommended

- Dedicated or discrete GPU (NVIDIA or Apple Silicon)
- 16GB+ RAM
- SSD storage
- Stable internet with <50ms latency to peers

### GPU Compatibility

Any NVIDIA GPU with CUDA support (GTX 1060+), any Apple Silicon Mac (M1+), or CPU-only mode for machines without a GPU.
