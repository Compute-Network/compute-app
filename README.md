# Compute

**Decentralized GPU Infrastructure** — Runs local and distributed inference across contributed machines, with the current stable path focused on single-node serving through the Compute orchestrator.

## Quick Start

```bash
# Build from source
cargo build --release

# First-time setup
./target/release/compute init

# Start the daemon
./target/release/compute start

# Open the TUI dashboard
./target/release/compute dashboard

# Run a hardware benchmark
./target/release/compute benchmark

# Check system health
./target/release/compute doctor
```

## Commands

| Command | Description |
|---------|-------------|
| `compute` | Launch TUI (splash + dashboard) |
| `compute start` | Start the daemon |
| `compute stop` | Stop the daemon |
| `compute status` | Quick status check |
| `compute dashboard` | Full TUI dashboard |
| `compute init` | First-time setup wizard |
| `compute benchmark` | Hardware benchmark |
| `compute hardware` | Hardware info (JSON) |
| `compute doctor` | Diagnose issues |
| `compute config show` | Show configuration |
| `compute wallet login` | Connect wallet in browser and authorize this node |
| `compute earnings` | View earnings |
| `compute pipeline` | Pipeline status |
| `compute logs -f` | Follow daemon logs |
| `compute service install` | Auto-start on login |
| `compute update` | Self-update |

## Architecture

```
crates/
  compute-cli/       CLI entry point + TUI (ratatui)
  compute-daemon/    Daemon, hardware detection, idle monitoring
  compute-network/   P2P QUIC transport, pipeline scheduler, model catalog
  compute-solana/    Solana wallet validation (read-only)
```

### Inference Modes

The repo contains both the stable single-node path and the experimental multi-node pipeline path.

Single-node today:
- local `llama-server` on the node
- orchestrator scheduling and relay
- wallet-authenticated node sessions

Multi-node research/code:
- daisy-chained consumer GPUs across the internet to run models no single machine could handle
- QUIC/P2P transport and staged layer assignment remain in the repo, but are not the primary production path today
- multi-node v1 is planned around stage-based pipeline parallelism; the current llama.cpp RPC path should be treated as experimental

- Models are split across N nodes, each running a contiguous range of transformer layers
- Activations flow through the pipeline: Node1 -> Node2 -> Node3
- QUIC transport for low-latency P2P communication between stages
- Water-filling scheduler allocates layers proportional to each node's compute power

### GPU Backends

| Platform | Backend | Method |
|----------|---------|--------|
| NVIDIA (Linux/Windows) | CUDA | Docker + nvidia-container-toolkit |
| Apple Silicon (macOS) | Metal/MLX | Native inference (no Docker) |
| CPU fallback (all) | CPU | llama.cpp CPU mode |

## Build

```bash
# Debug build
cargo build

# Release build (LTO + strip, ~1.5MB binary)
cargo build --release

# Run tests
cargo test --workspace

# Lint
cargo clippy --workspace

# Format
cargo fmt --all
```

### Cross-Compilation Targets

- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`
- `x86_64-pc-windows-msvc`
- `aarch64-pc-windows-msvc`

## Configuration

Config file: `~/.compute/config.toml` (macOS/Linux) or `%APPDATA%\compute\config.toml` (Windows)

```bash
compute config show           # View all settings
compute config set <key> <v>  # Change a setting
```

## License

MIT
