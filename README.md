# Compute

**Decentralized GPU Infrastructure** — Aggregates idle GPU/CPU resources and daisy-chains them via pipeline parallelism to run large AI models no single machine could handle.

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
| `compute wallet set <addr>` | Set Solana wallet |
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

### Pipeline Parallelism

Compute's core architecture is **distributed pipeline parallelism** — daisy-chaining consumer GPUs across the internet to run models no single machine could handle:

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
