# COMPUTE — Development Plan

## Overview

Compute is a DePIN (Decentralized Physical Infrastructure Network) terminal application that aggregates idle GPU/CPU resources from users' machines and sells that compute to AI inference providers and enterprises. Revenue flows back to contributors via the $COMPUTE token on Solana.

The app is a CLI-first daemon + TUI, installable via a single curl command, built primarily in Rust with a React frontend for the web dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPUTE NETWORK                             │
│                                                                 │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│  │  Node A    │───▶│  Node B    │───▶│  Node C    │  ...       │
│  │  (RTX4090) │    │  (M3 Max)  │    │  (RTX3080) │            │
│  │  Layers 0-15│    │ Layers 16-31│   │ Layers 32-47│           │
│  └─────┬──────┘    └─────┬──────┘    └─────┬──────┘            │
│        │                 │                 │                    │
│        └────────────┬────┘─────────────────┘                    │
│                    │                                            │
│         ┌──────────┴───────────┐                                │
│         │   Orchestrator API   │                                │
│         │  (Job Queue, Match,  │                                │
│         │   Pipeline Sched.)   │                                │
│         └──────────┬───────────┘                                │
│                    │                                            │
│         ┌──────────┴───────────┐                                │
│         │   Public API Layer   │                                │
│         │  (OpenAI-compatible) │                                │
│         └──────────┬───────────┘                                │
│                    │                                            │
│    ┌───────────────┴────────────────┐                           │
│    │    Solana Settlement Layer     │                           │
│    │  (Payments, Rewards, Staking,  │                           │
│    │   Burn-Mint Equilibrium)       │                           │
│    └────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
     ┌──────────┐ ┌───────────┐ ┌──────────────┐
     │ Direct   │ │ Enterprise│ │ DePIN/Market │
     │ API      │ │ Contracts │ │ Supply       │
     │ Users    │ │ OpenRouter│ │ (Akash,io.net│
     │ (devs,   │ │ Together  │ │  Render)     │
     │ startups)│ │ Fireworks │ │              │
     └──────────┘ └───────────┘ └──────────────┘
```

### Compute Model — Clustered Pipeline-Parallel Inference

Compute's core architecture is **distributed pipeline parallelism** — daisy-chaining consumer GPUs across the internet to run models no single machine could handle alone.

- Multiple consumer GPUs stitched together via pipeline parallelism
- Enables running 70B+ models across nodes that individually couldn't handle them
- Uses pipeline parallelism (NOT tensor parallelism — too bandwidth-hungry for public internet)
- Optimised for batch/throughput workloads, not real-time chat
- Research refs: Prime Intellect (PRIME-VLLM), Gradient Network (PARALLAX), Petals
- Narrative: "Your gaming PC becomes part of a supercomputer"
- Every node participates as a pipeline stage — there is no standalone single-node inference mode

### Revenue Channels

1. **Direct API Sales** — OpenAI-compatible inference API sold to devs/startups (primary)
2. **Enterprise Contracts** — Sell bulk compute capacity to companies like OpenRouter, Together AI, Fireworks AI as a lower-cost inference backend they can route to
3. **DePIN Network Supply** — Supply GPUs to Akash, io.net, Render as a provider to earn their tokens + diversify revenue
4. **Token Economics** — Creator fees from $COMPUTE trading on PumpSwap, Burn-Mint Equilibrium on compute purchases

---

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| CLI/Daemon | Rust | Performance, safety, single binary distribution, cross-platform |
| TUI Dashboard | Ratatui (Rust) | Native terminal UI, no runtime deps, animated rendering |
| ASCII Art/Animations | Custom + Ratatui | Startup animation, live stats display |
| Workload Isolation | Docker containers | Industry standard, same as SaladCloud |
| GPU Inference Runtime | Ollama / vLLM / llama.cpp | Model serving within containers |
| Pipeline Parallelism | Custom over libp2p or QUIC | P2P communication between pipeline stages |
| Orchestrator API | Rust (Axum) or TypeScript (Hono) | Job matching, scheduling, health checks |
| Public API | Rust (Axum) | OpenAI-compatible endpoints |
| Web Dashboard | React + Tailwind | Earnings, stats, node management |
| Blockchain | Solana (Anchor framework) | Token, payments, staking, governance |
| P2P Networking | libp2p or custom QUIC | Node discovery, health, pipeline communication |
| Installer Script | Bash (install.sh) | curl-based install like Rust/Bun/Deno |

---

## Phase 1: CLI Foundation & Installer

### 1.1 — Project Scaffolding

```
compute/
├── Cargo.toml                 # Workspace root
├── crates/
│   ├── compute-cli/           # CLI entry point + TUI
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── cli.rs         # Clap command definitions
│   │   │   ├── tui/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── app.rs     # TUI app state + event loop
│   │   │   │   ├── ascii.rs   # ASCII art + startup animation
│   │   │   │   ├── dashboard.rs # Live stats view
│   │   │   │   └── widgets.rs # Custom ratatui widgets
│   │   │   └── config.rs      # Config file handling
│   │   └── Cargo.toml
│   ├── compute-daemon/        # Background daemon process
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── daemon.rs      # Daemon lifecycle
│   │   │   ├── idle.rs        # Idle detection (CPU/GPU/input)
│   │   │   ├── benchmark.rs   # Hardware benchmarking
│   │   │   ├── workload.rs    # Workload execution manager
│   │   │   ├── container.rs   # Docker container management
│   │   │   ├── health.rs      # Heartbeat + health reporting
│   │   │   └── metrics.rs     # GPU utilisation, temps, earnings
│   │   └── Cargo.toml
│   ├── compute-network/       # Networking + orchestrator comms
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── api_client.rs  # Orchestrator API client
│   │   │   ├── p2p.rs         # P2P layer (Tier 2 pipeline)
│   │   │   └── auth.rs        # Wallet-based auth (Solana keypair)
│   │   └── Cargo.toml
│   ├── compute-solana/        # Solana integration (read-only)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── wallet.rs      # Public address + balance reading
│   │   │   └── rewards.rs     # Earnings tracking
│   │   └── Cargo.toml
│   └── compute-orchestrator/  # Central orchestrator service
│       ├── src/
│       │   ├── lib.rs
│       │   ├── registry.rs    # Node registration + capabilities
│       │   ├── scheduler.rs   # Pipeline formation + job matching
│       │   ├── api.rs         # Public API (OpenAI-compatible)
│       │   └── health.rs      # Node health tracking
│       └── Cargo.toml
├── install.sh                 # curl installer script
├── scripts/
│   ├── build-release.sh       # Cross-compile for all targets
│   └── publish-release.sh     # Upload binaries to GitHub/CDN
└── docs/
    └── ...
```

### 1.2 — CLI Commands

```bash
# Installation
curl -fsSL https://compute.sh/install.sh | bash

# Core commands
compute start              # Start daemon (shows ASCII splash, then detaches)
compute stop               # Stop daemon gracefully
compute status             # Quick one-line status (running/stopped, earnings, GPU)
compute dashboard          # Open full TUI dashboard (live stats, animated)
compute logs               # Tail daemon logs

# Configuration
compute init               # First-time setup wizard (wallet, preferences)
compute config set <k> <v> # Set config values
compute config get <k>     # Get config value
compute config show        # Show all config

# Wallet & Earnings
compute wallet             # Show wallet address + balances
compute wallet set <addr>  # Set/update Solana public address for receiving rewards
compute earnings           # Show earnings summary (today, week, month, all-time)
compute claim              # Claim pending $COMPUTE rewards

# Hardware & Benchmarking
compute benchmark          # Run GPU/CPU benchmark, report capabilities
compute hardware           # Show detected hardware info

# Advanced
compute pipeline           # Show current pipeline status and peers
compute update             # Self-update to latest version
compute uninstall          # Clean uninstall
compute doctor             # Diagnose common issues (Docker, GPU drivers, network)
```

### 1.3 — install.sh Script

The installer should:

1. Detect OS (macOS, Linux) and architecture (x86_64, aarch64/arm64)
2. Download the correct pre-built binary from GitHub Releases or CDN
3. Place binary in `/usr/local/bin/compute` (or `~/.compute/bin/` if no sudo)
4. Add to PATH if needed
5. Verify the binary works (`compute --version`)
6. Print a welcome message with next steps

Reference implementations to study:
- Rust's `rustup` installer: https://sh.rustup.rs
- Bun's installer: https://bun.sh/install
- Deno's installer: https://deno.land/install.sh

```bash
#!/bin/bash
set -euo pipefail

# --- Configuration ---
REPO="compute-network/compute"
INSTALL_DIR="/usr/local/bin"
BINARY_NAME="compute"
BASE_URL="https://compute.sh/releases/latest"

# --- Detect platform ---
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)   PLATFORM="linux" ;;
  Darwin)  PLATFORM="darwin" ;;
  MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
  *)       echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
  x86_64)        ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *)             echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

FILENAME="compute-${PLATFORM}-${ARCH}.tar.gz"
DOWNLOAD_URL="${BASE_URL}/${FILENAME}"

# --- Download & install ---
echo "Downloading Compute CLI for ${PLATFORM}-${ARCH}..."
TMPDIR=$(mktemp -d)
curl -fsSL "$DOWNLOAD_URL" -o "${TMPDIR}/${FILENAME}"
tar -xzf "${TMPDIR}/${FILENAME}" -C "$TMPDIR"

if [ -w "$INSTALL_DIR" ]; then
  mv "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
else
  sudo mv "${TMPDIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
fi

chmod +x "${INSTALL_DIR}/${BINARY_NAME}"
rm -rf "$TMPDIR"

# --- Verify ---
echo ""
echo "Compute CLI installed successfully!"
echo ""
${BINARY_NAME} --version
echo ""
echo "Get started:"
echo "  compute init        # First-time setup"
echo "  compute start       # Start contributing compute"
echo "  compute dashboard   # View live stats"
```

Host `install.sh` at: `https://compute.sh/install.sh` (or whatever the domain is)
Host release binaries at: `https://compute.sh/releases/latest/compute-{platform}-{arch}.tar.gz`

### 1.4 — ASCII Art, Animated Globe & Startup Animation

**Design principle:** Match the website's clean, minimal aesthetic (see Branding-Guide/design-language.md). The terminal UI uses a
consistent split layout: **left 1/3 = spinning ASCII globe**, **right 2/3 = content/info/menu**.
This mirrors the website's hero layout and gives the CLI a distinctive, recognisable look.

**TUI design tokens (mapped from web design language):**
- Monochromatic palette: white/light grey for continent outlines, no bright brand colours
- Typography hierarchy via weight/size, not colour — headings in bold, labels in dim/muted
- Borders as structure: use box-drawing characters for panel dividers, not heavy fills
- Generous whitespace within panels — sections breathe, don't cram
- Flat, sharp aesthetic: no rounded box corners (use `┌─┐` not curves)
- Mono font labels: section headers in UPPERCASE for the systematic/technical feel
- Subtle animation: globe rotates slowly, sparklines update smoothly — nothing flashy
- One "dark section" break: the globe panel can use a darker background to create contrast

#### Animated ASCII Globe

The globe is a continuously rotating wireframe Earth rendered in ASCII/Unicode braille
characters. It occupies the left ~33% of the terminal at all times (startup splash AND
dashboard). Nodes on the network can be shown as highlighted dots on the globe, giving a
real-time visual of the distributed network.

**Globe implementation approach:**

The globe is a sphere rendered from a simplified world map coordinate set. The rotation is
achieved by applying a Y-axis rotation matrix to the 3D coordinates each frame, then
projecting to 2D and rendering with braille Unicode characters (⠀⠁⠂⠃...⣿) for smooth
sub-character resolution. Each braille character is a 2x4 dot grid, giving effectively
2x the horizontal and 4x the vertical resolution of regular characters.

```
Frame rendering pipeline:
1. Load world map as array of (lat, lon) points
2. Convert to 3D cartesian coordinates on unit sphere
3. Each frame: rotate all points by increment around Y axis
4. Project visible points (z > 0) to 2D screen coordinates
5. Map to braille dot grid
6. Render frame to ratatui canvas widget
7. Overlay node positions as highlighted/coloured dots
```

Reference implementations to study:
- `globe-cli` (Rust): https://github.com/nicknisi/globe-cli — rotating ASCII globe in Rust
- `world.rs` from the `ratatui` examples — braille canvas rendering
- OpenAI Codex CLI source — for the typewriter-style text animation approach

**Rust crates for globe rendering:**
- `ratatui` Canvas widget with BrailleGrid for the actual rendering
- Pre-compute multiple rotation frames and cycle through them for performance
- Or compute rotation in real-time (it's just matrix math, very cheap)

**Globe visual style:**
- Continent outlines rendered in white/light grey braille dots
- Ocean areas are empty (transparent / terminal background)
- Active node locations shown as bright green or accent-coloured dots
- A subtle glow or pulse effect on nodes currently serving inference
- Globe rotates slowly (~1 revolution per 30-60 seconds)
- During startup: globe fades in while spinning up to speed

#### Startup Splash Layout (compute start)

```
┌──────────────────────┬─────────────────────────────────────────────┐
│                      │                                             │
│        ·  · ·        │   ██████╗ ██████╗ ███╗   ███╗██████╗       │
│      ·  ·    · ·     │  ██╔════╝██╔═══██╗████╗ ████║██╔══██╗      │
│    · ·  ····  · ·    │  ██║     ██║   ██║██╔████╔██║██████╔╝      │
│   · ···  ·  ···  ·   │  ██║     ██║   ██║██║╚██╔╝██║██╔═══╝       │
│    · ··    ··  · ·   │  ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║           │
│     ·  · ··  · ·     │   ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝           │
│       ·  · ·  ·      │                                             │
│         · · ·        │  Decentralized GPU Infrastructure           │
│          · ·         │  v0.1.0                                     │
│                      │                                             │
│    [rotating globe]  │  ▸ Detecting hardware...          ✓        │
│                      │  ▸ GPU: NVIDIA RTX 4090 (24GB)    ✓        │
│                      │  ▸ Connecting to network...        ✓        │
│                      │  ▸ 12,847 nodes online                      │
│                      │                                             │
│                      │  Daemon started. Earning $COMPUTE...        │
│                      │                                             │
└──────────────────────┴─────────────────────────────────────────────┘
```

The startup sequence animates step by step:
1. Globe fades in and starts rotating (0.5s)
2. COMPUTE logo types in (0.3s)
3. Tagline and version appear
4. Each detection step appears with a spinner, then resolves to ✓
5. Final "Daemon started" message
6. If `compute start`: holds for 1s then exits to shell (daemon runs in background)
7. If `compute dashboard`: transitions smoothly into the full dashboard layout

#### Dashboard Layout (compute dashboard)

The dashboard maintains the same left-1/3 globe, right-2/3 content split:

```
┌──────────────────────┬─────────────────────────────────────────────┐
│                      │                                             │
│                      │  COMPUTE              ● ACTIVE     v0.1.0  │
│        ·  · ·        │                                             │
│      ·  ·    · ·     ├─ Node ──────────────────────────────────────┤
│    · ·  ····  · ·    │                                             │
│   · ···  ·  ···  ·   │  GPU     NVIDIA RTX 4090 (24GB)            │
│    · ··    ··  · ·   │  Tier    Pipeline Stage 3/5                  │
│     ·  · ··  · ·     │  Uptime  4d 12h 33m                        │
│       ·  · ·  ·      │  Load    ▓▓▓▓▓▓░░░░ 62%                   │
│         · · ·        │                                             │
│          · ·         ├─ Earnings ──────────────────────────────────┤
│                      │                                             │
│   12,847 nodes       │  Today      142.5 $COMPUTE    ≈ $45.20     │
│   ● 847 PF peak     │  This Week  891.2 $COMPUTE    ≈ $284.50    │
│                      │  All Time   12,450 $COMPUTE                 │
│    ·                 │  Pending    23.4 $COMPUTE  [c]laim          │
│     · (you)         │                                             │
│                      ├─ Workload ──────────────────────────────────┤
│                      │                                             │
│                      │  Model     Llama-3.1-8B-Q4                  │
│                      │  Served    1,247 requests                   │
│                      │  Latency   124ms avg                        │
│                      │  VRAM      18 / 24 GB                       │
│                      │  Temp      67°C    Power  180W / 350W       │
│                      │                                             │
│                      ├─ Throughput ─────────────────────────────────┤
│                      │  ▁▂▃▅▇█▇▅▃▂▁▂▃▅▆▇█▇▅▃▁▂▃▅▇█▇▅▃▂▁▂▃▅▇    │
│                      │                              47.2 tok/s     │
│                      │                                             │
├──────────────────────┴─────────────────────────────────────────────┤
│ [q]uit  [p]ause  [l]ogs  [e]arnings  [c]laim  [i]nfo             │
└────────────────────────────────────────────────────────────────────┘
```

**Globe in dashboard mode:**
- Continues rotating
- Shows "you" marker — the user's node highlighted on the globe based on their IP geolocation
- Shows aggregate network stats below the globe (total nodes, peak capacity)
- Node dots pulse/animate when serving requests
- The globe is the visual anchor that ties the CLI to the website's branding

**Right 2/3 panels:**
- Clean, minimal layout matching the website's typography feel (lots of whitespace)
- Sections: Node info, Earnings, Current Workload, Throughput sparkline
- Bottom bar with keyboard shortcuts
- Data updates every 2-5 seconds via daemon IPC

#### Implementation Details

**ratatui layout:**
```rust
// Main horizontal split: 33% globe, 67% content
let chunks = Layout::default()
    .direction(Direction::Horizontal)
    .constraints([
        Constraint::Percentage(33),
        Constraint::Percentage(67),
    ])
    .split(frame.area());

// Left panel: globe canvas
let globe_area = chunks[0];

// Right panel: vertical stack of info sections
let right_chunks = Layout::default()
    .direction(Direction::Vertical)
    .constraints([
        Constraint::Length(3),   // Header (COMPUTE + status)
        Constraint::Length(6),   // Node info
        Constraint::Length(6),   // Earnings
        Constraint::Length(7),   // Workload
        Constraint::Length(4),   // Throughput sparkline
        Constraint::Length(1),   // Keyboard shortcuts
    ])
    .split(chunks[1]);
```

**Globe rendering with ratatui Canvas:**
```rust
use ratatui::widgets::canvas::{Canvas, Points};

// Render globe as braille dots on Canvas widget
let globe = Canvas::default()
    .block(Block::default().borders(Borders::NONE))
    .paint(|ctx| {
        // Draw continent outlines
        ctx.draw(&Points {
            coords: &visible_continent_points,
            color: Color::White,
        });
        // Draw active nodes
        ctx.draw(&Points {
            coords: &active_node_points,
            color: Color::Green,
        });
        // Draw "you" marker
        ctx.draw(&Points {
            coords: &[my_node_point],
            color: Color::Yellow,
        });
    })
    .x_bounds([-1.0, 1.0])
    .y_bounds([-1.0, 1.0]);
```

**Globe rotation state:**
```rust
struct GlobeState {
    angle: f64,                          // Current Y-axis rotation (radians)
    rotation_speed: f64,                 // Radians per frame (~0.01)
    continent_points: Vec<(f64, f64, f64)>, // 3D lat/lon → cartesian
    node_positions: Vec<(f64, f64, f64)>,   // Active nodes from orchestrator
    my_position: (f64, f64, f64),           // This node's position
}

impl GlobeState {
    fn tick(&mut self) {
        self.angle += self.rotation_speed;
        if self.angle > std::f64::consts::TAU {
            self.angle -= std::f64::consts::TAU;
        }
    }

    fn project(&self, point: (f64, f64, f64)) -> Option<(f64, f64)> {
        // Apply Y-axis rotation
        let x = point.0 * self.angle.cos() - point.2 * self.angle.sin();
        let z = point.0 * self.angle.sin() + point.2 * self.angle.cos();
        let y = point.1;

        // Only render front-facing points (z > 0)
        if z > 0.0 {
            Some((x, y))
        } else {
            None
        }
    }
}
```

**World map data:** Use a simplified coastline dataset (~2000-3000 points is enough for
ASCII resolution). Can be extracted from Natural Earth dataset and baked into the binary
as a const array. The `globe-cli` crate has a usable dataset.

For the `compute start` command specifically (daemon mode):
1. Show the splash layout with globe + detection steps (2-3 seconds total)
2. Each step animates in sequence
3. Final message: "Daemon started. Earning $COMPUTE..."
4. Exit to shell (daemon continues in background)

For `compute dashboard` (interactive TUI):
1. Show splash briefly (1-2 seconds)
2. Smooth transition into full dashboard (globe stays, right panel morphs)
3. Dashboard runs until user presses `q`

---

## Phase 2: Daemon & Hardware Detection

### 2.1 — Idle Detection System

The daemon must detect when the machine is idle vs active and scale compute usage accordingly.

**Signals to monitor:**
- CPU usage (system-wide)
- GPU usage (nvidia-smi / Metal performance counters)
- Keyboard/mouse input activity (last input timestamp)
- Active fullscreen application (gaming detection)
- Battery status (don't run on battery / low battery)

**Behaviour:**
- **Idle** (no input for X minutes + low CPU/GPU): Use maximum available GPU/CPU
- **Light use** (browsing, docs): Use available GPU, throttle CPU usage
- **Heavy use** (gaming, rendering, compiling): Pause all workloads immediately
- **User override**: `compute pause` / `compute resume` manual control

**Implementation:**
- Linux: Read `/proc/stat` for CPU, `nvidia-smi` for GPU, `/dev/input/` for input
- macOS: `IOKit` for idle time, `Metal` performance shaders for GPU, `powermetrics`
- Windows: `GetLastInputInfo` for idle time, `nvidia-smi` or NVML for GPU, WMI for CPU
- Cross-platform: Poll every 2-5 seconds, debounce state changes

### 2.2 — Hardware Benchmarking

On first run (`compute init` or `compute benchmark`), detect and benchmark:

- GPU model, VRAM, driver version, CUDA/Metal support
- CPU model, cores, clock speed
- RAM total/available
- Disk space available
- Network bandwidth (upload/download speed test)
- Docker availability and version

Benchmark results determine:
- Which workloads the node can accept
- Pipeline eligibility (stable connection, decent upload speed, 6GB+ VRAM)
- Expected earnings range

### 2.3 — Workload Execution

Workloads run in Docker containers for isolation. The daemon:

1. Receives workload assignment from orchestrator
2. Pulls the container image (cached locally after first pull)
3. Starts container with GPU passthrough (nvidia-docker or Metal)
4. Routes inference requests to the container
5. Reports results back to orchestrator
6. Monitors container health, restarts if needed

Container images are pre-built and hosted. Users don't build their own containers — the orchestrator assigns them based on hardware capabilities.

**Initial supported workload containers:**
- `compute/llm-inference` — vLLM or llama.cpp based, serves OpenAI-compatible API
- `compute/image-gen` — Stable Diffusion / Flux inference
- `compute/whisper` — Audio transcription
- `compute/embeddings` — Text embedding generation

---

## Phase 3: Orchestrator & API

### Infrastructure Stack

The orchestrator and API are **TypeScript (Hono on Node)**, separate from the Rust CLI workspace:

```
compute-orchestrator/       (TypeScript)
  src/
    api/                    # OpenAI-compatible public API
    orchestrator/           # Node registry, pipeline scheduling, health
    workers/                # Background jobs (reward calculation, etc.)
  supabase/
    migrations/             # Database schema
```

| Component | Service | Notes |
|-----------|---------|-------|
| Database | **Supabase** (Postgres) | Auth, realtime subscriptions, Row Level Security. Supports Solana wallet auth for future web portal (https://supabase.com/docs/guides/auth/auth-web3) |
| Orchestrator + API | **Railway** | Single service to start, auto-scales replicas. Persistent WebSocket connections for node heartbeats |
| Binary CDN | **GitHub Releases + Cloudflare R2** | install.sh and release binaries |
| Monitoring | **Grafana Cloud** (free tier) | Add when needed |
| Domain/DNS | **Cloudflare** | Edge caching on API if needed later |

See `cost-estimates.md` for scaling costs at 10/100/1,000/10,000+ nodes.

### 3.1 — Orchestrator Service

Central service (hosted by Compute team) that:

- Maintains registry of all active nodes + capabilities
- Receives inference requests from the public API
- Matches requests to capable nodes based on: model requirements, VRAM, latency, location, reliability score
- Handles load balancing across nodes
- Tracks work completed for reward distribution
- Manages pipeline formation (grouping nodes into pipelines for distributed inference)

### 3.2 — Public API (Revenue Source #1: Direct Sales)

OpenAI-compatible API that developers can use as a drop-in replacement:

```
POST https://api.compute.sh/v1/chat/completions
POST https://api.compute.sh/v1/completions
POST https://api.compute.sh/v1/embeddings
POST https://api.compute.sh/v1/images/generations
POST https://api.compute.sh/v1/audio/transcriptions
```

Pricing: 50-80% cheaper than OpenAI/Replicate/Together for equivalent models.
Payment: $COMPUTE token, SOL, or USDC. Potentially credit card via Stripe for non-crypto users.

### 3.3 — Enterprise API (Revenue Source #2: Big Companies)

For companies like OpenRouter, Together AI, Fireworks AI — offer Compute as a backend inference provider:

- **OpenRouter integration**: Become a provider on OpenRouter so their 250K+ apps can route to Compute's network for cheaper inference. OpenRouter already supports multiple providers and routes by cost/speed — Compute would compete on price.
- **Together AI / Fireworks**: Offer bulk compute capacity via B2B contracts. They need GPU capacity for serving open-source models. Compute offers lower cost through distributed consumer hardware.
- **Direct enterprise contracts**: Custom SLAs, guaranteed capacity, volume pricing.

The API is the same — the business relationship is different (contractual, bulk, SLAs).

### 3.4 — DePIN Network Supply (Revenue Source #3)

Run Compute nodes as providers on existing networks simultaneously:

- **Akash Network**: Bid on compute jobs via Akash's reverse auction
- **io.net**: Register as GPU supplier
- **Render Network**: Operate as a render node

This diversifies revenue and fills idle gaps when Compute's own API doesn't have enough demand.

---

## Phase 4: Pipeline Parallelism (Core Architecture)

Pipeline parallelism is the **core compute model**, not an optional tier. Every inference job runs as a distributed pipeline.

### 4.1 — Pipeline Parallelism Over Internet

Combine N consumer GPUs across the internet to run models as a daisy-chained pipeline.

**How it works:**
- Split model layers across N nodes (e.g., 70B model across 5x 24GB GPUs)
- Each node runs a subset of layers
- Activations flow through the pipeline: Node1 → Node2 → Node3 → ...
- Use QUIC or libp2p for low-latency P2P communication between pipeline stages

**Key challenges:**
- Latency: ~20-100ms per hop on public internet. With 5 nodes = 100-500ms added per token
- Reliability: If one node drops, the whole pipeline breaks. Need fast failover.
- Heterogeneity: Different GPU speeds create pipeline bubbles (fast GPUs wait for slow ones)
- Scheduling: Need to form optimal pipelines from available nodes (geography, speed, VRAM)

**Research to build on (open source):**
- Prime Intellect's PRIME-VLLM: Pipeline-parallel vLLM over public networks
- Gradient Network's PARALLAX: Distributed inference on consumer Macs
- Petals: BitTorrent-style collaborative LLM inference

**Target workloads:**
- Batch inference on 70B+ models (not real-time chat)
- Synthetic data generation
- Bulk document processing / summarisation
- Fine-tuning (distributed, async gradient updates)

### 4.2 — Pipeline Formation

The orchestrator groups nodes into pipelines:

1. When an inference job comes in (e.g., "run Llama 70B inference on 10K prompts")
2. Orchestrator selects N nodes with compatible hardware + good connectivity
3. Runs a quick latency test between candidate nodes
4. Forms pipeline, distributes model shards
5. Nodes download their layer weights (cached after first download)
6. Pipeline processes micro-batches with asynchronous scheduling
7. Results stream back to the API caller

---

## Phase 5: Solana Integration & Token

### 5.1 — Wallet Integration (Read-Only)

- User provides their Solana public address during `compute init`
- No private key management — Compute never holds keys
- Read token balances and earnings via Solana RPC
- Display wallet address and balances in dashboard
- Rewards are sent to the user's public address by the orchestrator/program

### 5.2 — Reward Distribution

- Orchestrator tracks GPU-hours, requests served, and quality scores per node
- Rewards distributed periodically (hourly or daily) in $COMPUTE
- On-chain settlement via Solana program (Anchor framework)
- Rewards formula weighs: uptime, work completed, reliability score, staking multiplier

### 5.3 — Burn-Mint Equilibrium

When a buyer purchases compute via the API:
1. Payment converts to $COMPUTE on-chain
2. A percentage is burned (reducing supply)
3. Contributors receive newly minted $COMPUTE as rewards
4. Net effect: real compute usage = buy pressure + burn = token value tied to real demand

### 5.4 — Staking

- Stake $COMPUTE to earn higher reward multipliers
- Staked nodes get priority for high-paying workloads
- Slashing for provably bad behaviour (serving wrong results, excessive downtime)

---

## Phase 6: Web Dashboard & Docs

### 6.1 — Website (already exists)

The existing website needs:
- Download / install instructions (the curl command prominently displayed)
- Live network stats (total nodes, GPU-hours served, tokens distributed)
- Documentation section
- **Future: Token claim portal** — web dashboard where node operators sign in with their Solana wallet (via Supabase wallet auth) to view earnings and claim $COMPUTE rewards. ZK Compression via Light Protocol for cheap batch reward distribution vs traditional airdrops.

### 6.2 — Documentation Site

Host at `docs.compute.sh` or as a section of the main site:

- Getting Started (install, init, start)
- CLI Reference (all commands documented)
- Configuration Guide
- Hardware Requirements
- Earnings FAQ
- API Documentation (for compute buyers)
- Tier 2 Explainer
- Tokenomics
- Architecture Overview
- Troubleshooting / `compute doctor`

---

## Development Priorities (Ordered)

### Sprint 1 — Skeleton + ASCII Splash (Week 1-2)
- [ ] Rust workspace scaffolding with all crates
- [ ] CLI argument parsing with clap
- [ ] ASCII art logo + animated startup sequence (ratatui + crossterm)
- [ ] `compute --version`, `compute --help`
- [ ] Basic config file creation (`~/.compute/config.toml`)
- [ ] `compute init` wizard (placeholder — just wallet generation + hardware detection)

### Sprint 2 — Installer + Distribution (Week 2-3)
- [ ] Cross-compilation setup (linux x86_64, linux aarch64, darwin x86_64, darwin aarch64)
- [ ] GitHub Actions CI/CD for building release binaries
- [ ] `install.sh` script (see spec above)
- [ ] Host install.sh and binaries on CDN / website
- [ ] `compute update` self-update mechanism
- [ ] `compute uninstall` cleanup

### Sprint 3 — Hardware Detection + Benchmarking (Week 3-4)
- [ ] GPU detection (NVIDIA via nvidia-smi, Apple Silicon via system_profiler)
- [ ] CPU/RAM/disk detection
- [ ] Network speed test
- [ ] Docker availability check
- [ ] `compute benchmark` command with formatted output
- [ ] `compute hardware` command
- [ ] `compute doctor` diagnostic checks

### Sprint 4 — Daemon + Idle Detection (Week 4-6)
- [ ] Daemon process management (start, stop, daemonise)
- [ ] Idle detection system (CPU, GPU, input activity)
- [ ] Dynamic resource scaling based on idle state
- [ ] Heartbeat system (periodic check-in with orchestrator)
- [ ] Basic logging infrastructure
- [ ] PID file management + lockfile

### Sprint 5 — TUI Dashboard (Week 5-7)
- [ ] Full ratatui TUI with panels (see layout spec above)
- [ ] Live GPU metrics display (temp, utilisation, VRAM, power)
- [ ] Earnings display (mock data initially)
- [ ] Throughput sparkline chart
- [ ] Log tail view
- [ ] Keyboard navigation (quit, pause, switch views)
- [ ] Graceful terminal resize handling

### Sprint 6 — Pipeline Parallelism Foundation (Week 7-10)
- [ ] P2P communication layer (QUIC-based)
- [ ] Pipeline formation logic (node grouping, latency testing)
- [ ] Model sharding + layer distribution to pipeline stages
- [ ] Activation forwarding between pipeline stages
- [ ] Basic micro-batch scheduling

### Sprint 7 — Workload Execution (Week 10-14)
- [ ] Docker container management (pull, start, stop, health check)
- [ ] GPU passthrough configuration (nvidia-docker / Metal)
- [ ] First workload: llama.cpp based LLM inference in pipeline mode
- [ ] Request routing from daemon to pipeline stages
- [ ] Container auto-restart on failure
- [ ] Failover handling (node drops out of pipeline)
- [ ] Benchmark: Llama 70B across 5 consumer GPUs

### Sprint 8 — Orchestrator MVP (Week 14-18)
- [ ] Orchestrator service (Rust or TypeScript)
- [ ] Node registration + capability reporting
- [ ] Job queue + matching algorithm
- [ ] Pipeline scheduling across nodes
- [ ] Public API layer (OpenAI-compatible endpoints)
- [ ] Basic auth (API keys for buyers)
- [ ] Work tracking for reward calculations

### Sprint 9 — Solana Integration (Week 18-24+)
- [ ] Anchor program for rewards distribution
- [ ] Wallet integration in CLI
- [ ] Reward claiming flow
- [ ] Earnings tracking (on-chain + off-chain hybrid)
- [ ] Basic staking mechanism
- [ ] Token burn on compute purchases

### Sprint 10 — Enterprise & Integrations (Week 24+)
- [ ] OpenRouter provider integration
- [ ] Akash Network provider agent
- [ ] io.net supplier integration
- [ ] Enterprise API features (SLAs, bulk pricing, dashboards)
- [ ] Admin dashboard for enterprise clients

---

## Build & Release

### Cross-Compilation Targets

```bash
# Linux
x86_64-unknown-linux-gnu
aarch64-unknown-linux-gnu

# macOS
x86_64-apple-darwin
aarch64-apple-darwin

# Windows
x86_64-pc-windows-msvc
aarch64-pc-windows-msvc
```

Use `cross` (https://github.com/cross-rs/cross) for Linux cross-compilation. Use GitHub Actions matrix build for all targets. Windows builds use MSVC toolchain.

### Release Process

1. Tag version in git (`v0.1.0`)
2. GitHub Actions builds all 4 binaries
3. Creates GitHub Release with binaries attached
4. Updates `https://compute.sh/releases/latest/` symlinks
5. `compute update` checks GitHub Releases API for newer versions

### Binary Size Target

Aim for <20MB single binary. Rust with LTO + strip should achieve this. No runtime dependencies except Docker (for workload execution, not for the CLI itself).

---

## Configuration

Config file: `~/.compute/config.toml` (Linux/macOS) or `%APPDATA%\compute\config.toml` (Windows)

```toml
[node]
name = "my-gaming-rig"          # Optional friendly name
max_gpu_usage = 90               # Max GPU % to use when idle
max_cpu_usage = 50               # Max CPU % to use when idle
idle_threshold_minutes = 5       # Minutes of no input before "idle"
pause_on_battery = true          # Don't run on battery power
pause_on_fullscreen = true       # Pause when fullscreen app detected

[wallet]
public_address = ""              # Solana public address for receiving rewards

[network]
orchestrator_url = "https://api.compute.sh"
region = "auto"                  # Auto-detect or manual override

[docker]
socket = "/var/run/docker.sock"  # Docker socket path
image_cache_dir = "~/.compute/images"

[logging]
level = "info"                   # debug, info, warn, error
file = "~/.compute/logs/compute.log"
max_size_mb = 100
```

---

## Testing Strategy

- **Unit tests**: Core logic (idle detection, config parsing, reward calculations)
- **Integration tests**: Docker container lifecycle, API client, daemon start/stop
- **E2E tests**: Full flow from `compute init` → `compute start` → serve inference → verify results
- **Benchmarks**: Inference latency, throughput, memory usage
- **Platform CI**: Test on Linux x86_64, macOS ARM, macOS x86_64, Windows x86_64 in GitHub Actions

---

## Key Dependencies (Rust Crates)

```toml
# CLI
clap = { version = "4", features = ["derive"] }

# TUI
ratatui = "0.29"
crossterm = "0.28"

# Async runtime
tokio = { version = "1", features = ["full"] }

# HTTP client/server
reqwest = { version = "0.12", features = ["json"] }
axum = "0.7"

# Serialisation
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"

# Solana
solana-sdk = "2"
anchor-client = "0.30"

# Docker
bollard = "0.18"          # Docker API client

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# System info
sysinfo = "0.32"          # CPU, memory, disk
nvml-wrapper = "0.10"     # NVIDIA GPU info (Linux)

# Networking
libp2p = "0.54"           # P2P (pipeline communication)
quinn = "0.11"            # QUIC transport
```

---

## Platform-Specific Notes

### macOS (Apple Silicon)
- Docker GPU passthrough doesn't work natively on macOS for NVIDIA GPUs
- Use llama.cpp with Metal backend or MLX natively (not in Docker)
- Apple Silicon is powerful for inference (M3 Max = ~90 tok/s on 7B models) but the ecosystem is MLX/Metal not CUDA
- Reference Parallax's custom Metal paged attention shaders (Apache 2.0)

### Windows
- NVIDIA GPU passthrough via Docker Desktop with WSL2 backend (nvidia-container-toolkit works under WSL2)
- Alternative: native CUDA inference without Docker (simpler for users who don't want Docker)
- Windows Terminal supports Unicode braille characters for the globe TUI — but verify terminal compatibility (cmd.exe has limited Unicode support, Windows Terminal / PowerShell 7 are fine)
- Installer: provide both a PowerShell install script (`install.ps1`) and an `.msi` / `.exe` installer for non-technical users
- Idle detection: Win32 `GetLastInputInfo` for input activity, `SetThreadExecutionState` awareness for preventing sleep
- Daemon: run as a Windows Service (via `windows-service` crate) or as a background process with system tray icon
- Config path: `%APPDATA%\compute\config.toml` (not `~/.compute/`)

### Linux
- NVIDIA: Docker + nvidia-container-toolkit (standard path)
- AMD ROCm: potential future support via Docker + ROCm runtime
- Idle detection: `/proc/stat` for CPU, `nvidia-smi` for GPU, `/dev/input/` for input activity
- Daemon: systemd service unit or background process with PID file

### Cross-platform execution paths
Three GPU backends, chosen at runtime:
1. **NVIDIA (Linux/Windows):** Docker containers with CUDA GPU passthrough
2. **Apple Silicon (macOS):** Native Metal/MLX inference (no Docker)
3. **CPU fallback (all platforms):** llama.cpp CPU mode for nodes without a GPU (lower performance, still useful for pipeline stages with small layer counts)

---

## Security Considerations

- All workloads MUST be containerised — no arbitrary code execution on host
- Docker containers run with minimal privileges (no host network, no privileged mode)
- Model weights are read-only mounted
- Network access from containers is restricted to orchestrator + API responses only
- No private keys stored — wallet integration is read-only (public address only)
- Node-to-orchestrator communication over TLS
- Pipeline P2P traffic encrypted end-to-end

---

## Notes

- The website already exists — focus is on the terminal app
- Token will launch on pump.fun — tokenomics designed around creator fee revenue + Burn-Mint Equilibrium
- Pipeline parallelism is the core model from day one — all nodes participate as pipeline stages in daisy-chained inference
- Enterprise sales (OpenRouter etc) can begin once the API is stable with enough nodes for reliable uptime
- The "idle compute" narrative is the marketing hook — make the onboarding experience (install → init → start → earning) feel magical
