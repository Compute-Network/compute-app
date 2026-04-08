# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Compute is a DePIN terminal application plus control plane for serving local and distributed inference. Revenue flows back to contributors via the $COMPUTE token on Solana. This repo (`compute-app`) contains the Rust CLI/daemon/TUI, the live TypeScript `orchestrator/`, and the related research/docs used to evolve both.

**Current status:** Active codebase, not planning-only.
- Production orchestrator is deployed on Railway.
- Supabase is the backing database.
- The stable live path today is primarily single-node inference with node auth, relay scheduling, billing, and rewards.
- Multi-node pipeline code and docs exist, but that path is still experimental relative to the single-node path.

## Core Architecture

- **Single-node first in practice** — the stable production path today is one node serving one model through the orchestrator relay.
- **Multi-node pipeline work exists** — the repo also contains pipeline/sharding research and scheduler code for future multi-node inference, but treat that path as experimental unless the task is explicitly about distributed inference.
- **TUI layout:** Left 1/3 = spinning ASCII globe (braille unicode, ratatui Canvas widget), right 2/3 = content/info/stats panels. This split is consistent across startup splash and dashboard.
- **macOS/Apple Silicon:** Uses native llama.cpp with Metal backend, NOT Docker+CUDA. Separate execution paths needed.
- **Apple Silicon path matters** — local Metal-backed `llama-server` on macOS is a first-class workflow in this repo.
- **Config:** `~/.compute/config.toml`. Daemon uses PID/lockfile at `~/.compute/compute.pid`.
- **Binary target:** <20MB with LTO + strip. No runtime deps except Docker.

## Current Tech Stack

- **Language:** Rust
- **CLI:** Clap 4
- **TUI:** Ratatui 0.29 + Crossterm 0.28
- **Async:** Tokio 1
- **HTTP:** Reqwest 0.12 for Rust clients, Hono for the orchestrator API/relay
- **Serialization:** Serde + TOML
- **Blockchain:** Solana SDK 2 + Anchor Client 0.30
- **Docker:** Bollard 0.18
- **System:** Sysinfo 0.32 + NVML Wrapper 0.10
- **Relay/control plane:** orchestrator WebSocket relay + Supabase-backed persistence
- **Logging:** Tracing 0.1

## Workspace Structure

```
Cargo.toml (workspace root)
crates/
  compute-cli/          # CLI entry point + TUI
  compute-daemon/       # Background daemon process
  compute-network/      # P2P pipeline communication
  compute-solana/       # Solana read-only (public address + balance reading)
orchestrator/            # Central orchestrator (node registry, scheduling, relay, public API)
```

## Build Targets

- Linux: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`
- macOS: `x86_64-apple-darwin`, `aarch64-apple-darwin`
- Windows: `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`

Uses `cross` crate for Linux cross-compilation. Windows builds use MSVC toolchain.

## Platform Notes

- **NVIDIA (Linux/Windows):** Docker containers with CUDA GPU passthrough (WSL2 backend on Windows)
- **Apple Silicon (macOS):** Native Metal/MLX inference, no Docker. Reference Parallax's Metal paged attention shaders.
- **CPU fallback (all platforms):** llama.cpp CPU mode for nodes without a GPU
- **Config:** `~/.compute/config.toml` on macOS/Linux
- **Logs:** `~/.compute/logs/compute.log`
- **Daemon runtime:** local `llama-server` process management is part of the normal single-node path

## TUI Design Principles (from design-language.md)

- Monochromatic palette — white/light grey for content, no bright brand colours
- Borders as structure: box-drawing characters for panel dividers, not heavy fills
- Flat, sharp aesthetic: square corners (`┌─┐`), no curves
- Generous whitespace within panels
- Section headers in UPPERCASE mono for systematic/technical feel
- Globe panel can use darker background for contrast (mirrors website's dark section break)
- Subtle animation only: slow globe rotation, smooth sparkline updates

## Branding

Design specs in `Branding-Guide/design-language.md`. Key: monochromatic stone palette, DM Sans headings, Inter body, IBM Plex Mono for code/terminal output.

## Practical Guidance

- Do not assume old planning docs reflect the live system; read the current code first.
- For node auth and control-plane behavior, prefer the orchestrator paths over legacy direct-Supabase client paths.
- When improving reliability, prioritize the single-node path before expanding distributed-system surface area.
- `compute-code` and `compute-website` are sibling repos and matter for wallet auth, API-key UX, and end-to-end testing.
