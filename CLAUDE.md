# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Compute is a DePIN terminal application that aggregates idle GPU/CPU resources and daisy-chains them via pipeline parallelism to run large AI models no single machine could handle. Revenue flows back to contributors via the $COMPUTE token on Solana. This repo (`compute-app`) contains the CLI daemon + TUI application. The sibling `compute-website` repo contains the marketing landing page.

**Current status:** Planning phase. Architecture is documented in `compute-plan.md`. No application code has been written yet.

## Core Architecture

- **Pipeline parallelism only** — every node participates as a stage in a daisy-chained inference pipeline. There is no standalone single-node inference mode. NOT tensor parallelism (too bandwidth-hungry for public internet).
- **TUI layout:** Left 1/3 = spinning ASCII globe (braille unicode, ratatui Canvas widget), right 2/3 = content/info/stats panels. This split is consistent across startup splash and dashboard.
- **macOS/Apple Silicon:** Uses native llama.cpp with Metal backend, NOT Docker+CUDA. Separate execution paths needed.
- **All workloads containerized** (Docker) except Apple Silicon Metal inference.
- **Config:** `~/.compute/config.toml`. Daemon uses PID/lockfile at `~/.compute/compute.pid`.
- **Binary target:** <20MB with LTO + strip. No runtime deps except Docker.

## Planned Tech Stack (CLI/Daemon)

- **Language:** Rust
- **CLI:** Clap 4
- **TUI:** Ratatui 0.29 + Crossterm 0.28
- **Async:** Tokio 1
- **HTTP:** Reqwest 0.12 (client) + Axum 0.7 (server)
- **Serialization:** Serde + TOML
- **Blockchain:** Solana SDK 2 + Anchor Client 0.30
- **Docker:** Bollard 0.18
- **System:** Sysinfo 0.32 + NVML Wrapper 0.10
- **P2P:** libp2p 0.54 + Quinn 0.11 (QUIC) — for pipeline stage communication
- **Logging:** Tracing 0.1

## Planned Workspace Structure

```
Cargo.toml (workspace root)
crates/
  compute-cli/          # CLI entry point + TUI
  compute-daemon/       # Background daemon process
  compute-network/      # P2P pipeline communication
  compute-solana/       # Solana read-only (public address + balance reading)
  compute-orchestrator/ # Central orchestrator (node registry, pipeline scheduling, public API)
```

## Build Targets

- Linux: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`
- macOS: `x86_64-apple-darwin`, `aarch64-apple-darwin`
- Windows: `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`

Uses `cross` crate for Linux cross-compilation. Windows builds use MSVC toolchain.

## Platform-Specific GPU Paths

- **NVIDIA (Linux/Windows):** Docker containers with CUDA GPU passthrough (WSL2 backend on Windows)
- **Apple Silicon (macOS):** Native Metal/MLX inference, no Docker. Reference Parallax's Metal paged attention shaders.
- **CPU fallback (all platforms):** llama.cpp CPU mode for nodes without a GPU
- **Windows daemon:** Windows Service (via `windows-service` crate) or background process. Config at `%APPDATA%\compute\config.toml`.
- **Windows TUI:** Requires Windows Terminal or PowerShell 7 for Unicode braille globe rendering.

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
