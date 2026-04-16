# Llama Stage Gateway Ops — April 2026

This is the operator runbook for the experimental `llama-stage-gateway` backend in `compute-daemon`.

Current validated scope:
- model: `gemma-4-E4B-it-Q4_K_M.gguf`
- split: head `0-20`, tail `21-41`
- transport: patched `llama.cpp` stage nodes behind the TCP gateway
- daemon boundary: orchestrator WebSocket -> relay -> gateway

This is not the old custom `ggml` staged runtime. Compute stays inside patched `llama.cpp`. `compute-daemon` only routes requests to the gateway.

## What Exists

Validated paths:
- relay non-streaming through gateway
- relay streaming through gateway
- daemon assignment roundtrip against the gateway
- delayed gateway startup and reconnect-on-assignment
- installed sidecar lookup from `~/.compute/bin`
- classified `node_error` on unreachable gateway

Important boundary:
- `compute-daemon` does not expose a separate local inference HTTP server for this mode
- the real request surface is orchestrator WebSocket -> [relay.rs](/Users/macintosh/Documents/projects/Compute/compute-app/crates/compute-daemon/src/relay.rs)

## Required Artifacts

Model:
- `~/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf`

Installed sidecars:
- `~/.compute/bin/llama_stage_tcp_node`
- `~/.compute/bin/llama_stage_gateway_tcp_node`

Those sidecars are produced from the sibling `compute-backend` repo.

## Sidecar Install

From `compute-backend`:

```bash
cd /Users/macintosh/Documents/projects/Compute/compute-backend
cargo build -p llama-stage-backend --bins
cargo run -q -p llama-stage-backend --bin llama_stage_install_sidecars
```

That installs:
- `~/.compute/bin/llama_stage_tcp_node`
- `~/.compute/bin/llama_stage_gateway_tcp_node`

Resolution order used by the daemon:
1. explicit config path
2. `~/.compute/bin`
3. sibling `compute-backend/target/{debug,release}` as dev fallback

## Daemon Config

Relevant `~/.compute/config.toml` keys:

```toml
[experimental]
stage_mode_enabled = true
stage_backend = "llama-stage-gateway"

# Remote gateway mode
stage_gateway_addr = "10.0.0.20:9300"

# Or local autostart mode
stage_gateway_autostart = false
stage_gateway_model_path = "/Users/macintosh/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf"

# Optional explicit sidecar paths
stage_gateway_stage_node_bin = ""
stage_gateway_gateway_bin = ""

# Remote gateway timing knobs
stage_gateway_connect_timeout_ms = 2000
stage_gateway_retry_window_ms = 30000
stage_gateway_retry_interval_ms = 250
stage_gateway_startup_grace_ms = 0
```

Operational meaning:
- `stage_gateway_connect_timeout_ms`
  - timeout for one connection attempt to the gateway
- `stage_gateway_retry_window_ms`
  - total time budget for connect/reconnect before the daemon emits `node_error`
- `stage_gateway_retry_interval_ms`
  - backoff between failed attempts
- `stage_gateway_startup_grace_ms`
  - optional boot-time grace window for external gateways
  - daemon startup does one immediate connect attempt
  - if that fails and this grace is nonzero, the daemon starts relay normally and keeps retrying in the background for this grace window before logging a warning
  - assignment/poll reconnect still uses the normal retry window above

## Two-Machine LAN Run

Machine B: run the installed sidecars.

Stage nodes:
```bash
~/.compute/bin/llama_stage_tcp_node \
  --model ~/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf \
  --bind 0.0.0.0:9201 \
  --stage-id stage-0-20 \
  --start-layer 0 \
  --end-layer 20 \
  --head
```

```bash
~/.compute/bin/llama_stage_tcp_node \
  --model ~/.compute/models/gemma-4-E4B-it-Q4_K_M.gguf \
  --bind 0.0.0.0:9202 \
  --stage-id stage-21-41 \
  --start-layer 21 \
  --end-layer 41 \
  --tail
```

Gateway:
```bash
~/.compute/bin/llama_stage_gateway_tcp_node \
  --head 127.0.0.1:9201 \
  --tail 127.0.0.1:9202 \
  --bind 0.0.0.0:9300
```

Machine A: point `compute-daemon` at the remote gateway:

```toml
[experimental]
stage_mode_enabled = true
stage_backend = "llama-stage-gateway"
stage_gateway_addr = "10.0.0.20:9300"
stage_gateway_connect_timeout_ms = 2000
stage_gateway_retry_window_ms = 30000
stage_gateway_retry_interval_ms = 250
```

## Positive Proof Commands

Non-stream assignment proof through the daemon/orchestrator path:

```bash
cd /Users/macintosh/Documents/projects/Compute/compute-app
cargo run -q -p compute-daemon --bin llama_stage_gateway_daemon_assignment_roundtrip -- \
  --gateway 10.0.0.20:9300 \
  --max-tokens 4
```

Expected result:
- `match=true`
- `overall=PASS`

Streaming proof through the same path:

```bash
cd /Users/macintosh/Documents/projects/Compute/compute-app
cargo run -q -p compute-daemon --bin llama_stage_gateway_daemon_assignment_roundtrip -- \
  --gateway 10.0.0.20:9300 \
  --max-tokens 4 \
  --stream
```

Expected result:
- `streamed_text` matches baseline
- `saw_finish_reason=true`
- `saw_done=true`
- `overall=PASS`

Delayed-start proof with startup grace:

```bash
cd /Users/macintosh/Documents/projects/Compute/compute-app
cargo run -q -p compute-daemon --bin llama_stage_gateway_daemon_assignment_roundtrip -- \
  --max-tokens 4 \
  --delay-gateway-start-ms 3000 \
  --gateway-startup-grace-ms 5000
```

Expected result:
- `gateway_startup_grace_ms=5000`
- `delay_gateway_start_ms=3000`
- `match=true`
- `overall=PASS`

Direct relay WS streaming proof:

```bash
cd /Users/macintosh/Documents/projects/Compute/compute-app
cargo run -q -p compute-daemon --bin llama_stage_gateway_relay_ws_stream_roundtrip -- \
  --gateway 10.0.0.20:9300 \
  --max-tokens 4
```

Expected result:
- exact token/text parity on the built-in prompt set
- `overall=PASS`

## Failure-Path Proof

Use an unreachable address and a short retry budget to verify `node_error` classification:

```bash
cd /Users/macintosh/Documents/projects/Compute/compute-app
cargo run -q -p compute-daemon --bin llama_stage_gateway_daemon_assignment_roundtrip -- \
  --gateway 127.0.0.1:65500 \
  --max-tokens 4 \
  --gateway-connect-timeout-ms 200 \
  --gateway-retry-window-ms 1000 \
  --gateway-retry-interval-ms 100 \
  --expect-node-error "connect failed"
```

Expected result:
- `node_error_model=gemma-4-e4b-q4`
- `node_error` contains `connect failed`
- `overall=PASS`

## Error Classes

The daemon now distinguishes these gateway failures:
- connect failure
  - DNS, address resolution, TCP connect timeout, refused connection
- protocol/version mismatch
  - sidecar protocol version does not match the daemon expectation
- model mismatch
  - gateway reports a different model than the assignment expects
- reachable but unusable gateway
  - gateway answered, but its internal stage node state or response contract is invalid

These surface in daemon logs and the `node_error.error` field sent back to the orchestrator.

## Operational Notes

- `llama-stage-gateway` is currently the only validated staged llama.cpp path in `compute-app`
- local autostart is validated, but remote `stage_gateway_addr` is the cleaner production shape
- streaming final `inference_response` is usage/timings only; generated text arrives through `inference_stream_chunk`
- if the gateway is unavailable at daemon boot, the daemon now retries on assignment and poll paths instead of staying dead

## Current Known Limits

- validated model path is Gemma 4 E4B Q4
- stage split is fixed to the current 2-stage proof
- this runbook does not cover arbitrary multi-stage fanout
- rollout still depends on sidecar binaries being installed or explicitly configured
