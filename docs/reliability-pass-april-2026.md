# Reliability Pass - April 2026

## Scope

Single-node production-oriented validation for:

- node auth enforcement
- Railway deploy / orchestrator restart behavior
- node relay reconnect behavior
- node heartbeat correctness
- dashboard/control-plane read paths

## What Was Verified

### 1. Node auth is enforced on production endpoints

Verified against `api.computenetwork.sh`:

- `POST /v1/nodes/register` without bearer token returns `401 Missing node session token`
- `POST /v1/nodes/:wallet/heartbeat` with invalid bearer token returns `401 Invalid or expired node session token`

This confirms the node control plane is no longer open to unauthenticated registration/heartbeat writes.

### 2. Relay reconnect after Railway deploy is observed

Railway logs showed:

- `[relay] Node 5fca306a-2787-4aba-92a6-f9759ce3133e connected (1 total)`

This is a positive signal that the node WebSocket relay reconnect path survives orchestrator deploy/restart.

### 3. `GET /v1/nodes/stats` production bug was fixed and redeployed

Problem found during validation:

- `GET /v1/nodes/stats` was being swallowed by the `/:wallet` route and returning `404`/`null`

Fix:

- reordered routes in `orchestrator/src/routes/nodes.ts`
- deployed to Railway in deployment `87e159d9-8a7d-435f-b952-934b5c83990b`

Post-fix verification:

- `GET /v1/nodes/stats` now returns JSON like:
  - `{"total":2,"online":0,"totalTflops":0}`

## Reliability Issues Still Present

### 1. Live daemon heartbeats are being rejected every 10s

Observed repeatedly in Railway HTTP logs:

- `POST /v1/nodes/<wallet>/heartbeat 400`

Root cause:

- Rust `HeartbeatPayload` was serializing `Option::None` fields as explicit `null`
- orchestrator `HeartbeatPayload` Zod schema accepts omitted fields, not `null`

Fix made locally:

- added `#[serde(skip_serializing_if = "Option::is_none")]` to optional fields in `crates/compute-network/src/client.rs`

Impact:

- until the updated daemon/client is rebuilt and run locally, the node remains effectively offline in control-plane state even though relay auth may succeed

### 2. Production node state is stale / partially inconsistent

Current live read:

- relay shows the node can connect
- node row still shows `status: offline`
- last heartbeat is stale

Interpretation:

- WebSocket reconnect works
- heartbeat/status plane is still blocked by the old client payload

### 3. Helius pricefeed reconnect churn still exists

Observed in Railway logs:

- `Helius websocket disconnected, reconnecting in 5s...`
- `Helius websocket connected`

This is not currently fatal, but it remains an instability source in the orchestrator process.

## Code Changes Made In This Pass

### Orchestrator

- fixed node route ordering in `orchestrator/src/routes/nodes.ts`
- wallet dashboards now resolve account-backed keys in:
  - `orchestrator/src/routes/apikeys.ts`
  - `orchestrator/src/services/apikeys.ts`
  - `orchestrator/src/services/billing.ts`

### Rust client / daemon

- improved node session error surfacing in:
  - `crates/compute-network/src/client.rs`
  - `crates/compute-daemon/src/relay.rs`
- deprecated public-address-only CLI path in:
  - `crates/compute-cli/src/main.rs`
  - `crates/compute-cli/src/cli.rs`
- removed TUI wallet editing path in:
  - `crates/compute-cli/src/tui/dashboard.rs`
- fixed heartbeat serialization in:
  - `crates/compute-network/src/client.rs`

### Website

Locally implemented but not deployed from this session:

- split dashboard auth surface into:
  - `Nodes`
  - `Authorised Apps`
  - `API Keys`

File:

- `compute-website/src/dashboard/Dashboard.tsx`

## What Must Happen Next

1. Rebuild and run the updated local `compute` client so heartbeats stop failing.
2. Verify that the node transitions back to `online` in production.
3. Re-run deploy/restart inference tests once heartbeats are healthy:
   - Railway deploy during active inference
   - orchestrator restart with active node relay
   - websocket drop/reconnect during idle and during request handling
   - llama-server crash while connected
4. Deploy the website changes so account-backed `compute-code` sessions appear under `Authorised Apps`.

## Current Bottom Line

The node auth boundary is substantially better and production reads are more coherent, but the current live daemon still cannot maintain healthy control-plane status because its heartbeat payload format is stale. The next validation pass should happen only after running the updated local binary.

## Follow-up Pass

### What improved

- explicit model inference on production is healthy again after daemon restart
- relay-side client cancel now propagates end-to-end:
  - client abort
  - orchestrator relay cancel
  - daemon request cancel
  - node becomes available for the next request quickly

Evidence:

- local daemon log recorded:
  - `[relay] Cancelled request req-...`
- after aborting a long streaming request, the next request succeeded in about 6 seconds instead of hanging behind the cancelled generation

### What is fixed locally but requires another daemon restart

- throughput/token-per-second display in the TUI

Root cause:

- the daemon only exposed the final TPS sample for roughly one metrics tick, then cleared it
- the dashboard chart and numeric label almost always missed that sample, so the chart looked dead even when the node was serving

Fix made locally:

- hold the most recent measured TPS for 5 seconds in `crates/compute-daemon/src/runtime.rs`
- this preserves the existing chart design while making the data visible

### Compute Code auth cleanup

Anthropic / Claude auth was disabled locally in `compute-code` so the app defaults to the Compute OpenAI-compatible path instead of falling back to Anthropic account flows.

Key local changes:

- `openclaude/src/utils/model/providers.ts`
- `openclaude/src/utils/auth.ts`
- `openclaude/src/entrypoints/cli.tsx`

This should stop the random Claude browser auth windows and keep `/login` and `/logout` Compute-only.

### Remaining reliability issue

- Railway deploy during active inference still does not complete cleanly

Observed behavior after the graceful-drain changes:

- the node reconnects after deploy
- the client stream no longer explodes with the earlier immediate 502 behavior
- but the in-flight stream is still truncated and does not end with `data: [DONE]`

Interpretation:

- request cancellation is now working
- graceful draining improved behavior but did not fully solve in-flight stream survival across a Railway replacement
- this likely needs a deeper approach than simple process drain, such as resumable requests or a different deployment/runtime strategy

## Latest Root Cause

### Restarted daemon still timed out on completions

After a clean daemon restart, explicit `/v1/chat/completions` requests started hanging for the full 120 second readiness timeout again.

Observed behavior:

- production `POST /v1/chat/completions` entered the orchestrator and then returned `503` after 120s
- the node stayed online and heartbeats continued succeeding
- local llama health on `http://127.0.0.1:8090/health` was healthy
- the daemon log for the latest restart showed:
  - `Pipeline reassigned: pre-warm -> ... (same model ..., keeping server)`
  - but no matching `[health] llama-server ready after ...` line after that restart

Root cause:

- the daemon has two assignment paths:
  - WebSocket push
  - fallback assignment polling
- the WebSocket push path already ran `spawn_ready_probe(...)` and emitted `node_ready`
- the fallback polling path updated the inference assignment but did not emit `node_ready`
- on startup, the polling path can win the race before the WebSocket push path is active
- that leaves the orchestrator waiting in `waitForNodeReady(...)` for the full timeout even though llama is already healthy

Fix made locally:

- updated `crates/compute-daemon/src/runtime.rs`
- `check_assignment(...)` now receives the WS outbound sender
- when poll-based assignment detection changes the active pipeline/model, it now runs the same ready probe used by the WS push path

Impact:

- after the next daemon restart on the updated binary, startup/reconnect should no longer hang for 120s waiting on a missing `node_ready`

### Claude browser popup investigation

Code search in `compute-app` did not find any browser-launch behavior in the daemon crates.

What was found:

- browser opening exists in:
  - `crates/compute-cli/src/main.rs`
  - `crates/compute-cli/src/tui/onboarding.rs`
  - `crates/compute-cli/src/tui/splash.rs`
  - `crates/compute-cli/src/tui/dashboard.rs`
- those are all CLI/TUI wallet-login or claim-page flows
- no equivalent browser-open path was found in:
  - `crates/compute-daemon`
  - `crates/compute-network`

Current conclusion:

- the random browser popup in `compute-app` does not appear to originate from the daemon relay/runtime code path
- if it still happens after the Compute-only auth cleanup elsewhere, the next step is to reproduce the exact CLI/TUI action that triggers it and trace that specific path

## Latest Validation

### Restart-after-build regression is fixed

## Latest Stabilization Sweep

### Runtime / control-plane cleanup

- background maintenance loops in the orchestrator now use jittered recurring scheduling instead of many fixed `setInterval(...)` calls
- this reduces synchronized timer spikes and makes reconnect/cleanup work less bursty under load
- files updated:
  - `orchestrator/src/index.ts`
  - `orchestrator/src/routes/auth.ts`
  - `orchestrator/src/services/apikeys.ts`
  - `orchestrator/src/middleware/walletAuth.ts`
  - `orchestrator/src/services/rewards.ts`
  - `orchestrator/src/services/timers.ts`

### Pricefeed reconnect behavior is less noisy

- Helius websocket reconnect now uses bounded backoff + jitter instead of a fixed 5 second loop
- repeated disconnect logs are rate-limited so transient churn is less noisy operationally
- fallback polling now uses the jittered recurring scheduler too
- file:
  - `orchestrator/src/services/pricefeed.ts`

### CLI startup path drift reduced

- node registration logic is now shared between the TUI path and the CLI `compute start` path instead of being implemented twice
- this reduces the chance of auth/registration behavior diverging between dashboard mode and foreground daemon mode
- files:
  - `crates/compute-cli/src/tui/app.rs`
  - `crates/compute-cli/src/main.rs`

### Legacy Supabase compile-time coupling reduced

- the active Rust orchestrator client no longer imports read-model structs from the legacy `supabase.rs` module
- this does not remove the legacy module yet, but it reduces coupling between the active control-plane path and the old direct-Supabase path
- file:
  - `crates/compute-network/src/client.rs`

### Solana / billing drift reduced

- billing crypto top-up responses now report the cluster derived from the configured RPC URL instead of always claiming `devnet`
- Solana service startup logs now report the derived cluster
- airdrop behavior is gated to dev/test style clusters only
- files:
  - `orchestrator/src/services/solana.ts`
  - `orchestrator/src/routes/billing.ts`
  - `orchestrator/src/services/crypto-deposits.ts`

### Docs and operator guidance updated

- `CLAUDE.md` no longer describes the repo as planning-only
- `README.md` and `docs/technical-overview.md` now point to `compute wallet login` instead of the removed `compute wallet set`
- CLI help now correctly describes `compute start` as foreground rather than detached

## Current Remaining Gaps

- secret rotation is still an operational task, not solved in code
- process-local auth/rate-limit/lock state is still in-memory rather than shared/durable
- in-flight requests still do not survive Railway replacements; they now fail fast and clean instead of hanging
- multi-node code surface still exceeds what production currently exercises

After restarting the daemon on the new build:

- the daemon log now includes a post-start assignment readiness line:
  - `[health] llama-server ready after 2.5s`
- a direct non-streaming completion against `qwen3.5-27b-q4` returned successfully in about 4.5 seconds instead of hanging for 120 seconds

This confirms the poll-path `node_ready` fix in `crates/compute-daemon/src/runtime.rs` is working.

### Cancel still works after the restart fix

Validation:

- started a long streaming request
- aborted the client after ~3s
- daemon logged:
  - `[relay] Cancelled request req-...`
- immediate follow-up request succeeded in about 6 seconds

Conclusion:

- cancel propagation remains healthy after the latest daemon changes

### Deploy during active inference still does not survive

Live Railway validation still failed for in-flight streaming requests:

- during deploy, the relay disconnected and later reconnected
- orchestrator logs showed the affected request ending as:
  - `POST /v1/chat/completions 503 120s`
- the captured stream output in `/tmp/compute_deploy_probe.out` was truncated
- there was no final `data: [DONE]`

Conclusion:

- graceful drain and reconnect behavior are improved
- but active in-flight inference still does not survive a Railway replacement

### New protocol issue exposed by the probes

The OpenAI-compatible response shape is still leaking reasoning text from the model:

- non-streaming responses returned empty `message.content`
- streamed and non-streamed responses emitted `reasoning_content`
- example probe output showed internal “Thinking Process” text instead of the requested plain answer

Impact:

- this is a correctness bug for `compute-code` and any OpenAI-compatible client
- the relay/orchestrator path needs a response normalization layer, or the underlying llama/openai-compatible settings need to be changed so visible output lands in `content` rather than `reasoning_content`

### Live workload metrics fix is coded but not yet running on the node

I added a daemon-side fix so non-streaming requests expose live work state immediately:

- relay now tracks in-flight request count, not just a boolean
- runtime heartbeats now send `inference_slots_busy` using the max of llama slot metrics and relay in-flight requests
- the daemon sends an immediate heartbeat when busy-slot count changes, instead of waiting for the next 10s interval
- the dashboard now treats `active_requests > 0` as `ACTIVE`, even before a completed TPS sample lands

Validation against the live node showed why the issue was still visible:

- during a 1200-token Gemma request, the node record still reported `inference_slots_busy: 0`
- the running process was `target/release/compute` started at `13:34`
- the updated release binary was built afterward, so the active daemon process was still the pre-fix binary

Current conclusion:

- the fix is in code and the orchestrator schema is deployed
- one more daemon restart is required before live non-streaming requests will show busy state and immediate dashboard activity

## Latest Follow-up

### Non-streaming busy-state is now verified live

After restarting the daemon on the rebuilt binary and probing a long Gemma request:

- the live node record showed `inference_slots_busy: 1` mid-flight
- the node also reported `inference_slots_total: 2`
- the request completed normally at about `32.8 tok/s`

Conclusion:

- the daemon/orchestrator metrics path is now exposing real in-flight work for non-streaming requests
- the dashboard should now be able to show active work before completion

### Request cancellation is confirmed end to end

I force-closed a streaming client connection mid-generation and verified:

- the daemon logged `[relay] Cancelled request req-...`
- the relay cancel path reached the node promptly

Conclusion:

- dropped/aborted clients now cancel active generation instead of leaving it running in the background

### Deploy-time behavior still does not preserve in-flight work, but it now fails fast

After adding explicit relay shutdown handling during orchestrator drain:

- deploy-time requests no longer hung for 120 seconds
- the client now receives an SSE error followed by `data: [DONE]`
- observed response:
  - `{"error":{"message":"Node is loading the model. Please retry in a moment.","type":"server_error"}}`

Conclusion:

- Railway replacements still interrupt active work
- but the system now fails fast and clean instead of leaving clients hanging until timeout

### `requests_served` counter now updates again

The previous RPC-based metric increment path was not updating the node row reliably in production.

Current behavior after the direct DB fallback:

- `requests_served` advanced from `67` to `68` after a successful Gemma request
- `tokens_per_second` was updated at the same time

Conclusion:

- node accounting is now truthful again for the single-node production path
