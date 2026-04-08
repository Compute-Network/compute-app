# Stabilization Review — April 2026

Point-in-time review of `compute-app` and the live `compute-orchestrator` deployment after end-to-end inspection of:

- local code in `compute-app`
- sibling auth flows in `compute-code` and `compute-website`
- live Railway production service
- live Supabase production project

This document is meant to preserve the current findings before refactors begin.

---

## Findings

1. Secrets handling is unsafe in current operational practice.
   `JWT_SECRET`, `SUPABASE_SERVICE_KEY`, `SOLANA_KEYPAIR`, and Helius keys are present in Railway env, and some were surfaced during CLI inspection. Any accidental logging, screen share, copied terminal output, or debug dump can compromise auth, database access, or treasury control.

   Relevant code:
   - `orchestrator/src/services/db.ts`
   - `orchestrator/src/middleware/enterpriseAuth.ts`
   - `orchestrator/src/services/solana.ts`

   Immediate action:
   - Rotate `SUPABASE_SERVICE_KEY`
   - Rotate `JWT_SECRET`
   - Rotate `SOLANA_KEYPAIR`
   - Rotate Helius keys if they have been broadly exposed

2. The node trust boundary is weak.
   Node registration and heartbeat are effectively accepted by wallet string, not proven wallet ownership. The Rust client can write directly via the Supabase anon client, and orchestrator node endpoints do not require cryptographic proof.

   Relevant code:
   - `crates/compute-network/src/supabase.rs`
   - `orchestrator/src/routes/nodes.ts`

   Impact:
   - spoofed nodes
   - hijacked heartbeats
   - polluted scheduler state

3. Production is still single-node inference.
   Live `pipeline_stages` rows are all `stage_index = 0` covering the full model layer range. Multi-node pipeline logic exists in code but is not the live execution path today.

   Relevant code:
   - `docs/pipeline-shard-planning.md`
   - `orchestrator/src/services/scheduler.ts`
   - `crates/compute-daemon/src/relay.rs`
   - `crates/compute-daemon/src/inference/manager.rs`

   Impact:
   - large unproven code surface
   - more failure surface than current production needs

4. The architecture is split-brain between Supabase and orchestrator control.
   The daemon uses Supabase directly for registration, heartbeat, assignment checks, stats, and earnings, while also using the orchestrator for relay traffic and API control.

   Relevant code:
   - `crates/compute-network/src/supabase.rs`
   - `crates/compute-daemon/src/runtime.rs`

   Impact:
   - two control planes
   - duplicated business logic
   - harder debugging and weaker reliability

5. Important auth and rate-limiting state is process-local.
   Device codes, auth rate limits, API key cache, API rate limits, claim locks, circuit breakers, and pipeline formation locks live in memory inside one orchestrator process.

   Relevant code:
   - `orchestrator/src/routes/auth.ts`
   - `orchestrator/src/services/apikeys.ts`
   - `orchestrator/src/services/rewards.ts`
   - `orchestrator/src/services/scheduler.ts`

   Impact:
   - resets on restart
   - incorrect behavior across replicas
   - weak scalability

6. Billing pre-authorization is not truly enforced.
   Streaming requests can consume compute before billing is finalized because `preAuthorize()` currently returns an estimate without creating an actual hold.

   Relevant code:
   - `orchestrator/src/services/billing.ts`

   Impact:
   - softer billing guarantees than intended
   - disconnect and abuse edge cases remain under-protected

7. Docs and runtime behavior have drifted.
   Solana and billing docs still describe devnet-oriented behavior in places, while live infra uses mainnet Helius URLs and a real mint.

   Relevant code:
   - `orchestrator/src/services/solana.ts`
   - `orchestrator/src/routes/billing.ts`

   Impact:
   - operator confusion
   - incident response mistakes
   - incorrect assumptions during debugging

8. The pricefeed shows live websocket instability.
   Railway logs show repeated Helius websocket disconnect/reconnect churn.

   Relevant code:
   - `orchestrator/src/services/pricefeed.ts`

   Impact:
   - background service instability
   - needs graceful degradation and monitoring

9. Startup behavior is duplicated across CLI paths.
   The splash/dashboard path and `compute start` are similar but not identical, which creates a classic “works in one path, fails in another” risk.

   Relevant code:
   - `crates/compute-cli/src/tui/app.rs`
   - `crates/compute-cli/src/main.rs`

10. Some project guidance docs are stale enough to mislead engineering work.
    `CLAUDE.md` still describes the repo as planning-only with no code written.

    Relevant code:
    - `CLAUDE.md`

---

## What I'd Do Next

1. Collapse to one control plane.
   Make the orchestrator the source of truth for registration, heartbeat, assignment, usage, and earnings. Avoid direct client writes to Supabase for authoritative state.

2. Harden the single-node path before investing more in multi-node.
   The live product today is effectively “one machine serves one model reliably through the relay.” Optimize that path first:
   - startup latency
   - reconnect behavior
   - llama-server crash recovery
   - billing correctness
   - API responsiveness

3. Add real auth to node identity.
   Reuse the existing wallet-auth pattern already used by `compute-website` and `compute-code`, but apply it to the node plane.

4. Feature-flag or cut back unfinished distributed complexity.
   Multi-node code should not be treated as production-critical until it is tested as a real path.

5. Move correctness-critical process-local state into durable/shared storage.
   Minimum candidates:
   - device auth state
   - claim idempotency / lock state
   - rate limiting
   - pipeline formation coordination

6. Build an explicit reliability test matrix for the current single-machine setup.

   Test cases:
   - orchestrator restart during inference
   - websocket drop during inference
   - llama-server 503
   - llama-server crash
   - slow llama health
   - Supabase latency or failure
   - Railway cold start / deploy during active node session

---

## Open Questions

- Is the immediate target “single trusted operator on one machine” or “ready for outside node operators”?
- Is low latency for the current solo setup more important right now than clean distributed architecture?
- Are we willing to simplify aggressively, even if that means disabling or removing multi-node paths temporarily?

---

## Bottom Line

The project is close to being a usable single-operator system, but it is carrying too much distributed-system scaffolding for what production is actually doing today.

The highest-leverage path to “fast and reliable” is:

- simplify around the proven single-node path
- centralize authority in the orchestrator
- harden node auth
- harden reconnect and crash behavior
- tighten billing correctness

Only after that should multi-node pipeline behavior become a primary focus again.

---

## Constraint For Refactor

Any auth/control-plane refactor must preserve the current provider onboarding flow:

1. A user can enter a public wallet address in the app before they have a website account.
2. The node can start earning under that wallet address immediately.
3. When the user later connects that same wallet on the website for the first time, the Supabase account is created and linked automatically.

Refactors should improve trust and control-plane integrity without breaking that account-less-first onboarding path.
