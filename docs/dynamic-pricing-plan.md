# Dynamic Pricing Based on Node Availability

## Overview

Adjust credit costs per token dynamically based on real-time supply (available nodes) and demand (request volume). When nodes are plentiful, users get discounts. When nodes are scarce, prices increase to manage demand and incentivize more nodes to come online.

## Pricing Tiers

| Availability | Condition | Multiplier | User sees |
|---|---|---|---|
| Abundant | >10 idle nodes for model | 0.5x | "50% off — network has capacity" |
| Normal | 3-10 idle nodes | 1.0x | Standard rate |
| Scarce | 1-2 idle nodes | 1.5x | "High demand — 1.5x rate" |
| Critical | 0 idle nodes (queue only) | 2.0x | "Very high demand — 2x rate" |

## Data Needed

- **Per-model idle node count** — already available via `downloaded_models` heartbeat field + pipeline assignments
- **Request queue depth** — count of in-flight requests per model (add a counter in completions.ts)
- **Rolling request rate** — requests/min over last 5 minutes per model (sliding window counter)
- **Node TFLOPS capacity** — already tracked, use to weight availability (a 4090 node counts more than a 3060)

## Implementation Plan

### Phase 1: Metrics Collection
- Add `requestsInFlight` counter per model in completions handler (increment on entry, decrement on completion)
- Add rolling 5-min request rate tracker (simple ring buffer of timestamps)
- New endpoint: `GET /v1/pricing/live` returns per-model multiplier + reasoning
- Store multiplier snapshots in Supabase `pricing_history` table for analytics

### Phase 2: Multiplier Calculation
```
availableCapacity = sum(idle_node_tflops for nodes with model) / model_required_tflops
demandPressure = requestsPerMinute / baselineRequestsPerMinute

supplyScore = clamp(availableCapacity, 0, 10)  // 0 = no nodes, 10 = abundant
demandScore = clamp(demandPressure, 0, 5)       // 0 = quiet, 5 = heavy

multiplier = baseMultiplier * (demandScore / supplyScore)
multiplier = clamp(multiplier, 0.5, 2.0)        // floor and ceiling
```

### Phase 3: Apply to Billing
- `deductCredits()` in billing.ts reads current multiplier for the model
- Multiplier stored in `credit_transactions` table for auditability
- Show effective rate in API response headers: `X-Compute-Price-Multiplier: 0.5`
- `/v1/pricing/live` endpoint for clients to check before sending requests

### Phase 4: User-Facing
- compute-code splash screen shows current multiplier if != 1.0x
- `/model` picker shows "50% off" or "1.5x demand" badges next to each model
- API response includes `pricing` field in usage object:
  ```json
  {
    "usage": {
      "prompt_tokens": 100,
      "completion_tokens": 200,
      "credits_charged": 12,
      "price_multiplier": 0.5
    }
  }
  ```

### Phase 5: Node Incentives (Mirrors Pricing)
- When demand is high (multiplier > 1.0x), increase reward rate for nodes serving that model
- This creates a market signal: high demand → higher rewards → attracts more nodes → supply normalizes → price drops
- Reward multiplier = `max(1.0, priceMultiplier * 0.8)` (nodes get 80% of the surge)

## Anti-Gaming Protections

- **Smoothing**: Multiplier changes gradually (max ±0.1 per minute) to prevent oscillation
- **Minimum observation window**: Need 5+ minutes of data before applying non-standard pricing
- **Node spoofing**: Nodes must actually serve requests to count as "available" — heartbeat alone doesn't count. Use `last_request_served_at` freshness check
- **Self-dealing**: A node operator can't create API keys and send requests to their own node at inflated prices — reward deduction already accounts for this via wallet-level caps

## Database Schema

```sql
CREATE TABLE pricing_history (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  model_id text NOT NULL,
  multiplier numeric(4,2) NOT NULL,
  idle_nodes integer NOT NULL,
  requests_per_min numeric(8,2) NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Add to credit_transactions
ALTER TABLE credit_transactions ADD COLUMN price_multiplier numeric(4,2) DEFAULT 1.0;
```

## Open Questions

1. Should we give existing users a grace period before dynamic pricing kicks in?
2. Should pre-paid credit bundles lock in the 1.0x rate at purchase time?
3. Do we want per-model pricing or a single network-wide multiplier?
4. Should the floor go below 0.5x to aggressively attract early users?
