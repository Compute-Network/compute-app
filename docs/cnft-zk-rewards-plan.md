# cNFTs + ZK-Compressed Rewards — Implementation Plan

Status: **Evaluate later** (not urgent until 50+ nodes)

---

## Current Approach

- Nodes register via wallet address in Supabase (off-chain)
- Rewards recorded as `reward_events` rows in Supabase
- On-chain distribution via individual SPL transfers from vault
- Users claim via `/claim` → build tx → sign → `/confirm`
- Double-claim race condition fixed with per-wallet lock (April 2026)

## Proposed Approach

### Phase 1: Merkle Distributor for Claims (at 50+ nodes)

Replace individual on-chain transfers with periodic Merkle root posting:

1. Accumulate rewards off-chain in Supabase (no change)
2. Every hour, compute Merkle root of all pending rewards
3. Post single root transaction on-chain (~0.000005 SOL regardless of node count)
4. Users claim by submitting Merkle proof of their reward amount
5. On-chain program verifies proof against root, transfers tokens

**Tools:** Jupiter Merkle Distributor SDK (open-source, battle-tested)

**Cost at 10K nodes:**
- Current: ~4,320 SOL/day (individual pushes)
- Merkle: ~0.12 SOL/day (24 hourly roots) + ~0.003 SOL per user claim

### Phase 2: cNFTs for Node Identity (at 500+ nodes)

Each registered node gets a compressed NFT on Solana:

- **Proves ownership:** Only the registering wallet can hold the cNFT
- **Contains metadata:** GPU model, VRAM, region, registration date, trust score
- **Updatable:** Metadata updates cost ~0.00001 SOL
- **Cost:** 1 million cNFTs for ~1 SOL (proven by Helium at 991K hotspot scale)
- **How:** Merkle tree with roots on-chain, individual NFT data in ledger

**Enables:**
- Cryptographic node ownership proof (no impersonation)
- On-chain staking/slashing (stake against cNFT)
- Immutable hardware claims
- TOPLOC trust scores as updatable metadata
- Future governance (cNFT holders vote on protocol parameters)

### Phase 3: ZK Compression (at 1000+ nodes)

Switch from standard Merkle to ZK-compressed distribution:

- **Helius AirShip:** Open-source ZK-compressed airdrop tool
- **Cost:** 10K distributions for ~1.83 SOL (vs ~3 SOL standard Merkle)
- **50%+ further reduction** on claim costs

## Cost Comparison at Scale

| Approach | 100 nodes/day | 1K nodes/day | 10K nodes/day |
|----------|--------------|-------------|--------------|
| Current (per-event push) | ~43 SOL | ~432 SOL | ~4,320 SOL |
| Phase 1 (Merkle claims) | ~0.42 SOL | ~3.12 SOL | ~30.12 SOL |
| Phase 2+3 (ZK compressed) | ~0.21 SOL | ~1.95 SOL | ~18.45 SOL |

## Why Not Now

1. At single-node scale, the overhead is negligible
2. The Merkle Distributor adds client-side complexity (proof generation)
3. cNFT infrastructure requires additional Solana program changes
4. Current per-wallet lock fix addresses the immediate double-claim bug

## When to Revisit

- **50 nodes:** Implement Phase 1 (Merkle distributor)
- **500 nodes:** Implement Phase 2 (cNFTs for identity)
- **1000 nodes:** Implement Phase 3 (ZK compression)
- **Or:** When on-chain distribution costs exceed 10 SOL/day

## References

- [Solana DePIN Quickstart Guide](https://solana.com/developers/guides/depin/getting-started)
- [Helius AirShip (ZK-compressed airdrops)](https://www.helius.dev/docs/airship/overview)
- [Helium cNFT Implementation (991K hotspots)](https://www.helius.dev/blog/all-you-need-to-know-about-compression-on-solana)
- [Jupiter Merkle Distributor SDK](https://github.com/jup-ag/merkle-distributor)
- [ZK Compression Solana Docs](https://www.zkcompression.com/)
