# Solana Program Deployment

## Deploy Wallet

- **Public Key**: `J7i4p29QD1cBygQmwJe7VFTM8r5BNn2rzHPcRhzQUUxZ`
- **Keypair File**: `deploy-keypair.json` (gitignored, DO NOT commit)
- **Seed Phrase**: stored separately, not in repo
- **Network**: Devnet

## Program

- **Program ID**: `J2hnoTyYweXqWQtF2zW7SQ2h7BTeKzBUsmKUq8T3HpQP`
- **Binary Size**: 272 KB
- **Name**: `compute_rewards`

## Deployment Cost Estimate (Devnet)

Solana charges rent for program storage. The cost is based on binary size:

- **Program account rent**: ~272 KB × 6.96 lamports/byte × 2 years = ~3.79 SOL
- **Buffer account** (temp, during deploy): same size, refunded after deploy
- **Transaction fees**: ~0.01 SOL
- **Total needed**: ~4 SOL (devnet, free via airdrop)

On mainnet with SOL at ~$150:
- ~4 SOL ≈ $600 one-time deployment cost
- Program can be upgraded without re-paying rent

## How to Deploy

```bash
# 1. Ensure you have devnet SOL
solana airdrop 2 J7i4p29QD1cBygQmwJe7VFTM8r5BNn2rzHPcRhzQUUxZ --url devnet
# (may need to use faucet.solana.com if rate limited)

# 2. Build
cd solana-program
anchor build

# 3. Deploy to devnet
anchor deploy --provider.cluster devnet --provider.wallet ./deploy-keypair.json

# 4. Verify
solana program show J2hnoTyYweXqWQtF2zW7SQ2h7BTeKzBUsmKUq8T3HpQP --url devnet
```

## Program Instructions

| Instruction | Who calls it | Purpose |
|-------------|-------------|---------|
| `initialize_pool` | Authority (you) | One-time setup of reward pool |
| `register_node` | Node owner | Register wallet for rewards |
| `distribute_rewards` | Orchestrator | Record earned rewards after pipeline work |
| `claim_rewards` | Node owner | Transfer pending $COMPUTE to wallet |
| `update_pool` | Authority | Adjust rates, pause/unpause |

## Accounts

- **RewardPool** (PDA: `["pool", authority]`): Global pool state, rates, totals
- **NodeAccount** (PDA: `["node", owner]`): Per-node earnings and pending rewards
- **Reward Vault**: Token account holding $COMPUTE for distribution
