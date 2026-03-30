import { supabase } from "./db.js";
import type { Pipeline, PipelineStage } from "../types/pipeline.js";
import { distributeRewardsOnChain } from "./solana.js";

// Maximum reward per single inference request (in $COMPUTE)
const MAX_REWARD_PER_REQUEST = 100;

// Reward rate configuration — all server-controlled, easily adjustable.
const REWARD_CONFIG = {
  // Base reward per token per layer (in $COMPUTE)
  baseRatePerTokenLayer: 0.001,

  // Minimum reward per token for any participant (floor rate)
  floorRatePerToken: 0.0005,

  // Multipliers
  busyHourMultiplier: 1.0, // Set > 1.0 during peak hours
  uptimeMultiplier: 1.0, // 1.0 - 1.2 based on reliability

  // Anti-whale: diminishing returns per additional node from same wallet
  antiWhaleEnabled: false,
  antiWhaleFactor: 0.1, // Each additional node earns 1/(1 + factor * count) less
};

/**
 * Calculate and record rewards for a completed inference request.
 *
 * Called after tokens are generated through a pipeline. Each node
 * in the pipeline receives rewards proportional to their layer contribution.
 */
export async function recordRequestReward(
  pipeline: Pipeline,
  tokensGenerated: number,
  tokensPrompt: number
): Promise<void> {
  const totalTokens = tokensGenerated + tokensPrompt;
  if (totalTokens === 0) return;

  // Calculate total request reward
  const totalRequestReward =
    totalTokens *
    pipeline.total_layers *
    REWARD_CONFIG.baseRatePerTokenLayer *
    REWARD_CONFIG.busyHourMultiplier;

  // Distribute across stages (one event per stage per request)
  const events = pipeline.stages.map((stage) => {
      const layersServed = stage.end_layer - stage.start_layer + 1;
      const layerProportion = layersServed / pipeline.total_layers;

      // Proportional reward
      let reward = totalRequestReward * layerProportion;

      // Apply floor rate
      const floorReward = totalTokens * REWARD_CONFIG.floorRatePerToken;
      reward = Math.max(reward, floorReward);

      // Apply multipliers
      reward *= REWARD_CONFIG.uptimeMultiplier;

      // Apply per-request cap
      if (reward > MAX_REWARD_PER_REQUEST) {
        console.warn(
          `[rewards] Capping reward for node ${stage.node_id} from ${reward.toFixed(4)} to ${MAX_REWARD_PER_REQUEST} $COMPUTE`
        );
        reward = MAX_REWARD_PER_REQUEST;
      }

      return {
        node_id: stage.node_id,
        wallet_address: stage.wallet_address,
        pipeline_id: pipeline.id,
        model_id: pipeline.model_id,
        layers_served: layersServed,
        total_layers: pipeline.total_layers,
        tokens_generated: tokensGenerated,
        tokens_prompt: tokensPrompt,
        base_reward: totalRequestReward * layerProportion,
        multiplier:
          REWARD_CONFIG.uptimeMultiplier * REWARD_CONFIG.busyHourMultiplier,
        final_reward: reward,
        status: "pending",
      };
    });

  // Batch insert reward events
  const { error } = await supabase.from("reward_events").insert(events);

  if (error) {
    console.error("Failed to record rewards:", error.message);
    return;
  }

  // Also distribute on-chain (best-effort, doesn't block if node not registered)
  for (const event of events) {
    distributeRewardsOnChain(
      event.wallet_address,
      event.final_reward,
      event.layers_served,
      event.total_layers,
      event.tokens_generated
    ).catch((e) =>
      console.warn(`[solana] On-chain distribute failed: ${e.message}`)
    );
  }

  // Ensure wallet has an account (for credit balance lookups)
  const wallets = [...new Set(events.map((e) => e.wallet_address))];
  for (const wallet of wallets) {
    const { data } = await supabase
      .from("accounts")
      .select("id")
      .eq("wallet_address", wallet)
      .single();
    if (!data) {
      const { error } = await supabase
        .from("accounts")
        .insert({ account_type: "wallet", wallet_address: wallet })
        .select("id")
        .single();
      if (error) console.debug("[rewards] Account creation skipped:", error.message);
    }
  }
}

/**
 * Get pending (unclaimed) rewards for a wallet.
 */
export async function getPendingRewards(
  walletAddress: string
): Promise<{ total: number; count: number }> {
  const { data, error } = await supabase
    .from("reward_events")
    .select("final_reward")
    .eq("wallet_address", walletAddress)
    .eq("status", "pending");

  if (error) throw new Error(`Failed to get rewards: ${error.message}`);

  const total = (data ?? []).reduce((sum, r) => sum + r.final_reward, 0);
  return { total, count: data?.length ?? 0 };
}

/**
 * Get reward history for a wallet.
 */
export async function getRewardHistory(
  walletAddress: string,
  limit = 50
): Promise<any[]> {
  const { data, error } = await supabase
    .from("reward_events")
    .select("*")
    .eq("wallet_address", walletAddress)
    .order("created_at", { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to get history: ${error.message}`);
  return data ?? [];
}

/**
 * Mark pending events for a wallet as "claiming" to prevent double-claims.
 * Returns the event IDs that were marked.
 *
 * TODO: Add a cleanup job that reverts "claiming" events back to "pending"
 * if they aren't confirmed within 5 minutes. This prevents permanently
 * locked events if the user abandons the claim flow.
 */
export async function markEventsAsClaiming(
  walletAddress: string
): Promise<string[]> {
  const { data, error } = await supabase
    .from("reward_events")
    .update({ status: "claiming" })
    .eq("wallet_address", walletAddress)
    .eq("status", "pending")
    .select("id");

  if (error) throw new Error(`Failed to mark as claiming: ${error.message}`);
  return (data ?? []).map((e) => e.id);
}

/**
 * Get the total reward amount for specific claiming events.
 */
export async function getClaimingTotal(
  walletAddress: string,
  eventIds: string[]
): Promise<number> {
  const { data, error } = await supabase
    .from("reward_events")
    .select("final_reward")
    .eq("wallet_address", walletAddress)
    .eq("status", "claiming")
    .in("id", eventIds);

  if (error) throw new Error(`Failed to get claiming total: ${error.message}`);
  return (data ?? []).reduce((sum, r) => sum + r.final_reward, 0);
}

/**
 * Revert "claiming" events back to "pending" (e.g., on claim build failure).
 */
export async function revertClaimingEvents(
  walletAddress: string
): Promise<void> {
  const { error } = await supabase
    .from("reward_events")
    .update({ status: "pending" })
    .eq("wallet_address", walletAddress)
    .eq("status", "claiming");

  if (error)
    console.error(`Failed to revert claiming events: ${error.message}`);
}

/**
 * Mark rewards as claimed (called after Solana transaction confirmed).
 * Only marks events that are in "claiming" status with the given IDs.
 */
export async function markRewardsClaimed(
  walletAddress: string,
  eventIds: string[]
): Promise<{ totalClaimed: number }> {
  const { data, error } = await supabase
    .from("reward_events")
    .update({ status: "claimed" })
    .eq("wallet_address", walletAddress)
    .in("id", eventIds)
    .eq("status", "claiming")
    .select("final_reward");

  if (error) throw new Error(`Failed to mark claimed: ${error.message}`);

  const totalClaimed = (data ?? []).reduce(
    (sum, r) => sum + r.final_reward,
    0
  );

  // Deduct from node's pending_compute
  await supabase
    .from("nodes")
    .update({ pending_compute: 0 })
    .eq("wallet_address", walletAddress);

  return { totalClaimed };
}

/**
 * Get network-wide reward stats.
 */
export async function getRewardStats(): Promise<{
  totalDistributed: number;
  totalPending: number;
  totalClaimed: number;
}> {
  const { data, error } = await supabase
    .from("reward_events")
    .select("final_reward, status");

  if (error) throw new Error(`Failed to get stats: ${error.message}`);

  const events = data ?? [];
  const totalDistributed = events.reduce((s, e) => s + e.final_reward, 0);
  const totalPending = events
    .filter((e) => e.status === "pending")
    .reduce((s, e) => s + e.final_reward, 0);
  const totalClaimed = events
    .filter((e) => e.status === "claimed")
    .reduce((s, e) => s + e.final_reward, 0);

  return { totalDistributed, totalPending, totalClaimed };
}

/**
 * Update reward configuration at runtime.
 * This is the "lever" — all parameters adjustable without redeployment.
 */
export function updateRewardConfig(
  updates: Partial<typeof REWARD_CONFIG>
): void {
  Object.assign(REWARD_CONFIG, updates);
  console.log("Reward config updated:", REWARD_CONFIG);
}

/**
 * Get current reward configuration.
 */
export function getRewardConfig(): typeof REWARD_CONFIG {
  return { ...REWARD_CONFIG };
}
