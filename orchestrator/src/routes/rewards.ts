import { Hono } from "hono";
import * as rewards from "../services/rewards.js";
import { transferReward, getMintAddress, getTreasuryBalance } from "../services/solana.js";

export const rewardsRouter = new Hono();

// Get pending rewards for a wallet
rewardsRouter.get("/:wallet", async (c) => {
  const wallet = c.req.param("wallet");

  try {
    const pending = await rewards.getPendingRewards(wallet);
    return c.json({ ...pending, mint: getMintAddress() });
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Get reward history for a wallet
rewardsRouter.get("/:wallet/history", async (c) => {
  const wallet = c.req.param("wallet");
  const limit = parseInt(c.req.query("limit") ?? "50", 10);

  try {
    const history = await rewards.getRewardHistory(wallet, limit);
    return c.json({ events: history, count: history.length });
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Claim rewards — transfers $COMPUTE on Solana devnet, marks events as claimed in DB
rewardsRouter.post("/:wallet/claim", async (c) => {
  const wallet = c.req.param("wallet");

  try {
    // Get all pending events for this wallet
    const pending = await rewards.getPendingRewards(wallet);
    if (pending.total <= 0) {
      return c.json({ error: "No pending rewards to claim" }, 400);
    }

    // Get event IDs to mark as claimed
    const history = await rewards.getRewardHistory(wallet, 10000);
    const pendingEvents = history.filter((e) => e.status === "pending");
    const eventIds = pendingEvents.map((e) => e.id);

    if (eventIds.length === 0) {
      return c.json({ error: "No pending reward events" }, 400);
    }

    // Transfer tokens on Solana
    const signature = await transferReward(wallet, pending.total);

    // Mark events as claimed in DB
    const result = await rewards.markRewardsClaimed(wallet, eventIds);

    return c.json({
      success: true,
      amount: pending.total,
      signature,
      mint: getMintAddress(),
      events_claimed: eventIds.length,
      ...result,
    });
  } catch (e: any) {
    console.error(`[claim] Failed for ${wallet}:`, e.message);
    return c.json({ error: e.message }, 500);
  }
});

// Network-wide reward stats
rewardsRouter.get("/", async (c) => {
  try {
    const stats = await rewards.getRewardStats();
    const treasury = await getTreasuryBalance();
    return c.json({ ...stats, mint: getMintAddress(), treasury });
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Get/update reward config (admin only — no auth for now)
rewardsRouter.get("/config", (c) => {
  return c.json(rewards.getRewardConfig());
});

rewardsRouter.patch("/config", async (c) => {
  const updates = await c.req.json();
  rewards.updateRewardConfig(updates);
  return c.json(rewards.getRewardConfig());
});
