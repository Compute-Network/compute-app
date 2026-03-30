import { Hono } from "hono";
import * as rewards from "../services/rewards.js";
import {
  buildClaimTransaction,
  getMintAddress,
  getPoolAddress,
  getVaultBalance,
  getProgramId,
} from "../services/solana.js";

export const rewardsRouter = new Hono();

// Get pending rewards for a wallet
rewardsRouter.get("/:wallet", async (c) => {
  const wallet = c.req.param("wallet");

  try {
    const pending = await rewards.getPendingRewards(wallet);
    return c.json({
      ...pending,
      mint: getMintAddress(),
      program: getProgramId(),
    });
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

// Build claim transaction — returns partially-signed tx for user to co-sign
rewardsRouter.post("/:wallet/claim", async (c) => {
  const wallet = c.req.param("wallet");

  try {
    // Get pending rewards from DB
    const pending = await rewards.getPendingRewards(wallet);
    if (pending.total <= 0) {
      return c.json({ error: "No pending rewards to claim" }, 400);
    }

    // Build partially-signed transaction via Anchor program
    const result = await buildClaimTransaction(wallet, pending.total);

    return c.json({
      transaction: result.transaction, // base64, partially signed by authority
      amount: result.amount,
      mint: getMintAddress(),
      program: getProgramId(),
      message: result.message,
    });
  } catch (e: any) {
    console.error(`[claim] Failed for ${wallet}:`, e.message);
    return c.json({ error: e.message }, 500);
  }
});

// Confirm claim — called after user signs + submits tx, marks events as claimed in DB
rewardsRouter.post("/:wallet/confirm", async (c) => {
  const wallet = c.req.param("wallet");
  const { signature } = await c.req.json<{ signature: string }>();

  if (!signature) {
    return c.json({ error: "signature is required" }, 400);
  }

  try {
    // Get all pending event IDs
    const history = await rewards.getRewardHistory(wallet, 10000);
    const pendingEvents = history.filter((e) => e.status === "pending");
    const eventIds = pendingEvents.map((e) => e.id);

    if (eventIds.length === 0) {
      return c.json({ error: "No pending events to confirm" }, 400);
    }

    // Mark as claimed
    const result = await rewards.markRewardsClaimed(wallet, eventIds);

    return c.json({
      success: true,
      signature,
      events_claimed: eventIds.length,
      ...result,
    });
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Network-wide reward stats
rewardsRouter.get("/", async (c) => {
  try {
    const stats = await rewards.getRewardStats();
    const vault = await getVaultBalance();
    return c.json({
      ...stats,
      mint: getMintAddress(),
      program: getProgramId(),
      pool: getPoolAddress(),
      vault,
    });
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Get/update reward config
rewardsRouter.get("/config", (c) => {
  return c.json(rewards.getRewardConfig());
});

rewardsRouter.patch("/config", async (c) => {
  const updates = await c.req.json();
  rewards.updateRewardConfig(updates);
  return c.json(rewards.getRewardConfig());
});
