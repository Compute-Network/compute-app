import { Hono } from "hono";
import * as rewards from "../services/rewards.js";
import {
  buildClaimTransaction,
  getMintAddress,
  getPoolAddress,
  getVaultBalance,
  getProgramId,
  verifyClaimTransaction,
} from "../services/solana.js";
import {
  verifyWalletRequest,
  verifyAdminRequest,
} from "../middleware/walletAuth.js";

export const rewardsRouter = new Hono();

// ── Public read-only endpoints ────────────────────────────────────

// Get pending rewards for a wallet
rewardsRouter.get("/:wallet", async (c) => {
  const wallet = c.req.param("wallet")!;

  try {
    const pending = await rewards.getPendingRewards(wallet);
    return c.json({
      ...pending,
      mint: getMintAddress(),
      program: getProgramId(),
    });
  } catch (e: any) {
    console.error(`[rewards] GET /${wallet} error:`, e);
    return c.json({ error: "Failed to fetch pending rewards" }, 500);
  }
});

// Get reward history for a wallet
rewardsRouter.get("/:wallet/history", async (c) => {
  const wallet = c.req.param("wallet")!;
  const limit = parseInt(c.req.query("limit") ?? "50", 10);

  try {
    const history = await rewards.getRewardHistory(wallet, limit);
    return c.json({ events: history, count: history.length });
  } catch (e: any) {
    console.error(`[rewards] GET /${wallet}/history error:`, e);
    return c.json({ error: "Failed to fetch reward history" }, 500);
  }
});

// Get reward config
rewardsRouter.get("/config", (c) => {
  return c.json(rewards.getRewardConfig());
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
    console.error("[rewards] GET / error:", e);
    return c.json({ error: "Failed to fetch reward stats" }, 500);
  }
});

// ── Protected endpoints (wallet signature + rate limiting) ────────

// Build claim transaction
rewardsRouter.post("/:wallet/claim", async (c) => {
  const authError = verifyWalletRequest(c);
  if (authError) return authError;

  const wallet = c.req.param("wallet")!;

  try {
    const pending = await rewards.getPendingRewards(wallet);
    if (pending.total <= 0) {
      return c.json({ error: "No pending rewards to claim" }, 400);
    }

    const eventIds = await rewards.markEventsAsClaiming(wallet);
    if (eventIds.length === 0) {
      return c.json({ error: "No pending rewards to claim" }, 400);
    }

    const claimingTotal = await rewards.getClaimingTotal(wallet, eventIds);
    const result = await buildClaimTransaction(wallet, claimingTotal);

    return c.json({
      transaction: result.transaction,
      amount: result.amount,
      event_ids: eventIds,
      mint: getMintAddress(),
      program: getProgramId(),
      message: result.message,
    });
  } catch (e: any) {
    console.error(`[claim] Failed for ${wallet}:`, e);
    try {
      await rewards.revertClaimingEvents(wallet);
    } catch (revertErr) {
      console.error(`[claim] Failed to revert claiming events:`, revertErr);
    }
    return c.json({ error: "Failed to build claim transaction" }, 500);
  }
});

// Confirm claim
rewardsRouter.post("/:wallet/confirm", async (c) => {
  const authError = verifyWalletRequest(c);
  if (authError) return authError;

  const wallet = c.req.param("wallet")!;

  let body: { signature: string; event_ids: string[] };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid request body" }, 400);
  }

  const { signature, event_ids } = body;

  if (!signature) {
    return c.json({ error: "signature is required" }, 400);
  }
  if (!event_ids || !Array.isArray(event_ids) || event_ids.length === 0) {
    return c.json({ error: "event_ids array is required" }, 400);
  }

  try {
    const txValid = await verifyClaimTransaction(signature, wallet);
    if (!txValid) {
      return c.json({ error: "Transaction not found or not valid" }, 400);
    }

    const result = await rewards.markRewardsClaimed(wallet, event_ids);

    return c.json({
      success: true,
      signature,
      events_claimed: event_ids.length,
      ...result,
    });
  } catch (e: any) {
    console.error(`[confirm] Failed for ${wallet}:`, e);
    return c.json({ error: "Failed to confirm claim" }, 500);
  }
});

// Update reward config — admin only
rewardsRouter.patch("/config", async (c) => {
  const authError = verifyAdminRequest(c);
  if (authError) return authError;

  try {
    const updates = await c.req.json();
    rewards.updateRewardConfig(updates);
    return c.json(rewards.getRewardConfig());
  } catch (e: any) {
    console.error("[rewards] PATCH /config error:", e);
    return c.json({ error: "Failed to update config" }, 500);
  }
});
