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
  walletAuth,
  adminAuth,
  claimRateLimit,
} from "../middleware/walletAuth.js";

export const rewardsRouter = new Hono();

// Get pending rewards for a wallet (public read-only)
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

// Get reward history for a wallet (public read-only)
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

// Build claim transaction — requires wallet signature auth + rate limiting
rewardsRouter.post("/:wallet/claim", walletAuth, claimRateLimit, async (c) => {
  const wallet = c.req.param("wallet")!;

  try {
    // Get pending rewards from DB
    const pending = await rewards.getPendingRewards(wallet);
    if (pending.total <= 0) {
      return c.json({ error: "No pending rewards to claim" }, 400);
    }

    // Get pending event IDs and mark them as "claiming" atomically
    const eventIds = await rewards.markEventsAsClaiming(wallet);
    if (eventIds.length === 0) {
      return c.json({ error: "No pending rewards to claim" }, 400);
    }

    // Recalculate total from the events that were just locked
    const claimingTotal = await rewards.getClaimingTotal(wallet, eventIds);

    // Build partially-signed transaction via Anchor program
    const result = await buildClaimTransaction(wallet, claimingTotal);

    return c.json({
      transaction: result.transaction, // base64, partially signed by authority
      amount: result.amount,
      event_ids: eventIds, // client must send these back in /confirm
      mint: getMintAddress(),
      program: getProgramId(),
      message: result.message,
    });
  } catch (e: any) {
    console.error(`[claim] Failed for ${wallet}:`, e);
    // Revert claiming events back to pending on error
    try {
      await rewards.revertClaimingEvents(wallet);
    } catch (revertErr) {
      console.error(`[claim] Failed to revert claiming events:`, revertErr);
    }
    return c.json({ error: "Failed to build claim transaction" }, 500);
  }
});

// Confirm claim — requires wallet signature auth + rate limiting
rewardsRouter.post(
  "/:wallet/confirm",
  walletAuth,
  claimRateLimit,
  async (c) => {
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
      // Verify the Solana transaction on-chain before marking as claimed
      const txValid = await verifyClaimTransaction(signature, wallet);
      if (!txValid) {
        return c.json(
          { error: "Transaction not found or not valid" },
          400
        );
      }

      // Only mark the specific event IDs as claimed (must be in "claiming" status)
      const result = await rewards.markRewardsClaimed(
        wallet,
        event_ids
      );

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
  }
);

// Network-wide reward stats (public read-only)
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

// Get reward config (public read-only)
rewardsRouter.get("/config", (c) => {
  return c.json(rewards.getRewardConfig());
});

// Update reward config — admin only
rewardsRouter.patch("/config", adminAuth, async (c) => {
  try {
    const updates = await c.req.json();
    rewards.updateRewardConfig(updates);
    return c.json(rewards.getRewardConfig());
  } catch (e: any) {
    console.error("[rewards] PATCH /config error:", e);
    return c.json({ error: "Failed to update config" }, 500);
  }
});
