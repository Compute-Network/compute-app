import { Hono } from "hono";
import * as rewards from "../services/rewards.js";

export const rewardsRouter = new Hono();

// Get pending rewards for a wallet
rewardsRouter.get("/:wallet", async (c) => {
  const wallet = c.req.param("wallet");

  try {
    const pending = await rewards.getPendingRewards(wallet);
    return c.json(pending);
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

// Mark rewards as claimed (called after Solana tx confirmation)
rewardsRouter.post("/:wallet/claim", async (c) => {
  const wallet = c.req.param("wallet");
  const { event_ids } = await c.req.json<{ event_ids: string[] }>();

  if (!event_ids?.length) {
    return c.json({ error: "event_ids array is required" }, 400);
  }

  try {
    const result = await rewards.markRewardsClaimed(wallet, event_ids);
    return c.json(result);
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Network-wide reward stats
rewardsRouter.get("/", async (c) => {
  try {
    const stats = await rewards.getRewardStats();
    return c.json(stats);
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
