import { Hono } from "hono";
import {
  getBalance,
  getUsageHistory,
  getUsageSummary,
  getTransactionHistory,
  calculateCredits,
  PRICING,
  getAccountById,
} from "../services/billing.js";
import { createCheckoutSession } from "../services/stripe.js";

export const billingRouter = new Hono();

function requireAccountId(c: any): string | null {
  const id = (c as any).get("accountId");
  if (!id) {
    c.json({ error: { message: "Unauthorized", type: "authentication_error" } }, 401);
    return null;
  }
  return id as string;
}

// Get credits balance
billingRouter.get("/balance", async (c) => {
  const accountId = requireAccountId(c);
  if (!accountId) return c.res;
  try {
    const balance = await getBalance(accountId);
    return c.json({
      credits_balance: balance.total,
      purchased_credits: balance.purchased,
      reward_credits: balance.from_rewards,
      pending_compute: balance.pending_compute,
    });
  } catch (e: any) {
    console.error("[billing] Balance error:", e);
    return c.json({ error: { message: "Failed to get balance", type: "server_error" } }, 500);
  }
});

// Get transaction history (paginated)
billingRouter.get("/transactions", async (c) => {
  const accountId = requireAccountId(c);
  if (!accountId) return c.res;
  const limit = parseInt(c.req.query("limit") ?? "50", 10);
  const offset = parseInt(c.req.query("offset") ?? "0", 10);

  try {
    const transactions = await getTransactionHistory(accountId, limit, offset);
    return c.json({ transactions, count: transactions.length });
  } catch (e: any) {
    console.error("[billing] Transactions error:", e);
    return c.json({ error: { message: "Failed to get transactions", type: "server_error" } }, 500);
  }
});

// Get daily usage analytics
billingRouter.get("/usage", async (c) => {
  const accountId = requireAccountId(c);
  if (!accountId) return c.res;
  const days = parseInt(c.req.query("days") ?? "30", 10);

  try {
    const usage = await getUsageHistory(accountId, days);
    return c.json({ usage, days });
  } catch (e: any) {
    console.error("[billing] Usage error:", e);
    return c.json({ error: { message: "Failed to get usage", type: "server_error" } }, 500);
  }
});

// Get usage summary (this month vs last month)
billingRouter.get("/usage/summary", async (c) => {
  const accountId = requireAccountId(c);
  if (!accountId) return c.res;
  try {
    const summary = await getUsageSummary(accountId);
    return c.json(summary);
  } catch (e: any) {
    console.error("[billing] Summary error:", e);
    return c.json({ error: { message: "Failed to get summary", type: "server_error" } }, 500);
  }
});

// Create Stripe Checkout session for top-up
billingRouter.post("/topup/stripe", async (c) => {
  const accountId = requireAccountId(c);
  if (!accountId) return c.res;

  let body: { amount: number; success_url?: string; cancel_url?: string };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid request body", type: "invalid_request_error" } }, 400);
  }

  const { amount, success_url, cancel_url } = body;
  if (!amount || typeof amount !== "number" || amount < PRICING.minTopupDollars || amount > 100_000) {
    return c.json({
      error: {
        message: `Minimum top-up is $${PRICING.minTopupDollars}`,
        type: "invalid_request_error",
      },
    }, 400);
  }

  try {
    const account = await getAccountById(accountId);
    const email = account?.email || "unknown@compute.network";

    const result = await createCheckoutSession(
      accountId,
      amount,
      email,
      success_url,
      cancel_url
    );

    const credits = calculateCredits(amount);

    return c.json({
      checkout_url: result.url,
      session_id: result.sessionId,
      amount_dollars: amount,
      credits_preview: credits,
    });
  } catch (e: any) {
    console.error("[billing] Stripe checkout error:", e);
    return c.json({ error: { message: "Failed to create checkout session", type: "server_error" } }, 500);
  }
});

// Create crypto deposit intent
billingRouter.post("/topup/crypto", async (c) => {
  const accountId = requireAccountId(c);
  if (!accountId) return c.res;

  let body: { token: string; amount_usd?: number };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid request body", type: "invalid_request_error" } }, 400);
  }

  const { token, amount_usd } = body;
  if (!token || !["USDC", "SOL", "COMPUTE"].includes(token)) {
    return c.json({
      error: { message: "Token must be USDC, SOL, or COMPUTE", type: "invalid_request_error" },
    }, 400);
  }

  if (!PRICING.treasuryWallet) {
    return c.json({
      error: { message: "Crypto deposits not yet configured", type: "server_error" },
    }, 503);
  }

  const credits = amount_usd ? calculateCredits(amount_usd, token) : null;

  return c.json({
    treasury_address: PRICING.treasuryWallet,
    token,
    network: "solana",
    cluster: "devnet",
    credits_preview: credits,
    note: token === "COMPUTE"
      ? `20% bonus credits when paying with $COMPUTE`
      : `Send ${token} to the treasury address. Credits will be applied within 1-2 minutes.`,
  });
});

// Get current pricing (public — no auth needed, but mounted under accountAuth)
// We'll add a separate public endpoint in index.ts
billingRouter.get("/pricing", (c) => {
  return c.json({
    credits_per_dollar: PRICING.creditsPerDollar,
    credits_per_token: PRICING.creditsPerToken,
    compute_token_bonus: PRICING.computeTokenBonus,
    volume_tiers: PRICING.volumeTiers,
    min_topup_dollars: PRICING.minTopupDollars,
  });
});
