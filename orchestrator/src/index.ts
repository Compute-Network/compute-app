import "dotenv/config";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { logger } from "hono/logger";
import { createAdaptorServer } from "@hono/node-server";
import { createNodeWebSocket } from "@hono/node-ws";

import { nodesRouter } from "./routes/nodes.js";
import { pipelinesRouter } from "./routes/pipelines.js";
import { completionsRouter } from "./routes/completions.js";
import { rewardsRouter } from "./routes/rewards.js";
import { apiKeysRouter } from "./routes/apikeys.js";
import { billingRouter } from "./routes/billing.js";
import { authRouter } from "./routes/auth.js";
import { webhooksRouter } from "./routes/webhooks.js";
import { createWsRoute, setUpgradeWebSocket } from "./routes/ws.js";
import { apiKeyAuth } from "./middleware/auth.js";
import { accountAuth } from "./middleware/enterpriseAuth.js";
import { markStaleNodesOffline } from "./services/nodes.js";
import { initScheduler } from "./services/scheduler.js";
import { initSolana } from "./services/solana.js";

const app = new Hono();

// Create WebSocket helper
const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });
setUpgradeWebSocket(upgradeWebSocket);

// Global middleware
app.use("*", cors());
app.use("*", logger());

// Health check (no auth)
app.get("/health", (c) => {
  return c.json({ status: "ok", version: "0.1.0" });
});

// Node management (no auth — nodes use anon key)
app.route("/v1/nodes", nodesRouter);

// Pipeline management (internal, no auth for now)
app.route("/v1/pipelines", pipelinesRouter);

// Rewards
app.route("/v1/rewards", rewardsRouter);

// API key management (wallet-authenticated)
app.route("/v1/api-keys", apiKeysRouter);

// Stripe webhooks (no auth — verified via Stripe signature)
app.route("/v1/webhooks", webhooksRouter);

// Auth (public — signup, login)
app.route("/v1/auth", authRouter);

// Billing (requires account auth — JWT or wallet signature)
app.use("/v1/billing/*", accountAuth);
app.route("/v1/billing", billingRouter);

// Public pricing endpoint (no auth)
app.get("/v1/pricing", async (c) => {
  const { PRICING } = await import("./services/billing.js");
  const { getComputePrice } = await import("./services/pricefeed.js");
  const computePrice = await getComputePrice();
  return c.json({
    credits_per_dollar: PRICING.creditsPerDollar,
    credits_per_token: PRICING.creditsPerToken,
    compute_token_bonus: PRICING.computeTokenBonus,
    compute_price_usd: computePrice,
    volume_tiers: PRICING.volumeTiers,
    min_topup_dollars: PRICING.minTopupDollars,
  });
});

// WebSocket relay for nodes
app.route("/v1/ws", createWsRoute());

// OpenAI-compatible API (requires API key)
app.use("/v1/chat/*", apiKeyAuth);
app.use("/v1/models", apiKeyAuth);
app.route("/v1", completionsRouter);

// 404 fallback
app.notFound((c) => {
  return c.json({ error: "Not found" }, 404);
});

// Error handler
app.onError((err, c) => {
  console.error("Unhandled error:", err);
  return c.json({ error: "Internal server error" }, 500);
});

// Initialize Solana ($COMPUTE token on devnet)
initSolana().catch((e) => console.error("[solana] Init failed:", e.message));

// Initialize scheduler (load active pipelines from DB)
initScheduler().catch(console.error);

// Periodic tasks
import { checkPendingDeposits } from "./services/crypto-deposits.js";

const STALE_CHECK_INTERVAL = 60_000; // 1 minute
setInterval(async () => {
  try {
    const count = await markStaleNodesOffline();
    if (count > 0) {
      console.log(`Marked ${count} stale nodes offline`);
    }
  } catch (e) {
    console.error("Stale node check failed:", e);
  }
}, STALE_CHECK_INTERVAL);

// Check for crypto deposits every 30 seconds
setInterval(async () => {
  try {
    await checkPendingDeposits();
  } catch (e) {
    console.error("[crypto] Deposit check failed:", e);
  }
}, 30_000);

// Start $COMPUTE price feed (DexScreener)
import { startPriceFeed } from "./services/pricefeed.js";
startPriceFeed();

// Start server
const port = parseInt(process.env.PORT ?? "3000", 10);

console.log(`
  ┌──────────────────────────────────────────┐
  │   COMPUTE ORCHESTRATOR v0.1.0            │
  │                                          │
  │   Endpoints:                             │
  │   POST /v1/nodes/register                │
  │   POST /v1/nodes/:wallet/heartbeat       │
  │   GET  /v1/nodes/stats                   │
  │   POST /v1/pipelines/form                │
  │   GET  /v1/rewards/:wallet                │
  │   POST /v1/rewards/:wallet/claim         │
  │   POST /v1/api-keys/:wallet  (wallet)   │
  │   GET  /v1/api-keys/:wallet  (wallet)   │
  │   POST /v1/auth/signup       (public)   │
  │   POST /v1/auth/login        (public)   │
  │   GET  /v1/billing/balance   (account)  │
  │   POST /v1/billing/topup/*   (account)  │
  │   GET  /v1/pricing           (public)   │
  │   POST /v1/webhooks/stripe   (stripe)   │
  │   POST /v1/chat/completions  (API key)   │
  │   GET  /v1/models            (API key)   │
  │   WS   /v1/ws                (relay)     │
  │   GET  /health                           │
  │                                          │
  │   Port: ${port}                              │
  └──────────────────────────────────────────┘
`);

const server = createAdaptorServer({ fetch: app.fetch });
injectWebSocket(server);
server.listen(port);
