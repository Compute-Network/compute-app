import { Hono } from "hono";
import { cors } from "hono/cors";
import { logger } from "hono/logger";
import { serve } from "@hono/node-server";

import { nodesRouter } from "./routes/nodes.js";
import { pipelinesRouter } from "./routes/pipelines.js";
import { completionsRouter } from "./routes/completions.js";
import { apiKeyAuth } from "./middleware/auth.js";
import { markStaleNodesOffline } from "./services/nodes.js";

const app = new Hono();

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

// Periodic tasks
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
  │   POST /v1/chat/completions  (API key)   │
  │   GET  /v1/models            (API key)   │
  │   GET  /health                           │
  │                                          │
  │   Port: ${port}                              │
  └──────────────────────────────────────────┘
`);

serve({ fetch: app.fetch, port });
