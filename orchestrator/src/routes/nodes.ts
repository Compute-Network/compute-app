import { Hono } from "hono";
import { NodeRegistration, HeartbeatPayload } from "../types/node.js";
import * as nodes from "../services/nodes.js";

export const nodesRouter = new Hono();

// Register a node
nodesRouter.post("/register", async (c) => {
  const body = await c.req.json();
  const parsed = NodeRegistration.safeParse(body);

  if (!parsed.success) {
    return c.json({ error: "Invalid registration", details: parsed.error.issues }, 400);
  }

  try {
    const result = await nodes.registerNode(parsed.data);
    return c.json({ node_id: result.id, status: "registered" }, 201);
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Heartbeat from a node
nodesRouter.post("/:wallet/heartbeat", async (c) => {
  const wallet = c.req.param("wallet");
  const body = await c.req.json();
  const parsed = HeartbeatPayload.safeParse(body);

  if (!parsed.success) {
    return c.json({ error: "Invalid heartbeat", details: parsed.error.issues }, 400);
  }

  try {
    await nodes.heartbeat(wallet, parsed.data);
    return c.json({ status: "ok" });
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// List online nodes
nodesRouter.get("/", async (c) => {
  try {
    const online = c.req.query("status") !== "all";
    const nodeList = online
      ? await nodes.getOnlineNodes()
      : await nodes.getAvailableNodes();
    return c.json({ nodes: nodeList, count: nodeList.length });
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Network stats
nodesRouter.get("/stats", async (c) => {
  try {
    const stats = await nodes.getNetworkStats();
    return c.json(stats);
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});
