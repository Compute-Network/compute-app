import { Hono } from "hono";
import { verifyWalletRequest } from "../middleware/walletAuth.js";
import * as apikeys from "../services/apikeys.js";

export const apiKeysRouter = new Hono();

// Create a new API key (wallet-authenticated)
apiKeysRouter.post("/:wallet", async (c) => {
  const authError = verifyWalletRequest(c);
  if (authError) return authError;

  const wallet = c.req.param("wallet")!;
  let name = "default";

  try {
    const body = await c.req.json();
    if (body.name) name = String(body.name).slice(0, 64);
  } catch {
    // No body is fine, use default name
  }

  try {
    const result = await apikeys.createApiKey(wallet, name);
    return c.json({
      key: result.key,
      id: result.id,
      prefix: result.prefix,
      name,
      message: "Store this key securely. It will not be shown again.",
    });
  } catch (e: any) {
    if (e.message.includes("Maximum")) {
      return c.json({ error: { message: e.message, type: "limit_error" } }, 400);
    }
    console.error(`[apikeys] POST /${wallet} error:`, e);
    return c.json({ error: { message: "Failed to create API key", type: "server_error" } }, 500);
  }
});

// List API keys for a wallet (wallet-authenticated)
apiKeysRouter.get("/:wallet", async (c) => {
  const authError = verifyWalletRequest(c);
  if (authError) return authError;

  const wallet = c.req.param("wallet")!;

  try {
    const keys = await apikeys.listApiKeys(wallet);
    return c.json({ keys });
  } catch (e: any) {
    console.error(`[apikeys] GET /${wallet} error:`, e);
    return c.json({ error: { message: "Failed to list API keys", type: "server_error" } }, 500);
  }
});

// Revoke an API key (wallet-authenticated)
apiKeysRouter.delete("/:wallet/:keyId", async (c) => {
  const authError = verifyWalletRequest(c);
  if (authError) return authError;

  const wallet = c.req.param("wallet")!;
  const keyId = c.req.param("keyId")!;

  try {
    const revoked = await apikeys.revokeApiKey(wallet, keyId);
    if (!revoked) {
      return c.json({ error: { message: "Key not found", type: "not_found" } }, 404);
    }
    return c.json({ success: true, message: "API key revoked" });
  } catch (e: any) {
    console.error(`[apikeys] DELETE /${wallet}/${keyId} error:`, e);
    return c.json({ error: { message: "Failed to revoke API key", type: "server_error" } }, 500);
  }
});
