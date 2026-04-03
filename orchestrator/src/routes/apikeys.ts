import { Hono } from "hono";
import { verifyWalletRequest } from "../middleware/walletAuth.js";
import { verifyJwt } from "../middleware/enterpriseAuth.js";
import { supabase } from "../services/db.js";
import * as apikeys from "../services/apikeys.js";

export const apiKeysRouter = new Hono();

/**
 * Resolve auth from either JWT (enterprise) or wallet signature.
 * Returns { accountId, walletAddress } on success, or a 401 Response on failure.
 */
async function resolveAuth(c: any): Promise<{ accountId: string; walletAddress: string | null } | Response> {
  // Try JWT first (enterprise users)
  const authHeader = c.req.header("Authorization");
  if (authHeader?.startsWith("Bearer ")) {
    const token = authHeader.slice(7);
    const payload = verifyJwt(token);
    if (payload?.accountId) {
      // Look up account to get wallet_address if linked
      const { data: account } = await supabase
        .from("accounts")
        .select("id, wallet_address")
        .eq("id", payload.accountId)
        .single();
      return {
        accountId: payload.accountId,
        walletAddress: account?.wallet_address || null,
      };
    }
    // JWT was provided but invalid
    if (!c.req.header("X-Wallet-Signature")) {
      return c.json({ error: { message: "Invalid or expired token", type: "authentication_error" } }, 401);
    }
  }

  // Fall back to wallet signature auth
  const authError = verifyWalletRequest(c);
  if (authError) return authError;

  const wallet = c.req.param("wallet")!;
  return { accountId: "", walletAddress: wallet };
}

// Create a new API key (wallet or enterprise authenticated)
apiKeysRouter.post("/:wallet", async (c) => {
  const auth = await resolveAuth(c);
  if (auth instanceof Response) return auth;

  const walletParam = c.req.param("wallet")!;
  // For enterprise users, use their linked wallet or null
  const wallet = auth.accountId ? (auth.walletAddress || null) : walletParam;
  let name = "default";

  try {
    const body = await c.req.json();
    if (body.name) name = String(body.name).slice(0, 64);
  } catch {
    // No body is fine, use default name
  }

  try {
    const result = await apikeys.createApiKey(wallet, name, auth.accountId || undefined);
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
    console.error(`[apikeys] POST error:`, e);
    return c.json({ error: { message: "Failed to create API key", type: "server_error" } }, 500);
  }
});

// List API keys (wallet or enterprise authenticated)
apiKeysRouter.get("/:wallet", async (c) => {
  const auth = await resolveAuth(c);
  if (auth instanceof Response) return auth;

  try {
    if (auth.accountId) {
      // Enterprise: list by account_id
      const keys = await apikeys.listApiKeysByAccount(auth.accountId);
      return c.json({ keys });
    }
    const wallet = c.req.param("wallet")!;
    const keys = await apikeys.listApiKeys(wallet);
    return c.json({ keys });
  } catch (e: any) {
    console.error(`[apikeys] GET error:`, e);
    return c.json({ error: { message: "Failed to list API keys", type: "server_error" } }, 500);
  }
});

// Revoke an API key (wallet or enterprise authenticated)
apiKeysRouter.delete("/:wallet/:keyId", async (c) => {
  const auth = await resolveAuth(c);
  if (auth instanceof Response) return auth;

  const keyId = c.req.param("keyId")!;

  try {
    let revoked: boolean;
    if (auth.accountId) {
      revoked = await apikeys.revokeApiKeyByAccount(auth.accountId, keyId);
    } else {
      const wallet = c.req.param("wallet")!;
      revoked = await apikeys.revokeApiKey(wallet, keyId);
    }
    if (!revoked) {
      return c.json({ error: { message: "Key not found", type: "not_found" } }, 404);
    }
    return c.json({ success: true, message: "API key revoked" });
  } catch (e: any) {
    console.error(`[apikeys] DELETE error:`, e);
    return c.json({ error: { message: "Failed to revoke API key", type: "server_error" } }, 500);
  }
});
