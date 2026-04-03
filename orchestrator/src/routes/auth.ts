import { Hono } from "hono";
import { randomBytes } from "crypto";
import bcrypt from "bcryptjs";
import { getOrCreateAccount, getAccountByEmail, getAccountById, linkWallet } from "../services/billing.js";
import { signJwt } from "../middleware/enterpriseAuth.js";
import { accountAuth } from "../middleware/enterpriseAuth.js";
import { PublicKey } from "@solana/web3.js";
import nacl from "tweetnacl";

export const authRouter = new Hono();

// ── Device code flow for CLI auth ──────────────────────────────────
// 1. CLI calls POST /device/start → gets { device_code, auth_url }
// 2. CLI opens browser to auth_url, polls GET /device/poll/:code
// 3. User authenticates on web page
// 4. Web page calls POST /device/complete with device_code + auth
// 5. CLI poll returns { api_key }

interface PendingDevice {
  code: string;
  createdAt: number;
  apiKey: string | null;  // null = pending, string = completed
}

const pendingDevices = new Map<string, PendingDevice>();

// Cleanup expired device codes every 60s
setInterval(() => {
  const now = Date.now();
  for (const [code, device] of pendingDevices) {
    if (now - device.createdAt > 5 * 60 * 1000) pendingDevices.delete(code);
  }
}, 60_000);

// ── Rate limiting for auth endpoints ────────────────────────────────
const authRateLimits = new Map<string, { count: number; resetAt: number }>();
const AUTH_MAX_PER_WINDOW = 10; // 10 attempts per window
const AUTH_WINDOW_MS = 5 * 60 * 1000; // 5 minutes

function checkAuthRateLimit(key: string): boolean {
  const now = Date.now();
  let entry = authRateLimits.get(key);
  if (!entry || now >= entry.resetAt) {
    entry = { count: 0, resetAt: now + AUTH_WINDOW_MS };
    authRateLimits.set(key, entry);
  }
  entry.count++;
  return entry.count <= AUTH_MAX_PER_WINDOW;
}

// Cleanup every 10 minutes
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of authRateLimits) {
    if (now >= v.resetAt) authRateLimits.delete(k);
  }
}, 10 * 60 * 1000);

// Enterprise signup
authRouter.post("/signup", async (c) => {
  // Rate limit by IP
  const ip = c.req.header("x-forwarded-for") || "unknown";
  if (!checkAuthRateLimit(`signup:${ip}`)) {
    return c.json({ error: { message: "Too many signup attempts. Try again later.", type: "rate_limit_error" } }, 429);
  }

  let body: { email: string; password: string };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid request body", type: "invalid_request_error" } }, 400);
  }

  const { email, password } = body;
  if (!email || !password) {
    return c.json({ error: { message: "Email and password required", type: "invalid_request_error" } }, 400);
  }

  if (password.length < 8) {
    return c.json({ error: { message: "Password must be at least 8 characters", type: "invalid_request_error" } }, 400);
  }

  // Check if email already exists
  const existing = await getAccountByEmail(email);
  if (existing) {
    return c.json({ error: { message: "Email already registered", type: "conflict_error" } }, 409);
  }

  try {
    const passwordHash = await bcrypt.hash(password, 12);
    const account = await getOrCreateAccount({ email, passwordHash });
    const token = signJwt(account.id, email);

    return c.json({
      token,
      account: {
        id: account.id,
        email: account.email,
        account_type: account.account_type,
        credits_balance: account.credits_balance,
      },
    });
  } catch (e: any) {
    console.error("[auth] Signup error:", e);
    return c.json({ error: { message: "Signup failed", type: "server_error" } }, 500);
  }
});

// Enterprise login
authRouter.post("/login", async (c) => {
  const ip = c.req.header("x-forwarded-for") || "unknown";
  if (!checkAuthRateLimit(`login:${ip}`)) {
    return c.json({ error: { message: "Too many login attempts. Try again later.", type: "rate_limit_error" } }, 429);
  }

  let body: { email: string; password: string };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid request body", type: "invalid_request_error" } }, 400);
  }

  const { email, password } = body;
  if (!email || !password) {
    return c.json({ error: { message: "Email and password required", type: "invalid_request_error" } }, 400);
  }

  // Also rate limit per email to prevent targeted brute force
  if (!checkAuthRateLimit(`login:email:${email.toLowerCase()}`)) {
    return c.json({ error: { message: "Too many login attempts for this account. Try again later.", type: "rate_limit_error" } }, 429);
  }

  const account = await getAccountByEmail(email);
  if (!account || !account.password_hash) {
    return c.json({ error: { message: "Invalid email or password", type: "authentication_error" } }, 401);
  }

  const valid = await bcrypt.compare(password, account.password_hash);
  if (!valid) {
    return c.json({ error: { message: "Invalid email or password", type: "authentication_error" } }, 401);
  }

  const token = signJwt(account.id, email);

  return c.json({
    token,
    account: {
      id: account.id,
      email: account.email,
      account_type: account.account_type,
      credits_balance: account.credits_balance,
      wallet_address: account.wallet_address,
    },
  });
});

// Get current account (requires auth)
authRouter.get("/account", accountAuth, async (c) => {
  const accountId = (c as any).get("accountId") as string;
  const account = await getAccountById(accountId);
  if (!account) {
    return c.json({ error: { message: "Account not found", type: "not_found" } }, 404);
  }

  return c.json({
    id: account.id,
    account_type: account.account_type,
    email: account.email,
    wallet_address: account.wallet_address,
    credits_balance: account.credits_balance,
    created_at: account.created_at,
  });
});

// ── Device code endpoints ──────────────────────────────────────────

// CLI calls this to start the flow
authRouter.post("/device/start", async (c) => {
  const code = randomBytes(16).toString("hex");
  pendingDevices.set(code, { code, createdAt: Date.now(), apiKey: null });
  return c.json({
    device_code: code,
    auth_url: `https://computenetwork.sh/auth/code?device_code=${code}`,
    expires_in: 300,
    poll_interval: 2,
  });
});

// CLI polls this until the user completes auth
authRouter.get("/device/poll/:code", async (c) => {
  const code = c.req.param("code")!;
  const device = pendingDevices.get(code);

  if (!device) {
    return c.json({ error: { message: "Device code not found or expired", type: "not_found" } }, 404);
  }

  if (Date.now() - device.createdAt > 5 * 60 * 1000) {
    pendingDevices.delete(code);
    return c.json({ error: { message: "Device code expired", type: "expired" } }, 410);
  }

  if (!device.apiKey) {
    return c.json({ status: "pending" }, 202);
  }

  // Auth completed — return key and clean up
  const apiKey = device.apiKey;
  pendingDevices.delete(code);
  return c.json({ status: "complete", api_key: apiKey });
});

// Web page calls this after user authenticates
authRouter.post("/device/complete", accountAuth, async (c) => {
  const accountId = (c as any).get("accountId") as string;

  let body: { device_code: string };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid request body", type: "invalid_request_error" } }, 400);
  }

  const { device_code } = body;
  if (!device_code) {
    return c.json({ error: { message: "device_code required", type: "invalid_request_error" } }, 400);
  }

  const device = pendingDevices.get(device_code);
  if (!device) {
    return c.json({ error: { message: "Device code not found or expired", type: "not_found" } }, 404);
  }

  if (device.apiKey) {
    return c.json({ error: { message: "Device code already completed", type: "conflict_error" } }, 409);
  }

  try {
    // Create an API key for this account
    const { createApiKey } = await import("../services/apikeys.js");
    const result = await createApiKey(null, "compute-code", accountId);
    device.apiKey = result.key;
    return c.json({ success: true, message: "API key created for CLI" });
  } catch (e: any) {
    console.error("[auth] Device complete error:", e);
    return c.json({ error: { message: "Failed to create API key", type: "server_error" } }, 500);
  }
});

// Link a Solana wallet to an enterprise account
authRouter.post("/link-wallet", accountAuth, async (c) => {
  const accountId = (c as any).get("accountId") as string;

  let body: { wallet_address: string; signature: string };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: { message: "Invalid request body", type: "invalid_request_error" } }, 400);
  }

  const { wallet_address, signature } = body;
  if (!wallet_address || !signature) {
    return c.json({ error: { message: "wallet_address and signature required", type: "invalid_request_error" } }, 400);
  }

  // Verify the wallet signature proves ownership
  try {
    const pubkey = new PublicKey(wallet_address);
    const sigBytes = Uint8Array.from(Buffer.from(signature, "base64"));
    const message = new TextEncoder().encode(pubkey.toBase58());
    const valid = nacl.sign.detached.verify(message, sigBytes, pubkey.toBytes());
    if (!valid) {
      return c.json({ error: { message: "Invalid wallet signature", type: "authentication_error" } }, 401);
    }
  } catch {
    return c.json({ error: { message: "Invalid wallet address or signature", type: "invalid_request_error" } }, 400);
  }

  try {
    await linkWallet(accountId, wallet_address);
    return c.json({ success: true, wallet_address });
  } catch (e: any) {
    if (e.message.includes("unique") || e.message.includes("duplicate")) {
      return c.json({ error: { message: "Wallet already linked to another account", type: "conflict_error" } }, 409);
    }
    console.error("[auth] Link wallet error:", e);
    return c.json({ error: { message: "Failed to link wallet", type: "server_error" } }, 500);
  }
});
