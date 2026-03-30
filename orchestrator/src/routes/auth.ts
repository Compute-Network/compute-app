import { Hono } from "hono";
import bcrypt from "bcryptjs";
import { getOrCreateAccount, getAccountByEmail, getAccountById, linkWallet } from "../services/billing.js";
import { signJwt } from "../middleware/enterpriseAuth.js";
import { accountAuth } from "../middleware/enterpriseAuth.js";
import { PublicKey } from "@solana/web3.js";
import nacl from "tweetnacl";

export const authRouter = new Hono();

// Enterprise signup
authRouter.post("/signup", async (c) => {
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
