import type { Context, Next } from "hono";
import jwt from "jsonwebtoken";
import { PublicKey } from "@solana/web3.js";
import nacl from "tweetnacl";
import { supabase } from "../services/db.js";

const JWT_SECRET: string = (() => {
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    console.error("FATAL: JWT_SECRET environment variable is required");
    process.exit(1);
  }
  return secret;
})();

export interface JwtPayload {
  accountId: string;
  email?: string;
  iat: number;
  exp: number;
}

/**
 * Sign a JWT for an account.
 */
export function signJwt(accountId: string, email?: string): string {
  return jwt.sign(
    { accountId, email },
    JWT_SECRET,
    { expiresIn: "7d" }
  );
}

/**
 * Verify a JWT token.
 */
export function verifyJwt(token: string): JwtPayload | null {
  try {
    return jwt.verify(token, JWT_SECRET) as unknown as JwtPayload;
  } catch {
    return null;
  }
}

/**
 * Unified account auth middleware.
 * Accepts either:
 *   1. JWT in Authorization: Bearer <jwt> (enterprise users)
 *   2. Wallet signature headers X-Wallet-Signature + X-Wallet-Public-Key (wallet users)
 *
 * On success, sets (c as any).set("accountId", ...) and (c as any).set("accountType", ...).
 */
export async function accountAuth(c: Context, next: Next) {
  // Try JWT first
  const authHeader = c.req.header("Authorization");
  if (authHeader?.startsWith("Bearer ")) {
    const token = authHeader.slice(7);
    const payload = verifyJwt(token);
    if (payload?.accountId) {
      (c as any).set("accountId", payload.accountId);
      (c as any).set("accountType", "enterprise");
      return next();
    }
    // If JWT failed, don't fall through — it was an explicit attempt
    // Unless it looks like a wallet signature header is also present
    if (!c.req.header("X-Wallet-Signature")) {
      return c.json(
        { error: { message: "Invalid or expired token", type: "authentication_error" } },
        401
      );
    }
  }

  // Try wallet signature
  const signatureB64 = c.req.header("X-Wallet-Signature");
  const publicKeyB58 = c.req.header("X-Wallet-Public-Key");

  if (signatureB64 && publicKeyB58) {
    let pubkey: PublicKey;
    try {
      pubkey = new PublicKey(publicKeyB58);
    } catch {
      return c.json(
        { error: { message: "Invalid public key", type: "authentication_error" } },
        401
      );
    }

    let signatureBytes: Uint8Array;
    try {
      signatureBytes = Uint8Array.from(Buffer.from(signatureB64, "base64"));
    } catch {
      return c.json(
        { error: { message: "Invalid signature", type: "authentication_error" } },
        401
      );
    }

    const message = new TextEncoder().encode(pubkey.toBase58());
    const valid = nacl.sign.detached.verify(message, signatureBytes, pubkey.toBytes());
    if (!valid) {
      return c.json(
        { error: { message: "Invalid wallet signature", type: "authentication_error" } },
        401
      );
    }

    // Look up or create account by wallet
    const walletAddress = pubkey.toBase58();
    const { data: account } = await supabase
      .from("accounts")
      .select("id")
      .eq("wallet_address", walletAddress)
      .single();

    if (account) {
      (c as any).set("accountId", account.id);
      (c as any).set("accountType", "wallet");
      return next();
    }

    // No auto-creation — wallet must have an account already
    // (created via API key generation or explicit registration)
    return c.json(
      { error: { message: "No account found for this wallet. Generate an API key first.", type: "authentication_error" } },
      401
    );
  }

  return c.json(
    { error: { message: "Authentication required", type: "authentication_error" } },
    401
  );
}
