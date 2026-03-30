import type { Context, Next } from "hono";
import { PublicKey } from "@solana/web3.js";
import nacl from "tweetnacl";

/**
 * Wallet signature authentication middleware.
 *
 * Requires headers:
 *   X-Wallet-Signature  — base58-encoded Ed25519 signature of the wallet address
 *   X-Wallet-Public-Key — base58-encoded public key of the signer
 *
 * Verifies the signature, then checks the public key matches the :wallet route param.
 * On success, sets c.set("verifiedWallet", walletAddress).
 */
export async function walletAuth(c: Context, next: Next) {
  const signatureB58 = c.req.header("X-Wallet-Signature");
  const publicKeyB58 = c.req.header("X-Wallet-Public-Key");

  if (!signatureB58 || !publicKeyB58) {
    return c.json(
      {
        error: {
          message:
            "Missing X-Wallet-Signature or X-Wallet-Public-Key headers",
          type: "authentication_error",
        },
      },
      401
    );
  }

  let pubkey: PublicKey;
  try {
    pubkey = new PublicKey(publicKeyB58);
  } catch {
    return c.json(
      {
        error: {
          message: "Invalid public key",
          type: "authentication_error",
        },
      },
      401
    );
  }

  // Verify the signature is a valid Ed25519 signature of the wallet address
  // Accepts base64-encoded signature
  let signatureBytes: Uint8Array;
  try {
    signatureBytes = Uint8Array.from(
      Buffer.from(signatureB58, "base64")
    );
  } catch {
    return c.json(
      {
        error: {
          message: "Invalid signature encoding",
          type: "authentication_error",
        },
      },
      401
    );
  }

  const message = new TextEncoder().encode(pubkey.toBase58());
  const valid = nacl.sign.detached.verify(
    message,
    signatureBytes,
    pubkey.toBytes()
  );

  if (!valid) {
    return c.json(
      {
        error: {
          message: "Invalid wallet signature",
          type: "authentication_error",
        },
      },
      401
    );
  }

  // Ensure the public key matches the :wallet route parameter
  const walletParam = c.req.param("wallet");
  if (walletParam && pubkey.toBase58() !== walletParam) {
    return c.json(
      {
        error: {
          message: "Public key does not match wallet parameter",
          type: "authorization_error",
        },
      },
      403
    );
  }

  c.set("verifiedWallet", pubkey.toBase58());
  await next();
}

/**
 * Admin authentication middleware.
 *
 * Verifies the caller is the authority wallet (the SOLANA_KEYPAIR's public key
 * or the ADMIN_WALLET env var). Uses the same signature scheme as walletAuth.
 */
export async function adminAuth(c: Context, next: Next) {
  const signatureB58 = c.req.header("X-Wallet-Signature");
  const publicKeyB58 = c.req.header("X-Wallet-Public-Key");

  if (!signatureB58 || !publicKeyB58) {
    return c.json(
      {
        error: {
          message:
            "Missing X-Wallet-Signature or X-Wallet-Public-Key headers",
          type: "authentication_error",
        },
      },
      401
    );
  }

  let pubkey: PublicKey;
  try {
    pubkey = new PublicKey(publicKeyB58);
  } catch {
    return c.json(
      {
        error: {
          message: "Invalid public key",
          type: "authentication_error",
        },
      },
      401
    );
  }

  // Verify signature (base64-encoded)
  let signatureBytes: Uint8Array;
  try {
    signatureBytes = Uint8Array.from(
      Buffer.from(signatureB58, "base64")
    );
  } catch {
    return c.json(
      {
        error: {
          message: "Invalid signature encoding",
          type: "authentication_error",
        },
      },
      401
    );
  }

  const message = new TextEncoder().encode(pubkey.toBase58());
  const valid = nacl.sign.detached.verify(
    message,
    signatureBytes,
    pubkey.toBytes()
  );

  if (!valid) {
    return c.json(
      {
        error: {
          message: "Invalid wallet signature",
          type: "authentication_error",
        },
      },
      401
    );
  }

  // Check if the caller is the admin
  const adminWallet = getAdminWallet();
  if (!adminWallet) {
    return c.json(
      {
        error: {
          message: "Admin wallet not configured",
          type: "server_error",
        },
      },
      500
    );
  }

  if (pubkey.toBase58() !== adminWallet) {
    return c.json(
      {
        error: {
          message: "Unauthorized: not an admin wallet",
          type: "authorization_error",
        },
      },
      403
    );
  }

  c.set("verifiedWallet", pubkey.toBase58());
  await next();
}

/**
 * Verify wallet ownership — returns a Response if unauthorized, null if OK.
 * Call at the start of any protected handler.
 */
export function verifyWalletRequest(c: Context): Response | null {
  const signatureB64 = c.req.header("X-Wallet-Signature");
  const publicKeyB58 = c.req.header("X-Wallet-Public-Key");

  if (!signatureB64 || !publicKeyB58) {
    return c.json({ error: { message: "Missing auth headers", type: "authentication_error" } }, 401);
  }

  let pubkey: PublicKey;
  try { pubkey = new PublicKey(publicKeyB58); } catch {
    return c.json({ error: { message: "Invalid public key", type: "authentication_error" } }, 401);
  }

  let signatureBytes: Uint8Array;
  try { signatureBytes = Uint8Array.from(Buffer.from(signatureB64, "base64")); } catch {
    return c.json({ error: { message: "Invalid signature", type: "authentication_error" } }, 401);
  }

  const message = new TextEncoder().encode(pubkey.toBase58());
  const valid = nacl.sign.detached.verify(message, signatureBytes, pubkey.toBytes());
  if (!valid) {
    return c.json({ error: { message: "Invalid wallet signature", type: "authentication_error" } }, 401);
  }

  const walletParam = c.req.param("wallet");
  if (walletParam && pubkey.toBase58() !== walletParam) {
    return c.json({ error: { message: "Public key does not match wallet", type: "authorization_error" } }, 403);
  }

  // Rate limit check
  const wallet = walletParam || pubkey.toBase58();
  const now = Date.now();
  let entry = claimRateLimits.get(wallet);
  if (!entry || now >= entry.resetAt) {
    entry = { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS };
    claimRateLimits.set(wallet, entry);
  }
  entry.count++;
  if (entry.count > MAX_CLAIMS_PER_HOUR) {
    return c.json({ error: { message: "Rate limit exceeded. Max 5 claims per hour.", type: "rate_limit_error" } }, 429);
  }

  return null;
}

/**
 * Synchronous admin auth check — returns a Response if unauthorized, null if OK.
 * Call at the start of any admin-only handler.
 */
export function verifyAdminRequest(c: Context): Response | null {
  const signatureB64 = c.req.header("X-Wallet-Signature");
  const publicKeyB58 = c.req.header("X-Wallet-Public-Key");

  if (!signatureB64 || !publicKeyB58) {
    return c.json({ error: { message: "Missing auth headers", type: "authentication_error" } }, 401);
  }

  let pubkey: PublicKey;
  try { pubkey = new PublicKey(publicKeyB58); } catch {
    return c.json({ error: { message: "Invalid public key", type: "authentication_error" } }, 401);
  }

  let signatureBytes: Uint8Array;
  try { signatureBytes = Uint8Array.from(Buffer.from(signatureB64, "base64")); } catch {
    return c.json({ error: { message: "Invalid signature", type: "authentication_error" } }, 401);
  }

  const message = new TextEncoder().encode(pubkey.toBase58());
  const valid = nacl.sign.detached.verify(message, signatureBytes, pubkey.toBytes());
  if (!valid) {
    return c.json({ error: { message: "Invalid wallet signature", type: "authentication_error" } }, 401);
  }

  const adminWallet = getAdminWallet();
  if (!adminWallet || pubkey.toBase58() !== adminWallet) {
    return c.json({ error: { message: "Unauthorized: not admin", type: "authorization_error" } }, 403);
  }

  return null; // Auth passed
}

/**
 * Returns the admin wallet address from ADMIN_WALLET env var,
 * or derives it from SOLANA_KEYPAIR if available.
 */
function getAdminWallet(): string | null {
  if (process.env.ADMIN_WALLET) {
    return process.env.ADMIN_WALLET;
  }

  // Derive from SOLANA_KEYPAIR
  const envKey = process.env.SOLANA_KEYPAIR;
  if (envKey) {
    try {
      const { Keypair } = require("@solana/web3.js");
      const kp = Keypair.fromSecretKey(Uint8Array.from(JSON.parse(envKey)));
      return kp.publicKey.toBase58();
    } catch {
      return null;
    }
  }

  return null;
}

// ── Rate limiting ────────────────────────────────────────────────────

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

const claimRateLimits = new Map<string, RateLimitEntry>();

const MAX_CLAIMS_PER_HOUR = 5;
const RATE_LIMIT_WINDOW_MS = 60 * 60 * 1000; // 1 hour

// Clean up expired entries every 10 minutes
setInterval(() => {
  const now = Date.now();
  for (const [key, entry] of claimRateLimits) {
    if (now >= entry.resetAt) {
      claimRateLimits.delete(key);
    }
  }
}, 10 * 60 * 1000);

/**
 * Rate limiting middleware for claim endpoints.
 * Max 5 claims per wallet per hour.
 */
export async function claimRateLimit(c: Context, next: Next) {
  const wallet = c.req.param("wallet");
  if (!wallet) {
    return c.json({ error: "Wallet parameter required" }, 400);
  }

  const now = Date.now();
  let entry = claimRateLimits.get(wallet);

  if (!entry || now >= entry.resetAt) {
    entry = { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS };
    claimRateLimits.set(wallet, entry);
  }

  entry.count++;

  if (entry.count > MAX_CLAIMS_PER_HOUR) {
    return c.json(
      {
        error: {
          message: "Rate limit exceeded. Max 5 claim requests per hour.",
          type: "rate_limit_error",
        },
      },
      429
    );
  }

  await next();
}
