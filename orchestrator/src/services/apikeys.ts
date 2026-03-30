import { supabase } from "./db.js";
import { createHash, randomBytes } from "crypto";

const KEY_PREFIX = "cpu_";

/**
 * Generate a new API key.
 * Accepts walletAddress and/or accountId. Auto-resolves/creates account if needed.
 * Returns the raw key (shown once) and the stored record.
 */
export async function createApiKey(
  walletAddress: string | null,
  name: string = "default",
  accountId?: string
): Promise<{ key: string; id: string; prefix: string }> {
  // Resolve account_id if not provided
  if (!accountId && walletAddress) {
    const { data: account } = await supabase
      .from("accounts")
      .select("id")
      .eq("wallet_address", walletAddress)
      .single();

    if (account) {
      accountId = account.id;
    } else {
      // Auto-create wallet account
      const { data: newAccount } = await supabase
        .from("accounts")
        .insert({ account_type: "wallet", wallet_address: walletAddress })
        .select("id")
        .single();
      accountId = newAccount?.id;
    }
  }

  // Check existing key count (max 5 per account)
  const filterCol = accountId ? "account_id" : "wallet_address";
  const filterVal = accountId || walletAddress;
  const { count } = await supabase
    .from("api_keys")
    .select("id", { count: "exact", head: true })
    .eq(filterCol, filterVal)
    .eq("active", true);

  if ((count ?? 0) >= 5) {
    throw new Error("Maximum 5 active API keys per account");
  }

  // Generate key: cp_<32 random hex chars>
  const raw = randomBytes(24).toString("hex");
  const key = `${KEY_PREFIX}${raw}`;
  const prefix = key.slice(0, 8);
  const keyHash = hashKey(key);

  const { data, error } = await supabase
    .from("api_keys")
    .insert({
      wallet_address: walletAddress,
      account_id: accountId || null,
      key_hash: keyHash,
      name,
      prefix,
    })
    .select("id")
    .single();

  if (error) throw new Error(`Failed to create API key: ${error.message}`);

  return { key, id: data.id, prefix };
}

/**
 * Validate an API key. Returns the key record if valid, null otherwise.
 */
export async function validateApiKey(key: string): Promise<ApiKeyRecord | null> {
  if (!key.startsWith(KEY_PREFIX) && !key.startsWith("cp_")) return null;

  const keyHash = hashKey(key);

  const { data, error } = await supabase
    .from("api_keys")
    .select("id, wallet_address, account_id, name, prefix, rate_limit_per_min, monthly_token_limit, tokens_total, active")
    .eq("key_hash", keyHash)
    .single();

  if (error || !data || !data.active) return null;

  // Update last_used_at (fire-and-forget)
  supabase
    .from("api_keys")
    .update({ last_used_at: new Date().toISOString() })
    .eq("id", data.id)
    .then(() => {});

  return data as ApiKeyRecord;
}

/**
 * Record usage after a request completes.
 */
export async function recordUsage(
  keyId: string,
  tokens: number
): Promise<void> {
  const { error } = await supabase.rpc("increment_api_key_usage", {
    key_id: keyId,
    token_count: tokens,
  });

  // Fallback: manual increment if RPC fails
  if (error) {
    const { data } = await supabase
      .from("api_keys")
      .select("requests_total, tokens_total")
      .eq("id", keyId)
      .single();

    if (data) {
      await supabase
        .from("api_keys")
        .update({
          requests_total: (data.requests_total ?? 0) + 1,
          tokens_total: (data.tokens_total ?? 0) + tokens,
        })
        .eq("id", keyId);
    }
  }
}

/**
 * List API keys for a wallet (never returns the actual key).
 */
export async function listApiKeys(walletAddress: string): Promise<ApiKeyInfo[]> {
  const { data, error } = await supabase
    .from("api_keys")
    .select("id, name, prefix, requests_total, tokens_total, rate_limit_per_min, active, created_at, last_used_at")
    .eq("wallet_address", walletAddress)
    .order("created_at", { ascending: false });

  if (error) throw new Error(`Failed to list keys: ${error.message}`);
  return (data ?? []) as ApiKeyInfo[];
}

/**
 * Revoke an API key.
 */
export async function revokeApiKey(
  walletAddress: string,
  keyId: string
): Promise<boolean> {
  const { data, error } = await supabase
    .from("api_keys")
    .update({ active: false })
    .eq("id", keyId)
    .eq("wallet_address", walletAddress)
    .select("id")
    .single();

  if (error || !data) return false;
  return true;
}

function hashKey(key: string): string {
  return createHash("sha256").update(key).digest("hex");
}

// ── Rate limiting (in-memory, per API key) ──────────────────────────

interface RateWindow {
  count: number;
  resetAt: number;
}

const rateLimits = new Map<string, RateWindow>();

/**
 * Check rate limit for an API key. Returns true if allowed.
 */
export function checkRateLimit(keyId: string, limitPerMin: number): boolean {
  const now = Date.now();
  let window = rateLimits.get(keyId);

  if (!window || now >= window.resetAt) {
    window = { count: 0, resetAt: now + 60_000 };
    rateLimits.set(keyId, window);
  }

  window.count++;
  return window.count <= limitPerMin;
}

// Clean up expired rate limit entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  for (const [key, window] of rateLimits) {
    if (now >= window.resetAt) rateLimits.delete(key);
  }
}, 5 * 60_000);

// ── Types ───────────────────────────────────────────────────────────

export interface ApiKeyRecord {
  id: string;
  account_id: string | null;
  wallet_address: string | null;
  name: string;
  prefix: string;
  rate_limit_per_min: number;
  monthly_token_limit: number | null;
  tokens_total: number;
  active: boolean;
}

export interface ApiKeyInfo {
  id: string;
  name: string;
  prefix: string;
  requests_total: number;
  tokens_total: number;
  rate_limit_per_min: number;
  active: boolean;
  created_at: string;
  last_used_at: string | null;
}
