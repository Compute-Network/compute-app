import { supabase } from "./db.js";

// ── Pricing configuration ──────────────────────────────────────────

export const PRICING = {
  // 1 USD = 1,000,000 credits (1 credit ≈ 1 inference token)
  creditsPerDollar: 1_000_000,

  // Cost per inference token in credits
  creditsPerToken: 1,

  // $COMPUTE token bonus: 20% more credits
  computeTokenBonus: 0.20,

  // Volume discount tiers
  volumeTiers: [
    { minDollars: 0, discount: 0 },
    { minDollars: 100, discount: 0.05 },
    { minDollars: 500, discount: 0.10 },
    { minDollars: 2000, discount: 0.15 },
    { minDollars: 10000, discount: 0.20 },
  ],

  minTopupDollars: 5,

  // Treasury wallet for crypto deposits
  treasuryWallet: process.env.TREASURY_WALLET || "",

  // Feature flag: enforce credits for wallet users (false = legacy unlimited)
  enforceCreditsForWalletUsers: process.env.ENFORCE_CREDITS_WALLET_USERS === "true",
};

// ── Account management ─────────────────────────────────────────────

export interface Account {
  id: string;
  account_type: "wallet" | "enterprise";
  wallet_address: string | null;
  email: string | null;
  stripe_customer_id: string | null;
  credits_balance: number;
  created_at: string;
}

export async function getOrCreateAccount(
  opts: { walletAddress?: string; email?: string; passwordHash?: string }
): Promise<Account> {
  // Try to find existing account
  if (opts.walletAddress) {
    const { data } = await supabase
      .from("accounts")
      .select("*")
      .eq("wallet_address", opts.walletAddress)
      .single();
    if (data) return data as Account;
  }

  if (opts.email) {
    const { data } = await supabase
      .from("accounts")
      .select("*")
      .eq("email", opts.email)
      .single();
    if (data) return data as Account;
  }

  // Create new account
  const accountType = opts.email ? "enterprise" : "wallet";
  const { data, error } = await supabase
    .from("accounts")
    .insert({
      account_type: accountType,
      wallet_address: opts.walletAddress || null,
      email: opts.email || null,
      password_hash: opts.passwordHash || null,
    })
    .select("*")
    .single();

  if (error) throw new Error(`Failed to create account: ${error.message}`);
  return data as Account;
}

export async function getAccountById(id: string): Promise<Account | null> {
  const { data } = await supabase
    .from("accounts")
    .select("*")
    .eq("id", id)
    .single();
  return data as Account | null;
}

export async function getAccountByEmail(email: string): Promise<(Account & { password_hash: string }) | null> {
  const { data } = await supabase
    .from("accounts")
    .select("*")
    .eq("email", email)
    .single();
  return data as any;
}

export async function linkWallet(accountId: string, walletAddress: string): Promise<void> {
  const { error } = await supabase
    .from("accounts")
    .update({ wallet_address: walletAddress, updated_at: new Date().toISOString() })
    .eq("id", accountId);
  if (error) throw new Error(`Failed to link wallet: ${error.message}`);
}

// ── Credits ────────────────────────────────────────────────────────

export async function getBalance(accountId: string): Promise<number> {
  const { data } = await supabase
    .from("accounts")
    .select("credits_balance")
    .eq("id", accountId)
    .single();
  return data?.credits_balance ?? 0;
}

export async function deductCredits(
  accountId: string,
  tokensUsed: number,
  apiKeyId: string,
  modelId: string
): Promise<number> {
  const creditsToDeduct = tokensUsed * PRICING.creditsPerToken;
  if (creditsToDeduct <= 0) return 0;

  const { data, error } = await supabase.rpc("deduct_credits", {
    p_account_id: accountId,
    p_amount: creditsToDeduct,
    p_api_key_id: apiKeyId,
    p_tokens_used: tokensUsed,
    p_model_id: modelId,
  });

  if (error) throw new Error(`Deduction failed: ${error.message}`);
  if (data === -1) throw new Error("Insufficient credits");
  return data as number;
}

export async function topUpCredits(
  accountId: string,
  amount: number,
  type: string,
  description?: string,
  stripeSessionId?: string,
  cryptoTx?: string,
  cryptoToken?: string
): Promise<number> {
  const { data, error } = await supabase.rpc("topup_credits", {
    p_account_id: accountId,
    p_amount: amount,
    p_type: type,
    p_description: description || null,
    p_stripe_session_id: stripeSessionId || null,
    p_crypto_tx: cryptoTx || null,
    p_crypto_token: cryptoToken || null,
  });

  if (error) throw new Error(`Top-up failed: ${error.message}`);
  return data as number;
}

// ── Pricing calculations ───────────────────────────────────────────

export function calculateCredits(amountDollars: number, token?: string): {
  baseCredits: number;
  bonusCredits: number;
  totalCredits: number;
  discount: number;
} {
  // Find applicable volume discount
  let discount = 0;
  for (const tier of [...PRICING.volumeTiers].reverse()) {
    if (amountDollars >= tier.minDollars) {
      discount = tier.discount;
      break;
    }
  }

  const baseCredits = Math.floor(amountDollars * PRICING.creditsPerDollar * (1 + discount));

  // $COMPUTE token bonus
  let bonusCredits = 0;
  if (token === "COMPUTE") {
    bonusCredits = Math.floor(baseCredits * PRICING.computeTokenBonus);
  }

  return {
    baseCredits,
    bonusCredits,
    totalCredits: baseCredits + bonusCredits,
    discount,
  };
}

// ── Usage & transaction history ────────────────────────────────────

export async function getUsageHistory(
  accountId: string,
  days: number = 30
): Promise<{ date: string; requests: number; tokens_used: number; credits_used: number }[]> {
  const since = new Date();
  since.setDate(since.getDate() - days);

  const { data, error } = await supabase
    .from("usage_daily")
    .select("date, requests, tokens_used, credits_used")
    .eq("account_id", accountId)
    .gte("date", since.toISOString().split("T")[0])
    .order("date", { ascending: true });

  if (error) throw new Error(`Failed to get usage: ${error.message}`);
  return data ?? [];
}

export async function getUsageSummary(accountId: string): Promise<{
  this_month: { requests: number; tokens: number; credits: number };
  last_month: { requests: number; tokens: number; credits: number };
}> {
  const now = new Date();
  const thisMonthStart = new Date(now.getFullYear(), now.getMonth(), 1).toISOString().split("T")[0];
  const lastMonthStart = new Date(now.getFullYear(), now.getMonth() - 1, 1).toISOString().split("T")[0];

  const { data } = await supabase
    .from("usage_daily")
    .select("date, requests, tokens_used, credits_used")
    .eq("account_id", accountId)
    .gte("date", lastMonthStart);

  const rows = data ?? [];
  const thisMonth = rows.filter((r) => r.date >= thisMonthStart);
  const lastMonth = rows.filter((r) => r.date >= lastMonthStart && r.date < thisMonthStart);

  const sum = (arr: typeof rows) => ({
    requests: arr.reduce((s, r) => s + r.requests, 0),
    tokens: arr.reduce((s, r) => s + r.tokens_used, 0),
    credits: arr.reduce((s, r) => s + r.credits_used, 0),
  });

  return { this_month: sum(thisMonth), last_month: sum(lastMonth) };
}

export async function getTransactionHistory(
  accountId: string,
  limit: number = 50,
  offset: number = 0
): Promise<any[]> {
  const { data, error } = await supabase
    .from("credit_transactions")
    .select("id, amount, balance_after, type, crypto_token, tokens_used, model_id, description, created_at")
    .eq("account_id", accountId)
    .order("created_at", { ascending: false })
    .range(offset, offset + limit - 1);

  if (error) throw new Error(`Failed to get transactions: ${error.message}`);
  return data ?? [];
}
