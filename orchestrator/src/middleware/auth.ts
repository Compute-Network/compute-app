import type { Context, Next } from "hono";
import { validateApiKey, checkRateLimit } from "../services/apikeys.js";
import { getBalance, PRICING } from "../services/billing.js";

/**
 * API key authentication middleware.
 * Validates Bearer token against Supabase api_keys table.
 * Checks credit balance (402 if zero and not legacy wallet).
 * Sets apiKey, apiKeyId, apiKeyWallet, accountId on context.
 */
export async function apiKeyAuth(c: Context, next: Next) {
  const authHeader = c.req.header("Authorization");

  if (!authHeader?.startsWith("Bearer ")) {
    return c.json(
      {
        error: {
          message: "Missing or invalid Authorization header. Use: Bearer <api-key>",
          type: "authentication_error",
        },
      },
      401
    );
  }

  const apiKey = authHeader.slice(7);
  if (!apiKey) {
    return c.json(
      {
        error: {
          message: "Empty API key",
          type: "authentication_error",
        },
      },
      401
    );
  }

  const record = await validateApiKey(apiKey);
  if (!record) {
    return c.json(
      {
        error: {
          message: "Invalid API key",
          type: "authentication_error",
        },
      },
      401
    );
  }

  // Rate limit check
  if (!checkRateLimit(record.id, record.rate_limit_per_min ?? 60)) {
    return c.json(
      {
        error: {
          message: `Rate limit exceeded. Max ${record.rate_limit_per_min} requests per minute.`,
          type: "rate_limit_error",
        },
      },
      429
    );
  }

  c.set("apiKey", apiKey);
  c.set("apiKeyId", record.id);
  c.set("apiKeyWallet", record.wallet_address);

  // Credit balance check
  if (record.account_id) {
    const isLegacyWallet = record.wallet_address && !PRICING.enforceCreditsForWalletUsers;

    if (!isLegacyWallet) {
      const balance = await getBalance(record.account_id);
      if (balance <= 0) {
        return c.json(
          {
            error: {
              message: "Insufficient credits. Please top up your account at https://computenetwork.sh/dashboard",
              type: "billing_error",
            },
          },
          402
        );
      }
    }

    (c as any).set("accountId", record.account_id);
  }

  await next();
}
