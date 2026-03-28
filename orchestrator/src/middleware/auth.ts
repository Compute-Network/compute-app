import type { Context, Next } from "hono";

/**
 * API key authentication middleware.
 * Checks for Bearer token in Authorization header.
 *
 * For now, accepts any non-empty key. In production, validate
 * against a stored API keys table in Supabase.
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

  // TODO: Validate API key against Supabase api_keys table
  // For now, accept any non-empty key during development
  c.set("apiKey", apiKey);

  await next();
}
