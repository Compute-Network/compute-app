import { Hono } from "hono";
import { handleWebhook } from "../services/stripe.js";

export const webhooksRouter = new Hono();

// Stripe webhook — no auth middleware (verified via Stripe signature)
webhooksRouter.post("/stripe", async (c) => {
  const signature = c.req.header("stripe-signature");
  if (!signature) {
    return c.json({ error: "Missing stripe-signature header" }, 400);
  }

  try {
    const rawBody = await c.req.text();
    const result = await handleWebhook(rawBody, signature);

    if (result.handled) {
      console.log(`[webhook] Handled Stripe event: ${result.event}`);
    }

    return c.json({ received: true });
  } catch (e: any) {
    console.error("[webhook] Stripe error:", e.message);
    return c.json({ error: e.message }, 400);
  }
});
