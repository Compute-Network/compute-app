import Stripe from "stripe";
import { supabase } from "./db.js";
import { topUpCredits, calculateCredits } from "./billing.js";

let stripe: Stripe | null = null;

function getStripe(): Stripe {
  if (!stripe) {
    const key = process.env.STRIPE_SECRET_KEY;
    if (!key) throw new Error("STRIPE_SECRET_KEY not configured");
    stripe = new Stripe(key, { apiVersion: "2025-03-31.basil" as any });
  }
  return stripe;
}

/**
 * Create or retrieve a Stripe Customer for an account.
 */
export async function getOrCreateCustomer(
  accountId: string,
  email: string
): Promise<string> {
  // Check if account already has a Stripe customer
  const { data: account } = await supabase
    .from("accounts")
    .select("stripe_customer_id")
    .eq("id", accountId)
    .single();

  if (account?.stripe_customer_id) return account.stripe_customer_id;

  // Create new customer
  const customer = await getStripe().customers.create({
    email,
    metadata: { account_id: accountId },
  });

  // Store on account
  await supabase
    .from("accounts")
    .update({ stripe_customer_id: customer.id })
    .eq("id", accountId);

  return customer.id;
}

/**
 * Create a Stripe Checkout session for a one-time credit top-up.
 */
export async function createCheckoutSession(
  accountId: string,
  amountDollars: number,
  email: string,
  successUrl?: string,
  cancelUrl?: string
): Promise<{ url: string; sessionId: string }> {
  const s = getStripe();
  const customerId = await getOrCreateCustomer(accountId, email);

  const { totalCredits, discount } = calculateCredits(amountDollars);
  const amountCents = Math.round(amountDollars * 100);

  const session = await s.checkout.sessions.create({
    customer: customerId,
    mode: "payment",
    payment_method_types: ["card"],
    line_items: [
      {
        price_data: {
          currency: "usd",
          unit_amount: amountCents,
          product_data: {
            name: "Compute API Credits",
            description: `${totalCredits.toLocaleString()} credits${discount > 0 ? ` (${Math.round(discount * 100)}% volume bonus)` : ""}`,
          },
        },
        quantity: 1,
      },
    ],
    metadata: {
      account_id: accountId,
      credits: totalCredits.toString(),
      amount_dollars: amountDollars.toString(),
    },
    success_url: successUrl || process.env.STRIPE_SUCCESS_URL || "https://computenetwork.sh/dashboard?topup=success",
    cancel_url: cancelUrl || process.env.STRIPE_CANCEL_URL || "https://computenetwork.sh/dashboard?topup=cancel",
  });

  if (!session.url) throw new Error("Failed to create checkout session");

  return { url: session.url, sessionId: session.id };
}

/**
 * Handle Stripe webhook events.
 * Call this with the raw body and Stripe-Signature header.
 */
export async function handleWebhook(
  rawBody: string,
  signature: string
): Promise<{ handled: boolean; event?: string }> {
  const s = getStripe();
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!webhookSecret) throw new Error("STRIPE_WEBHOOK_SECRET not configured");

  let event: Stripe.Event;
  try {
    event = s.webhooks.constructEvent(rawBody, signature, webhookSecret);
  } catch (err: any) {
    throw new Error(`Webhook signature verification failed: ${err.message}`);
  }

  if (event.type === "checkout.session.completed") {
    const session = event.data.object as Stripe.Checkout.Session;
    const accountId = session.metadata?.account_id;
    const credits = parseInt(session.metadata?.credits ?? "0", 10);
    const amountDollars = session.metadata?.amount_dollars ?? "0";

    if (!accountId || credits <= 0) {
      console.warn("[stripe] Checkout completed but missing metadata:", session.id);
      return { handled: false, event: event.type };
    }

    // Idempotency: check if we already processed this session
    const { data: existing } = await supabase
      .from("credit_transactions")
      .select("id")
      .eq("stripe_checkout_session_id", session.id)
      .limit(1);

    if (existing && existing.length > 0) {
      console.log("[stripe] Already processed session:", session.id);
      return { handled: true, event: event.type };
    }

    await topUpCredits(
      accountId,
      credits,
      "stripe_topup",
      `$${amountDollars} top-up via Stripe`,
      session.id
    );

    console.log(`[stripe] Credited ${credits} to account ${accountId} (session: ${session.id})`);
    return { handled: true, event: event.type };
  }

  return { handled: false, event: event.type };
}
