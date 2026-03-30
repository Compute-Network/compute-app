/**
 * Live $COMPUTE price feed via DexScreener.
 * Auto-adjusts creditsPerToken to maintain ~5% loss on self-requests.
 */

import { PRICING } from "./billing.js";

// The $COMPUTE token mint — will be updated when launching on pump.fun
const COMPUTE_MINT = process.env.COMPUTE_MINT || "7Jun2ULc2mRTBh6M5nPyzGHj7ziFianr3y1x4MSLiiK1";

let lastPrice: number = parseFloat(process.env.PRICE_COMPUTE_USD || "0.001");
let lastFetchedAt: number = 0;

/**
 * Fetch the current $COMPUTE price from DexScreener.
 * Caches for 60 seconds to avoid rate limiting.
 */
export async function getComputePrice(): Promise<number> {
  const now = Date.now();
  if (now - lastFetchedAt < 60_000) return lastPrice;

  try {
    const res = await fetch(`https://api.dexscreener.com/latest/dex/tokens/${COMPUTE_MINT}`);
    if (!res.ok) return lastPrice;

    const data = await res.json();
    const pairs = data?.pairs;
    if (!pairs || pairs.length === 0) return lastPrice;

    // Use the pair with highest liquidity
    const sorted = pairs.sort((a: any, b: any) =>
      (b.liquidity?.usd ?? 0) - (a.liquidity?.usd ?? 0)
    );

    const price = parseFloat(sorted[0].priceUsd);
    if (isNaN(price) || price <= 0) return lastPrice;

    const oldPrice = lastPrice;
    lastPrice = price;
    lastFetchedAt = now;

    // Update the env-level price for crypto deposit calculations
    process.env.PRICE_COMPUTE_USD = price.toString();

    if (Math.abs(price - oldPrice) / oldPrice > 0.1) {
      console.log(`[pricefeed] $COMPUTE price updated: $${oldPrice.toFixed(6)} → $${price.toFixed(6)}`);
    }

    return price;
  } catch (e: any) {
    console.debug("[pricefeed] Fetch failed:", e.message);
    return lastPrice;
  }
}

/**
 * Recalculate creditsPerToken based on current $COMPUTE price.
 *
 * Formula: providers earn (layers × baseRate × price × creditsPerDollar × (1 + bonus))
 * credits per token served. We set the cost to 105% of that so self-requesting
 * with the bonus yields a ~5% loss.
 */
export function recalculateTokenPrice(): void {
  const price = lastPrice;
  const layers = 32; // Default model layers (llama-3.1-8b)
  const baseRate = 0.001; // REWARD_CONFIG.baseRatePerTokenLayer

  // Credits earned per token served (with 20% $COMPUTE bonus)
  const earnedPerToken = layers * baseRate * price * PRICING.creditsPerDollar * (1 + PRICING.computeTokenBonus);

  // Cost = 105% of earned (5% loss on self-requests)
  const newCost = Math.max(Math.ceil(earnedPerToken * 1.05), 1);

  if (newCost !== PRICING.creditsPerToken) {
    console.log(`[pricefeed] creditsPerToken: ${PRICING.creditsPerToken} → ${newCost} (price: $${price.toFixed(6)})`);
    PRICING.creditsPerToken = newCost;
  }
}

/**
 * Start the price feed — fetches every 60 seconds and recalculates pricing.
 */
export function startPriceFeed(): void {
  // Initial fetch
  getComputePrice().then(() => recalculateTokenPrice());

  setInterval(async () => {
    await getComputePrice();
    recalculateTokenPrice();
  }, 60_000);

  console.log("[pricefeed] Started — updating every 60s from DexScreener");
}
