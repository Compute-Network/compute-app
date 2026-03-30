/**
 * Live $COMPUTE price feed via DexScreener.
 * Auto-adjusts the reward rate so self-requesting with the 20% bonus
 * yields a ~5% loss. API credit pricing stays fixed for customers.
 */

import { PRICING } from "./billing.js";
import { updateRewardConfig, getRewardConfig } from "./rewards.js";

// The $COMPUTE token mint — update when launching on pump.fun
const COMPUTE_MINT = process.env.COMPUTE_MINT || "7Jun2ULc2mRTBh6M5nPyzGHj7ziFianr3y1x4MSLiiK1";

// The base reward rate at the reference price ($0.001)
// This is the starting point — as price moves, the rate adjusts inversely
const REFERENCE_PRICE = 0.001;
const REFERENCE_BASE_RATE = 0.001; // baseRatePerTokenLayer at reference price

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
      console.log(`[pricefeed] $COMPUTE price: $${oldPrice.toFixed(6)} → $${price.toFixed(6)}`);
    }

    return price;
  } catch (e: any) {
    console.debug("[pricefeed] Fetch failed:", e.message);
    return lastPrice;
  }
}

/**
 * Recalculate the reward rate based on current $COMPUTE price.
 *
 * As token price goes up, providers earn fewer tokens per request
 * (but each token is worth more). This keeps the USD value of rewards
 * roughly constant and prevents self-request profitability.
 *
 * Formula: newRate = referenceRate × (referencePrice / currentPrice) × 0.95
 * The 0.95 factor ensures ~5% loss on self-requests with the 20% bonus.
 *
 * Example:
 *   At $0.001: rate = 0.001 × 0.95 = 0.00095 → earn 0.03 $COMPUTE/token
 *   At $0.01:  rate = 0.0001 × 0.95 = 0.000095 → earn 0.003 $COMPUTE/token
 *   At $0.10:  rate = 0.00001 × 0.95 = 0.0000095 → earn 0.0003 $COMPUTE/token
 *   USD value earned stays ~constant, credit value stays below API cost.
 */
export function recalculateRewardRate(): void {
  const price = lastPrice;
  const priceRatio = REFERENCE_PRICE / price;

  // Scale reward rate inversely with price, with 5% haircut
  const newBaseRate = REFERENCE_BASE_RATE * priceRatio * 0.95;
  const newFloorRate = (REFERENCE_BASE_RATE * 0.5) * priceRatio * 0.95;

  const currentConfig = getRewardConfig();
  const oldRate = currentConfig.baseRatePerTokenLayer;

  // Only update if change is > 1% to avoid log spam
  if (Math.abs(newBaseRate - oldRate) / oldRate > 0.01) {
    updateRewardConfig({
      baseRatePerTokenLayer: newBaseRate,
      floorRatePerToken: newFloorRate,
    });
    console.log(
      `[pricefeed] Reward rate: ${oldRate.toFixed(6)} → ${newBaseRate.toFixed(6)} ` +
      `(price: $${price.toFixed(6)}, ${(priceRatio).toFixed(1)}x from reference)`
    );
  }
}

/**
 * Start the price feed — fetches every 60 seconds and recalculates reward rate.
 */
export function startPriceFeed(): void {
  // Initial fetch
  getComputePrice().then(() => recalculateRewardRate());

  setInterval(async () => {
    await getComputePrice();
    recalculateRewardRate();
  }, 60_000);

  console.log("[pricefeed] Started — adjusting reward rate every 60s from DexScreener");
}
