/**
 * Live $COMPUTE price feed.
 * Primary: Helius RPC (Jupiter on-chain quote)
 * Fallback: DexScreener polling
 *
 * Auto-adjusts the reward rate inversely with token price so
 * self-requesting with the 20% bonus yields a ~5% loss.
 * Credit pricing (creditsPerToken) stays fixed for customers.
 */

import { Connection } from "@solana/web3.js";
import { updateRewardConfig, getRewardConfig } from "./rewards.js";

const COMPUTE_MINT = process.env.COMPUTE_MINT || "7Jun2ULc2mRTBh6M5nPyzGHj7ziFianr3y1x4MSLiiK1";
const USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"; // mainnet USDC

const HELIUS_RPC = process.env.HELIUS_RPC_URL || "";
const HELIUS_WS = process.env.HELIUS_WS_URL || "";

const REFERENCE_PRICE = 0.001;
const REFERENCE_BASE_RATE = 0.001;

let lastPrice: number = parseFloat(process.env.PRICE_COMPUTE_USD || "0.001");
let lastFetchedAt: number = 0;
let wsConnected = false;

/**
 * Get the current $COMPUTE price. Uses cached value if fresh.
 */
export function getComputePrice(): number {
  return lastPrice;
}

/**
 * Fetch price via Jupiter quote API (works for any Solana token including pump.fun).
 * Asks: how much USDC for 1,000,000 $COMPUTE (6 decimals)?
 */
async function fetchPriceJupiter(): Promise<number | null> {
  try {
    // 1 $COMPUTE = 1_000_000 base units (6 decimals)
    const amount = 1_000_000;
    const url = `https://quote-api.jup.ag/v6/quote?inputMint=${COMPUTE_MINT}&outputMint=${USDC_MINT}&amount=${amount}&slippageBps=100`;
    const res = await fetch(url);
    if (!res.ok) return null;

    const data = await res.json();
    const outAmount = parseInt(data.outAmount ?? "0", 10);
    if (outAmount <= 0) return null;

    // USDC has 6 decimals
    const price = outAmount / 1_000_000;
    return price;
  } catch {
    return null;
  }
}

/**
 * Fetch price via DexScreener (free, no auth, works for pump.fun).
 */
async function fetchPriceDexScreener(): Promise<number | null> {
  try {
    const res = await fetch(`https://api.dexscreener.com/latest/dex/tokens/${COMPUTE_MINT}`);
    if (!res.ok) return null;

    const data = await res.json();
    const pairs = data?.pairs;
    if (!pairs || pairs.length === 0) return null;

    const sorted = pairs.sort((a: any, b: any) =>
      (b.liquidity?.usd ?? 0) - (a.liquidity?.usd ?? 0)
    );

    const price = parseFloat(sorted[0].priceUsd);
    return isNaN(price) || price <= 0 ? null : price;
  } catch {
    return null;
  }
}

/**
 * Update the price from best available source.
 */
async function refreshPrice(): Promise<void> {
  // Try Jupiter first (most accurate, works for pump.fun)
  let price = await fetchPriceJupiter();

  // Fallback to DexScreener
  if (!price) {
    price = await fetchPriceDexScreener();
  }

  if (!price) return;

  const oldPrice = lastPrice;
  lastPrice = price;
  lastFetchedAt = Date.now();
  process.env.PRICE_COMPUTE_USD = price.toString();

  if (Math.abs(price - oldPrice) / Math.max(oldPrice, 0.0000001) > 0.05) {
    console.log(`[pricefeed] $COMPUTE: $${oldPrice.toFixed(8)} → $${price.toFixed(8)}`);
    recalculateRewardRate();
  }
}

/**
 * Subscribe to token account changes via Helius websocket.
 * When the pool state changes (trade happens), we refresh the price.
 */
function startWebsocket(): void {
  if (!HELIUS_WS) {
    console.log("[pricefeed] No HELIUS_WS_URL — using polling only");
    return;
  }

  const connect = () => {
    try {
      const ws = new WebSocket(HELIUS_WS);

      ws.onopen = () => {
        wsConnected = true;
        console.log("[pricefeed] Helius websocket connected");

        // Subscribe to the token mint account for changes
        ws.send(JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "accountSubscribe",
          params: [
            COMPUTE_MINT,
            { encoding: "jsonParsed", commitment: "confirmed" }
          ]
        }));
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data.toString());
          if (msg.method === "accountNotification") {
            // Token account changed — likely a trade, refresh price
            refreshPrice();
          }
        } catch {}
      };

      ws.onclose = () => {
        wsConnected = false;
        console.log("[pricefeed] Helius websocket disconnected, reconnecting in 5s...");
        setTimeout(connect, 5000);
      };

      ws.onerror = () => {
        wsConnected = false;
        ws.close();
      };
    } catch (e: any) {
      console.debug("[pricefeed] Websocket error:", e.message);
      setTimeout(connect, 10000);
    }
  };

  connect();
}

/**
 * Recalculate the reward rate based on current $COMPUTE price.
 * Rate scales inversely with price × 0.95 factor for 5% loss.
 */
export function recalculateRewardRate(): void {
  const price = lastPrice;
  const priceRatio = REFERENCE_PRICE / price;

  const newBaseRate = REFERENCE_BASE_RATE * priceRatio * 0.95;
  const newFloorRate = (REFERENCE_BASE_RATE * 0.5) * priceRatio * 0.95;

  const currentConfig = getRewardConfig();
  const oldRate = currentConfig.baseRatePerTokenLayer;

  if (Math.abs(newBaseRate - oldRate) / Math.max(oldRate, 0.0000001) > 0.01) {
    updateRewardConfig({
      baseRatePerTokenLayer: newBaseRate,
      floorRatePerToken: newFloorRate,
    });
    console.log(
      `[pricefeed] Reward rate: ${oldRate.toFixed(8)} → ${newBaseRate.toFixed(8)} ` +
      `(price: $${price.toFixed(8)})`
    );
  }
}

/**
 * Start the price feed.
 * - Websocket for real-time updates (Helius)
 * - Polling every 10s as backup
 */
export function startPriceFeed(): void {
  // Initial fetch
  refreshPrice().then(() => recalculateRewardRate());

  // Websocket for real-time
  startWebsocket();

  // Polling backup every 10s
  setInterval(refreshPrice, 10_000);

  console.log("[pricefeed] Started — websocket + 10s polling fallback");
}
