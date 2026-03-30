import { Connection, PublicKey, ParsedTransactionWithMeta } from "@solana/web3.js";
import { supabase } from "./db.js";
import { topUpCredits, calculateCredits, PRICING } from "./billing.js";

// Known token mints (devnet)
const USDC_MINT = process.env.USDC_MINT || "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"; // devnet USDC
const COMPUTE_MINT = process.env.COMPUTE_MINT || "7Jun2ULc2mRTBh6M5nPyzGHj7ziFianr3y1x4MSLiiK1";

// Price feeds (hardcoded for devnet; in production, use an oracle)
const TOKEN_PRICES_USD: Record<string, number> = {
  SOL: 150, // approximate
  USDC: 1,
  COMPUTE: 0.001, // placeholder
};

let lastSignature: string | undefined;

/**
 * Check for new deposits to the treasury wallet.
 * Called periodically (every 30-60 seconds).
 */
export async function checkPendingDeposits(): Promise<void> {
  const treasuryAddress = PRICING.treasuryWallet;
  if (!treasuryAddress) return;

  const rpcUrl = process.env.SOLANA_RPC_URL || "https://api.devnet.solana.com";
  const connection = new Connection(rpcUrl, "confirmed");
  const treasuryPubkey = new PublicKey(treasuryAddress);

  try {
    // Get recent signatures for the treasury
    const signatures = await connection.getSignaturesForAddress(treasuryPubkey, {
      limit: 20,
      until: lastSignature,
    });

    if (signatures.length === 0) return;

    // Update cursor to most recent
    lastSignature = signatures[0].signature;

    for (const sig of signatures) {
      // Check if we already processed this signature
      const { data: existing } = await supabase
        .from("credit_transactions")
        .select("id")
        .eq("crypto_tx_signature", sig.signature)
        .limit(1);

      if (existing && existing.length > 0) continue;

      // Fetch the full transaction
      const tx = await connection.getParsedTransaction(sig.signature, {
        maxSupportedTransactionVersion: 0,
      });

      if (!tx || tx.meta?.err) continue;

      await processTransaction(tx, sig.signature, treasuryAddress);
    }
  } catch (e: any) {
    // Don't spam logs for RPC errors on devnet
    if (!e.message?.includes("429")) {
      console.error("[crypto-deposits] Check failed:", e.message);
    }
  }
}

async function processTransaction(
  tx: ParsedTransactionWithMeta,
  signature: string,
  treasuryAddress: string
): Promise<void> {
  if (!tx.meta) return;

  const instructions = tx.transaction.message.instructions;

  for (const ix of instructions) {
    // Check for SOL transfer to treasury
    if ("parsed" in ix && ix.program === "system" && ix.parsed?.type === "transfer") {
      const { destination, lamports, source } = ix.parsed.info;
      if (destination === treasuryAddress && lamports > 0) {
        const solAmount = lamports / 1e9;
        const usdValue = solAmount * TOKEN_PRICES_USD.SOL;
        if (usdValue < PRICING.minTopupDollars) continue;

        await creditDeposit(source, "SOL", solAmount, usdValue, signature);
        return;
      }
    }

    // Check for SPL token transfer to treasury (USDC or COMPUTE)
    if ("parsed" in ix && ix.program === "spl-token" && ix.parsed?.type === "transfer") {
      const { amount, authority } = ix.parsed.info;
      // We need to check the mint from the token account
      // For simplicity, check post-token balances
      const postBalances = tx.meta.postTokenBalances ?? [];
      for (const balance of postBalances) {
        if (balance.owner === treasuryAddress && balance.mint) {
          const tokenAmount = Number(amount);
          let token: string;
          let decimals: number;
          let priceKey: string;

          if (balance.mint === USDC_MINT) {
            token = "USDC";
            decimals = 6;
            priceKey = "USDC";
          } else if (balance.mint === COMPUTE_MINT) {
            token = "COMPUTE";
            decimals = 6;
            priceKey = "COMPUTE";
          } else {
            continue; // Unknown token
          }

          const humanAmount = tokenAmount / Math.pow(10, decimals);
          const usdValue = humanAmount * TOKEN_PRICES_USD[priceKey];
          if (usdValue < PRICING.minTopupDollars) continue;

          await creditDeposit(authority, token, humanAmount, usdValue, signature);
          return;
        }
      }
    }
  }
}

async function creditDeposit(
  senderWallet: string,
  token: string,
  tokenAmount: number,
  usdValue: number,
  txSignature: string
): Promise<void> {
  // Find account by wallet address
  const { data: account } = await supabase
    .from("accounts")
    .select("id")
    .eq("wallet_address", senderWallet)
    .single();

  if (!account) {
    console.log(`[crypto-deposits] No account for wallet ${senderWallet}, skipping`);
    return;
  }

  const { totalCredits } = calculateCredits(usdValue, token);

  await topUpCredits(
    account.id,
    totalCredits,
    "crypto_topup",
    `${tokenAmount} ${token} deposit`,
    undefined, // no stripe session
    txSignature,
    token
  );

  console.log(
    `[crypto-deposits] Credited ${totalCredits} credits to account ${account.id} ` +
    `(${tokenAmount} ${token} = $${usdValue.toFixed(2)}, tx: ${txSignature.slice(0, 16)}...)`
  );
}
