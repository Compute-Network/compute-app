/**
 * Solana service — manages the $COMPUTE SPL token on devnet.
 *
 * On startup: loads or creates the token mint + treasury ATA.
 * On claim:   transfers tokens from treasury to the user's ATA.
 */

import {
  Connection,
  Keypair,
  PublicKey,
  clusterApiUrl,
  LAMPORTS_PER_SOL,
} from "@solana/web3.js";
import {
  createMint,
  getOrCreateAssociatedTokenAccount,
  mintTo,
  transfer,
  getMint,
  getAccount,
} from "@solana/spl-token";
import * as fs from "fs";
import * as path from "path";

const DECIMALS = 6; // $COMPUTE has 6 decimal places
const MINT_SUPPLY = 1_000_000_000; // 1B tokens initial supply for devnet
const CONFIG_PATH = path.resolve(process.cwd(), ".compute-token.json");

let connection: Connection;
let authority: Keypair;
let mintAddress: PublicKey;
let treasuryAta: PublicKey; // authority's associated token account

interface TokenConfig {
  mint: string;
  authority: string; // base58 pubkey for verification
}

function loadKeypair(): Keypair {
  // Try SOLANA_KEYPAIR env (JSON array), else use deploy-keypair.json
  const envKey = process.env.SOLANA_KEYPAIR;
  if (envKey) {
    const secret = JSON.parse(envKey);
    return Keypair.fromSecretKey(Uint8Array.from(secret));
  }

  const keypairPath = path.resolve(
    process.cwd(),
    "../solana-program/deploy-keypair.json"
  );
  if (fs.existsSync(keypairPath)) {
    const secret = JSON.parse(fs.readFileSync(keypairPath, "utf-8"));
    return Keypair.fromSecretKey(Uint8Array.from(secret));
  }

  throw new Error(
    "No Solana keypair found. Set SOLANA_KEYPAIR env or place deploy-keypair.json"
  );
}

function loadConfig(): TokenConfig | null {
  // Check env first
  if (process.env.COMPUTE_MINT) {
    return {
      mint: process.env.COMPUTE_MINT,
      authority: authority.publicKey.toBase58(),
    };
  }
  if (!fs.existsSync(CONFIG_PATH)) return null;
  try {
    return JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));
  } catch {
    return null;
  }
}

function saveConfig(config: TokenConfig) {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
}

async function ensureDevnetSol() {
  const balance = await connection.getBalance(authority.publicKey);
  if (balance < 0.5 * LAMPORTS_PER_SOL) {
    console.log("[solana] Requesting devnet airdrop...");
    try {
      const sig = await connection.requestAirdrop(
        authority.publicKey,
        2 * LAMPORTS_PER_SOL
      );
      await connection.confirmTransaction(sig, "confirmed");
      console.log("[solana] Airdrop confirmed");
    } catch (e: any) {
      console.warn("[solana] Airdrop failed (may be rate-limited):", e.message);
    }
  }
}

/**
 * Initialize the Solana connection, load or create the $COMPUTE token.
 */
export async function initSolana(): Promise<void> {
  const rpcUrl = process.env.SOLANA_RPC_URL || clusterApiUrl("devnet");
  connection = new Connection(rpcUrl, "confirmed");
  authority = loadKeypair();

  console.log(
    `[solana] Authority: ${authority.publicKey.toBase58()}`
  );
  console.log(`[solana] RPC: ${rpcUrl}`);

  // Ensure we have SOL for fees
  await ensureDevnetSol();

  const config = loadConfig();

  if (config) {
    // Verify existing mint
    mintAddress = new PublicKey(config.mint);
    try {
      const mintInfo = await getMint(connection, mintAddress);
      console.log(
        `[solana] $COMPUTE mint loaded: ${mintAddress.toBase58()} (supply: ${
          Number(mintInfo.supply) / 10 ** DECIMALS
        })`
      );
    } catch {
      console.warn("[solana] Saved mint not found on-chain, creating new one");
      await createToken();
    }
  } else {
    await createToken();
  }

  // Ensure treasury ATA exists and has tokens
  const ata = await getOrCreateAssociatedTokenAccount(
    connection,
    authority,
    mintAddress,
    authority.publicKey
  );
  treasuryAta = ata.address;

  const balance = Number(ata.amount) / 10 ** DECIMALS;
  console.log(
    `[solana] Treasury ATA: ${treasuryAta.toBase58()} (balance: ${balance.toLocaleString()} $COMPUTE)`
  );

  // Top up treasury if low (devnet only)
  if (balance < 10_000) {
    console.log("[solana] Minting tokens to treasury...");
    await mintTo(
      connection,
      authority,
      mintAddress,
      treasuryAta,
      authority,
      BigInt(MINT_SUPPLY) * BigInt(10 ** DECIMALS)
    );
    console.log(
      `[solana] Minted ${MINT_SUPPLY.toLocaleString()} $COMPUTE to treasury`
    );
  }
}

async function createToken(): Promise<void> {
  console.log("[solana] Creating $COMPUTE token mint...");

  mintAddress = await createMint(
    connection,
    authority, // payer
    authority.publicKey, // mint authority
    null, // freeze authority (none)
    DECIMALS
  );

  saveConfig({
    mint: mintAddress.toBase58(),
    authority: authority.publicKey.toBase58(),
  });

  console.log(`[solana] $COMPUTE mint created: ${mintAddress.toBase58()}`);
}

/**
 * Transfer $COMPUTE tokens from treasury to a user's wallet.
 * Creates the user's ATA if it doesn't exist.
 *
 * @param recipientWallet - The user's Solana wallet address (base58)
 * @param amount - Amount in human-readable $COMPUTE (e.g. 1.5)
 * @returns Transaction signature
 */
export async function transferReward(
  recipientWallet: string,
  amount: number
): Promise<string> {
  const recipient = new PublicKey(recipientWallet);
  const rawAmount = BigInt(Math.round(amount * 10 ** DECIMALS));

  // Get or create recipient's ATA
  const recipientAta = await getOrCreateAssociatedTokenAccount(
    connection,
    authority, // payer for ATA creation
    mintAddress,
    recipient
  );

  // Transfer from treasury
  const sig = await transfer(
    connection,
    authority, // payer + authority
    treasuryAta, // from
    recipientAta.address, // to
    authority, // owner of source
    rawAmount
  );

  console.log(
    `[solana] Transferred ${amount} $COMPUTE to ${recipientWallet} (tx: ${sig})`
  );

  return sig;
}

/**
 * Get the $COMPUTE token mint address.
 */
export function getMintAddress(): string {
  return mintAddress?.toBase58() ?? "";
}

/**
 * Get the treasury balance.
 */
export async function getTreasuryBalance(): Promise<number> {
  if (!treasuryAta) return 0;
  const account = await getAccount(connection, treasuryAta);
  return Number(account.amount) / 10 ** DECIMALS;
}
