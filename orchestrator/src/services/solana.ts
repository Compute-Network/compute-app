/**
 * Solana service — interacts with the Compute Rewards Anchor program on devnet.
 *
 * On startup:  loads keypair, ensures pool + vault are initialized on-chain.
 * On inference: calls distribute_rewards to record rewards on-chain.
 * On claim:     builds a partially-signed tx (register_node? + claim_rewards),
 *               returns it for the user to co-sign via wallet adapter.
 *
 * The mint is temporary (devnet) — will be replaced by pump.fun token on mainnet.
 */

import {
  Connection,
  Keypair,
  PublicKey,
  SystemProgram,
  Transaction,
  TransactionInstruction,
  clusterApiUrl,
  LAMPORTS_PER_SOL,
} from "@solana/web3.js";
import {
  TOKEN_PROGRAM_ID,
  createMint,
  getOrCreateAssociatedTokenAccount,
  getAssociatedTokenAddress,
  mintTo,
  getAccount,
} from "@solana/spl-token";
import * as fs from "fs";
import * as path from "path";

// ── Program constants ───────────────────────────────────────────────
const PROGRAM_ID = new PublicKey(
  "8socMypA9fyApkdKwoK8SiP4LngUR1gjh5JqcCzQVb4q"
);
const DECIMALS = 6;
const MINT_SUPPLY = 1_000_000_000; // 1B tokens for devnet
const CONFIG_PATH = path.resolve(process.cwd(), ".compute-token.json");

// Max tokens per single distribute_rewards transaction (10,000 $COMPUTE)
const MAX_DISTRIBUTE_PER_TX = BigInt(10_000) * BigInt(10 ** DECIMALS);

// Instruction discriminators from IDL
const IX_INITIALIZE_POOL = Buffer.from([95, 180, 10, 172, 84, 174, 232, 40]);
const IX_REGISTER_NODE = Buffer.from([102, 85, 117, 114, 194, 188, 211, 168]);
const IX_DISTRIBUTE_REWARDS = Buffer.from([
  97, 6, 227, 255, 124, 165, 3, 148,
]);
const IX_CLAIM_REWARDS = Buffer.from([4, 144, 132, 71, 116, 23, 151, 80]);

// Account discriminators for existence checks
const NODE_ACCOUNT_DISCRIMINATOR = Buffer.from([
  125, 166, 18, 146, 195, 127, 86, 220,
]);

// ── NodeAccount layout ─────────────────────────────────────────────
// Defines the on-chain NodeAccount struct fields and their byte sizes.
// Used to safely calculate offsets instead of hardcoding magic numbers.
const NODE_ACCOUNT_LAYOUT = {
  discriminator: { offset: 0, size: 8 },
  owner: { offset: 8, size: 32 },
  total_earned: { offset: 40, size: 8 },
  total_claimed: { offset: 48, size: 8 },
  pending_rewards: { offset: 56, size: 8 },
} as const;

// ── Module state ────────────────────────────────────────────────────
let connection: Connection;
let authority: Keypair;
let mintAddress: PublicKey;
let poolPDA: PublicKey;
let poolBump: number;
let vaultAddress: PublicKey;

// ── PDA derivation ──────────────────────────────────────────────────
function derivePoolPDA(auth: PublicKey): [PublicKey, number] {
  return PublicKey.findProgramAddressSync(
    [Buffer.from("pool"), auth.toBuffer()],
    PROGRAM_ID
  );
}

function deriveNodePDA(owner: PublicKey): [PublicKey, number] {
  return PublicKey.findProgramAddressSync(
    [Buffer.from("node"), owner.toBuffer()],
    PROGRAM_ID
  );
}

// ── Helpers ─────────────────────────────────────────────────────────
function loadKeypair(): Keypair {
  const envKey = process.env.SOLANA_KEYPAIR;
  if (envKey) {
    return Keypair.fromSecretKey(Uint8Array.from(JSON.parse(envKey)));
  }
  const keypairPath = path.resolve(
    process.cwd(),
    "../solana-program/deploy-keypair.json"
  );
  if (fs.existsSync(keypairPath)) {
    return Keypair.fromSecretKey(
      Uint8Array.from(JSON.parse(fs.readFileSync(keypairPath, "utf-8")))
    );
  }
  throw new Error("No Solana keypair found");
}

interface TokenConfig {
  mint: string;
}

function loadConfig(): TokenConfig | null {
  if (process.env.COMPUTE_MINT) return { mint: process.env.COMPUTE_MINT };
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

async function ensureDevnetSol(pubkey: PublicKey, minSol = 0.5) {
  const balance = await connection.getBalance(pubkey);
  if (balance < minSol * LAMPORTS_PER_SOL) {
    console.log(
      `[solana] Requesting devnet airdrop for ${pubkey.toBase58().slice(0, 8)}...`
    );
    try {
      const sig = await connection.requestAirdrop(
        pubkey,
        2 * LAMPORTS_PER_SOL
      );
      await connection.confirmTransaction(sig, "confirmed");
    } catch (e: any) {
      console.warn("[solana] Airdrop failed:", e.message);
    }
  }
}

// ── Serialization helpers for instruction args ──────────────────────
function encodeU64(value: bigint): Buffer {
  const buf = Buffer.alloc(8);
  buf.writeBigUInt64LE(value);
  return buf;
}

function encodeU32(value: number): Buffer {
  const buf = Buffer.alloc(4);
  buf.writeUInt32LE(value);
  return buf;
}

// ── Initialization ──────────────────────────────────────────────────
export async function initSolana(): Promise<void> {
  const rpcUrl = process.env.SOLANA_RPC_URL || clusterApiUrl("devnet");
  connection = new Connection(rpcUrl, "confirmed");
  authority = loadKeypair();

  console.log(`[solana] Authority: ${authority.publicKey.toBase58()}`);
  console.log(`[solana] Program:   ${PROGRAM_ID.toBase58()}`);

  await ensureDevnetSol(authority.publicKey);

  // Load or create mint
  const config = loadConfig();
  if (config) {
    mintAddress = new PublicKey(config.mint);
    console.log(`[solana] Mint: ${mintAddress.toBase58()}`);
  } else {
    console.log("[solana] Creating $COMPUTE token mint...");
    mintAddress = await createMint(
      connection,
      authority,
      authority.publicKey,
      null,
      DECIMALS
    );
    saveConfig({ mint: mintAddress.toBase58() });
    console.log(`[solana] Mint created: ${mintAddress.toBase58()}`);
  }

  // Derive pool PDA
  [poolPDA, poolBump] = derivePoolPDA(authority.publicKey);
  console.log(`[solana] Pool PDA: ${poolPDA.toBase58()}`);

  // Check if pool exists on-chain
  const poolAccount = await connection.getAccountInfo(poolPDA);

  if (!poolAccount) {
    await initializePool();
  } else {
    console.log("[solana] Pool already initialized");
  }

  // Ensure vault has tokens
  vaultAddress = await getAssociatedTokenAddress(
    mintAddress,
    poolPDA,
    true // allowOwnerOffCurve
  );

  try {
    const vaultAccount = await getAccount(connection, vaultAddress);
    const balance = Number(vaultAccount.amount) / 10 ** DECIMALS;
    console.log(
      `[solana] Vault: ${vaultAddress.toBase58()} (${balance.toLocaleString()} $COMPUTE)`
    );

    if (balance < 10_000) {
      await topUpVault();
    }
  } catch {
    // Vault doesn't exist yet — pool was just created, create + fund it
    await topUpVault();
  }
}

async function initializePool(): Promise<void> {
  console.log("[solana] Initializing reward pool on-chain...");

  // Create vault ATA owned by pool PDA
  const vault = await getOrCreateAssociatedTokenAccount(
    connection,
    authority,
    mintAddress,
    poolPDA,
    true // allowOwnerOffCurve (PDA owner)
  );
  vaultAddress = vault.address;

  // Build initialize_pool instruction
  const data = Buffer.concat([
    IX_INITIALIZE_POOL,
    encodeU64(BigInt(1000)), // reward_rate (base units)
    encodeU64(BigInt(500)), // floor_rate
    encodeU64(BigInt(1_000_000) * BigInt(10 ** DECIMALS)), // max_claim_amount: 1M tokens
  ]);

  const ix = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      { pubkey: poolPDA, isSigner: false, isWritable: true },
      { pubkey: mintAddress, isSigner: false, isWritable: false },
      { pubkey: vaultAddress, isSigner: false, isWritable: false },
      { pubkey: authority.publicKey, isSigner: true, isWritable: true },
      { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
    ],
    data,
  });

  const tx = new Transaction().add(ix);
  tx.feePayer = authority.publicKey;
  tx.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
  tx.sign(authority);

  const sig = await connection.sendRawTransaction(tx.serialize());
  await connection.confirmTransaction(sig, "confirmed");
  console.log(`[solana] Pool initialized (tx: ${sig})`);
}

async function topUpVault(): Promise<void> {
  // Ensure vault ATA exists
  const vault = await getOrCreateAssociatedTokenAccount(
    connection,
    authority,
    mintAddress,
    poolPDA,
    true
  );
  vaultAddress = vault.address;

  console.log("[solana] Minting tokens to vault...");
  await mintTo(
    connection,
    authority,
    mintAddress,
    vaultAddress,
    authority,
    BigInt(MINT_SUPPLY) * BigInt(10 ** DECIMALS)
  );
  console.log(
    `[solana] Minted ${MINT_SUPPLY.toLocaleString()} $COMPUTE to vault`
  );
}

// ── Node registration check ─────────────────────────────────────────
export async function nodeExistsOnChain(
  ownerWallet: string
): Promise<boolean> {
  const owner = new PublicKey(ownerWallet);
  const [nodePDA] = deriveNodePDA(owner);
  const account = await connection.getAccountInfo(nodePDA);
  return account !== null;
}

// ── Validate NodeAccount discriminator ──────────────────────────────
function validateNodeAccountDiscriminator(data: Buffer): boolean {
  if (data.length < NODE_ACCOUNT_LAYOUT.discriminator.size) return false;
  const disc = data.subarray(0, NODE_ACCOUNT_LAYOUT.discriminator.size);
  return disc.equals(NODE_ACCOUNT_DISCRIMINATOR);
}

// ── Distribute rewards (orchestrator calls after inference) ─────────
export async function distributeRewardsOnChain(
  nodeOwnerWallet: string,
  amount: number,
  layersServed: number,
  totalLayers: number,
  tokensGenerated: number
): Promise<string | null> {
  const owner = new PublicKey(nodeOwnerWallet);
  const [nodePDA] = deriveNodePDA(owner);

  // Check if node is registered on-chain
  const nodeAccount = await connection.getAccountInfo(nodePDA);
  if (!nodeAccount) {
    console.log(
      `[solana] Node ${nodeOwnerWallet.slice(0, 8)}... not registered on-chain, skipping distribute`
    );
    return null;
  }

  let rawAmount = BigInt(Math.round(amount * 10 ** DECIMALS));

  // Enforce per-distribution cap
  if (rawAmount > MAX_DISTRIBUTE_PER_TX) {
    console.warn(
      `[solana] Distribution of ${amount} $COMPUTE exceeds cap of ${Number(MAX_DISTRIBUTE_PER_TX) / 10 ** DECIMALS}. Capping.`
    );
    rawAmount = MAX_DISTRIBUTE_PER_TX;
  }

  const data = Buffer.concat([
    IX_DISTRIBUTE_REWARDS,
    encodeU64(rawAmount),
    encodeU32(layersServed),
    encodeU32(totalLayers),
    encodeU32(tokensGenerated),
  ]);

  const ix = new TransactionInstruction({
    programId: PROGRAM_ID,
    keys: [
      { pubkey: poolPDA, isSigner: false, isWritable: true },
      { pubkey: nodePDA, isSigner: false, isWritable: true },
      { pubkey: authority.publicKey, isSigner: true, isWritable: false },
    ],
    data,
  });

  const tx = new Transaction().add(ix);
  tx.feePayer = authority.publicKey;
  tx.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
  tx.sign(authority);

  try {
    const sig = await connection.sendRawTransaction(tx.serialize());
    await connection.confirmTransaction(sig, "confirmed");
    console.log(
      `[solana] Distributed ${amount} $COMPUTE to ${nodeOwnerWallet.slice(0, 8)}... (tx: ${sig})`
    );
    return sig;
  } catch (e: any) {
    console.error(`[solana] distribute_rewards failed:`, e.message);
    return null;
  }
}

// ── Verify claim transaction on-chain ───────────────────────────────
/**
 * Verifies a claim transaction exists on Solana and is valid:
 * - Transaction is confirmed and succeeded (no errors)
 * - Transaction includes the correct program ID in its instructions
 * - Transaction involves the expected wallet
 */
export async function verifyClaimTransaction(
  signature: string,
  wallet: string
): Promise<boolean> {
  try {
    const tx = await connection.getTransaction(signature, {
      commitment: "confirmed",
      maxSupportedTransactionVersion: 0,
    });

    if (!tx) {
      console.warn(
        `[solana] Claim verification: tx ${signature} not found`
      );
      return false;
    }

    // Check transaction succeeded (no error)
    if (tx.meta?.err) {
      console.warn(
        `[solana] Claim verification: tx ${signature} has error:`,
        tx.meta.err
      );
      return false;
    }

    // Check the transaction includes our program ID
    const accountKeys = tx.transaction.message.getAccountKeys();
    const programIdStr = PROGRAM_ID.toBase58();
    let programFound = false;

    for (let i = 0; i < accountKeys.length; i++) {
      if (accountKeys.get(i)?.toBase58() === programIdStr) {
        programFound = true;
        break;
      }
    }

    if (!programFound) {
      console.warn(
        `[solana] Claim verification: tx ${signature} does not include program ${programIdStr}`
      );
      return false;
    }

    // Check the transaction involves the expected wallet
    const walletPubkey = new PublicKey(wallet);
    let walletFound = false;

    for (let i = 0; i < accountKeys.length; i++) {
      if (accountKeys.get(i)?.equals(walletPubkey)) {
        walletFound = true;
        break;
      }
    }

    if (!walletFound) {
      console.warn(
        `[solana] Claim verification: tx ${signature} does not involve wallet ${wallet}`
      );
      return false;
    }

    return true;
  } catch (e: any) {
    console.error(
      `[solana] Claim verification error for ${signature}:`,
      e.message
    );
    return false;
  }
}

// ── Build claim transaction (partially signed by authority) ─────────
/**
 * Builds a transaction for the user to claim rewards.
 * Includes register_node if needed, distribute_rewards for pending DB amount,
 * and claim_rewards. Authority partially signs; user co-signs via wallet adapter.
 *
 * @returns base64-encoded partially-signed transaction + claim amount
 */
export async function buildClaimTransaction(
  recipientWallet: string,
  pendingAmount: number
): Promise<{
  transaction: string;
  amount: number;
  message: string;
}> {
  const owner = new PublicKey(recipientWallet);
  const [nodePDA, nodeBump] = deriveNodePDA(owner);
  let rawAmount = BigInt(Math.round(pendingAmount * 10 ** DECIMALS));

  // Enforce per-distribution cap on the claim transaction
  if (rawAmount > MAX_DISTRIBUTE_PER_TX) {
    console.warn(
      `[solana] Claim of ${pendingAmount} $COMPUTE exceeds cap of ${Number(MAX_DISTRIBUTE_PER_TX) / 10 ** DECIMALS}. Capping.`
    );
    rawAmount = MAX_DISTRIBUTE_PER_TX;
  }

  // Verify vault balance before building the transaction
  const vaultBalance = await getVaultBalance();
  const claimAmountHuman = Number(rawAmount) / 10 ** DECIMALS;
  if (vaultBalance < claimAmountHuman) {
    throw new Error(
      `Insufficient vault balance: ${vaultBalance.toFixed(4)} $COMPUTE available, ${claimAmountHuman.toFixed(4)} requested`
    );
  }

  const instructions: TransactionInstruction[] = [];
  let message = "";

  // 1. Register node if not already on-chain
  const nodeAccount = await connection.getAccountInfo(nodePDA);
  if (!nodeAccount) {
    // Airdrop SOL to user for rent (devnet only)
    await ensureDevnetSol(owner, 0.01);

    instructions.push(
      new TransactionInstruction({
        programId: PROGRAM_ID,
        keys: [
          { pubkey: nodePDA, isSigner: false, isWritable: true },
          { pubkey: owner, isSigner: true, isWritable: true },
          {
            pubkey: SystemProgram.programId,
            isSigner: false,
            isWritable: false,
          },
        ],
        data: IX_REGISTER_NODE,
      })
    );
    message += "Registering node on-chain. ";
  }

  // 2. Distribute only the difference between DB pending and on-chain pending
  //    (rewards already distributed in real-time after inference shouldn't be doubled)
  let onChainPending = 0n;
  if (nodeAccount) {
    // Validate account discriminator before reading data
    if (!validateNodeAccountDiscriminator(nodeAccount.data)) {
      throw new Error(
        "Invalid NodeAccount discriminator — account data does not match expected format"
      );
    }

    const pendingOffset = NODE_ACCOUNT_LAYOUT.pending_rewards.offset;
    onChainPending = nodeAccount.data.readBigUInt64LE(pendingOffset);
  }

  const distributeAmount =
    rawAmount > onChainPending ? rawAmount - onChainPending : 0n;

  // Cap the distribute amount too
  const cappedDistributeAmount =
    distributeAmount > MAX_DISTRIBUTE_PER_TX
      ? MAX_DISTRIBUTE_PER_TX
      : distributeAmount;

  if (cappedDistributeAmount > 0n) {
    if (cappedDistributeAmount < distributeAmount) {
      console.warn(
        `[solana] Distribute in claim capped from ${Number(distributeAmount) / 10 ** DECIMALS} to ${Number(cappedDistributeAmount) / 10 ** DECIMALS}`
      );
    }
    const humanAmount = Number(cappedDistributeAmount) / 10 ** DECIMALS;
    const distributeData = Buffer.concat([
      IX_DISTRIBUTE_REWARDS,
      encodeU64(cappedDistributeAmount),
      encodeU32(1), // layers_served
      encodeU32(1), // total_layers
      encodeU32(0), // tokens_generated (aggregate)
    ]);

    instructions.push(
      new TransactionInstruction({
        programId: PROGRAM_ID,
        keys: [
          { pubkey: poolPDA, isSigner: false, isWritable: true },
          { pubkey: nodePDA, isSigner: false, isWritable: true },
          { pubkey: authority.publicKey, isSigner: true, isWritable: false },
        ],
        data: distributeData,
      })
    );
    message += `Distributing ${humanAmount.toFixed(4)} $COMPUTE. `;
  }

  // 3. Claim rewards (user signs)
  const userATA = await getAssociatedTokenAddress(mintAddress, owner);

  // Check if user ATA exists, if not add create instruction
  try {
    await getAccount(connection, userATA);
  } catch {
    // Import createAssociatedTokenAccountInstruction
    const { createAssociatedTokenAccountInstruction } = await import(
      "@solana/spl-token"
    );
    instructions.push(
      createAssociatedTokenAccountInstruction(
        authority.publicKey, // payer
        userATA,
        owner,
        mintAddress
      )
    );
    message += "Creating token account. ";
  }

  instructions.push(
    new TransactionInstruction({
      programId: PROGRAM_ID,
      keys: [
        { pubkey: poolPDA, isSigner: false, isWritable: true },
        { pubkey: nodePDA, isSigner: false, isWritable: true },
        { pubkey: mintAddress, isSigner: false, isWritable: false },
        { pubkey: vaultAddress, isSigner: false, isWritable: true },
        { pubkey: userATA, isSigner: false, isWritable: true },
        { pubkey: owner, isSigner: true, isWritable: false },
        { pubkey: TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
      ],
      data: IX_CLAIM_REWARDS,
    })
  );
  // The actual claim amount = on-chain pending + any new distribution
  const totalClaimRaw = onChainPending + cappedDistributeAmount;
  const totalClaimHuman = Number(totalClaimRaw) / 10 ** DECIMALS;
  message += `Claiming ${totalClaimHuman.toFixed(4)} $COMPUTE.`;

  // Build transaction — authority pays fees
  const tx = new Transaction();
  tx.add(...instructions);
  tx.feePayer = authority.publicKey;
  const { blockhash, lastValidBlockHeight } =
    await connection.getLatestBlockhash();
  tx.recentBlockhash = blockhash;
  tx.lastValidBlockHeight = lastValidBlockHeight;

  // Partially sign with authority
  tx.partialSign(authority);

  // Serialize (with requireAllSignatures = false since user hasn't signed)
  const serialized = tx
    .serialize({ requireAllSignatures: false })
    .toString("base64");

  return {
    transaction: serialized,
    amount: totalClaimHuman,
    message,
  };
}

// ── Getters ─────────────────────────────────────────────────────────
export function getMintAddress(): string {
  return mintAddress?.toBase58() ?? "";
}

export function getPoolAddress(): string {
  return poolPDA?.toBase58() ?? "";
}

export async function getVaultBalance(): Promise<number> {
  if (!vaultAddress) return 0;
  try {
    const account = await getAccount(connection, vaultAddress);
    return Number(account.amount) / 10 ** DECIMALS;
  } catch {
    return 0;
  }
}

export function getProgramId(): string {
  return PROGRAM_ID.toBase58();
}
