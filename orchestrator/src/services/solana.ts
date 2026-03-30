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
    console.log(`[solana] Requesting devnet airdrop for ${pubkey.toBase58().slice(0, 8)}...`);
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
  tx.recentBlockhash = (
    await connection.getLatestBlockhash()
  ).blockhash;
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

  const rawAmount = BigInt(Math.round(amount * 10 ** DECIMALS));

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
  tx.recentBlockhash = (
    await connection.getLatestBlockhash()
  ).blockhash;
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
  const rawAmount = BigInt(Math.round(pendingAmount * 10 ** DECIMALS));

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
    // NodeAccount layout: 8 (discriminator) + 32 (owner) + 8 (total_earned) + 8 (total_claimed) + 8 (pending_rewards)
    const pendingOffset = 8 + 32 + 8 + 8;
    onChainPending = nodeAccount.data.readBigUInt64LE(pendingOffset);
  }

  const distributeAmount = rawAmount > onChainPending ? rawAmount - onChainPending : 0n;
  if (distributeAmount > 0n) {
    const humanAmount = Number(distributeAmount) / 10 ** DECIMALS;
    const distributeData = Buffer.concat([
      IX_DISTRIBUTE_REWARDS,
      encodeU64(distributeAmount),
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
  const totalClaimRaw = onChainPending + distributeAmount;
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
