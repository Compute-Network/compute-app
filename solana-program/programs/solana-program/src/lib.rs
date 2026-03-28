use anchor_lang::prelude::*;
use anchor_spl::token_interface::{self, Mint, TokenAccount, TokenInterface, TransferChecked};

declare_id!("8socMypA9fyApkdKwoK8SiP4LngUR1gjh5JqcCzQVb4q");

#[program]
pub mod compute_rewards {
    use super::*;

    /// Initialize the reward pool. Called once by the program authority.
    pub fn initialize_pool(
        ctx: Context<InitializePool>,
        reward_rate: u64,
        floor_rate: u64,
        max_claim_amount: u64,
    ) -> Result<()> {
        let pool = &mut ctx.accounts.pool;
        pool.authority = ctx.accounts.authority.key();
        pool.reward_mint = ctx.accounts.reward_mint.key();
        pool.reward_vault = ctx.accounts.reward_vault.key();
        pool.reward_rate = reward_rate;
        pool.floor_rate = floor_rate;
        pool.max_claim_amount = max_claim_amount;
        pool.total_distributed = 0;
        pool.total_claimed = 0;
        pool.paused = false;
        pool.bump = ctx.bumps.pool;

        msg!("Reward pool initialized. Rate: {}, Floor: {}", reward_rate, floor_rate);
        Ok(())
    }

    /// Register a compute node on-chain.
    pub fn register_node(ctx: Context<RegisterNode>) -> Result<()> {
        let node = &mut ctx.accounts.node_account;
        node.owner = ctx.accounts.owner.key();
        node.total_earned = 0;
        node.total_claimed = 0;
        node.pending_rewards = 0;
        node.last_claim_slot = 0;
        node.registered_at = Clock::get()?.slot;
        node.bump = ctx.bumps.node_account;

        msg!("Node registered: {}", ctx.accounts.owner.key());
        Ok(())
    }

    /// Record rewards for a node (orchestrator authority only).
    pub fn distribute_rewards(
        ctx: Context<DistributeRewards>,
        amount: u64,
        layers_served: u32,
        total_layers: u32,
        tokens_generated: u32,
    ) -> Result<()> {
        let pool = &ctx.accounts.pool;
        require!(!pool.paused, ComputeError::PoolPaused);
        require!(amount > 0, ComputeError::ZeroAmount);

        let node = &mut ctx.accounts.node_account;
        node.pending_rewards = node.pending_rewards.checked_add(amount).ok_or(ComputeError::Overflow)?;
        node.total_earned = node.total_earned.checked_add(amount).ok_or(ComputeError::Overflow)?;

        let pool_mut = &mut ctx.accounts.pool;
        pool_mut.total_distributed = pool_mut.total_distributed.checked_add(amount).ok_or(ComputeError::Overflow)?;

        emit!(RewardDistributed {
            node: node.owner,
            amount,
            layers_served,
            total_layers,
            tokens_generated,
            slot: Clock::get()?.slot,
        });

        Ok(())
    }

    /// Claim pending rewards — transfers $COMPUTE from vault to claimant.
    pub fn claim_rewards(ctx: Context<ClaimRewards>) -> Result<()> {
        let pool = &ctx.accounts.pool;
        require!(!pool.paused, ComputeError::PoolPaused);

        let node = &mut ctx.accounts.node_account;
        let amount = node.pending_rewards;
        require!(amount > 0, ComputeError::NothingToClaim);
        require!(amount <= pool.max_claim_amount, ComputeError::ExceedsMaxClaim);

        // Transfer tokens from vault to claimant via PDA signer
        let authority_key = pool.authority;
        let seeds = &[b"pool".as_ref(), authority_key.as_ref(), &[pool.bump]];
        let signer_seeds = &[&seeds[..]];

        let decimals = ctx.accounts.reward_mint.decimals;

        let cpi_ctx = CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            TransferChecked {
                from: ctx.accounts.reward_vault.to_account_info(),
                to: ctx.accounts.claimant_token_account.to_account_info(),
                authority: ctx.accounts.pool.to_account_info(),
                mint: ctx.accounts.reward_mint.to_account_info(),
            },
            signer_seeds,
        );
        token_interface::transfer_checked(cpi_ctx, amount, decimals)?;

        // Update state
        node.pending_rewards = 0;
        node.total_claimed = node.total_claimed.checked_add(amount).ok_or(ComputeError::Overflow)?;
        node.last_claim_slot = Clock::get()?.slot;

        let pool_mut = &mut ctx.accounts.pool;
        pool_mut.total_claimed = pool_mut.total_claimed.checked_add(amount).ok_or(ComputeError::Overflow)?;

        emit!(RewardClaimed {
            node: node.owner,
            amount,
            slot: Clock::get()?.slot,
        });

        Ok(())
    }

    /// Update pool parameters. Authority only.
    pub fn update_pool(
        ctx: Context<UpdatePool>,
        reward_rate: Option<u64>,
        floor_rate: Option<u64>,
        max_claim_amount: Option<u64>,
        paused: Option<bool>,
    ) -> Result<()> {
        let pool = &mut ctx.accounts.pool;

        if let Some(v) = reward_rate { pool.reward_rate = v; }
        if let Some(v) = floor_rate { pool.floor_rate = v; }
        if let Some(v) = max_claim_amount { pool.max_claim_amount = v; }
        if let Some(v) = paused { pool.paused = v; }

        msg!("Pool updated");
        Ok(())
    }
}

// ============================================================
// Accounts
// ============================================================

#[derive(Accounts)]
pub struct InitializePool<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + RewardPool::INIT_SPACE,
        seeds = [b"pool", authority.key().as_ref()],
        bump
    )]
    pub pool: Account<'info, RewardPool>,

    pub reward_mint: InterfaceAccount<'info, Mint>,

    #[account(constraint = reward_vault.mint == reward_mint.key())]
    pub reward_vault: InterfaceAccount<'info, TokenAccount>,

    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct RegisterNode<'info> {
    #[account(
        init,
        payer = owner,
        space = 8 + NodeAccount::INIT_SPACE,
        seeds = [b"node", owner.key().as_ref()],
        bump
    )]
    pub node_account: Account<'info, NodeAccount>,

    #[account(mut)]
    pub owner: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct DistributeRewards<'info> {
    #[account(mut, has_one = authority)]
    pub pool: Account<'info, RewardPool>,

    #[account(
        mut,
        seeds = [b"node", node_account.owner.as_ref()],
        bump = node_account.bump,
    )]
    pub node_account: Account<'info, NodeAccount>,

    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ClaimRewards<'info> {
    #[account(mut)]
    pub pool: Account<'info, RewardPool>,

    #[account(
        mut,
        seeds = [b"node", owner.key().as_ref()],
        bump = node_account.bump,
        has_one = owner,
    )]
    pub node_account: Account<'info, NodeAccount>,

    #[account(constraint = reward_mint.key() == pool.reward_mint)]
    pub reward_mint: InterfaceAccount<'info, Mint>,

    #[account(mut, constraint = reward_vault.key() == pool.reward_vault)]
    pub reward_vault: InterfaceAccount<'info, TokenAccount>,

    #[account(
        mut,
        constraint = claimant_token_account.owner == owner.key(),
        constraint = claimant_token_account.mint == pool.reward_mint,
    )]
    pub claimant_token_account: InterfaceAccount<'info, TokenAccount>,

    pub owner: Signer<'info>,
    pub token_program: Interface<'info, TokenInterface>,
}

#[derive(Accounts)]
pub struct UpdatePool<'info> {
    #[account(mut, has_one = authority)]
    pub pool: Account<'info, RewardPool>,
    pub authority: Signer<'info>,
}

// ============================================================
// State
// ============================================================

#[account]
#[derive(InitSpace)]
pub struct RewardPool {
    pub authority: Pubkey,
    pub reward_mint: Pubkey,
    pub reward_vault: Pubkey,
    pub reward_rate: u64,
    pub floor_rate: u64,
    pub max_claim_amount: u64,
    pub total_distributed: u64,
    pub total_claimed: u64,
    pub paused: bool,
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct NodeAccount {
    pub owner: Pubkey,
    pub total_earned: u64,
    pub total_claimed: u64,
    pub pending_rewards: u64,
    pub last_claim_slot: u64,
    pub registered_at: u64,
    pub bump: u8,
}

// ============================================================
// Events
// ============================================================

#[event]
pub struct RewardDistributed {
    pub node: Pubkey,
    pub amount: u64,
    pub layers_served: u32,
    pub total_layers: u32,
    pub tokens_generated: u32,
    pub slot: u64,
}

#[event]
pub struct RewardClaimed {
    pub node: Pubkey,
    pub amount: u64,
    pub slot: u64,
}

// ============================================================
// Errors
// ============================================================

#[error_code]
pub enum ComputeError {
    #[msg("Reward pool is paused")]
    PoolPaused,
    #[msg("Amount must be greater than zero")]
    ZeroAmount,
    #[msg("Nothing to claim")]
    NothingToClaim,
    #[msg("Claim amount exceeds maximum")]
    ExceedsMaxClaim,
    #[msg("Arithmetic overflow")]
    Overflow,
}
