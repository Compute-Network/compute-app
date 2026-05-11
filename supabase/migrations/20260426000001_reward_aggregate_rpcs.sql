-- PostgREST aggregates are disabled on the hosted project, so service code
-- must use explicit RPCs for reward summaries instead of final_reward.sum().

create or replace function public.get_pending_reward_summary(
  p_wallet_address text
)
returns table (
  total numeric,
  event_count bigint
)
language sql
stable
security definer
set search_path = public
as $$
  select
    coalesce(sum(final_reward), 0)::numeric as total,
    count(*)::bigint as event_count
  from public.reward_events
  where wallet_address = p_wallet_address
    and status = 'pending';
$$;

create or replace function public.get_pending_rewards_by_node(
  p_wallet_address text
)
returns table (
  node_id uuid,
  total numeric,
  event_count bigint
)
language sql
stable
security definer
set search_path = public
as $$
  select
    reward_events.node_id,
    coalesce(sum(final_reward), 0)::numeric as total,
    count(*)::bigint as event_count
  from public.reward_events
  where wallet_address = p_wallet_address
    and status = 'pending'
  group by reward_events.node_id;
$$;

create or replace function public.get_reward_stats_summary()
returns table (
  total_distributed numeric,
  total_pending numeric,
  total_claimed numeric
)
language sql
stable
security definer
set search_path = public
as $$
  select
    coalesce(sum(final_reward), 0)::numeric as total_distributed,
    coalesce(sum(final_reward) filter (where status = 'pending'), 0)::numeric as total_pending,
    coalesce(sum(final_reward) filter (where status = 'claimed'), 0)::numeric as total_claimed
  from public.reward_events;
$$;

revoke all on function public.get_pending_reward_summary(text) from public;
revoke all on function public.get_pending_reward_summary(text) from anon;
revoke all on function public.get_pending_reward_summary(text) from authenticated;
grant execute on function public.get_pending_reward_summary(text) to service_role;

revoke all on function public.get_pending_rewards_by_node(text) from public;
revoke all on function public.get_pending_rewards_by_node(text) from anon;
revoke all on function public.get_pending_rewards_by_node(text) from authenticated;
grant execute on function public.get_pending_rewards_by_node(text) to service_role;

revoke all on function public.get_reward_stats_summary() from public;
revoke all on function public.get_reward_stats_summary() from anon;
revoke all on function public.get_reward_stats_summary() from authenticated;
grant execute on function public.get_reward_stats_summary() to service_role;
