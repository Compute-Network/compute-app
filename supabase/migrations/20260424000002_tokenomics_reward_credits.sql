create extension if not exists pgcrypto;

alter table if exists public.reward_events
  add column if not exists value_usd numeric not null default 0,
  add column if not exists compute_price_usd numeric not null default 0,
  add column if not exists original_reward numeric,
  add column if not exists spent_as_credits_usd numeric not null default 0,
  add column if not exists spent_as_credits_compute numeric not null default 0;

update public.reward_events
set original_reward = final_reward
where original_reward is null;

create table if not exists public.protocol_revenue_events (
  id uuid primary key default gen_random_uuid(),
  pipeline_id text,
  model_id text,
  gross_usd numeric not null,
  provider_usd numeric not null,
  treasury_usd numeric not null,
  compute_price_usd numeric not null,
  provider_compute numeric not null,
  treasury_compute numeric not null,
  provider_share numeric not null default 0.8,
  treasury_share numeric not null default 0.2,
  status text not null default 'buyback_recorded',
  created_at timestamptz not null default now()
);

alter table public.protocol_revenue_events enable row level security;
revoke all on public.protocol_revenue_events from anon, authenticated;
grant select, insert, update, delete on public.protocol_revenue_events to service_role;

create or replace function public.spend_pending_reward_credits(
  p_wallet_address text,
  p_credits numeric,
  p_credits_per_dollar numeric,
  p_compute_price_usd numeric
)
returns numeric
language plpgsql
security definer
set search_path = public
as $$
declare
  remaining_compute numeric;
  spent_compute numeric := 0;
  row record;
  spend_compute numeric;
begin
  if p_credits <= 0 then
    return 0;
  end if;

  if p_wallet_address is null or length(p_wallet_address) = 0 then
    raise exception 'wallet_address_required';
  end if;

  if p_credits_per_dollar <= 0 or p_compute_price_usd <= 0 then
    raise exception 'invalid_pricing';
  end if;

  remaining_compute := (p_credits / p_credits_per_dollar) / p_compute_price_usd;

  for row in
    select
      id,
      final_reward,
      spent_as_credits_usd,
      spent_as_credits_compute
    from public.reward_events
    where wallet_address = p_wallet_address
      and status = 'pending'
      and final_reward > 0
    order by created_at asc
    for update
  loop
    exit when remaining_compute <= 0.000000001;

    spend_compute := least(row.final_reward, remaining_compute);

    update public.reward_events
    set
      final_reward = greatest(final_reward - spend_compute, 0),
      spent_as_credits_compute = coalesce(spent_as_credits_compute, 0) + spend_compute,
      spent_as_credits_usd = coalesce(spent_as_credits_usd, 0) + (spend_compute * p_compute_price_usd),
      status = case
        when greatest(final_reward - spend_compute, 0) <= 0.000000001 then 'spent_credit'
        else status
      end,
      updated_at = now()
    where id = row.id;

    remaining_compute := remaining_compute - spend_compute;
    spent_compute := spent_compute + spend_compute;
  end loop;

  if remaining_compute > 0.000000001 then
    raise exception 'insufficient_reward_credits';
  end if;

  update public.nodes
  set pending_compute = coalesce((
    select sum(final_reward)
    from public.reward_events
    where wallet_address = p_wallet_address
      and status = 'pending'
  ), 0)
  where wallet_address = p_wallet_address;

  return spent_compute;
end;
$$;

revoke all on function public.spend_pending_reward_credits(text, numeric, numeric, numeric) from public;
grant execute on function public.spend_pending_reward_credits(text, numeric, numeric, numeric) to service_role;
