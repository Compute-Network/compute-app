create extension if not exists pgcrypto;

create table if not exists public.dashboard_chat_sessions (
  id uuid primary key default gen_random_uuid(),
  account_id uuid not null references public.accounts(id) on delete cascade,
  wallet_address text,
  title text not null default 'New chat',
  model_id text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.dashboard_chat_messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.dashboard_chat_sessions(id) on delete cascade,
  account_id uuid not null references public.accounts(id) on delete cascade,
  role text not null check (role in ('system', 'user', 'assistant')),
  content text not null default '',
  reasoning text,
  meta text,
  model_id text,
  created_at timestamptz not null default now()
);

create index if not exists dashboard_chat_sessions_account_updated_idx
  on public.dashboard_chat_sessions (account_id, updated_at desc);

create index if not exists dashboard_chat_messages_session_created_idx
  on public.dashboard_chat_messages (session_id, created_at asc);

create index if not exists dashboard_chat_messages_account_created_idx
  on public.dashboard_chat_messages (account_id, created_at desc);

create or replace function public.touch_dashboard_chat_session()
returns trigger
language plpgsql
as $$
begin
  update public.dashboard_chat_sessions
  set updated_at = now()
  where id = new.session_id;
  return new;
end;
$$;

drop trigger if exists dashboard_chat_messages_touch_session
  on public.dashboard_chat_messages;

create trigger dashboard_chat_messages_touch_session
after insert on public.dashboard_chat_messages
for each row
execute function public.touch_dashboard_chat_session();

alter table public.dashboard_chat_sessions enable row level security;
alter table public.dashboard_chat_messages enable row level security;

revoke all on public.dashboard_chat_sessions from anon, authenticated;
revoke all on public.dashboard_chat_messages from anon, authenticated;
grant select, insert, update, delete on public.dashboard_chat_sessions to service_role;
grant select, insert, update, delete on public.dashboard_chat_messages to service_role;
