alter table if exists public.nodes
  add column if not exists auto_download_enabled boolean not null default true,
  add column if not exists current_backend text,
  add column if not exists network_down_mbps numeric,
  add column if not exists caffeinated boolean;
