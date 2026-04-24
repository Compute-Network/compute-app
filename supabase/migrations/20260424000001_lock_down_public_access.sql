-- Emergency lockdown: these service-owned tables are accessed through the
-- orchestrator service role, not directly from public Supabase clients.

BEGIN;

DO $$
DECLARE
  locked_tables text[] := ARRAY[
    'accounts',
    'api_keys',
    'credit_transactions',
    'nodes',
    'pipeline_stages',
    'pipelines',
    'reward_events',
    'usage_daily'
  ];
  table_name text;
  policy_record record;
BEGIN
  -- Remove any broad policies that currently make these tables readable via
  -- the anon key. No replacement policies are created here intentionally.
  FOR policy_record IN
    SELECT schemaname, tablename, policyname
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = ANY (locked_tables)
  LOOP
    EXECUTE format(
      'DROP POLICY IF EXISTS %I ON %I.%I',
      policy_record.policyname,
      policy_record.schemaname,
      policy_record.tablename
    );
  END LOOP;

  FOREACH table_name IN ARRAY locked_tables
  LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
    EXECUTE format('REVOKE ALL ON TABLE public.%I FROM PUBLIC', table_name);
    EXECUTE format('REVOKE ALL ON TABLE public.%I FROM anon', table_name);
    EXECUTE format('REVOKE ALL ON TABLE public.%I FROM authenticated', table_name);
    EXECUTE format('GRANT ALL ON TABLE public.%I TO service_role', table_name);
  END LOOP;
END $$;

DO $$
DECLARE
  function_record record;
BEGIN
  -- These RPCs mutate billing and API-key counters. They should only be called
  -- by the orchestrator using the service role.
  FOR function_record IN
    SELECT p.oid::regprocedure AS signature
    FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public'
      AND p.proname IN (
        'deduct_credits',
        'increment_api_key_usage',
        'topup_credits'
      )
  LOOP
    EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC', function_record.signature);
    EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM anon', function_record.signature);
    EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM authenticated', function_record.signature);
    EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', function_record.signature);
  END LOOP;
END $$;

-- Guard against future migrations accidentally exposing new public objects by
-- default. Object-specific grants should be added deliberately.
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM anon;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE EXECUTE ON FUNCTIONS FROM anon;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE EXECUTE ON FUNCTIONS FROM authenticated;

COMMIT;
