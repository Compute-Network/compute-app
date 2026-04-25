-- Allow one wallet to operate multiple node identities.
-- Native nodes keep node_key = wallet_address; browser nodes use
-- browser:<wallet>:<browser-install-id> while rewards still accrue to wallet_address.

ALTER TABLE public.nodes
  ADD COLUMN IF NOT EXISTS node_key TEXT;

UPDATE public.nodes
SET node_key = wallet_address
WHERE node_key IS NULL;

ALTER TABLE public.nodes
  ALTER COLUMN node_key SET NOT NULL;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conrelid = 'public.nodes'::regclass
      AND conname = 'nodes_wallet_address_key'
  ) THEN
    ALTER TABLE public.nodes DROP CONSTRAINT nodes_wallet_address_key;
  END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS nodes_node_key_key
  ON public.nodes (node_key);

CREATE INDEX IF NOT EXISTS nodes_wallet_address_idx
  ON public.nodes (wallet_address);

COMMENT ON COLUMN public.nodes.node_key IS
  'Stable node identity key. Defaults to wallet_address for native nodes; browser nodes use browser:<wallet>:<browser-id>.';
