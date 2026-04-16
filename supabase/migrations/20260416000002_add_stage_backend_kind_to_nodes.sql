-- Migration: add stage backend kind metadata to nodes
-- Supports daemon/orchestrator compatibility checks for staged inference backends.

ALTER TABLE nodes
  ADD COLUMN IF NOT EXISTS stage_backend_kind TEXT;

COMMENT ON COLUMN nodes.stage_backend_kind IS
  'Daemon-reported staged inference backend identifier, e.g. llama-stage-gateway.';
