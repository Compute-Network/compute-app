-- Migration: add placement fields to nodes
-- Supports the topology-aware placement engine in orchestrator/src/services/placement.ts
--
-- pipeline_capable:      node has advertised support for pipeline stage execution
-- memory_bandwidth_gbps: reported bandwidth; placement engine derives it from
--                        gpu_backend + vram if NULL

ALTER TABLE nodes
  ADD COLUMN IF NOT EXISTS pipeline_capable       BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS memory_bandwidth_gbps  FLOAT;

COMMENT ON COLUMN nodes.pipeline_capable IS
  'Node supports pipeline stage execution (stage-forward backend). '
  'Set via NodeRegistration.pipeline_capable on daemon startup.';

COMMENT ON COLUMN nodes.memory_bandwidth_gbps IS
  'Memory bandwidth in GB/s reported by the daemon, or NULL for engine-derived estimate. '
  'Used by placement engine to model per-stage decode compute time.';

-- Partial index: fast lookup of pipeline-capable online nodes
CREATE INDEX IF NOT EXISTS nodes_pipeline_capable_status_idx
  ON nodes (pipeline_capable, status)
  WHERE pipeline_capable = true;
