import { supabase } from "./db.js";
import type { Pipeline, PipelineStage } from "../types/pipeline.js";

/**
 * Persist a pipeline and its stages to Supabase.
 */
export async function savePipeline(pipeline: Pipeline): Promise<void> {
  // Insert pipeline
  const { error: pipeErr } = await supabase.from("pipelines").insert({
    id: pipeline.id,
    model_id: pipeline.model_id,
    total_layers: pipeline.total_layers,
    total_stages: pipeline.stages.length,
    estimated_latency_ms: pipeline.estimated_latency_ms,
    status: pipeline.status,
  });

  if (pipeErr) throw new Error(`Failed to save pipeline: ${pipeErr.message}`);

  // Insert stages
  const stages = pipeline.stages.map((s, i) => ({
    pipeline_id: pipeline.id,
    node_id: s.node_id,
    stage_index: i,
    start_layer: s.start_layer,
    end_layer: s.end_layer,
  }));

  const { error: stageErr } = await supabase
    .from("pipeline_stages")
    .insert(stages);

  if (stageErr)
    throw new Error(`Failed to save pipeline stages: ${stageErr.message}`);
}

/**
 * Update pipeline status in Supabase.
 */
export async function updatePipelineStatus(
  pipelineId: string,
  status: Pipeline["status"]
): Promise<void> {
  const update: Record<string, any> = { status };
  if (status === "terminated") {
    update.terminated_at = new Date().toISOString();
  }

  const { error } = await supabase
    .from("pipelines")
    .update(update)
    .eq("id", pipelineId);

  if (error)
    throw new Error(`Failed to update pipeline status: ${error.message}`);
}

/**
 * Load all active pipelines from Supabase (for startup recovery).
 */
export async function loadActivePipelines(): Promise<Pipeline[]> {
  const { data: pipelines, error: pipeErr } = await supabase
    .from("pipelines")
    .select("*")
    .in("status", ["forming", "active"]);

  if (pipeErr)
    throw new Error(`Failed to load pipelines: ${pipeErr.message}`);
  if (!pipelines?.length) return [];

  const result: Pipeline[] = [];

  for (const p of pipelines) {
    const { data: stages, error: stageErr } = await supabase
      .from("pipeline_stages")
      .select("*, nodes!inner(wallet_address, ip_address, listen_port, tflops_fp16)")
      .eq("pipeline_id", p.id)
      .order("stage_index");

    if (stageErr) {
      console.error(
        `Failed to load stages for ${p.id}: ${stageErr.message}`
      );
      continue;
    }

    result.push({
      id: p.id,
      model_id: p.model_id,
      total_layers: p.total_layers,
      estimated_latency_ms: p.estimated_latency_ms,
      status: p.status,
      created_at: p.created_at,
      stages: (stages ?? []).map((s: any) => ({
        node_id: s.node_id,
        wallet_address: s.nodes?.wallet_address ?? "",
        start_layer: s.start_layer,
        end_layer: s.end_layer,
        listen_addr: `${s.nodes?.ip_address ?? "0.0.0.0"}:${s.nodes?.listen_port ?? 9090}`,
        tflops_fp16: s.nodes?.tflops_fp16 ?? 0,
        estimated_latency_ms: 10,
      })),
    });
  }

  return result;
}

/**
 * Get pipeline history (terminated pipelines).
 */
export async function getPipelineHistory(
  limit = 20
): Promise<any[]> {
  const { data, error } = await supabase
    .from("pipelines")
    .select("*")
    .eq("status", "terminated")
    .order("terminated_at", { ascending: false })
    .limit(limit);

  if (error)
    throw new Error(`Failed to get pipeline history: ${error.message}`);
  return data ?? [];
}
