import type { Node } from "../types/node.js";
import type { Pipeline, PipelineStage, PipelineAssignment } from "../types/pipeline.js";
import { getAvailableNodes, assignPipeline, releasePipeline } from "./nodes.js";
import { savePipeline, updatePipelineStatus, loadActivePipelines } from "./pipelines.js";

// Model definitions — mirrors crates/compute-network/src/models.rs
interface ModelDef {
  id: string;
  name: string;
  total_layers: number;
  vram_per_layer_mb: number;
}

const MODELS: Record<string, ModelDef> = {
  "gemma-4-26b-a4b-q4": {
    id: "gemma-4-26b-a4b-q4",
    name: "Gemma 4 26B-A4B (Q4 MoE)",
    total_layers: 48,
    vram_per_layer_mb: 350,
  },
  "llama-3.1-8b-q4": {
    id: "llama-3.1-8b-q4",
    name: "Llama 3.1 8B (Q4)",
    total_layers: 32,
    vram_per_layer_mb: 150,
  },
  "llama-3.1-70b-q4": {
    id: "llama-3.1-70b-q4",
    name: "Llama 3.1 70B (Q4)",
    total_layers: 80,
    vram_per_layer_mb: 450,
  },
  "llama-3.1-70b-fp16": {
    id: "llama-3.1-70b-fp16",
    name: "Llama 3.1 70B (FP16)",
    total_layers: 80,
    vram_per_layer_mb: 1750,
  },
  "deepseek-r1-671b-q4": {
    id: "deepseek-r1-671b-q4",
    name: "DeepSeek R1 671B (Q4 MoE)",
    total_layers: 60,
    vram_per_layer_mb: 5600,
  },
  "qwen-2.5-72b-q4": {
    id: "qwen-2.5-72b-q4",
    name: "Qwen 2.5 72B (Q4)",
    total_layers: 80,
    vram_per_layer_mb: 460,
  },
  "mistral-7b-q4": {
    id: "mistral-7b-q4",
    name: "Mistral 7B (Q4)",
    total_layers: 32,
    vram_per_layer_mb: 140,
  },
};

// Active pipelines in memory
const activePipelines = new Map<string, Pipeline>();

export function getModel(modelId: string): ModelDef | undefined {
  return MODELS[modelId];
}

export function listModels(): ModelDef[] {
  return Object.values(MODELS);
}

/**
 * Water-filling layer allocation.
 * Assigns layers proportional to each node's TFLOPS, respecting VRAM.
 */
function allocateLayers(
  nodes: Node[],
  model: ModelDef
): PipelineStage[] | null {
  if (nodes.length === 0) return null;

  // Sort by TFLOPS descending
  const sorted = [...nodes].sort(
    (a, b) => (b.tflops_fp16 ?? 0) - (a.tflops_fp16 ?? 0)
  );

  const totalTflops = sorted.reduce(
    (sum, n) => sum + (n.tflops_fp16 ?? 1),
    0
  );

  const stages: PipelineStage[] = [];
  let currentLayer = 0;

  for (let i = 0; i < sorted.length; i++) {
    const node = sorted[i];
    const isLast = i === sorted.length - 1;

    const proportion = (node.tflops_fp16 ?? 1) / totalTflops;
    const idealLayers = Math.round(model.total_layers * proportion);

    // VRAM constraint
    const maxByVram = Math.floor(
      (node.gpu_vram_mb ?? 0) / Math.max(model.vram_per_layer_mb, 1)
    );

    let layers: number;
    if (isLast) {
      layers = model.total_layers - currentLayer;
    } else {
      layers = Math.min(
        Math.max(idealLayers, 1),
        maxByVram,
        model.total_layers - currentLayer
      );
    }

    if (layers <= 0) continue;

    stages.push({
      node_id: node.id,
      wallet_address: node.wallet_address,
      start_layer: currentLayer,
      end_layer: currentLayer + layers - 1,
      listen_addr: `${node.ip_address ?? "0.0.0.0"}:${node.listen_port ?? 9090}`,
      tflops_fp16: node.tflops_fp16 ?? 0,
      estimated_latency_ms: 10, // Placeholder — measure in production
    });

    currentLayer += layers;
    if (currentLayer >= model.total_layers) break;
  }

  // Verify full coverage
  if (currentLayer < model.total_layers) return null;

  return stages;
}

/**
 * Try to form a pipeline for the given model.
 * Finds available nodes, allocates layers, and assigns them.
 */
export async function formPipeline(
  modelId: string
): Promise<Pipeline | null> {
  const model = MODELS[modelId];
  if (!model) return null;

  const minVram = model.vram_per_layer_mb; // At least 1 layer must fit

  // Clear stale pipeline assignments — nodes assigned to pipelines that no longer
  // exist in memory (e.g. after orchestrator restart) get freed automatically
  const activePipelineIds = new Set(activePipelines.keys());
  const { supabase } = await import("./db.js");
  const { data: staleNodes } = await supabase
    .from("nodes")
    .select("id, pipeline_id")
    .in("status", ["online", "idle"])
    .not("pipeline_id", "is", null);

  const toFree = (staleNodes ?? []).filter(
    (n) => !activePipelineIds.has(n.pipeline_id)
  );
  if (toFree.length > 0) {
    await supabase
      .from("nodes")
      .update({ pipeline_id: null, model_name: null, pipeline_stage: null, pipeline_total_stages: null })
      .in("id", toFree.map((n) => n.id));
    console.log(`[scheduler] Freed ${toFree.length} nodes with stale pipeline assignments`);
  }

  const available = await getAvailableNodes(minVram);

  if (available.length === 0) return null;

  const stages = allocateLayers(available, model);
  if (!stages || stages.length === 0) return null;

  const pipelineId = `pipe-${Date.now().toString(36)}`;
  const totalLatency = stages.reduce(
    (sum, s) => sum + s.estimated_latency_ms,
    0
  );

  const pipeline: Pipeline = {
    id: pipelineId,
    model_id: modelId,
    stages,
    total_layers: model.total_layers,
    estimated_latency_ms: totalLatency,
    status: "forming",
    created_at: new Date().toISOString(),
  };

  // Assign nodes to the pipeline in Supabase
  for (let i = 0; i < stages.length; i++) {
    await assignPipeline(
      stages[i].node_id,
      pipelineId,
      i,
      stages.length,
      model.name
    );
  }

  pipeline.status = "active";

  // Persist to Supabase
  try {
    await savePipeline(pipeline);
  } catch (e) {
    console.error("Failed to persist pipeline:", e);
  }

  activePipelines.set(pipelineId, pipeline);

  return pipeline;
}

/**
 * Get the pipeline assignment for a specific node.
 */
export function getAssignment(
  nodeId: string
): PipelineAssignment | null {
  for (const pipeline of activePipelines.values()) {
    const stageIndex = pipeline.stages.findIndex(
      (s) => s.node_id === nodeId
    );
    if (stageIndex === -1) continue;

    const stage = pipeline.stages[stageIndex];
    return {
      pipeline_id: pipeline.id,
      model_id: pipeline.model_id,
      start_layer: stage.start_layer,
      end_layer: stage.end_layer,
      total_layers: pipeline.total_layers,
      upstream_addr:
        stageIndex > 0
          ? pipeline.stages[stageIndex - 1].listen_addr
          : null,
      downstream_addr:
        stageIndex < pipeline.stages.length - 1
          ? pipeline.stages[stageIndex + 1].listen_addr
          : null,
      stage_index: stageIndex,
      total_stages: pipeline.stages.length,
    };
  }
  return null;
}

/**
 * Terminate a pipeline and release all nodes.
 */
export async function terminatePipeline(pipelineId: string): Promise<void> {
  await releasePipeline(pipelineId);
  await updatePipelineStatus(pipelineId, "terminated").catch(console.error);
  const pipeline = activePipelines.get(pipelineId);
  if (pipeline) {
    pipeline.status = "terminated";
    activePipelines.delete(pipelineId);
  }
}

/**
 * Initialize the scheduler — load active pipelines from Supabase.
 * Call this on server startup.
 */
export async function initScheduler(): Promise<void> {
  try {
    const { supabase } = await import("./db.js");

    // Terminate all stale pipelines and clear node assignments
    // On restart, we have no in-memory state, so all DB pipelines are stale
    await supabase
      .from("pipelines")
      .update({ status: "terminated", terminated_at: new Date().toISOString() })
      .in("status", ["forming", "active"]);

    await supabase
      .from("nodes")
      .update({
        pipeline_id: null,
        pipeline_stage: null,
        pipeline_total_stages: null,
        model_name: null,
      })
      .not("pipeline_id", "is", null);

    console.log("Cleaned up stale pipelines and node assignments");
  } catch (e) {
    console.error("Failed to init scheduler:", e);
  }
}

/**
 * List all active pipelines.
 */
export function listPipelines(): Pipeline[] {
  return Array.from(activePipelines.values());
}
