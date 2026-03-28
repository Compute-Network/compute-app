import { supabase } from "./db.js";
import type { Node, NodeRegistration, HeartbeatPayload } from "../types/node.js";

const STALE_THRESHOLD_SECONDS = 120; // Mark offline after 2 min without heartbeat

export async function registerNode(
  reg: NodeRegistration
): Promise<{ id: string }> {
  const { data, error } = await supabase
    .from("nodes")
    .upsert(
      {
        wallet_address: reg.wallet_address,
        node_name: reg.node_name,
        status: "online",
        gpu_model: reg.gpu_model,
        gpu_vram_mb: reg.gpu_vram_mb,
        gpu_backend: reg.gpu_backend,
        cpu_model: reg.cpu_model,
        cpu_cores: reg.cpu_cores,
        memory_mb: reg.memory_mb,
        os: reg.os,
        app_version: reg.app_version,
        region: reg.region,
        tflops_fp16: reg.tflops_fp16,
        listen_port: reg.listen_port,
        last_heartbeat: new Date().toISOString(),
      },
      { onConflict: "wallet_address" }
    )
    .select("id")
    .single();

  if (error) throw new Error(`Registration failed: ${error.message}`);
  return { id: data.id };
}

export async function heartbeat(
  walletAddress: string,
  payload: HeartbeatPayload
): Promise<void> {
  const { error } = await supabase
    .from("nodes")
    .update({
      ...payload,
      last_heartbeat: new Date().toISOString(),
    })
    .eq("wallet_address", walletAddress);

  if (error) throw new Error(`Heartbeat failed: ${error.message}`);
}

export async function getOnlineNodes(): Promise<Node[]> {
  const { data, error } = await supabase
    .from("nodes")
    .select("*")
    .eq("status", "online")
    .order("tflops_fp16", { ascending: false, nullsFirst: false });

  if (error) throw new Error(`Failed to fetch nodes: ${error.message}`);
  return data ?? [];
}

export async function getAvailableNodes(minVramMb?: number): Promise<Node[]> {
  let query = supabase
    .from("nodes")
    .select("*")
    .in("status", ["online", "idle"])
    .is("pipeline_id", null) // Not already in a pipeline
    .order("tflops_fp16", { ascending: false, nullsFirst: false });

  if (minVramMb) {
    query = query.gte("gpu_vram_mb", minVramMb);
  }

  const { data, error } = await query;
  if (error) throw new Error(`Failed to fetch available nodes: ${error.message}`);
  return data ?? [];
}

export async function assignPipeline(
  nodeId: string,
  pipelineId: string,
  stage: number,
  totalStages: number,
  modelName: string
): Promise<void> {
  const { error } = await supabase
    .from("nodes")
    .update({
      pipeline_id: pipelineId,
      pipeline_stage: stage,
      pipeline_total_stages: totalStages,
      model_name: modelName,
    })
    .eq("id", nodeId);

  if (error) throw new Error(`Pipeline assignment failed: ${error.message}`);
}

export async function releasePipeline(pipelineId: string): Promise<void> {
  const { error } = await supabase
    .from("nodes")
    .update({
      pipeline_id: null,
      pipeline_stage: null,
      pipeline_total_stages: null,
      model_name: null,
    })
    .eq("pipeline_id", pipelineId);

  if (error) throw new Error(`Pipeline release failed: ${error.message}`);
}

export async function markStaleNodesOffline(): Promise<number> {
  const cutoff = new Date(
    Date.now() - STALE_THRESHOLD_SECONDS * 1000
  ).toISOString();

  const { data, error } = await supabase
    .from("nodes")
    .update({ status: "offline" })
    .eq("status", "online")
    .lt("last_heartbeat", cutoff)
    .select("id");

  if (error) {
    console.error("Failed to mark stale nodes offline:", error.message);
    return 0;
  }

  return data?.length ?? 0;
}

export async function getNetworkStats(): Promise<{
  total: number;
  online: number;
  totalTflops: number;
}> {
  const { data, error } = await supabase
    .from("nodes")
    .select("status, tflops_fp16");

  if (error) throw new Error(`Failed to get stats: ${error.message}`);

  const nodes = data ?? [];
  const online = nodes.filter((n) => n.status === "online");
  const totalTflops = online.reduce(
    (sum, n) => sum + (n.tflops_fp16 ?? 0),
    0
  );

  return {
    total: nodes.length,
    online: online.length,
    totalTflops,
  };
}
