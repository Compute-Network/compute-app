import { z } from "zod";

export const GpuBackend = z.enum(["cuda", "metal", "cpu"]);
export type GpuBackend = z.infer<typeof GpuBackend>;

export const NodeStatus = z.enum(["online", "offline", "idle", "paused"]);
export type NodeStatus = z.infer<typeof NodeStatus>;

export const NodeRegistration = z.object({
  wallet_address: z.string().min(32).max(44),
  node_name: z.string().optional(),
  gpu_model: z.string().optional(),
  gpu_vram_mb: z.number().int().optional(),
  gpu_backend: GpuBackend.optional(),
  cpu_model: z.string().optional(),
  cpu_cores: z.number().int().optional(),
  memory_mb: z.number().int().optional(),
  os: z.string().optional(),
  app_version: z.string().optional(),
  region: z.string().optional(),
  tflops_fp16: z.number().optional(),
  listen_port: z.number().int().optional(),
});
export type NodeRegistration = z.infer<typeof NodeRegistration>;

export const HeartbeatPayload = z.object({
  status: NodeStatus,
  cpu_usage_percent: z.number().optional(),
  gpu_usage_percent: z.number().optional(),
  gpu_temp_celsius: z.number().optional(),
  memory_used_mb: z.number().int().optional(),
  idle_state: z.string().optional(),
  uptime_seconds: z.number().int().optional(),
  pipeline_id: z.string().optional(),
  pipeline_stage: z.number().int().optional(),
  requests_served: z.number().int().optional(),
  tokens_per_second: z.number().optional(),
  downloaded_models: z.string().optional(),
});
export type HeartbeatPayload = z.infer<typeof HeartbeatPayload>;

export interface Node {
  id: string;
  wallet_address: string;
  node_name: string | null;
  status: NodeStatus;
  gpu_model: string | null;
  gpu_vram_mb: number | null;
  gpu_backend: GpuBackend | null;
  cpu_model: string | null;
  cpu_cores: number | null;
  memory_mb: number | null;
  os: string | null;
  app_version: string | null;
  region: string | null;
  tflops_fp16: number | null;
  listen_port: number | null;
  ip_address: string | null;
  cpu_usage_percent: number | null;
  gpu_usage_percent: number | null;
  gpu_temp_celsius: number | null;
  memory_used_mb: number | null;
  idle_state: string | null;
  uptime_seconds: number | null;
  pipeline_id: string | null;
  pipeline_stage: number | null;
  pipeline_total_stages: number | null;
  model_name: string | null;
  requests_served: number;
  tokens_per_second: number | null;
  downloaded_models: string | null;
  total_earned_compute: number;
  pending_compute: number;
  last_heartbeat: string | null;
  created_at: string;
  updated_at: string;
  user_id: string | null;
}
