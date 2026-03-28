import { z } from "zod";

export interface PipelineStage {
  node_id: string;
  wallet_address: string;
  start_layer: number;
  end_layer: number;
  listen_addr: string;
  tflops_fp16: number;
  estimated_latency_ms: number;
}

export interface Pipeline {
  id: string;
  model_id: string;
  stages: PipelineStage[];
  total_layers: number;
  estimated_latency_ms: number;
  status: "forming" | "active" | "draining" | "terminated";
  created_at: string;
}

export interface PipelineAssignment {
  pipeline_id: string;
  model_id: string;
  start_layer: number;
  end_layer: number;
  total_layers: number;
  upstream_addr: string | null;
  downstream_addr: string | null;
  stage_index: number;
  total_stages: number;
}

export const CompletionRequest = z.object({
  model: z.string(),
  prompt: z.string().optional(),
  messages: z
    .array(
      z.object({
        role: z.enum(["system", "user", "assistant"]),
        content: z.string(),
      })
    )
    .optional(),
  max_tokens: z.number().int().default(256),
  temperature: z.number().default(0.7),
  top_p: z.number().default(0.9),
  stream: z.boolean().default(false),
});
export type CompletionRequest = z.infer<typeof CompletionRequest>;

export interface CompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: "assistant";
      content: string;
    };
    finish_reason: "stop" | "length";
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
