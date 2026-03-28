import { Hono } from "hono";
import { CompletionRequest } from "../types/pipeline.js";
import type { CompletionResponse } from "../types/pipeline.js";
import * as scheduler from "../services/scheduler.js";

export const completionsRouter = new Hono();

// OpenAI-compatible chat completions endpoint
completionsRouter.post("/chat/completions", async (c) => {
  const body = await c.req.json();
  const parsed = CompletionRequest.safeParse(body);

  if (!parsed.success) {
    return c.json(
      { error: { message: "Invalid request", type: "invalid_request_error" } },
      400
    );
  }

  const req = parsed.data;

  // Find or form a pipeline for this model
  const model = scheduler.getModel(req.model);
  if (!model) {
    return c.json(
      {
        error: {
          message: `Model '${req.model}' not found. Available: ${scheduler.listModels().map((m) => m.id).join(", ")}`,
          type: "invalid_request_error",
        },
      },
      404
    );
  }

  // Check for an active pipeline, or try to form one
  let pipelines = scheduler.listPipelines();
  let pipeline = pipelines.find((p) => p.model_id === req.model && p.status === "active");

  if (!pipeline) {
    pipeline = (await scheduler.formPipeline(req.model)) ?? undefined;
    if (!pipeline) {
      return c.json(
        {
          error: {
            message: "No nodes available to serve this model. Try again later.",
            type: "server_error",
          },
        },
        503
      );
    }
  }

  // Build the prompt from messages
  const prompt = req.messages
    ? req.messages.map((m) => `${m.role}: ${m.content}`).join("\n")
    : req.prompt ?? "";

  // TODO: Actually route the request through the pipeline stages.
  // For now, return a placeholder response showing the pipeline is ready.
  const response: CompletionResponse = {
    id: `cmpl-${Date.now().toString(36)}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: req.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: `[Pipeline ${pipeline.id} active with ${pipeline.stages.length} stages serving ${model.name}. Inference routing not yet implemented.]`,
        },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: prompt.split(/\s+/).length,
      completion_tokens: 0,
      total_tokens: prompt.split(/\s+/).length,
    },
  };

  return c.json(response);
});

// List available models (OpenAI-compatible)
completionsRouter.get("/models", (c) => {
  const models = scheduler.listModels().map((m) => ({
    id: m.id,
    object: "model",
    created: 0,
    owned_by: "compute-network",
  }));

  return c.json({ object: "list", data: models });
});
