import { Hono } from "hono";
import { CompletionRequest } from "../types/pipeline.js";
import type { CompletionResponse } from "../types/pipeline.js";
import * as scheduler from "../services/scheduler.js";
import * as relay from "../services/relay.js";

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

  const firstStage = pipeline.stages[0];
  const nodeId = firstStage.node_id;

  // Build the inference request payload
  const inferenceBody = {
    model: req.model,
    messages: req.messages,
    prompt: req.prompt,
    max_tokens: req.max_tokens,
    temperature: req.temperature,
    top_p: req.top_p,
    stream: req.stream,
  };

  try {
    let result: any;

    if (relay.isConnected(nodeId)) {
      // Route through WebSocket relay
      const response = await relay.sendRequest(nodeId, inferenceBody);
      if (response.status !== 200) {
        return c.json(
          {
            error: {
              message: `Inference failed: ${JSON.stringify(response.body)}`,
              type: "server_error",
            },
          },
          502
        );
      }
      result = response.body;
    } else {
      // Fallback: direct HTTP (only works when orchestrator and node are co-located)
      const inferenceUrl = `http://127.0.0.1:8090/v1/chat/completions`;
      const inferenceResp = await fetch(inferenceUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inferenceBody),
      });

      if (!inferenceResp.ok) {
        const text = await inferenceResp.text();
        return c.json(
          {
            error: {
              message: `Inference failed: ${text}`,
              type: "server_error",
            },
          },
          502
        );
      }

      result = await inferenceResp.json();
    }

    // Record rewards and update node throughput
    const { recordRequestReward } = await import("../services/rewards.js");
    const usage = result.usage;
    const timings = result.timings;
    if (usage) {
      await recordRequestReward(
        pipeline,
        usage.completion_tokens ?? 0,
        usage.prompt_tokens ?? 0
      ).catch(console.error);
    }

    // Write tokens_per_second and requests_served to node in Supabase
    if (timings) {
      const tps = timings.predicted_per_second ?? 0;
      const { supabase } = await import("../services/db.js");
      try {
        const current = await supabase
          .from("nodes")
          .select("requests_served")
          .eq("id", firstStage.node_id)
          .single();

        await supabase
          .from("nodes")
          .update({
            tokens_per_second: Math.round(tps * 10) / 10,
            requests_served: (current.data?.requests_served ?? 0) + 1,
          })
          .eq("id", firstStage.node_id);
      } catch (e) {
        console.error("Failed to update node metrics:", e);
      }
    }

    return c.json(result);
  } catch (e: any) {
    return c.json(
      {
        error: {
          message: `Failed to reach inference node: ${e.message}`,
          type: "server_error",
        },
      },
      502
    );
  }
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
