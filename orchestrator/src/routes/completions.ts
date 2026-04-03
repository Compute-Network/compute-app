import { Hono } from "hono";
import { stream as honoStream } from "hono/streaming";
import { CompletionRequest } from "../types/pipeline.js";
import type { Pipeline } from "../types/pipeline.js";
import * as scheduler from "../services/scheduler.js";
import * as relay from "../services/relay.js";
import { recordUsage } from "../services/apikeys.js";
import type { RelayResponse } from "../services/relay.js";

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
  // API key ID and account ID set by auth middleware
  let apiKeyId: string | undefined;
  let accountId: string | undefined;
  try { apiKeyId = (c as any).get("apiKeyId"); } catch {}
  try { accountId = (c as any).get("accountId"); } catch {}

  // Pre-flight credit check — reject if account has no credits
  if (accountId) {
    try {
      const { getBalance } = await import("../services/billing.js");
      const balance = await getBalance(accountId);
      if (balance.total <= 0) {
        return c.json(
          { error: { message: "Insufficient credits. Top up at computenetwork.sh/dashboard", type: "insufficient_credits" } },
          402
        );
      }
    } catch (e: any) {
      console.error("[billing] Pre-flight check failed:", e.message);
      // Don't block on billing check failures — let inference proceed
    }
  }

  // Build the inference request payload
  const inferenceBody: Record<string, unknown> = {
    model: req.model,
    messages: req.messages,
    prompt: req.prompt,
    max_tokens: req.max_tokens,
    temperature: req.temperature,
    top_p: req.top_p,
    stream: req.stream,
  };
  if (req.tools) inferenceBody.tools = req.tools;
  if (req.tool_choice) inferenceBody.tool_choice = req.tool_choice;

  // ── Streaming path ──────────────────────────────────────────────
  if (req.stream) {
    return handleStreamingRequest(c, pipeline, nodeId, inferenceBody, apiKeyId, accountId, req.model);
  }

  // ── Non-streaming path ──────────────────────────────────────────
  try {
    let result: any;

    if (relay.isConnected(nodeId)) {
      const response = await relay.sendRequest(nodeId, inferenceBody);
      if (response.status !== 200) {
        return c.json(
          { error: { message: "Inference failed", type: "server_error" } },
          502
        );
      }
      result = response.body;
    } else {
      const inferenceUrl = `http://127.0.0.1:8090/v1/chat/completions`;
      const inferenceResp = await fetch(inferenceUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inferenceBody),
      });

      if (!inferenceResp.ok) {
        return c.json(
          { error: { message: "Inference failed", type: "server_error" } },
          502
        );
      }

      result = await inferenceResp.json();
    }

    // Post-processing: rewards + metrics + billing (non-blocking — never fail the response)
    postProcess(pipeline, firstStage.node_id, result, apiKeyId, accountId, req.model).catch(console.error);

    return c.json(result);
  } catch (e: any) {
    return c.json(
      { error: { message: "Failed to reach inference node", type: "server_error" } },
      502
    );
  }
});

/**
 * Handle SSE streaming response.
 * Proxies the llama-server SSE stream back to the client.
 */
function handleStreamingRequest(
  c: any,
  pipeline: Pipeline,
  nodeId: string,
  inferenceBody: any,
  apiKeyId: string | undefined,
  accountId: string | undefined,
  modelId: string
) {
  c.header("Content-Type", "text/event-stream");
  c.header("Cache-Control", "no-cache");
  c.header("Connection", "keep-alive");

  return honoStream(c, async (stream) => {
    let totalCompletionTokens = 0;
    let totalPromptTokens = 0;

    try {
      if (relay.isConnected(nodeId)) {
        // Real WebSocket streaming: daemon sends SSE chunks as they arrive
        const finalResponse: RelayResponse = await relay.sendStreamingRequest(
          nodeId,
          inferenceBody,
          (chunk: string) => {
            // Forward SSE chunk directly to client
            stream.write(chunk).catch(() => {});
          }
        );

        // Extract usage from final response
        const usage = (finalResponse.body as any)?.usage;
        if (usage) {
          totalCompletionTokens = usage.completion_tokens ?? 0;
          totalPromptTokens = usage.prompt_tokens ?? 0;
        }

        // Send [DONE] if daemon didn't already
        await stream.write("data: [DONE]\n\n");

      } else {
        // Direct HTTP path: true SSE proxy from llama-server
        const inferenceUrl = `http://127.0.0.1:8090/v1/chat/completions`;
        const inferenceResp = await fetch(inferenceUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(inferenceBody),
        });

        if (!inferenceResp.ok || !inferenceResp.body) {
          await stream.write(`data: ${JSON.stringify({ error: "Inference failed" })}\n\n`);
          await stream.write("data: [DONE]\n\n");
          return;
        }

        // Pipe llama-server SSE stream directly to client
        const reader = inferenceResp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE lines
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              await stream.write(line + "\n\n");

              // Parse for usage tracking
              const payload = line.slice(6).trim();
              if (payload !== "[DONE]") {
                try {
                  const chunk = JSON.parse(payload);
                  if (chunk.usage) {
                    totalCompletionTokens = chunk.usage.completion_tokens ?? totalCompletionTokens;
                    totalPromptTokens = chunk.usage.prompt_tokens ?? totalPromptTokens;
                  }
                  // Count tokens from deltas
                  if (chunk.choices?.[0]?.delta?.content) {
                    totalCompletionTokens++;
                  }
                } catch {}
              }
            }
          }
        }

        // Flush remaining buffer
        if (buffer.trim()) {
          await stream.write(buffer + "\n\n");
        }
      }

      // Post-process: rewards + API key usage + billing
      const mockResult = {
        usage: {
          completion_tokens: totalCompletionTokens,
          prompt_tokens: totalPromptTokens,
        },
      };
      await postProcess(pipeline, pipeline.stages[0].node_id, mockResult, apiKeyId, accountId, modelId);

    } catch (e: any) {
      console.error("[stream] Error:", e.message);
      await stream.write(`data: ${JSON.stringify({ error: "Stream failed" })}\n\n`);
      await stream.write("data: [DONE]\n\n");
    }
  });
}

/**
 * Post-process: record rewards, update node metrics, track API key usage.
 */
async function postProcess(
  pipeline: Pipeline,
  nodeId: string,
  result: any,
  apiKeyId: string | undefined,
  accountId: string | undefined,
  modelId: string
): Promise<void> {
  const { recordRequestReward } = await import("../services/rewards.js");
  const usage = result.usage;
  const timings = result.timings;

  if (usage) {
    await recordRequestReward(
      pipeline,
      usage.completion_tokens ?? 0,
      usage.prompt_tokens ?? 0
    ).catch(console.error);

    const totalTokens = (usage.completion_tokens ?? 0) + (usage.prompt_tokens ?? 0);

    // Track API key usage
    if (apiKeyId) {
      recordUsage(apiKeyId, totalTokens).catch(console.error);
    }

    // Deduct credits (awaited — ensures billing consistency)
    if (accountId && apiKeyId) {
      try {
        const { deductCredits } = await import("../services/billing.js");
        await deductCredits(accountId, totalTokens, apiKeyId, modelId);
      } catch (err: any) {
        // Log as error, not warning — billing failures need attention
        console.error(`[billing] DEDUCTION FAILED: account=${accountId} tokens=${totalTokens} error=${err.message}`);
      }
    }
  }

  if (timings) {
    const tps = timings.predicted_per_second ?? 0;
    const { supabase } = await import("../services/db.js");
    try {
      const current = await supabase
        .from("nodes")
        .select("requests_served")
        .eq("id", nodeId)
        .single();

      await supabase
        .from("nodes")
        .update({
          tokens_per_second: Math.round(tps * 10) / 10,
          requests_served: (current.data?.requests_served ?? 0) + 1,
        })
        .eq("id", nodeId);
    } catch (e) {
      console.error("Failed to update node metrics:", e);
    }
  }
}

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
