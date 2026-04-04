import { Hono } from "hono";
import { stream as honoStream } from "hono/streaming";
import { CompletionRequest } from "../types/pipeline.js";
import type { Pipeline } from "../types/pipeline.js";
import * as scheduler from "../services/scheduler.js";
import * as relay from "../services/relay.js";
import { recordUsage } from "../services/apikeys.js";
import type { RelayResponse } from "../services/relay.js";

export const completionsRouter = new Hono();

const MAX_RETRIES = 2;

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Find a usable pipeline for the model. If an existing active pipeline's
 * first-stage node is disconnected or circuit-broken, terminate it and
 * form a fresh one.
 */
async function findOrFormPipeline(modelId: string): Promise<Pipeline | null> {
  const pipelines = scheduler.listPipelines();

  // Check for an existing active pipeline for this model
  const existing = pipelines.find(
    (p) => p.model_id === modelId && p.status === "active"
  );

  if (existing) {
    const firstNode = existing.stages[0].node_id;
    if (relay.isConnected(firstNode) && !relay.isCircuitOpen(firstNode)) {
      return existing;
    }
    console.warn(
      `[completions] Pipeline ${existing.id} stale (node ${firstNode.slice(0, 8)} unreachable) — reforming`
    );
    await scheduler.terminatePipeline(existing.id).catch(console.error);
  }

  // Try to form a new pipeline
  let pipeline = await scheduler.formPipeline(modelId);
  if (pipeline) return pipeline;

  // No nodes available — they might be locked in pipelines for OTHER models.
  // Terminate those pipelines to free up nodes, then retry.
  const otherPipelines = pipelines.filter(
    (p) => p.model_id !== modelId && p.status === "active"
  );
  if (otherPipelines.length > 0) {
    console.log(
      `[completions] No free nodes for ${modelId} — releasing ${otherPipelines.length} pipeline(s) for other models`
    );
    for (const p of otherPipelines) {
      await scheduler.terminatePipeline(p.id).catch(console.error);
    }
    // Retry after freeing nodes
    pipeline = await scheduler.formPipeline(modelId);
  }

  return pipeline;
}

/**
 * Attempt a single non-streaming inference request against a pipeline.
 * Returns the result or throws on failure.
 */
async function attemptInference(
  pipeline: Pipeline,
  inferenceBody: Record<string, unknown>
): Promise<any> {
  const nodeId = pipeline.stages[0].node_id;

  if (relay.isConnected(nodeId)) {
    const response = await relay.sendRequest(nodeId, inferenceBody);
    if (response.status !== 200) {
      const errBody = response.body as any;
      const errMsg = errBody?.error?.message ?? `Inference returned status ${response.status}`;
      throw new Error(errMsg);
    }
    return response.body;
  }

  // Fallback: direct HTTP to local llama-server
  const inferenceUrl = `http://127.0.0.1:8090/v1/chat/completions`;
  const inferenceResp = await fetch(inferenceUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inferenceBody),
  });

  if (!inferenceResp.ok) {
    const text = await inferenceResp.text().catch(() => "");
    throw new Error(`Local inference failed (${inferenceResp.status}): ${text.slice(0, 200)}`);
  }

  return inferenceResp.json();
}

// ── Main endpoint ─────────────────────────────────────────────────────────────

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

  // Resolve "auto" model to the best available concrete model
  let resolvedModelId = req.model;
  if (resolvedModelId === "auto") {
    const autoModel = await scheduler.resolveAutoModel();
    if (!autoModel) {
      return c.json(
        { error: { message: "No nodes available for auto model selection", type: "server_error" } },
        503
      );
    }
    resolvedModelId = autoModel;
    console.log(`[completions] Auto-resolved model to: ${resolvedModelId}`);
  }

  // Validate model exists
  const model = scheduler.getModel(resolvedModelId);
  if (!model) {
    return c.json(
      {
        error: {
          message: `Model '${resolvedModelId}' not found. Available: ${scheduler.listModels().map((m) => m.id).join(", ")}`,
          type: "invalid_request_error",
        },
      },
      404
    );
  }

  // API key ID and account ID set by auth middleware
  let apiKeyId: string | undefined;
  let accountId: string | undefined;
  try { apiKeyId = (c as any).get("apiKeyId"); } catch {}
  try { accountId = (c as any).get("accountId"); } catch {}

  // Pre-flight credit check
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
    }
  }

  // Build the inference request payload
  const inferenceBody: Record<string, unknown> = {
    model: resolvedModelId,
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
    // For streaming, we do one attempt with pipeline health check
    const pipeline = await findOrFormPipeline(resolvedModelId);
    if (!pipeline) {
      return c.json(
        { error: { message: "No nodes available to serve this model. Try again later.", type: "server_error" } },
        503
      );
    }
    return handleStreamingRequest(c, pipeline, pipeline.stages[0].node_id, inferenceBody, apiKeyId, accountId, resolvedModelId);
  }

  // ── Non-streaming path with retry ──────────────────────────────
  let lastError: string = "Unknown error";

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const pipeline = await findOrFormPipeline(resolvedModelId);
    if (!pipeline) {
      return c.json(
        { error: { message: "No nodes available to serve this model. Try again later.", type: "server_error" } },
        503
      );
    }

    try {
      const result = await attemptInference(pipeline, inferenceBody);

      // Mark pipeline as recently used
      scheduler.touchPipeline(pipeline.id);

      // Success — post-process (non-blocking)
      postProcess(pipeline, pipeline.stages[0].node_id, result, apiKeyId, accountId, resolvedModelId).catch(console.error);

      return c.json(result);
    } catch (e: any) {
      lastError = e.message;
      console.warn(
        `[completions] Attempt ${attempt + 1}/${MAX_RETRIES + 1} failed: ${lastError}`
      );

      // Terminate the broken pipeline so next attempt gets a fresh one
      await scheduler.terminatePipeline(pipeline.id).catch(console.error);

      if (attempt < MAX_RETRIES) {
        // Brief pause before retry to let the scheduler find new nodes
        await new Promise((r) => setTimeout(r, 500));
      }
    }
  }

  return c.json(
    { error: { message: `Inference failed after ${MAX_RETRIES + 1} attempts: ${lastError}`, type: "server_error" } },
    502
  );
});

// ── Streaming handler ─────────────────────────────────────────────────────────

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
        const finalResponse: RelayResponse = await relay.sendStreamingRequest(
          nodeId,
          inferenceBody,
          (chunk: string) => {
            stream.write(chunk).catch(() => {});
          }
        );

        const usage = (finalResponse.body as any)?.usage;
        if (usage) {
          totalCompletionTokens = usage.completion_tokens ?? 0;
          totalPromptTokens = usage.prompt_tokens ?? 0;
        }

        await stream.write("data: [DONE]\n\n");

      } else {
        // Direct HTTP path
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

        const reader = inferenceResp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              await stream.write(line + "\n\n");

              const payload = line.slice(6).trim();
              if (payload !== "[DONE]") {
                try {
                  const chunk = JSON.parse(payload);
                  if (chunk.usage) {
                    totalCompletionTokens = chunk.usage.completion_tokens ?? totalCompletionTokens;
                    totalPromptTokens = chunk.usage.prompt_tokens ?? totalPromptTokens;
                  }
                  if (chunk.choices?.[0]?.delta?.content) {
                    totalCompletionTokens++;
                  }
                } catch {}
              }
            }
          }
        }

        if (buffer.trim()) {
          await stream.write(buffer + "\n\n");
        }
      }

      // Post-process
      const mockResult = {
        usage: {
          completion_tokens: totalCompletionTokens,
          prompt_tokens: totalPromptTokens,
        },
      };
      await postProcess(pipeline, pipeline.stages[0].node_id, mockResult, apiKeyId, accountId, modelId);

    } catch (e: any) {
      console.error("[stream] Error:", e.message);
      // Terminate broken pipeline so next request gets a fresh one
      scheduler.terminatePipeline(pipeline.id).catch(console.error);
      await stream.write(`data: ${JSON.stringify({ error: "Stream interrupted. Please retry." })}\n\n`);
      await stream.write("data: [DONE]\n\n");
    }
  });
}

// ── Post-processing ───────────────────────────────────────────────────────────

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
    // Run rewards and usage tracking in parallel (non-blocking)
    const totalTokens = (usage.completion_tokens ?? 0) + (usage.prompt_tokens ?? 0);

    const tasks: Promise<void>[] = [
      recordRequestReward(
        pipeline,
        usage.completion_tokens ?? 0,
        usage.prompt_tokens ?? 0
      ).catch((e) => console.error("[rewards] Record failed:", e.message)),
    ];

    if (apiKeyId) {
      tasks.push(
        recordUsage(apiKeyId, totalTokens).catch((e) =>
          console.error("[usage] Record failed:", e.message)
        )
      );
    }

    // Billing is critical — await it separately
    if (accountId && apiKeyId) {
      try {
        const { deductCredits } = await import("../services/billing.js");
        await deductCredits(accountId, totalTokens, apiKeyId, modelId);
      } catch (err: any) {
        console.error(`[billing] DEDUCTION FAILED: account=${accountId} tokens=${totalTokens} error=${err.message}`);
      }
    }

    // Wait for non-critical tasks (but don't fail the response)
    await Promise.allSettled(tasks);
  }

  if (timings) {
    const tps = timings.predicted_per_second ?? 0;
    const { supabase } = await import("../services/db.js");
    try {
      // Use atomic increment instead of read-then-write (race condition fix)
      await supabase.rpc("increment_requests_served", {
        p_node_id: nodeId,
        p_tps: Math.round(tps * 10) / 10,
      }).then(({ error }) => {
        // Fallback if RPC doesn't exist yet
        if (error) {
          return supabase
            .from("nodes")
            .update({
              tokens_per_second: Math.round(tps * 10) / 10,
            })
            .eq("id", nodeId);
        }
      });
    } catch (e) {
      console.error("Failed to update node metrics:", e);
    }
  }
}

// ── Account info (for splash screen) ──────────────────────────────────────────

completionsRouter.get("/account/info", async (c) => {
  const accountId = (c as any).get("accountId") as string | undefined;
  const wallet = (c as any).get("apiKeyWallet") as string | undefined;

  let credits = 0;
  if (accountId) {
    try {
      const { getBalance } = await import("../services/billing.js");
      const balance = await getBalance(accountId);
      credits = balance.total ?? 0;
    } catch {}
  }

  let nodesOnline = 0;
  try {
    const { getNetworkStats } = await import("../services/nodes.js");
    const stats = await getNetworkStats();
    nodesOnline = stats.online;
  } catch {}

  return c.json({
    wallet_address: wallet ?? null,
    credits,
    nodes_online: nodesOnline,
  });
});

// ── Models list ───────────────────────────────────────────────────────────────

completionsRouter.get("/models", (c) => {
  const models = [
    { id: "auto", object: "model", created: 0, owned_by: "compute-network" },
    ...scheduler.listModels().map((m) => ({
      id: m.id,
      object: "model",
      created: 0,
      owned_by: "compute-network",
    })),
  ];

  return c.json({ object: "list", data: models });
});
