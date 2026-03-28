import { Hono } from "hono";
import * as scheduler from "../services/scheduler.js";

export const pipelinesRouter = new Hono();

// List available models
pipelinesRouter.get("/models", (c) => {
  const models = scheduler.listModels();
  return c.json({ models });
});

// Form a new pipeline for a model
pipelinesRouter.post("/form", async (c) => {
  const { model_id } = await c.req.json<{ model_id: string }>();

  if (!model_id) {
    return c.json({ error: "model_id is required" }, 400);
  }

  const model = scheduler.getModel(model_id);
  if (!model) {
    return c.json({ error: `Unknown model: ${model_id}` }, 404);
  }

  try {
    const pipeline = await scheduler.formPipeline(model_id);
    if (!pipeline) {
      return c.json(
        { error: "Not enough available nodes to form pipeline" },
        503
      );
    }
    return c.json({ pipeline }, 201);
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});

// Get assignment for a node
pipelinesRouter.get("/assignment/:nodeId", (c) => {
  const nodeId = c.req.param("nodeId");
  const assignment = scheduler.getAssignment(nodeId);

  if (!assignment) {
    return c.json(null, 404);
  }

  return c.json(assignment);
});

// List active pipelines
pipelinesRouter.get("/", (c) => {
  const pipelines = scheduler.listPipelines();
  return c.json({ pipelines, count: pipelines.length });
});

// Terminate a pipeline
pipelinesRouter.delete("/:pipelineId", async (c) => {
  const pipelineId = c.req.param("pipelineId");

  try {
    await scheduler.terminatePipeline(pipelineId);
    return c.json({ status: "terminated" });
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
});
