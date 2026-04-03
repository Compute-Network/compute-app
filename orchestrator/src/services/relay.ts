import type { WSContext } from "hono/ws";

// Active WebSocket connections keyed by node_id
const connections = new Map<string, WSContext>();

// Pending request promises keyed by request ID
const pending = new Map<
  string,
  {
    resolve: (value: RelayResponse) => void;
    reject: (reason: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }
>();

export interface RelayRequest {
  id: string;
  type: "inference_request";
  method: string;
  path: string;
  body: unknown;
}

export interface RelayResponse {
  id: string;
  type: "inference_response";
  status: number;
  body: unknown;
}

export interface RelayStreamChunk {
  id: string;
  type: "inference_stream_chunk";
  chunk: string;
}

// Stream callbacks keyed by request ID
const streamCallbacks = new Map<string, (chunk: string) => void>();

// Per-node request queue — serializes requests to avoid slot contention
const nodeQueues = new Map<string, { busy: boolean; queue: Array<() => void> }>();

function getNodeQueue(nodeId: string) {
  let q = nodeQueues.get(nodeId);
  if (!q) {
    q = { busy: false, queue: [] };
    nodeQueues.set(nodeId, q);
  }
  return q;
}

function acquireSlot(nodeId: string): Promise<void> {
  const q = getNodeQueue(nodeId);
  if (!q.busy) {
    q.busy = true;
    console.log(`[relay-queue] ${nodeId.slice(0, 8)}: slot acquired (was free)`);
    return Promise.resolve();
  }
  console.log(`[relay-queue] ${nodeId.slice(0, 8)}: queued (${q.queue.length + 1} waiting)`);
  return new Promise<void>((resolve) => {
    q.queue.push(resolve);
  });
}

function releaseSlot(nodeId: string): void {
  const q = getNodeQueue(nodeId);
  const next = q.queue.shift();
  if (next) {
    console.log(`[relay-queue] ${nodeId.slice(0, 8)}: slot handed to next (${q.queue.length} remaining)`);
    next();
  } else {
    q.busy = false;
    console.log(`[relay-queue] ${nodeId.slice(0, 8)}: slot released (idle)`);
  }
}

let requestCounter = 0;

function generateRequestId(): string {
  return `req-${Date.now()}-${++requestCounter}`;
}

export function registerConnection(nodeId: string, ws: WSContext): void {
  // Close any existing connection for this node
  const existing = connections.get(nodeId);
  if (existing) {
    try {
      existing.close(1000, "replaced");
    } catch (_) {}
  }
  connections.set(nodeId, ws);
  console.log(`[relay] Node ${nodeId} connected (${connections.size} total)`);
}

export function removeConnection(nodeId: string): void {
  connections.delete(nodeId);
  console.log(`[relay] Node ${nodeId} disconnected (${connections.size} total)`);

  // Reject all pending requests for this node
  for (const [reqId, entry] of pending) {
    if (reqId.startsWith(`req-`)) {
      // We can't easily map request IDs to nodes, so we don't reject here.
      // The timeout will handle it.
    }
  }
}

export function isConnected(nodeId: string): boolean {
  return connections.has(nodeId);
}

export function getConnectedCount(): number {
  return connections.size;
}

/**
 * Send a streaming inference request through the WebSocket relay.
 * The onChunk callback receives raw SSE lines as they arrive.
 * Returns the final aggregated response.
 */
export async function sendStreamingRequest(
  nodeId: string,
  body: unknown,
  onChunk: (chunk: string) => void,
  path: string = "/v1/chat/completions",
  timeoutMs: number = 120_000
): Promise<RelayResponse> {
  // Queue: wait for the node's slot to be free
  await acquireSlot(nodeId);

  try {
    const ws = connections.get(nodeId);
    if (!ws) throw new Error(`Node ${nodeId} not connected`);

    const id = generateRequestId();
    const request: RelayRequest = {
      id,
      type: "inference_request",
      method: "POST",
      path,
      body,
    };

    // Register stream callback
    streamCallbacks.set(id, onChunk);

    return await new Promise<RelayResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        streamCallbacks.delete(id);
        reject(new Error(`Streaming request ${id} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      pending.set(id, { resolve, reject, timer });

      try {
        ws.send(JSON.stringify(request));
      } catch (e: any) {
        pending.delete(id);
        streamCallbacks.delete(id);
        clearTimeout(timer);
        reject(new Error(`Failed to send to node ${nodeId}: ${e.message}`));
      }
    });
  } finally {
    releaseSlot(nodeId);
  }
}

export async function sendRequest(
  nodeId: string,
  body: unknown,
  path: string = "/v1/chat/completions",
  timeoutMs: number = 120_000
): Promise<RelayResponse> {
  // Queue: wait for the node's slot to be free
  await acquireSlot(nodeId);

  try {
    const ws = connections.get(nodeId);
    if (!ws) {
      throw new Error(`Node ${nodeId} not connected`);
    }

    const id = generateRequestId();
    const request: RelayRequest = {
      id,
      type: "inference_request",
      method: "POST",
      path,
      body,
    };

    return await new Promise<RelayResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`Request ${id} to node ${nodeId} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      pending.set(id, { resolve, reject, timer });

      try {
        ws.send(JSON.stringify(request));
      } catch (e: any) {
        pending.delete(id);
        clearTimeout(timer);
        reject(new Error(`Failed to send to node ${nodeId}: ${e.message}`));
      }
    });
  } finally {
    releaseSlot(nodeId);
  }
}

export function handleMessage(nodeId: string, data: string): void {
  let msg: any;
  try {
    msg = JSON.parse(data);
  } catch {
    console.error(`[relay] Invalid JSON from node ${nodeId}`);
    return;
  }

  if (msg.type === "inference_stream_chunk") {
    const callback = streamCallbacks.get(msg.id);
    if (callback) {
      callback(msg.chunk);
    }
  } else if (msg.type === "inference_response") {
    const entry = pending.get(msg.id);
    if (entry) {
      clearTimeout(entry.timer);
      pending.delete(msg.id);
      streamCallbacks.delete(msg.id);
      entry.resolve(msg as RelayResponse);
    }
  } else if (msg.type === "pong") {
    // Keepalive response, ignore
  } else {
    console.warn(`[relay] Unknown message type from node ${nodeId}:`, msg.type);
  }
}
