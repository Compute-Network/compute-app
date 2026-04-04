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
    nodeId: string; // Track which node owns this request
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

// ── Circuit breaker ───────────────────────────────────────────────────────────
// Tracks recent failures per node. If a node fails too often, skip it temporarily.
interface CircuitState {
  failures: number;
  lastFailure: number;
  openUntil: number; // timestamp — skip this node until this time
}
const circuitBreakers = new Map<string, CircuitState>();

const CIRCUIT_FAILURE_THRESHOLD = 3;  // failures within the window to open circuit
const CIRCUIT_FAILURE_WINDOW = 5 * 60_000;  // 5 minutes
const CIRCUIT_OPEN_DURATION = 60_000;  // skip node for 60 seconds after tripping

export function isCircuitOpen(nodeId: string): boolean {
  const state = circuitBreakers.get(nodeId);
  if (!state) return false;
  if (Date.now() < state.openUntil) return true;
  // Circuit closed — reset if past the window
  if (Date.now() - state.lastFailure > CIRCUIT_FAILURE_WINDOW) {
    circuitBreakers.delete(nodeId);
  }
  return false;
}

function recordNodeFailure(nodeId: string): void {
  const now = Date.now();
  let state = circuitBreakers.get(nodeId);
  if (!state || now - state.lastFailure > CIRCUIT_FAILURE_WINDOW) {
    state = { failures: 0, lastFailure: now, openUntil: 0 };
  }
  state.failures++;
  state.lastFailure = now;
  if (state.failures >= CIRCUIT_FAILURE_THRESHOLD) {
    state.openUntil = now + CIRCUIT_OPEN_DURATION;
    console.warn(`[circuit-breaker] Node ${nodeId.slice(0, 8)} tripped (${state.failures} failures) — skipping for ${CIRCUIT_OPEN_DURATION / 1000}s`);
  }
  circuitBreakers.set(nodeId, state);
}

export function recordNodeSuccess(nodeId: string): void {
  // Reset circuit breaker on success
  circuitBreakers.delete(nodeId);
}

// ── Queue management ──────────────────────────────────────────────────────────

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
    return Promise.resolve();
  }
  if (q.queue.length === 0) {
    console.log(`[relay-queue] ${nodeId.slice(0, 8)}: request queued`);
  }
  return new Promise<void>((resolve) => {
    q.queue.push(resolve);
  });
}

function releaseSlot(nodeId: string): void {
  const q = getNodeQueue(nodeId);
  const next = q.queue.shift();
  if (next) {
    next();
  } else {
    q.busy = false;
  }
}

let requestCounter = 0;

function generateRequestId(): string {
  return `req-${Date.now()}-${++requestCounter}`;
}

// ── Connection management ─────────────────────────────────────────────────────

export function registerConnection(nodeId: string, ws: WSContext): void {
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

  // CRITICAL: Reject all pending requests for this node immediately.
  // Without this, requests hang for up to 120s until timeout.
  let rejected = 0;
  for (const [reqId, entry] of pending) {
    if (entry.nodeId === nodeId) {
      clearTimeout(entry.timer);
      pending.delete(reqId);
      streamCallbacks.delete(reqId);
      entry.reject(new Error(`Node ${nodeId.slice(0, 8)} disconnected during request`));
      rejected++;
    }
  }
  if (rejected > 0) {
    console.warn(`[relay] Rejected ${rejected} pending request(s) for disconnected node ${nodeId.slice(0, 8)}`);
    recordNodeFailure(nodeId);
  }

  // Release the queue slot so queued requests can proceed (and fail fast)
  const q = nodeQueues.get(nodeId);
  if (q) {
    q.busy = false;
    // Drain any queued callers — they'll fail when they try to send (no connection)
    while (q.queue.length > 0) {
      const next = q.queue.shift()!;
      next();
    }
    nodeQueues.delete(nodeId);
  }
}

export function isConnected(nodeId: string): boolean {
  return connections.has(nodeId);
}

export function getConnectedCount(): number {
  return connections.size;
}

// ── Inference requests ────────────────────────────────────────────────────────

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

    streamCallbacks.set(id, onChunk);

    return await new Promise<RelayResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        streamCallbacks.delete(id);
        recordNodeFailure(nodeId);
        reject(new Error(`Streaming request ${id} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      pending.set(id, { resolve, reject, timer, nodeId });

      try {
        ws.send(JSON.stringify(request));
      } catch (e: any) {
        pending.delete(id);
        streamCallbacks.delete(id);
        clearTimeout(timer);
        recordNodeFailure(nodeId);
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
        recordNodeFailure(nodeId);
        reject(new Error(`Request ${id} to node ${nodeId} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      pending.set(id, { resolve, reject, timer, nodeId });

      try {
        ws.send(JSON.stringify(request));
      } catch (e: any) {
        pending.delete(id);
        clearTimeout(timer);
        recordNodeFailure(nodeId);
        reject(new Error(`Failed to send to node ${nodeId}: ${e.message}`));
      }
    });
  } finally {
    releaseSlot(nodeId);
  }
}

// ── Message handling ──────────────────────────────────────────────────────────

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
      // Record success to reset circuit breaker
      if ((msg as RelayResponse).status === 200) {
        recordNodeSuccess(entry.nodeId);
      } else {
        recordNodeFailure(entry.nodeId);
      }
      entry.resolve(msg as RelayResponse);
    }
  } else if (msg.type === "pong") {
    // Keepalive response, ignore
  } else {
    console.warn(`[relay] Unknown message type from node ${nodeId}:`, msg.type);
  }
}

// ── Monitoring ────────────────────────────────────────────────────────────────

export function getStats(): {
  connections: number;
  pendingRequests: number;
  circuitBreakers: number;
} {
  return {
    connections: connections.size,
    pendingRequests: pending.size,
    circuitBreakers: circuitBreakers.size,
  };
}
