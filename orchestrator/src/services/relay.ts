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

export async function sendRequest(
  nodeId: string,
  body: unknown,
  path: string = "/v1/chat/completions",
  timeoutMs: number = 120_000
): Promise<RelayResponse> {
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

  return new Promise<RelayResponse>((resolve, reject) => {
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
}

export function handleMessage(nodeId: string, data: string): void {
  let msg: any;
  try {
    msg = JSON.parse(data);
  } catch {
    console.error(`[relay] Invalid JSON from node ${nodeId}`);
    return;
  }

  if (msg.type === "inference_response") {
    const entry = pending.get(msg.id);
    if (entry) {
      clearTimeout(entry.timer);
      pending.delete(msg.id);
      entry.resolve(msg as RelayResponse);
    }
  } else if (msg.type === "pong") {
    // Keepalive response, ignore
  } else {
    console.warn(`[relay] Unknown message type from node ${nodeId}:`, msg.type);
  }
}
