import { Hono } from "hono";
import type { WSContext } from "hono/ws";
import * as relay from "../services/relay.js";

// This will be set by index.ts after creating the WebSocket helper
let upgradeWebSocket: any;

export function setUpgradeWebSocket(fn: any) {
  upgradeWebSocket = fn;
}

export function createWsRoute(): Hono {
  const wsRouter = new Hono();

  wsRouter.get(
    "/",
    (c, next) => {
      if (!upgradeWebSocket) {
        return c.json({ error: "WebSocket not available" }, 500);
      }
      return upgradeWebSocket((c: any) => {
        let nodeId: string | null = null;
        let identifyTimer: ReturnType<typeof setTimeout>;

        return {
          onOpen(evt: Event, ws: WSContext) {
            // Node must identify within 10 seconds
            identifyTimer = setTimeout(() => {
              if (!nodeId) {
                console.warn("[ws] Connection closed: no identify received");
                ws.close(4001, "identify timeout");
              }
            }, 10_000);
          },

          onMessage(evt: MessageEvent, ws: WSContext) {
            const data = typeof evt.data === "string" ? evt.data : evt.data.toString();
            let msg: any;
            try {
              msg = JSON.parse(data);
            } catch {
              return;
            }

            if (msg.type === "identify") {
              clearTimeout(identifyTimer);
              nodeId = msg.node_id;
              if (!nodeId) {
                ws.close(4002, "missing node_id");
                return;
              }
              relay.registerConnection(nodeId, ws);
              ws.send(JSON.stringify({ type: "identified", node_id: nodeId }));
              console.log(`[ws] Node identified: ${nodeId}`);
              return;
            }

            if (!nodeId) {
              ws.close(4003, "not identified");
              return;
            }

            relay.handleMessage(nodeId, data);
          },

          onClose(evt: CloseEvent, ws: WSContext) {
            clearTimeout(identifyTimer);
            if (nodeId) {
              relay.removeConnection(nodeId);
            }
          },

          onError(evt: Event, ws: WSContext) {
            clearTimeout(identifyTimer);
            if (nodeId) {
              relay.removeConnection(nodeId);
            }
          },
        };
      })(c, next);
    }
  );

  return wsRouter;
}
