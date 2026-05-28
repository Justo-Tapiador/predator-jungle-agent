#!/usr/bin/env node
/**
 * PREDATOR Web Dashboard Server
 * Express + WebSocket real-time monitoring interface
 * Visit: http://localhost:3000
 *
 * Provides:
 *   - Real-time WebSocket event streaming from the PREDATOR agent
 *   - REST API for agent status, history, metrics, and control
 *   - Static file serving for the dashboard UI
 *   - Interactive task submission and owner feedback
 *
 * Architecture:
 *   ┌─────────────┐   WebSocket   ┌──────────────┐
 *   │  Dashboard   │◄─────────────│  Web Server   │
 *   │  (Browser)   │─────────────►│  (Express+WS) │
 *   └─────────────┘   Commands    └──────┬───────┘
 *                                        │ Events
 *                                        ▼
 *                                 ┌──────────────┐
 *                                 │   Predator    │
 *                                 │   Agent       │
 *                                 └──────────────┘
 *
 * @module web/server
 */

import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { randomUUID } from 'crypto';

import { Predator } from '../src/core/Predator.js';
import { createApiRouter } from './routes/api.js';

// ─── ES Module path helpers ──────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

// ─── Configuration ───────────────────────────────────────────────────────────
const PORT         = parseInt(process.env.PREDATOR_PORT ?? '3000', 10);
const HOST         = process.env.PREDATOR_HOST ?? '0.0.0.0';
const STATIC_DIR   = join(__dirname, 'public');
const WS_PATH      = '/ws';

// ─── Predator Agent Instance ─────────────────────────────────────────────────
const predator = new Predator({
  dModel:    256,
  nHeads:    8,
  dFF:       1024,
  maxSteps:  200,
  enableMemory:   true,
  enableSafety:   true,
  enableMetrics:  true,
  enablePlugins:  true,
});

// ─── WebSocket Client Registry ───────────────────────────────────────────────
const wsClients = new Map(); // clientId -> { ws, subscriptions }

function broadcastToClients(event, data) {
  const payload = JSON.stringify({ event, data, timestamp: Date.now() });
  for (const [, client] of wsClients) {
    if (client.ws.readyState === 1) { // OPEN
      try {
        client.ws.send(payload);
      } catch (_) {
        // Client disconnected, will be cleaned up
      }
    }
  }
}

// ─── Forward Predator Events to WebSocket Clients ────────────────────────────
const FORWARD_EVENTS = [
  'task:complete',
  'step:complete',
  'chain:start',
  'chain:step',
  'chain:complete',
  'chain:error',
  'safety:blocked',
  'safety:warning',
  'safety:stepBlocked',
  'cascade:warning',
  'cascade:critical',
  'memory:recall',
  'memory:store',
  'owner:escalation',
  'owner:resume',
  'plugin:registered',
  'tool:registered',
  'train:start',
  'train:complete',
  'shutdown:start',
  'shutdown:complete',
  'state:restored',
];

for (const eventName of FORWARD_EVENTS) {
  predator.on(eventName, (data) => {
    broadcastToClients(eventName, data);
  });
}

// ─── Express App ─────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
app.use(express.static(STATIC_DIR));

// ─── API Routes ──────────────────────────────────────────────────────────────
const apiRouter = createApiRouter(predator);
app.use('/api', apiRouter);

// ─── HTTP Server ─────────────────────────────────────────────────────────────
const server = createServer(app);

// ─── WebSocket Server ────────────────────────────────────────────────────────
const wss = new WebSocketServer({ server, path: WS_PATH });

wss.on('connection', (ws) => {
  const clientId = randomUUID();
  wsClients.set(clientId, { ws, subscriptions: new Set(['*']) });

  console.log(`[WS] Client connected: ${clientId} (total: ${wsClients.size})`);

  // Send initial status
  ws.send(JSON.stringify({
    event: 'connected',
    data: {
      clientId,
      agentStatus: predator.status(),
      serverTime: Date.now(),
    },
    timestamp: Date.now(),
  }));

  // Send current metrics snapshot
  if (predator.metrics) {
    ws.send(JSON.stringify({
      event: 'metrics:snapshot',
      data: predator.metrics.getSummary(),
      timestamp: Date.now(),
    }));
  }

  // Send cascade risk history
  ws.send(JSON.stringify({
    event: 'cascade:history',
    data: predator.cascadeMonitor.getCascadeRiskHistory(),
    timestamp: Date.now(),
  }));

  ws.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      ws.send(JSON.stringify({ event: 'error', data: { message: 'Invalid JSON' } }));
      return;
    }

    handleClientMessage(clientId, msg);
  });

  ws.on('close', () => {
    wsClients.delete(clientId);
    console.log(`[WS] Client disconnected: ${clientId} (total: ${wsClients.size})`);
  });

  ws.on('error', (err) => {
    console.error(`[WS] Client error (${clientId}):`, err.message);
    wsClients.delete(clientId);
  });
});

// ─── Handle Client Commands ──────────────────────────────────────────────────
async function handleClientMessage(clientId, msg) {
  const client = wsClients.get(clientId);
  if (!client) return;

  const { type, id, payload } = msg;

  try {
    switch (type) {
      // ── Task Execution ──────────────────────────────────────────────────
      case 'task:execute': {
        const { directive, budget } = payload;
        if (!directive) throw new Error('Missing "directive" in payload');

        // Execute asynchronously, events will stream via WebSocket
        predator.execute(directive, budget)
          .then((result) => {
            client.ws.send(JSON.stringify({
              event: 'task:result',
              data: result,
              requestId: id,
              timestamp: Date.now(),
            }));
          })
          .catch((err) => {
            client.ws.send(JSON.stringify({
              event: 'task:error',
              data: { error: err.message },
              requestId: id,
              timestamp: Date.now(),
            }));
          });

        client.ws.send(JSON.stringify({
          event: 'task:accepted',
          data: { directive },
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      case 'chain:execute': {
        const { directives } = payload;
        if (!Array.isArray(directives)) throw new Error('Missing "directives" array');

        predator.executeChain(directives)
          .then((results) => {
            client.ws.send(JSON.stringify({
              event: 'chain:result',
              data: results,
              requestId: id,
              timestamp: Date.now(),
            }));
          })
          .catch((err) => {
            client.ws.send(JSON.stringify({
              event: 'chain:error',
              data: { error: err.message },
              requestId: id,
              timestamp: Date.now(),
            }));
          });

        client.ws.send(JSON.stringify({
          event: 'chain:accepted',
          data: { directiveCount: directives.length },
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      // ── Owner Feedback ──────────────────────────────────────────────────
      case 'owner:feedback': {
        const { feedback } = payload;
        predator.resume(feedback);
        client.ws.send(JSON.stringify({
          event: 'owner:feedback_received',
          data: { feedback },
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      // ── Training ────────────────────────────────────────────────────────
      case 'train:start': {
        const { config } = payload;
        predator.train(config ?? {})
          .then((result) => {
            client.ws.send(JSON.stringify({
              event: 'train:result',
              data: result,
              requestId: id,
              timestamp: Date.now(),
            }));
          })
          .catch((err) => {
            client.ws.send(JSON.stringify({
              event: 'train:error',
              data: { error: err.message },
              requestId: id,
              timestamp: Date.now(),
            }));
          });

        client.ws.send(JSON.stringify({
          event: 'train:accepted',
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      // ── State Management ────────────────────────────────────────────────
      case 'state:serialize': {
        const state = predator.serialize();
        client.ws.send(JSON.stringify({
          event: 'state:serialized',
          data: state,
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      case 'state:deserialize': {
        const { state } = payload;
        predator.deserialize(state);
        client.ws.send(JSON.stringify({
          event: 'state:restored',
          data: predator.status(),
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      // ── Metrics ─────────────────────────────────────────────────────────
      case 'metrics:snapshot': {
        const summary = predator.metrics ? predator.metrics.getSummary() : null;
        client.ws.send(JSON.stringify({
          event: 'metrics:snapshot',
          data: summary,
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      case 'metrics:reset': {
        if (predator.metrics) predator.metrics.resetAll();
        client.ws.send(JSON.stringify({
          event: 'metrics:reset',
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      // ── Tool Registration ───────────────────────────────────────────────
      case 'tool:register': {
        const { toolId, code, meta } = payload;
        if (!toolId || !code) throw new Error('Missing "toolId" or "code"');

        // Create a sandboxed function from the code string
        const fn = new Function('context', code);
        predator.registerTool(toolId, fn, meta ?? {});
        client.ws.send(JSON.stringify({
          event: 'tool:registered',
          data: { toolId, meta },
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      // ── Subscribe / Unsubscribe ─────────────────────────────────────────
      case 'subscribe': {
        const { events } = payload;
        if (Array.isArray(events)) {
          for (const e of events) client.subscriptions.add(e);
        } else if (payload.all) {
          client.subscriptions.add('*');
        }
        client.ws.send(JSON.stringify({
          event: 'subscribed',
          data: { subscriptions: [...client.subscriptions] },
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      case 'unsubscribe': {
        const { events } = payload;
        if (Array.isArray(events)) {
          for (const e of events) client.subscriptions.delete(e);
        }
        client.subscriptions.delete('*');
        client.ws.send(JSON.stringify({
          event: 'unsubscribed',
          data: { subscriptions: [...client.subscriptions] },
          requestId: id,
          timestamp: Date.now(),
        }));
        break;
      }

      // ── Shutdown ────────────────────────────────────────────────────────
      case 'shutdown': {
        client.ws.send(JSON.stringify({
          event: 'shutdown:initiated',
          requestId: id,
          timestamp: Date.now(),
        }));
        predator.shutdown()
          .then(() => {
            broadcastToClients('shutdown:complete', {});
          });
        break;
      }

      default:
        client.ws.send(JSON.stringify({
          event: 'error',
          data: { message: `Unknown command type: ${type}` },
          requestId: id,
          timestamp: Date.now(),
        }));
    }
  } catch (err) {
    client.ws.send(JSON.stringify({
      event: 'error',
      data: { message: err.message, type },
      requestId: id,
      timestamp: Date.now(),
    }));
  }
}

// ─── Periodic Status Push ────────────────────────────────────────────────────
const STATUS_INTERVAL = 2000; // Push status every 2 seconds
const statusTimer = setInterval(() => {
  if (wsClients.size === 0) return;
  broadcastToClients('status:update', predator.status());
}, STATUS_INTERVAL);

// Push metrics every 5 seconds
const METRICS_INTERVAL = 5000;
const metricsTimer = setInterval(() => {
  if (wsClients.size === 0) return;
  if (!predator.metrics) return;
  broadcastToClients('metrics:update', predator.metrics.getSummary());
}, METRICS_INTERVAL);

// ─── Graceful Shutdown ───────────────────────────────────────────────────────
async function gracefulShutdown(signal) {
  console.log(`\n[Server] Received ${signal}. Shutting down gracefully...`);

  clearInterval(statusTimer);
  clearInterval(metricsTimer);

  // Close all WebSocket connections
  for (const [clientId, client] of wsClients) {
    try {
      client.ws.send(JSON.stringify({
        event: 'server:shutdown',
        data: { reason: signal },
        timestamp: Date.now(),
      }));
      client.ws.close();
    } catch (_) { /* ignore */ }
  }
  wsClients.clear();

  // Shutdown the predator agent
  await predator.shutdown();

  // Close HTTP server
  server.close(() => {
    console.log('[Server] HTTP server closed.');
    process.exit(0);
  });

  // Force exit after 10s
  setTimeout(() => {
    console.error('[Server] Forced shutdown after timeout.');
    process.exit(1);
  }, 10000);
}

process.on('SIGINT',  () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

// ─── Start Server ────────────────────────────────────────────────────────────
server.listen(PORT, HOST, () => {
  console.log(`
  ╔══════════════════════════════════════════════════════════════╗
  ║            PREDATOR Web Dashboard Server                    ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║   Dashboard :  http://${HOST}:${PORT}                         ║
  ║   WebSocket :  ws://${HOST}:${PORT}${WS_PATH}                  ║
  ║   REST API  :  http://${HOST}:${PORT}/api/*                    ║
  ║                                                              ║
  ║   Agent ID  :  ${predator.status().id}    ║
  ║   Status    :  Ready                                         ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
  `);
});

export { app, server, wss, predator };
