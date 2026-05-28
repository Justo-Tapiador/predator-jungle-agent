/**
 * API Router — REST endpoints for the PREDATOR Web Dashboard
 *
 * Endpoints:
 *   GET  /api/status          — Current agent status
 *   GET  /api/history         — Task execution history
 *   GET  /api/metrics         — Full metrics snapshot
 *   GET  /api/metrics/counters — Counters only
 *   GET  /api/metrics/gauges  — Gauges only
 *   GET  /api/metrics/histograms — Histograms with percentiles
 *   GET  /api/cascade         — Cascade monitor info
 *   GET  /api/cascade/history — Cascade risk history
 *   GET  /api/cascade/extinctions — Extinction log
 *   GET  /api/cascade/interventions — Intervention log
 *   GET  /api/memory/:query   — Memory recall
 *   POST /api/task            — Submit a new task
 *   POST /api/chain           — Submit a task chain
 *   POST /api/feedback        — Provide owner feedback
 *   POST /api/train           — Start training
 *   GET  /api/state           — Serialize agent state
 *   POST /api/state           — Restore agent state
 *   POST /api/metrics/reset   — Reset all metrics
 *   GET  /api/tools           — List registered tools
 *   GET  /api/plugins         — List registered plugins
 *
 * @module web/routes/api
 */

import { Router } from 'express';

export function createApiRouter(predator) {
  const router = Router();

  // ── Agent Status ────────────────────────────────────────────────────────

  router.get('/status', (_req, res) => {
    try {
      res.json({
        ok: true,
        data: predator.status(),
      });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Task History ────────────────────────────────────────────────────────

  router.get('/history', (_req, res) => {
    try {
      const history = predator.history();
      res.json({
        ok: true,
        data: history,
        count: history.length,
      });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Metrics ─────────────────────────────────────────────────────────────

  router.get('/metrics', (_req, res) => {
    try {
      if (!predator.metrics) {
        return res.json({ ok: true, data: null, message: 'Metrics disabled' });
      }
      res.json({
        ok: true,
        data: predator.metrics.getSummary(),
      });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.get('/metrics/counters', (_req, res) => {
    try {
      if (!predator.metrics) return res.json({ ok: true, data: {} });
      const summary = predator.metrics.getSummary();
      res.json({ ok: true, data: summary.counters });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.get('/metrics/gauges', (_req, res) => {
    try {
      if (!predator.metrics) return res.json({ ok: true, data: {} });
      const summary = predator.metrics.getSummary();
      res.json({ ok: true, data: summary.gauges });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.get('/metrics/histograms', (_req, res) => {
    try {
      if (!predator.metrics) return res.json({ ok: true, data: {} });
      const summary = predator.metrics.getSummary();
      res.json({ ok: true, data: summary.histograms });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.post('/metrics/reset', (_req, res) => {
    try {
      if (predator.metrics) predator.metrics.resetAll();
      res.json({ ok: true, message: 'Metrics reset' });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Cascade Monitor ─────────────────────────────────────────────────────

  router.get('/cascade', (_req, res) => {
    try {
      const cm = predator.cascadeMonitor;
      res.json({
        ok: true,
        data: {
          rhoWarn: cm.rhoWarn,
          rhoModerate: cm.rhoModerate,
          rhoCritical: cm.rhoCritical,
          selfHealing: cm.selfHealing,
          extinctionCount: cm.getExtinctionLog().length,
          interventionCount: cm.getInterventions().length,
          layerRiskTrends: cm.getLayerRiskTrends(),
        },
      });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.get('/cascade/history', (_req, res) => {
    try {
      const limit = Math.min(parseInt(_req.query.limit ?? '100', 10), 1000);
      const history = predator.cascadeMonitor.getCascadeRiskHistory().slice(-limit);
      res.json({ ok: true, data: history, count: history.length });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.get('/cascade/extinctions', (_req, res) => {
    try {
      const log = predator.cascadeMonitor.getExtinctionLog();
      res.json({ ok: true, data: log, count: log.length });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.get('/cascade/interventions', (_req, res) => {
    try {
      const interventions = predator.cascadeMonitor.getInterventions();
      res.json({ ok: true, data: interventions, count: interventions.length });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Memory ──────────────────────────────────────────────────────────────

  router.get('/memory/:query', async (req, res) => {
    try {
      if (!predator.memory) {
        return res.json({ ok: true, data: [], message: 'Memory disabled' });
      }
      const memories = await predator.memory.recall(req.params.query, {
        limit: parseInt(req.query.limit ?? '10', 10),
      });
      res.json({ ok: true, data: memories, count: memories.length });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Task Submission ─────────────────────────────────────────────────────

  router.post('/task', async (req, res) => {
    try {
      const { directive, budget } = req.body;
      if (!directive) {
        return res.status(400).json({ ok: false, error: 'Missing "directive" field' });
      }
      // Execute asynchronously and return immediately
      const result = await predator.execute(directive, budget);
      res.json({ ok: true, data: result });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.post('/chain', async (req, res) => {
    try {
      const { directives } = req.body;
      if (!Array.isArray(directives) || directives.length === 0) {
        return res.status(400).json({ ok: false, error: 'Missing "directives" array' });
      }
      const results = await predator.executeChain(directives);
      res.json({ ok: true, data: results });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Owner Feedback ──────────────────────────────────────────────────────

  router.post('/feedback', (req, res) => {
    try {
      const { feedback } = req.body;
      if (!feedback) {
        return res.status(400).json({ ok: false, error: 'Missing "feedback" field' });
      }
      predator.resume(feedback);
      res.json({ ok: true, message: 'Feedback delivered to agent' });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Training ────────────────────────────────────────────────────────────

  router.post('/train', async (req, res) => {
    try {
      const { config } = req.body;
      const result = await predator.train(config ?? {});
      res.json({ ok: true, data: result });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── State Serialization ─────────────────────────────────────────────────

  router.get('/state', (_req, res) => {
    try {
      const state = predator.serialize();
      res.json({ ok: true, data: state });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  router.post('/state', (req, res) => {
    try {
      const { state } = req.body;
      if (!state) {
        return res.status(400).json({ ok: false, error: 'Missing "state" field' });
      }
      predator.deserialize(state);
      res.json({ ok: true, data: predator.status() });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Tools ───────────────────────────────────────────────────────────────

  router.get('/tools', (_req, res) => {
    try {
      const status = predator.status();
      res.json({
        ok: true,
        data: {
          customToolCount: status.customToolCount,
        },
      });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── Plugins ─────────────────────────────────────────────────────────────

  router.get('/plugins', (_req, res) => {
    try {
      const status = predator.status();
      res.json({
        ok: true,
        data: {
          pluginsEnabled: status.pluginsEnabled,
        },
      });
    } catch (err) {
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  return router;
}
