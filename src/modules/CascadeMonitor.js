/**
 * CascadeMonitor.js  (v2.0 – Enhanced)
 * ─────────────────────────────────────────────────────────────────────────────
 * Enhanced background monitor that evaluates cascade risk for each AJN layer
 * and triggers interventions when thresholds are exceeded.
 *
 * IMPROVEMENTS over v0.1:
 *   - Per-layer cascade risk tracking with history
 *   - Predictive cascade risk estimation (trend analysis)
 *   - Graduated intervention levels (not just warn/critical)
 *   - Self-healing: automatic stimulus injection with learned patterns
 *   - Cascade risk history for analysis and debugging
 *   - Configurable intervention strategies
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';

export class CascadeMonitor extends EventEmitter {
  /**
   * @param {object} opts
   * @param {number} [opts.rhoWarn=0.35]      – Warning threshold for rho_ext
   * @param {number} [opts.rhoCritical=0.65]  – Critical threshold
   * @param {number} [opts.rhoModerate=0.50]  – Moderate threshold (new)
   * @param {number} [opts.pollMs=500]        – Polling interval (ms)
   * @param {boolean} [opts.selfHealing=true] – Enable self-healing stimulus injection
   * @param {number} [opts.historySize=100]   – Max cascade risk history entries
   */
  constructor(opts = {}) {
    super();
    this.rhoWarn     = opts.rhoWarn     ?? 0.35;
    this.rhoModerate = opts.rhoModerate  ?? 0.50;
    this.rhoCritical = opts.rhoCritical ?? 0.65;
    this.pollMs      = opts.pollMs      ?? 500;
    this.selfHealing = opts.selfHealing ?? true;
    this.historySize = opts.historySize ?? 100;

    this.extinctionLog = [];
    this.interventions = [];
    this.cascadeRiskHistory = [];  // { timestamp, layerRisks: {} }
    this._timer        = null;
    this._backbone     = null;
    this._paused       = false;
    this._stimulusPatterns = []; // Learned successful stimulus patterns
  }

  /** Attach to a running ANNPsi instance */
  attach(backbone) {
    this._backbone = backbone;

    backbone.on('extinction', (e) => {
      this.extinctionLog.push({
        unitId: e.id,
        t:      Date.now(),
        cause:  'stimulus_starvation',
        extinctions: e.extinctions,
      });
      this.emit('extinctionLogged', e);
    });
  }

  /** Start the periodic cascade risk poll */
  start() {
    if (this._timer) return;
    this._timer = setInterval(() => this._poll(), this.pollMs);
    this.emit('started');
  }

  /** Stop monitoring */
  stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    this.emit('stopped');
  }

  /**
   * Evaluate cascade risk from a forward-pass result.
   * Returns graduated intervention levels.
   */
  evaluate(forwardResult, directive) {
    const { cascadeRisk, layerTrace } = forwardResult;

    // Record cascade risk history
    const layerRisks = {};
    for (const t of (layerTrace ?? [])) {
      if (t.cascadeRisk !== undefined) layerRisks[t.layer] = t.cascadeRisk;
    }
    this.cascadeRiskHistory.push({ timestamp: Date.now(), overallRisk: cascadeRisk, layerRisks });
    if (this.cascadeRiskHistory.length > this.historySize) {
      this.cascadeRiskHistory.shift();
    }

    // Predictive analysis: check if risk is trending upward
    const trend = this._predictTrend();
    const predictedRisk = cascadeRisk + trend * 5; // Look 5 steps ahead

    // Graduated intervention levels
    if (predictedRisk >= this.rhoCritical || cascadeRisk >= this.rhoCritical) {
      return this._intervene('owner_escalation', Math.max(cascadeRisk, predictedRisk), directive);
    }
    if (cascadeRisk >= this.rhoModerate) {
      // Moderate: self-healing with targeted stimulus injection
      if (this.selfHealing) {
        this._injectSelfHealing(layerTrace, directive);
      }
      return this._intervene('stimulus_injection', cascadeRisk, directive);
    }
    if (cascadeRisk >= this.rhoWarn || predictedRisk >= this.rhoModerate) {
      return this._intervene('preventive_stimulus', cascadeRisk, directive);
    }
    return { action: 'none', cascadeRisk, predictedRisk, trend };
  }

  /** Record a successful stimulus pattern for self-healing */
  recordStimulusPattern(stimulus, outcome) {
    if (outcome.success && outcome.task_progress > 0.5) {
      this._stimulusPatterns.push({
        stimulus: { ...stimulus },
        progress: outcome.task_progress,
        timestamp: Date.now(),
      });
      if (this._stimulusPatterns.length > 50) {
        this._stimulusPatterns.shift();
      }
    }
  }

  getExtinctionLog() { return [...this.extinctionLog]; }
  getInterventions()  { return [...this.interventions]; }
  getCascadeRiskHistory() { return [...this.cascadeRiskHistory]; }

  /** Get per-layer risk trends */
  getLayerRiskTrends() {
    const trends = {};
    const history = this.cascadeRiskHistory.slice(-10);

    for (const entry of history) {
      for (const [layer, risk] of Object.entries(entry.layerRisks ?? {})) {
        if (!trends[layer]) trends[layer] = [];
        trends[layer].push(risk);
      }
    }

    const result = {};
    for (const [layer, risks] of Object.entries(trends)) {
      if (risks.length < 2) {
        result[layer] = { current: risks[risks.length - 1], trend: 0 };
        continue;
      }
      const recent = risks[risks.length - 1];
      const prev   = risks[risks.length - 2];
      result[layer] = {
        current: recent,
        trend: recent - prev,
        avg: risks.reduce((s, r) => s + r, 0) / risks.length,
      };
    }
    return result;
  }

  // ── Private ────────────────────────────────────────────────────────────────

  _poll() {
    if (!this._backbone) return;
    try {
      const snap = this._backbone.snapshot();
      const l12 = snap.layers?.L12;
      if (l12) {
        const rho = l12.units
          ? l12.units.filter(u => u.nFail > (u.tau ?? 20) / 2).length / (l12.units.length || 1)
          : 0;
        if (rho > this.rhoWarn) {
          this.emit('cascadeWarning', { layer: 'L12', rho, trend: this._predictTrend() });
        }
      }
    } catch (_) { /* backbone not ready */ }
  }

  _intervene(type, risk, directive) {
    const entry = { type, risk, directiveId: directive?.id, t: Date.now() };
    this.interventions.push(entry);
    this.emit('intervention', entry);

    switch (type) {
      case 'preventive_stimulus':
        this.emit('requestPreventiveStimulus', { risk, directive });
        return { action: 'preventive_stimulus', cascadeRisk: risk };

      case 'stimulus_injection':
        this.emit('requestStimulusInjection', { risk, directive });
        return { action: 'stimulus_injection', cascadeRisk: risk };

      case 'task_decomposition':
        this.emit('requestTaskDecomposition', { risk, directive });
        return { action: 'task_decomposition', cascadeRisk: risk };

      case 'owner_escalation':
        this._paused = true;
        this.emit('ownerEscalation', {
          risk,
          directive,
          message: `PREDATOR cascade risk critical (rho=${risk.toFixed(3)}). Awaiting Owner input.`,
        });
        return { action: 'owner_escalation', cascadeRisk: risk, paused: true };

      default:
        return { action: 'none', cascadeRisk: risk };
    }
  }

  _injectSelfHealing(layerTrace, directive) {
    // Use learned stimulus patterns to generate targeted stimulus
    if (this._stimulusPatterns.length === 0) return;

    // Find most successful pattern
    const best = this._stimulusPatterns.reduce((a, b) =>
      a.progress > b.progress ? a : b
    );

    this.emit('selfHealingInjection', {
      stimulus: best.stimulus,
      source: 'learned_pattern',
      progress: best.progress,
    });
  }

  _predictTrend() {
    const history = this.cascadeRiskHistory.slice(-5);
    if (history.length < 2) return 0;

    // Simple linear regression on recent cascade risks
    const n = history.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (let i = 0; i < n; i++) {
      sumX  += i;
      sumY  += history[i].overallRisk;
      sumXY += i * history[i].overallRisk;
      sumX2 += i * i;
    }
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return slope;
  }
}
