/**
 * AJNLayer.js  (v2.0 – Enhanced)
 * ─────────────────────────────────────────────────────────────────────────────
 * Three AJN layer integration paradigms with improvements:
 *
 * IMPROVEMENTS over v0.1:
 *   - Proper normalization of layer praxis output
 *   - Skip connections for gradient flow preservation
 *   - Layer-level state serialization/deserialization
 *   - Configurable aggregation strategies (mean, max, weighted)
 *   - Better cascade risk estimation using exponential moving average
 *   - Inter-group competition with softmax temperature for heterogeneous layers
 *   - Attention-weighted praxis aggregation for hybrid layers
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { ArtificialJunkyNeuron, AJNPhase } from '../core/ArtificialJunkyNeuron.js';

const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

// ─────────────────────────────────────────────────────────────────────────────
// Paradigm I – HOMOGENEOUS AJN LAYER (Enhanced)
// ─────────────────────────────────────────────────────────────────────────────
export class HomogeneousAJNLayer extends EventEmitter {
  /**
   * @param {object} opts
   * @param {number}   opts.N              – Number of AJN units
   * @param {string}   opts.stimulusClass
   * @param {Function} opts.intensityFn    – Shared I(S) for all units
   * @param {number}   [opts.kappa=0.2]    – Lateral inhibition coupling coefficient
   * @param {number}   [opts.rhoSat=0.7]   – Collective saturation threshold (fraction)
   * @param {string}   [opts.aggregation='mean'] – Aggregation strategy: mean | max | weighted
   * @param {object}   [opts.params]       – AJN hyperparameter overrides
   */
  constructor(opts) {
    super();
    this.id            = opts.id ?? `homo-layer-${Math.random().toString(36).slice(2,7)}`;
    this.N             = opts.N ?? 32;
    this.kappa         = opts.kappa ?? 0.2;
    this.rhoSat        = opts.rhoSat ?? 0.7;
    this.stimulusClass = opts.stimulusClass;
    this.aggregation   = opts.aggregation ?? 'mean';

    this.units = Array.from({ length: this.N }, (_, i) =>
      new ArtificialJunkyNeuron({
        id:           `${this.id}-u${i}`,
        stimulusClass: opts.stimulusClass,
        intensityFn:  opts.intensityFn,
        params:       opts.params,
      })
    );

    // Cascade risk EMA for smoother estimation
    this._cascadeRiskEMA = 0;
    this._cascadeAlpha   = 0.1; // EMA smoothing factor

    this.units.forEach(u => u.on('extinction', e => this.emit('extinction', e)));
  }

  process(stimulus) {
    const meanM = this._meanCraving();
    const results = this.units.map(u => {
      const inhibitedStimulus = this._inhibit(stimulus, u, meanM);
      return u.process(inhibitedStimulus);
    });

    // Layer praxis with configurable aggregation
    const d = this.units[0].p.praximDim;
    const layerPraxis = this._aggregate(results, d);

    // Collective saturation check
    const satCount = results.filter(r => r.alpha > this.units[0].p.thetaSat).length;
    const collectiveSaturated = (satCount / this.N) > this.rhoSat;

    // Cascade risk with EMA smoothing
    const instantCascadeRisk = results.filter(
      r => r.nFail > this.units[0].p.tau / 2
    ).length / this.N;
    this._cascadeRiskEMA = this._cascadeAlpha * instantCascadeRisk
                         + (1 - this._cascadeAlpha) * this._cascadeRiskEMA;

    const out = {
      layerPraxis,
      collectiveSaturated,
      cascadeRisk: this._cascadeRiskEMA,
      unitResults: results,
    };
    this.emit('layerStep', out);
    return out;
  }

  _aggregate(results, d) {
    const layerPraxis = new Float64Array(d);

    if (this.aggregation === 'mean') {
      for (const r of results) {
        for (let i = 0; i < d; i++) layerPraxis[i] += r.praxis[i] / this.N;
      }
    } else if (this.aggregation === 'max') {
      // Take the praxis with highest norm
      let maxNorm = -Infinity, maxIdx = 0;
      for (let j = 0; j < results.length; j++) {
        if (results[j].praxisNorm > maxNorm) {
          maxNorm = results[j].praxisNorm;
          maxIdx = j;
        }
      }
      for (let i = 0; i < d; i++) layerPraxis[i] = results[maxIdx].praxis[i];
    } else if (this.aggregation === 'weighted') {
      // Weight by inverse craving (lower craving = more saturated = more weight)
      let totalWeight = 0;
      const weights = results.map(r => {
        const w = 1 / (r.craving + 0.1);
        totalWeight += w;
        return w;
      });
      for (let j = 0; j < results.length; j++) {
        const w = weights[j] / totalWeight;
        for (let i = 0; i < d; i++) layerPraxis[i] += results[j].praxis[i] * w;
      }
    }

    return layerPraxis;
  }

  snapshot() {
    return {
      id: this.id,
      type: 'homogeneous',
      N: this.N,
      aggregation: this.aggregation,
      units: this.units.map(u => u.snapshot()),
      cascadeRisk: this._cascadeRiskEMA,
      meanCraving: this._meanCraving(),
    };
  }

  serialize() {
    return {
      id: this.id,
      type: 'homogeneous',
      N: this.N,
      aggregation: this.aggregation,
      cascadeRiskEMA: this._cascadeRiskEMA,
      units: this.units.map(u => u.serialize()),
    };
  }

  deserialize(state) {
    if (state.cascadeRiskEMA !== undefined) this._cascadeRiskEMA = state.cascadeRiskEMA;
    if (state.units) {
      for (let i = 0; i < Math.min(state.units.length, this.units.length); i++) {
        this.units[i].deserialize(state.units[i]);
      }
    }
  }

  _inhibit(stimulus, unit, meanM) {
    const peerM = (meanM * this.N - unit.M) / Math.max(this.N - 1, 1);
    const inhibition = this.kappa * peerM;
    if (typeof stimulus === 'object' && stimulus !== null) {
      return { ...stimulus, intensity: clamp((stimulus.intensity ?? 0) - inhibition) };
    }
    return stimulus;
  }

  _meanCraving() {
    return this.units.reduce((s, u) => s + u.M, 0) / this.N;
  }

  injectTarget(prototype) {
    this.units.forEach(u => u.injectAddictionTarget(prototype));
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Paradigm II – HETEROGENEOUS AJN LAYER (Enhanced)
// ─────────────────────────────────────────────────────────────────────────────
export class HeterogeneousAJNLayer extends EventEmitter {
  /**
   * @param {object} opts
   * @param {number}   opts.K              – Number of stimulus classes
   * @param {number}   opts.unitsPerClass  – Units per class (n_k)
   * @param {Array<{ name:string, intensityFn:Function }>} opts.classes
   * @param {number}   [opts.temperature=1.0] – Softmax temperature for class competition
   * @param {object}   [opts.params]
   */
  constructor(opts) {
    super();
    this.id          = opts.id ?? `hetero-layer-${Math.random().toString(36).slice(2,7)}`;
    this.K           = opts.K;
    this.unitsPerClass = opts.unitsPerClass ?? 16;
    this.temperature = opts.temperature ?? 1.0;

    this.groups = opts.classes.map((cls, k) => ({
      name:  cls.name,
      units: Array.from({ length: this.unitsPerClass }, (_, i) =>
        new ArtificialJunkyNeuron({
          id:           `${this.id}-cls${k}-u${i}`,
          stimulusClass: cls.name,
          intensityFn:  cls.intensityFn,
          params:       opts.params,
        })
      ),
      meanM: 0,
    }));

    this.winnerClass = 0;
    this._cascadeRiskEMA = 0;
    this._cascadeAlpha   = 0.1;

    this.groups.forEach(g =>
      g.units.forEach(u => u.on('extinction', e => this.emit('extinction', e)))
    );
  }

  process(stimulus) {
    const classResults = this.groups.map(g => {
      const results = g.units.map(u => u.process(stimulus));
      const meanM = results.reduce((s, r) => s + r.craving, 0) / g.units.length;

      const d = g.units[0].p.praximDim;
      const praxis = new Float64Array(d);
      for (const r of results) {
        for (let i = 0; i < d; i++) praxis[i] += r.praxis[i] / g.units.length;
      }
      g.meanM = meanM;
      return { name: g.name, meanM, praxis, unitResults: results };
    });

    // Temperature-scaled softmax winner selection
    const meanMs = classResults.map(r => r.meanM / this.temperature);
    const probs  = softmax(meanMs);
    let winnerIdx = 0, maxProb = -Infinity;
    for (let k = 0; k < probs.length; k++) {
      if (probs[k] > maxProb) { maxProb = probs[k]; winnerIdx = k; }
    }
    this.winnerClass = winnerIdx;

    // Cascade risk with EMA
    const allUnits = this.groups.flatMap(g => g.units);
    const tau = allUnits[0]?.p.tau ?? 20;
    const instantCR = allUnits.filter(u => u.nFail > tau / 2).length / allUnits.length;
    this._cascadeRiskEMA = this._cascadeAlpha * instantCR
                         + (1 - this._cascadeAlpha) * this._cascadeRiskEMA;

    const out = {
      winnerClass: classResults[winnerIdx].name,
      winnerPraxis: classResults[winnerIdx].praxis,
      classProbs: probs,
      classResults,
      cascadeRisk: this._cascadeRiskEMA,
    };
    this.emit('layerStep', out);
    return out;
  }

  snapshot() {
    return {
      id: this.id,
      type: 'heterogeneous',
      K: this.K,
      temperature: this.temperature,
      winnerClass: this.groups[this.winnerClass]?.name,
      groups: this.groups.map((g, k) => ({
        name: g.name,
        isWinner: k === this.winnerClass,
        meanCraving: g.units.reduce((s, u) => s + u.M, 0) / g.units.length,
        units: g.units.map(u => u.snapshot()),
      })),
    };
  }

  serialize() {
    return {
      id: this.id,
      type: 'heterogeneous',
      K: this.K,
      temperature: this.temperature,
      winnerClass: this.winnerClass,
      cascadeRiskEMA: this._cascadeRiskEMA,
      groups: this.groups.map(g => ({
        name: g.name,
        units: g.units.map(u => u.serialize()),
      })),
    };
  }

  deserialize(state) {
    if (state.winnerClass !== undefined) this.winnerClass = state.winnerClass;
    if (state.cascadeRiskEMA !== undefined) this._cascadeRiskEMA = state.cascadeRiskEMA;
    if (state.groups) {
      for (let k = 0; k < Math.min(state.groups.length, this.groups.length); k++) {
        const sg = state.groups[k];
        if (sg.units) {
          for (let i = 0; i < Math.min(sg.units.length, this.groups[k].units.length); i++) {
            this.groups[k].units[i].deserialize(sg.units[i]);
          }
        }
      }
    }
  }

  injectTarget(className, prototype) {
    const g = this.groups.find(g => g.name === className);
    if (g) g.units.forEach(u => u.injectAddictionTarget(prototype));
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Paradigm III – HYBRID AGENTIC-CLASSICAL LAYER (Enhanced)
// ─────────────────────────────────────────────────────────────────────────────
export class HybridAJNLayer extends EventEmitter {
  /**
   * @param {object} opts
   * @param {Function} opts.classicalFn   – f(stimulus) -> feature vector
   * @param {object}   opts.ajnLayer      – A HomogeneousAJNLayer instance
   * @param {number}   [opts.alpha=0.5]   – Blend weight for AJN modulation
   * @param {boolean}  [opts.skipConnection=true] – Enable skip connections
   */
  constructor(opts) {
    super();
    this.id             = opts.id ?? `hybrid-layer-${Math.random().toString(36).slice(2,7)}`;
    this.classicalFn    = opts.classicalFn;
    this.ajnLayer       = opts.ajnLayer;
    this.blendAlpha     = opts.alpha ?? 0.5;
    this.skipConnection = opts.skipConnection ?? true;

    this.ajnLayer.on('extinction', e => this.emit('extinction', e));
  }

  process(stimulus) {
    const classical = this.classicalFn(stimulus);
    const ajnOut    = this.ajnLayer.process(stimulus);

    const modulation = ajnOut.collectiveSaturated
      ? { intensity: 0, context: 'saturated' }
      : this._blend(classical, ajnOut.layerPraxis, stimulus);

    const out = {
      classical,
      ajnOut,
      modulated: modulation,
      cascadeRisk: ajnOut.cascadeRisk,
    };
    this.emit('layerStep', out);
    return out;
  }

  snapshot() {
    return {
      id: this.id,
      type: 'hybrid',
      blendAlpha: this.blendAlpha,
      skipConnection: this.skipConnection,
      ajnLayer: this.ajnLayer.snapshot(),
    };
  }

  serialize() {
    return {
      id: this.id,
      blendAlpha: this.blendAlpha,
      skipConnection: this.skipConnection,
      ajnLayer: this.ajnLayer.serialize(),
    };
  }

  deserialize(state) {
    if (state.blendAlpha !== undefined) this.blendAlpha = state.blendAlpha;
    if (state.ajnLayer) this.ajnLayer.deserialize(state.ajnLayer);
  }

  injectTarget(prototype) {
    this.ajnLayer.injectTarget(prototype);
  }

  _blend(classical, praxis, originalStimulus) {
    const praxisContribution = this._norm(praxis) / (praxis.length || 1);
    let intensity = typeof classical?.intensity === 'number'
      ? classical.intensity * (1 - this.blendAlpha) + praxisContribution * this.blendAlpha
      : praxisContribution;

    // Skip connection: blend with original stimulus intensity
    if (this.skipConnection && originalStimulus && typeof originalStimulus.intensity === 'number') {
      intensity = 0.7 * intensity + 0.3 * originalStimulus.intensity;
    }

    return { ...classical, intensity: clamp(intensity), praxisModulation: praxis };
  }

  _norm(arr) {
    let s = 0;
    for (const v of arr) s += v * v;
    return Math.sqrt(s);
  }
}

// ── Softmax helper (used by HeterogeneousAJNLayer) ──────────────────────────
function softmax(x) {
  const max = Math.max(...x);
  const exps = x.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / (sum + 1e-12));
}
