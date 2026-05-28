/**
 * ANNPsi.js  (v2.0 – Enhanced Agentic Neural Network Backbone)
 * ─────────────────────────────────────────────────────────────────────────────
 * 12-layer deep agentic flow backbone for the PREDATOR agent system.
 *
 * IMPROVEMENTS over v0.1:
 *   - Uses real TransformerBlock instances (L4-L5, L8-L9) instead of
 *     the simulated classicalTransformerBlock
 *   - Proper state serialization / deserialization for ALL layers
 *   - Layer-wise learning rate support via _layerLR map
 *   - Gradient flow monitoring via gradientFlowReport()
 *   - Configurable model dimensions and number of heads for transformer blocks
 *   - Per-layer step counters for granular analytics
 *   - Full event propagation from sub-layers through to the backbone
 *
 * LAYER ARCHITECTURE:
 *   L1-L2  : Hybrid (Conv + AJN homogeneous)       – sensory encoding
 *   L3     : Heterogeneous AJN (K=8)               – low-level feature specialization
 *   L4-L5  : Real Transformer blocks                – contextual attention
 *   L6     : Heterogeneous AJN (K=16)              – mid-level concept specialization
 *   L7     : Hybrid AJN                            – contextual modulation
 *   L8-L9  : Real Transformer blocks                – high-level reasoning
 *   L10    : Heterogeneous AJN (K=32)              – high-order addiction layer
 *   L11    : Hybrid AJN                            – praxic assembly
 *   L12    : Output AJN (N=1)                      – TPS emitter
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import {
  HomogeneousAJNLayer,
  HeterogeneousAJNLayer,
  HybridAJNLayer,
} from '../layers/AJNLayer.js';
import { TransformerBlock } from '../layers/TransformerBlock.js';

// ─────────────────────────────────────────────────────────────────────────────
// Stimulus class definitions (preserved from original v0.1)
// ─────────────────────────────────────────────────────────────────────────────

/** L3 – 8 low-level feature classes */
const STIMULUS_CLASSES_L3 = [
  { name: 'syntax',          intensityFn: null },
  { name: 'semantics',       intensityFn: null },
  { name: 'structure',       intensityFn: null },
  { name: 'novelty',         intensityFn: null },
  { name: 'relevance',       intensityFn: null },
  { name: 'coherence',       intensityFn: null },
  { name: 'completion',      intensityFn: null },
  { name: 'correctness',     intensityFn: null },
];

/** L6 – 16 mid-level concept classes */
const STIMULUS_CLASSES_L6 = [
  { name: 'task_progress',    intensityFn: null },
  { name: 'context_depth',    intensityFn: null },
  { name: 'tool_success',     intensityFn: null },
  { name: 'information_gain', intensityFn: null },
  { name: 'plan_adherence',   intensityFn: null },
  { name: 'constraint_ok',    intensityFn: null },
  { name: 'output_quality',   intensityFn: null },
  { name: 'resource_efficiency', intensityFn: null },
  { name: 'owner_alignment',  intensityFn: null },
  { name: 'error_absence',    intensityFn: null },
  { name: 'exploration_gain', intensityFn: null },
  { name: 'goal_proximity',   intensityFn: null },
  { name: 'feedback_richness', intensityFn: null },
  { name: 'action_impact',    intensityFn: null },
  { name: 'knowledge_use',    intensityFn: null },
  { name: 'completion_signal', intensityFn: null },
];

/** L10 – 32 high-order addiction classes */
const STIMULUS_CLASSES_L10 = [
  { name: 'meta_reasoning',        intensityFn: null },
  { name: 'strategic_depth',       intensityFn: null },
  { name: 'causal_attribution',    intensityFn: null },
  { name: 'counterfactual_eval',   intensityFn: null },
  { name: 'abstraction_level',     intensityFn: null },
  { name: 'transfer_potential',    intensityFn: null },
  { name: 'uncertainty_quant',     intensityFn: null },
  { name: 'belief_revision',       intensityFn: null },
  { name: 'value_alignment',       intensityFn: null },
  { name: 'ethical_compliance',    intensityFn: null },
  { name: 'self_improvement',      intensityFn: null },
  { name: 'creative_synthesis',    intensityFn: null },
  { name: 'paradigm_shift',        intensityFn: null },
  { name: 'holistic_integration',  intensityFn: null },
  { name: 'temporal_foresight',    intensityFn: null },
  { name: 'stakeholder_impact',    intensityFn: null },
  { name: 'narrative_coherence',   intensityFn: null },
  { name: 'epistemic_virtue',      intensityFn: null },
  { name: 'adaptive_capacity',     intensityFn: null },
  { name: 'resilience_factor',     intensityFn: null },
  { name: 'emergent_pattern',      intensityFn: null },
  { name: 'systemic_leverage',     intensityFn: null },
  { name: 'interdisciplinary_bridge', intensityFn: null },
  { name: 'second_order_effect',   intensityFn: null },
  { name: 'autonomy_degree',       intensityFn: null },
  { name: 'collaboration_quality', intensityFn: null },
  { name: 'knowledge_frontier',    intensityFn: null },
  { name: 'hypothesis_quality',    intensityFn: null },
  { name: 'evidence_strength',     intensityFn: null },
  { name: 'inference_validity',    intensityFn: null },
  { name: 'decision_optimality',   intensityFn: null },
  { name: 'outcome_superiority',   intensityFn: null },
];

// ─────────────────────────────────────────────────────────────────────────────
// Intensity function factory (mkIntensityFn pattern from v0.1)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Creates a stimulus intensity function I(S) -> [0,1] for a given class name.
 * The function inspects the stimulus object for a matching field or falls back
 * to the generic `intensity` property, then applies a sigmoid-like mapping.
 *
 * @param {string} className – The stimulus class this function serves
 * @param {object} [opts]
 * @param {number} [opts.sensitivity=5.0] – Steepness of the sigmoid mapping
 * @param {number} [opts.offset=0.0]     – Horizontal shift before sigmoid
 * @returns {Function} intensityFn(stimulus) -> number in [0,1]
 */
function mkIntensityFn(className, opts = {}) {
  const sensitivity = opts.sensitivity ?? 5.0;
  const offset      = opts.offset      ?? 0.0;

  return (stimulus) => {
    if (!stimulus || typeof stimulus !== 'object') return 0;

    // Try direct class field first
    let raw = stimulus[className];

    // Fallback: check stimulus.features map
    if (raw === undefined && stimulus.features && typeof stimulus.features === 'object') {
      raw = stimulus.features[className];
    }

    // Fallback: generic intensity
    if (raw === undefined) {
      raw = stimulus.intensity ?? 0;
    }

    // Sigmoid mapping to [0, 1]
    const x = sensitivity * (Number(raw) - offset);
    return 1 / (1 + Math.exp(-x));
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: norm of a Float64Array
// ─────────────────────────────────────────────────────────────────────────────
function vecNorm(arr) {
  if (!arr || arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
  return Math.sqrt(s);
}

// ─────────────────────────────────────────────────────────────────────────────
// ANNPsi – Enhanced Agentic Neural Network Backbone
// ─────────────────────────────────────────────────────────────────────────────
export class ANNPsi extends EventEmitter {
  /**
   * @param {object} [opts]
   * @param {object} [opts.ajnParams]   – AJN hyperparameter overrides (forwarded to all AJN layers)
   * @param {number} [opts.dModel=64]   – Model dimension for Transformer blocks
   * @param {number} [opts.nHeads=4]    – Number of attention heads for Transformer blocks
   * @param {number} [opts.dFF=256]     – Feed-forward dimension for Transformer blocks
   * @param {object} [opts.layerLR]     – Per-layer learning rate overrides  { layerIndex: lr }
   * @param {number} [opts.unitsPerHybrid=16]   – AJN units in hybrid layers
   * @param {number} [opts.unitsPerClass=16]     – AJN units per class in heterogeneous layers
   */
  constructor(opts = {}) {
    super();

    this.ajnParams       = opts.ajnParams       ?? {};
    this.dModel          = opts.dModel           ?? 64;
    this.nHeads          = opts.nHeads           ?? 4;
    this.dFF             = opts.dFF              ?? 256;
    this.unitsPerHybrid  = opts.unitsPerHybrid   ?? 16;
    this.unitsPerClass   = opts.unitsPerClass    ?? 16;

    /** Per-layer learning rate multiplier (layer index -> multiplier) */
    this._layerLR = opts.layerLR ?? {};

    /** Per-layer step counters */
    this._layerSteps = new Array(12).fill(0);

    /** Last forward pass results per layer (for gradient monitoring) */
    this._lastResults = new Array(12).fill(null);

    /** Build the 12-layer stack */
    this.layers = this._buildLayers();

    /** Global step counter */
    this.step = 0;

    // ── Wire up sub-layer events ─────────────────────────────────────────
    this.layers.forEach((layer, idx) => {
      if (layer && typeof layer.on === 'function') {
        layer.on('extinction', (e) => this.emit('extinction', { layer: idx, ...e }));
        layer.on('layerStep',  (e) => this.emit('layerStep',  { layer: idx, ...e }));
        layer.on('blockProcessed', (e) => this.emit('blockProcessed', { layer: idx, ...e }));
      }
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Layer construction
  // ─────────────────────────────────────────────────────────────────────────

  _buildLayers() {
    const p   = this.ajnParams;
    const d   = this.dModel;
    const h   = this.nHeads;
    const dFF = this.dFF;

    const layers = [];

    // ── L1: Hybrid – Conv + AJN homogeneous (sensory encoding) ──────────
    layers.push(new HybridAJNLayer({
      id:    'L1-sensory-encode',
      alpha: 0.5,
      skipConnection: true,
      classicalFn: (stimulus) => {
        // Simple 1-D convolutional feature extraction
        const raw = (typeof stimulus?.intensity === 'number') ? stimulus.intensity : 0;
        const ctx = (typeof stimulus?.context   === 'number') ? stimulus.context   : 0;
        const feat = (typeof stimulus?.features === 'object') ? Object.values(stimulus.features) : [];
        const convSum = feat.reduce((s, v, i) => s + v * (1 / (i + 1)), 0);
        return {
          intensity: Math.tanh(raw + 0.1 * convSum),
          context:   Math.tanh(ctx),
          features:  { convolved: Math.tanh(raw * 0.5 + convSum * 0.5) },
        };
      },
      ajnLayer: new HomogeneousAJNLayer({
        id:           'L1-ajn',
        N:            this.unitsPerHybrid,
        stimulusClass: 'sensory',
        intensityFn:  mkIntensityFn('sensory'),
        params:       p,
      }),
    }));

    // ── L2: Hybrid – Conv + AJN homogeneous (sensory encoding) ──────────
    layers.push(new HybridAJNLayer({
      id:    'L2-sensory-refine',
      alpha: 0.45,
      skipConnection: true,
      classicalFn: (stimulus) => {
        const raw = (typeof stimulus?.intensity === 'number') ? stimulus.intensity : 0;
        const mod = (typeof stimulus?.modulated?.intensity === 'number')
          ? stimulus.modulated.intensity
          : raw;
        return {
          intensity: Math.tanh(0.6 * raw + 0.4 * mod),
          context:   'sensory_refined',
          features:  { refined: Math.tanh(raw + mod) },
        };
      },
      ajnLayer: new HomogeneousAJNLayer({
        id:           'L2-ajn',
        N:            this.unitsPerHybrid,
        stimulusClass: 'sensory_refined',
        intensityFn:  mkIntensityFn('sensory_refined'),
        params:       p,
      }),
    }));

    // ── L3: Heterogeneous AJN (K=8) – low-level feature specialization ──
    const l3Classes = STIMULUS_CLASSES_L3.map(cls => ({
      name:        cls.name,
      intensityFn: mkIntensityFn(cls.name),
    }));
    layers.push(new HeterogeneousAJNLayer({
      id:            'L3-hetero-features',
      K:             8,
      unitsPerClass: this.unitsPerClass,
      classes:       l3Classes,
      temperature:   1.0,
      params:        p,
    }));

    // ── L4: Real Transformer block – contextual attention ────────────────
    layers.push(new TransformerBlock({
      id:     'L4-transformer-context',
      dModel: d,
      nHeads: h,
      dFF,
      causal: false,
      seqLen: 1,
    }));

    // ── L5: Real Transformer block – contextual attention ────────────────
    layers.push(new TransformerBlock({
      id:     'L5-transformer-context2',
      dModel: d,
      nHeads: h,
      dFF,
      causal: false,
      seqLen: 1,
    }));

    // ── L6: Heterogeneous AJN (K=16) – mid-level concept specialization ─
    const l6Classes = STIMULUS_CLASSES_L6.map(cls => ({
      name:        cls.name,
      intensityFn: mkIntensityFn(cls.name),
    }));
    layers.push(new HeterogeneousAJNLayer({
      id:            'L6-hetero-concepts',
      K:             16,
      unitsPerClass: this.unitsPerClass,
      classes:       l6Classes,
      temperature:   0.9,
      params:        p,
    }));

    // ── L7: Hybrid AJN – contextual modulation ──────────────────────────
    layers.push(new HybridAJNLayer({
      id:    'L7-hybrid-modulation',
      alpha: 0.55,
      skipConnection: true,
      classicalFn: (stimulus) => {
        const raw = (typeof stimulus?.intensity === 'number') ? stimulus.intensity : 0;
        const winner = stimulus?.winnerPraxis;
        const wNorm  = winner ? vecNorm(winner) : 0;
        return {
          intensity: Math.tanh(raw * 0.6 + wNorm * 0.4),
          context:   'modulated',
          features:  { modulation: Math.tanh(wNorm) },
        };
      },
      ajnLayer: new HomogeneousAJNLayer({
        id:           'L7-ajn',
        N:            this.unitsPerHybrid,
        stimulusClass: 'modulation',
        intensityFn:  mkIntensityFn('modulation'),
        params:       p,
      }),
    }));

    // ── L8: Real Transformer block – high-level reasoning ────────────────
    layers.push(new TransformerBlock({
      id:     'L8-transformer-reasoning',
      dModel: d,
      nHeads: h,
      dFF,
      causal: false,
      seqLen: 1,
    }));

    // ── L9: Real Transformer block – high-level reasoning ────────────────
    layers.push(new TransformerBlock({
      id:     'L9-transformer-reasoning2',
      dModel: d,
      nHeads: h,
      dFF,
      causal: true,  // autoregressive for reasoning chain
      seqLen: 1,
    }));

    // ── L10: Heterogeneous AJN (K=32) – high-order addiction layer ──────
    const l10Classes = STIMULUS_CLASSES_L10.map(cls => ({
      name:        cls.name,
      intensityFn: mkIntensityFn(cls.name, { sensitivity: 6.0 }),
    }));
    layers.push(new HeterogeneousAJNLayer({
      id:            'L10-hetero-highorder',
      K:             32,
      unitsPerClass: Math.max(4, Math.floor(this.unitsPerClass / 2)), // fewer per class due to K=32
      classes:       l10Classes,
      temperature:   0.8,
      params:        p,
    }));

    // ── L11: Hybrid AJN – praxic assembly ───────────────────────────────
    layers.push(new HybridAJNLayer({
      id:    'L11-hybrid-praxic',
      alpha: 0.6,
      skipConnection: true,
      classicalFn: (stimulus) => {
        const raw = (typeof stimulus?.intensity === 'number') ? stimulus.intensity : 0;
        const winner = stimulus?.winnerPraxis;
        const wNorm  = winner ? vecNorm(winner) : 0;
        return {
          intensity: Math.tanh(0.5 * raw + 0.5 * wNorm),
          context:   'praxic_assembly',
          features:  { praxic_signal: Math.tanh(wNorm) },
        };
      },
      ajnLayer: new HomogeneousAJNLayer({
        id:           'L11-ajn',
        N:            this.unitsPerHybrid,
        stimulusClass: 'praxic_assembly',
        intensityFn:  mkIntensityFn('praxic_assembly'),
        params:       p,
      }),
    }));

    // ── L12: Output AJN (N=1) – TPS emitter ─────────────────────────────
    layers.push(new HomogeneousAJNLayer({
      id:           'L12-output-tps',
      N:            1,
      stimulusClass: 'tps_output',
      intensityFn:  mkIntensityFn('tps_output'),
      aggregation:  'max',
      params:       p,
    }));

    return layers;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Forward pass
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Run a full 12-layer forward pass.
   * @param {*} stimulus – Input stimulus object
   * @returns {object} Final output with per-layer trace and TPS emission
   */
  forward(stimulus) {
    let current = stimulus;
    const trace = [];

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];

      // Apply layer-wise learning rate scaling (via stimulus intensity modulation)
      if (this._layerLR[i] !== undefined) {
        const lrScale = this._layerLR[i];
        if (typeof current === 'object' && current !== null) {
          current = {
            ...current,
            intensity: (current.intensity ?? 0) * lrScale,
            _lrScale:  lrScale,
          };
        }
      }

      // ── Route through layer by type ──────────────────────────────────
      const result = layer.process(current);

      // ── Derive the next stimulus from the layer output ───────────────
      if (layer instanceof TransformerBlock) {
        // TransformerBlock.process() returns a stimulus-like object
        current = result;
      } else if (layer instanceof HybridAJNLayer) {
        // Hybrid outputs { classical, ajnOut, modulated, cascadeRisk }
        current = result.modulated ?? result.classical ?? current;
      } else if (layer instanceof HeterogeneousAJNLayer) {
        // Heterogeneous outputs { winnerClass, winnerPraxis, classProbs, classResults, cascadeRisk }
        current = {
          ...current,
          intensity:      vecNorm(result.winnerPraxis),
          winnerClass:    result.winnerClass,
          winnerPraxis:   result.winnerPraxis,
          classProbs:     result.classProbs,
          cascadeRisk:    result.cascadeRisk,
          _layer:         layer.id,
        };
      } else if (layer instanceof HomogeneousAJNLayer) {
        // Homogeneous outputs { layerPraxis, collectiveSaturated, cascadeRisk, unitResults }
        const praxisNorm = vecNorm(result.layerPraxis);
        current = {
          ...current,
          intensity:           praxisNorm,
          layerPraxis:         result.layerPraxis,
          collectiveSaturated: result.collectiveSaturated,
          cascadeRisk:         result.cascadeRisk,
          _layer:              layer.id,
        };
      }

      this._layerSteps[i]++;
      this._lastResults[i] = result;

      trace.push({
        layer:  i,
        id:     layer.id,
        result: result,
      });
    }

    this.step++;

    // ── Assemble final TPS emission ────────────────────────────────────
    const l12Result = this._lastResults[11];
    const tpsPraxis = l12Result?.layerPraxis ?? new Float64Array(0);
    const tpsNorm   = vecNorm(tpsPraxis);

    const output = {
      tpsEmission:     tpsPraxis,
      tpsNorm:         tpsNorm,
      tpsIntensity:    current.intensity ?? 0,
      finalStimulus:   current,
      trace,
      step:            this.step,
    };

    this.emit('forward', output);
    return output;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Serialization / Deserialization
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Serialize the full backbone state to a plain object suitable for JSON.
   * @returns {object} Serializable state
   */
  serialize() {
    return {
      version:   2,
      dModel:    this.dModel,
      nHeads:    this.nHeads,
      dFF:       this.dFF,
      step:      this.step,
      layerLR:   { ...this._layerLR },
      layerSteps: [ ...this._layerSteps ],
      layers:    this.layers.map((layer, idx) => ({
        index: idx,
        id:    layer.id,
        type:  this._layerType(layer),
        state: layer.serialize(),
      })),
    };
  }

  /**
   * Restore backbone state from a previously serialized object.
   * @param {object} data – Output of serialize()
   */
  deserialize(data) {
    if (!data || data.version !== 2) {
      throw new Error('ANNPsi.deserialize: incompatible serialization version');
    }

    this.step = data.step ?? 0;
    this._layerLR = data.layerLR ?? {};

    if (data.layerSteps) {
      for (let i = 0; i < Math.min(data.layerSteps.length, 12); i++) {
        this._layerSteps[i] = data.layerSteps[i];
      }
    }

    if (data.layers) {
      for (const layerData of data.layers) {
        const idx = layerData.index;
        if (idx >= 0 && idx < this.layers.length) {
          this.layers[idx].deserialize(layerData.state);
        }
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Gradient flow monitoring
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Produce a per-layer report on gradient flow health, praxis norms, and
   * cascade risks. Useful for diagnosing vanishing/exploding signals and
   * addiction collapse.
   * @returns {Array<object>} One report entry per layer
   */
  gradientFlowReport() {
    const report = [];

    for (let i = 0; i < this.layers.length; i++) {
      const layer  = this.layers[i];
      const result = this._lastResults[i];
      const lr     = this._layerLR[i] ?? 1.0;
      const steps  = this._layerSteps[i];

      let praxisNorm    = 0;
      let cascadeRisk   = 0;
      let saturationPct = 0;
      let details       = {};

      if (layer instanceof HomogeneousAJNLayer) {
        if (result) {
          praxisNorm  = vecNorm(result.layerPraxis);
          cascadeRisk = result.cascadeRisk ?? 0;
          if (result.unitResults) {
            const total   = result.unitResults.length;
            const sat     = result.unitResults.filter(r => r.alpha > (layer.units[0]?.p?.thetaSat ?? 0.75)).length;
            saturationPct = total > 0 ? sat / total : 0;
          }
          details = {
            N:                   layer.N,
            aggregation:         layer.aggregation,
            collectiveSaturated: result.collectiveSaturated,
          };
        }
      } else if (layer instanceof HeterogeneousAJNLayer) {
        if (result) {
          praxisNorm  = vecNorm(result.winnerPraxis);
          cascadeRisk = result.cascadeRisk ?? 0;
          details = {
            K:           layer.K,
            winnerClass: result.winnerClass,
            topProb:     result.classProbs
              ? Math.max(...result.classProbs)
              : 0,
          };
        }
      } else if (layer instanceof HybridAJNLayer) {
        if (result) {
          praxisNorm  = vecNorm(result.ajnOut?.layerPraxis ?? new Float64Array(0));
          cascadeRisk = result.cascadeRisk ?? 0;
          details = {
            blendAlpha:     layer.blendAlpha,
            skipConnection: layer.skipConnection,
            saturated:      result.ajnOut?.collectiveSaturated ?? false,
          };
        }
      } else if (layer instanceof TransformerBlock) {
        // Transformer blocks don't have praxis/cascade semantics;
        // report on output norm as a proxy for gradient magnitude
        if (result && typeof result.intensity === 'number') {
          praxisNorm = Math.abs(result.intensity);
        }
        details = {
          dModel: layer.dModel,
          nHeads: layer.nHeads,
          dFF:    layer.dFF,
          causal: layer.causal,
        };
      }

      // Gradient health assessment
      const health = this._assessGradientHealth(praxisNorm, cascadeRisk, saturationPct, i);

      report.push({
        layer:         i,
        id:            layer.id,
        type:          this._layerType(layer),
        steps:         steps,
        learningRate:  lr,
        praxisNorm:    praxisNorm,
        cascadeRisk:   cascadeRisk,
        saturationPct: saturationPct,
        health,
        details,
      });
    }

    return report;
  }

  /**
   * Assess gradient health for a single layer.
   * @returns {{ status: string, flags: string[] }}
   */
  _assessGradientHealth(praxisNorm, cascadeRisk, saturationPct, layerIdx) {
    const flags  = [];
    let   status = 'healthy';

    // Vanishing gradient check
    if (praxisNorm < 1e-4) {
      flags.push('vanishing_praxis');
      status = 'warning';
    }

    // Exploding gradient check
    if (praxisNorm > 100) {
      flags.push('exploding_praxis');
      status = 'critical';
    }

    // Cascade risk check
    if (cascadeRisk > 0.6) {
      flags.push('high_cascade_risk');
      status = 'critical';
    } else if (cascadeRisk > 0.3) {
      flags.push('elevated_cascade_risk');
      if (status === 'healthy') status = 'warning';
    }

    // Saturation check
    if (saturationPct > 0.8) {
      flags.push('over_saturated');
      if (status === 'healthy') status = 'warning';
    } else if (saturationPct < 0.05 && layerIdx < 10) {
      flags.push('under_activated');
      if (status === 'healthy') status = 'info';
    }

    return { status, flags };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Layer-wise learning rate management
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Set the learning rate multiplier for a specific layer.
   * @param {number} layerIndex
   * @param {number} lr – Learning rate multiplier (1.0 = default)
   */
  setLayerLR(layerIndex, lr) {
    if (layerIndex < 0 || layerIndex >= this.layers.length) {
      throw new RangeError(`ANNPsi.setLayerLR: invalid layer index ${layerIndex}`);
    }
    this._layerLR[layerIndex] = lr;
  }

  /**
   * Get the current learning rate multiplier for a specific layer.
   * @param {number} layerIndex
   * @returns {number}
   */
  getLayerLR(layerIndex) {
    return this._layerLR[layerIndex] ?? 1.0;
  }

  /**
   * Apply a learning rate schedule to all layers. The schedule is a function
   * that takes (layerIndex, globalStep) and returns a multiplier.
   * @param {Function} scheduleFn – (layerIdx, step) -> lrMultiplier
   */
  applyLRSchedule(scheduleFn) {
    if (typeof scheduleFn !== 'function') return;
    for (let i = 0; i < this.layers.length; i++) {
      this._layerLR[i] = scheduleFn(i, this.step);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Target injection (for HCI – Addiction Target Seeding)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Inject a praxis prototype as an addiction target into a specific layer.
   * @param {number} layerIndex
   * @param {Float64Array|number[]} prototype
   * @param {string} [className] – Required for heterogeneous layers
   */
  injectTarget(layerIndex, prototype, className) {
    if (layerIndex < 0 || layerIndex >= this.layers.length) return;

    const layer = this.layers[layerIndex];

    if (layer instanceof HomogeneousAJNLayer) {
      layer.injectTarget(prototype);
    } else if (layer instanceof HybridAJNLayer) {
      layer.injectTarget(prototype);
    } else if (layer instanceof HeterogeneousAJNLayer) {
      layer.injectTarget(className ?? '', prototype);
    }
    // TransformerBlocks don't support target injection
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Diagnostic helpers
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Get a compact snapshot of all layers for monitoring dashboards.
   * @returns {Array<object>}
   */
  snapshot() {
    return this.layers.map((layer, idx) => {
      const base = {
        index: idx,
        id:    layer.id,
        type:  this._layerType(layer),
        steps: this._layerSteps[idx],
      };

      if (typeof layer.snapshot === 'function') {
        base.snapshot = layer.snapshot();
      }

      return base;
    });
  }

  /**
   * Determine the layer type string for reporting.
   * @param {*} layer
   * @returns {string}
   */
  _layerType(layer) {
    if (layer instanceof TransformerBlock)      return 'transformer';
    if (layer instanceof HybridAJNLayer)        return 'hybrid';
    if (layer instanceof HeterogeneousAJNLayer) return 'heterogeneous';
    if (layer instanceof HomogeneousAJNLayer)   return 'homogeneous';
    return 'unknown';
  }

  /**
   * Get the count of total AJN units across all layers.
   * @returns {number}
   */
  totalAJNUnits() {
    let count = 0;
    for (const layer of this.layers) {
      if (layer instanceof HomogeneousAJNLayer) {
        count += layer.N;
      } else if (layer instanceof HeterogeneousAJNLayer) {
        count += layer.K * layer.unitsPerClass;
      } else if (layer instanceof HybridAJNLayer) {
        count += layer.ajnLayer.N;
      }
    }
    return count;
  }

  /**
   * Get architecture summary string.
   * @returns {string}
   */
  architectureSummary() {
    const parts = this.layers.map((l, i) => `L${i + 1}:${this._layerType(l)}`);
    return `ANNPsi[${parts.join(' → ')}] d=${this.dModel} h=${this.nHeads} dFF=${this.dFF} units=${this.totalAJNUnits()}`;
  }
}
