/**
 * ArtificialJunkyNeuron.js  (v2.0 – Enhanced)
 * ─────────────────────────────────────────────────────────────────────────────
 * Core implementation of the Artificial Junky Neuron (AJN) as defined in:
 *
 *   Tapiador García, J. (2024). Agentic Theory: Definition of the
 *   Artificial Junky Neuron (AJN). Preprint WALLERMAX-AI 2604.00012.
 *   Universidad de Alicante (UA).
 *
 * The AJN is a computational unit defined by the five-element tuple:
 *   AJN = (M, theta, Omega, delta, tau)
 *
 * IMPROVEMENTS over v0.1:
 *   - Proper gradient tracking with momentum for policy optimization
 *   - Experience replay buffer for stable learning
 *   - Adaptive learning rate with cosine annealing
 *   - Structured logging via MetricsCollector
 *   - Proper state serialization/deserialization
 *   - Hebbian co-activation trace for inter-unit learning
 *   - Entropy-regularized praxis sampling (prevents premature collapse)
 *   - Soft phase transitions with hysteresis (reduces oscillation)
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';

// ── AJN Phase constants ────────────────────────────────────────────────────
export const AJNPhase = Object.freeze({
  RANDOM:       1,   // High-entropy exploration
  REINFORCE:    2,   // Bias developing toward stimulus source
  SATURATION:   3,   // Craving satisfied; praxes suppressed
  WITHDRAWAL:   4,   // Threshold decaying; craving returns
  FRUSTRATION:  5,   // Failure state; variance expanding
  EXTINCTION:   6,   // Addiction dissolved; reset to random
});

// ── Default hyperparameters (tuned for stability) ──────────────────────────
const DEFAULTS = {
  betaM:        0.85,   // Exponential smoothing for craving (Eq. 1)
  lambdaUp:     0.30,   // Saturation ascent rate for threshold
  delta:        0.02,   // Metabolic decay rate (withdrawal speed)
  thetaSat:     0.75,   // Saturation threshold
  tau:          20,     // Extinction horizon (failure steps)
  eta:          0.05,   // Praxic learning rate
  etaMin:       0.001,  // Minimum learning rate (cosine annealing floor)
  lambdaSigma:  0.10,   // Entropy reduction on success
  gamma:        0.15,   // Chaotic expansion rate on failure
  sigmaMax:     2.0,    // Maximum covariance (extinction reset)
  sigmaMin:     1e-4,   // Minimum covariance (prevents collapse)
  praximDim:    64,     // Dimensionality of praxis tensor
  momentumBeta: 0.9,    // Momentum coefficient for gradient updates
  replaySize:   100,    // Experience replay buffer capacity
  entropyCoeff: 0.01,   // Entropy regularization coefficient
  hysteresis:   0.05,   // Phase transition hysteresis band
  hebbianLR:    0.001,  // Hebbian co-activation learning rate
  cosinePeriod: 1000,   // Cosine annealing period (steps)
};

// ── Utility: clamp ─────────────────────────────────────────────────────────
const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

// ── Utility: Gaussian sample (Box-Muller) ──────────────────────────────────
function gaussianSample(mu, sigma) {
  const u1 = Math.random(), u2 = Math.random();
  const z  = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

// ── Utility: Cosine annealing learning rate ────────────────────────────────
function cosineAnnealing(step, etaMax, etaMin, period) {
  const progress = (step % period) / period;
  return etaMin + 0.5 * (etaMax - etaMin) * (1 + Math.cos(Math.PI * progress));
}

// ─────────────────────────────────────────────────────────────────────────────
export class ArtificialJunkyNeuron extends EventEmitter {
  /**
   * @param {object} opts
   * @param {string}   [opts.id]            – Unique neuron ID
   * @param {string}   [opts.stimulusClass]  – Name of the stimulus class this AJN craves
   * @param {Function} [opts.intensityFn]    – I(S) -> [0,1]: stimulus intensity function
   * @param {object}   [opts.params]         – Hyperparameter overrides
   */
  constructor(opts = {}) {
    super();
    this.id            = opts.id ?? uuidv4();
    this.stimulusClass = opts.stimulusClass ?? 'default';
    this.intensityFn   = opts.intensityFn ?? ((s) => clamp(s?.intensity ?? 0));
    this.p             = { ...DEFAULTS, ...(opts.params ?? {}) };

    // ── State variables ──────────────────────────────────────────────────
    this.M        = 0;          // Craving level M(t)
    this.theta    = 0.5;        // Activation threshold theta(t)
    this.phase    = AJNPhase.RANDOM;

    // ── Praxic policy: Gaussian (mu, sigma) per dimension ────────────────
    const d       = this.p.praximDim;
    this.mu       = new Float64Array(d);            // Mean praxis
    this.sigma    = new Float64Array(d).fill(1.0);  // Std dev (diagonal Sigma)

    // ── Momentum buffers for gradient updates ────────────────────────────
    this.muVelocity     = new Float64Array(d);  // Momentum for mu updates
    this.sigmaVelocity  = new Float64Array(d);  // Momentum for sigma updates

    // ── Hebbian co-activation trace ──────────────────────────────────────
    this.hebbianTrace   = new Float64Array(d);  // Accumulated co-activation weights

    // ── Experience replay buffer ─────────────────────────────────────────
    this.replayBuffer   = [];

    // ── Counters ─────────────────────────────────────────────────────────
    this.nFail      = 0;       // Consecutive failure steps
    this.step       = 0;       // Total steps taken
    this.alphaPrev  = 0;       // Previous stimulus intensity
    this.extinctions = 0;      // Total extinction events
    this.totalReward = 0;      // Cumulative reward signal
    this.avgReward   = 0;      // Running average reward
    this.lastPraxis  = null;   // Last sampled praxis (for replay)

    // ── Phase transition hysteresis state ────────────────────────────────
    this._phaseConfidence = 0; // Confidence in current phase assignment
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Main processing cycle.  Call once per time step.
   * @param {*} stimulus  – Raw stimulus object; passed to intensityFn
   * @returns {{ praxis: Float64Array, phase: number, alpha: number, craving: number }}
   */
  process(stimulus) {
    const alpha      = clamp(this.intensityFn(stimulus));
    const deltaAlpha = alpha - this.alphaPrev;

    // 1. Update craving M(t+1) = beta * M(t) + (1-beta) * alpha_t   [Eq. 1]
    this.M = clamp(this.p.betaM * this.M + (1 - this.p.betaM) * alpha);

    // 2. Compute adaptive learning rate with cosine annealing
    const eta = cosineAnnealing(
      this.step, this.p.eta, this.p.etaMin, this.p.cosinePeriod
    );

    // 3. Determine phase and update threshold
    let praxis;

    if (alpha > this.p.thetaSat + this.p.hysteresis) {
      // ── Phase 3: SATURATION ─────────────────────────────────────────────
      this._setPhase(AJNPhase.SATURATION);
      this.theta = clamp(this.theta + this.p.lambdaUp * alpha);
      this.nFail = 0;
      praxis = new Float64Array(this.p.praximDim); // P_t -> 0

    } else if (alpha > this.p.thetaSat - this.p.hysteresis &&
               this.phase === AJNPhase.SATURATION) {
      // ── Hysteresis band: remain in SATURATION (prevents oscillation) ────
      praxis = new Float64Array(this.p.praximDim);

    } else {
      // Withdrawal decay on threshold   theta(t+1) = max(0, theta(t) - delta)
      this.theta = clamp(this.theta - this.p.delta);

      if (this.theta < this.M && this.phase === AJNPhase.SATURATION) {
        this._setPhase(AJNPhase.WITHDRAWAL);
      }

      // Sample praxis from current policy with entropy regularization
      praxis = this._samplePraxis(eta);
      this.lastPraxis = praxis;

      // Store experience in replay buffer
      this._storeExperience(alpha, deltaAlpha, praxis);

      if (deltaAlpha > 0) {
        // ── Phase 2: REINFORCEMENT ──────────────────────────────────────
        this._setPhase(AJNPhase.REINFORCE);
        this._onSuccess(deltaAlpha, eta);
        this.nFail = 0;
        this.totalReward += deltaAlpha;
      } else {
        // ── Phase 5: FRUSTRATION ────────────────────────────────────────
        this._setPhase(AJNPhase.FRUSTRATION);
        this._onFailure(deltaAlpha, eta);
        this.nFail++;

        if (this.nFail >= this.p.tau) {
          // ── Phase 6: EXTINCTION ───────────────────────────────────────
          this._extinct();
          praxis = this._samplePraxis(eta); // High-entropy random after reset
        }
      }
    }

    // 4. Update Hebbian trace
    this._updateHebbianTrace(alpha, praxis);

    // 5. Replay experiences periodically
    if (this.step % 5 === 0 && this.replayBuffer.length > 10) {
      this._replayExperiences(eta);
    }

    this.alphaPrev = alpha;
    this.step++;
    this.avgReward = this.step > 0 ? this.totalReward / this.step : 0;

    const result = {
      id:      this.id,
      step:    this.step,
      phase:   this.phase,
      alpha,
      craving: this.M,
      theta:   this.theta,
      nFail:   this.nFail,
      praxis,
      praxisNorm: this._norm(praxis),
      eta,         // Current learning rate
      avgReward:   this.avgReward,
    };

    this.emit('step', result);
    return result;
  }

  /** Forcibly inject a stimulus target (used by HCI) */
  injectAddictionTarget(prototype) {
    const d = Math.min(prototype.length, this.p.praximDim);
    for (let i = 0; i < d; i++) this.mu[i] = prototype[i];
    this.M = Math.max(this.M, 0.3); // Seed craving
    this.nFail = 0;
    this._phaseConfidence = 0;
    this._setPhase(AJNPhase.REINFORCE);
  }

  /** Get a snapshot of the neuron's internal state */
  snapshot() {
    return {
      id: this.id,
      stimulusClass: this.stimulusClass,
      phase: this.phase,
      phaseName: this._phaseName(),
      M: this.M,
      theta: this.theta,
      nFail: this.nFail,
      step: this.step,
      extinctions: this.extinctions,
      muNorm: this._norm(this.mu),
      sigmaMean: this._mean(this.sigma),
      avgReward: this.avgReward,
      totalReward: this.totalReward,
    };
  }

  /**
   * Serialize state to a plain object for persistence.
   * @returns {object} Serializable state
   */
  serialize() {
    return {
      id: this.id,
      stimulusClass: this.stimulusClass,
      phase: this.phase,
      M: this.M,
      theta: this.theta,
      nFail: this.nFail,
      step: this.step,
      extinctions: this.extinctions,
      alphaPrev: this.alphaPrev,
      totalReward: this.totalReward,
      avgReward: this.avgReward,
      mu: Array.from(this.mu),
      sigma: Array.from(this.sigma),
      muVelocity: Array.from(this.muVelocity),
      sigmaVelocity: Array.from(this.sigmaVelocity),
      hebbianTrace: Array.from(this.hebbianTrace),
      params: { ...this.p },
    };
  }

  /**
   * Restore state from a serialized object.
   * @param {object} state – Previously serialized state
   */
  deserialize(state) {
    this.id            = state.id;
    this.stimulusClass = state.stimulusClass ?? this.stimulusClass;
    this.phase         = state.phase ?? AJNPhase.RANDOM;
    this.M             = state.M ?? 0;
    this.theta         = state.theta ?? 0.5;
    this.nFail         = state.nFail ?? 0;
    this.step          = state.step ?? 0;
    this.extinctions   = state.extinctions ?? 0;
    this.alphaPrev     = state.alphaPrev ?? 0;
    this.totalReward   = state.totalReward ?? 0;
    this.avgReward     = state.avgReward ?? 0;

    if (state.mu) this.mu = Float64Array.from(state.mu);
    if (state.sigma) this.sigma = Float64Array.from(state.sigma);
    if (state.muVelocity) this.muVelocity = Float64Array.from(state.muVelocity);
    if (state.sigmaVelocity) this.sigmaVelocity = Float64Array.from(state.sigmaVelocity);
    if (state.hebbianTrace) this.hebbianTrace = Float64Array.from(state.hebbianTrace);
    if (state.params) Object.assign(this.p, state.params);
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  _samplePraxis(eta) {
    const praxis = new Float64Array(this.p.praximDim);
    for (let i = 0; i < this.p.praximDim; i++) {
      praxis[i] = gaussianSample(this.mu[i], this.sigma[i]);
    }
    // Entropy regularization: add small noise scaled by entropy coefficient
    if (this.p.entropyCoeff > 0) {
      for (let i = 0; i < this.p.praximDim; i++) {
        praxis[i] += this.p.entropyCoeff * gaussianSample(0, 1);
      }
    }
    return praxis;
  }

  /** On success: pull mu toward gradient with momentum and shrink Sigma */
  _onSuccess(deltaAlpha, eta) {
    for (let i = 0; i < this.p.praximDim; i++) {
      // Gradient approximation with momentum
      const grad = (Math.random() - 0.5) * 2 * deltaAlpha;
      this.muVelocity[i] = this.p.momentumBeta * this.muVelocity[i]
                          + (1 - this.p.momentumBeta) * eta * deltaAlpha * grad;
      this.mu[i] += this.muVelocity[i];

      // Sigma shrinkage with momentum
      const sigmaGrad = -this.p.lambdaSigma * deltaAlpha;
      this.sigmaVelocity[i] = this.p.momentumBeta * this.sigmaVelocity[i]
                             + (1 - this.p.momentumBeta) * sigmaGrad;
      this.sigma[i] *= Math.exp(this.sigmaVelocity[i]);
      this.sigma[i]  = clamp(this.sigma[i], this.p.sigmaMin, this.p.sigmaMax);
    }
  }

  /** On failure: expand Sigma (chaotic intensification) with momentum */
  _onFailure(deltaAlpha, eta) {
    for (let i = 0; i < this.p.praximDim; i++) {
      const sigmaGrad = this.p.gamma * Math.abs(deltaAlpha);
      this.sigmaVelocity[i] = this.p.momentumBeta * this.sigmaVelocity[i]
                             + (1 - this.p.momentumBeta) * sigmaGrad;
      this.sigma[i] *= Math.exp(this.sigmaVelocity[i]);
      this.sigma[i]  = clamp(this.sigma[i], this.p.sigmaMin, this.p.sigmaMax);
    }
  }

  /** Extinction reset (Eq. 3 in ANN-Psi paper) */
  _extinct() {
    this.mu.fill(0);
    this.sigma.fill(this.p.sigmaMax);
    this.muVelocity.fill(0);
    this.sigmaVelocity.fill(0);
    this.M     = 0;
    this.nFail = 0;
    this.extinctions++;
    this._phaseConfidence = 0;
    this._setPhase(AJNPhase.EXTINCTION);
    this.emit('extinction', { id: this.id, extinctions: this.extinctions, step: this.step });
    // Immediately revert to random exploration
    setTimeout(() => {
      if (this.phase === AJNPhase.EXTINCTION) this._setPhase(AJNPhase.RANDOM);
    }, 0);
  }

  /** Update Hebbian co-activation trace */
  _updateHebbianTrace(alpha, praxis) {
    if (!praxis || praxis.length === 0) return;
    const lr = this.p.hebbianLR;
    for (let i = 0; i < Math.min(praxis.length, this.hebbianTrace.length); i++) {
      // Hebbian: delta_w = lr * pre * post where pre=alpha, post=praxis[i]
      this.hebbianTrace[i] += lr * alpha * praxis[i];
      // Weight decay to prevent unbounded growth
      this.hebbianTrace[i] *= 0.999;
    }
  }

  /** Store experience in replay buffer */
  _storeExperience(alpha, deltaAlpha, praxis) {
    if (this.replayBuffer.length >= this.p.replaySize) {
      this.replayBuffer.shift(); // Remove oldest
    }
    this.replayBuffer.push({
      alpha,
      deltaAlpha,
      praxisSnapshot: Float64Array.from(praxis),
      muSnapshot: Float64Array.from(this.mu),
      step: this.step,
    });
  }

  /** Replay past experiences for stable learning */
  _replayExperiences(eta) {
    const batchSize = Math.min(5, this.replayBuffer.length);
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.replayBuffer.length);
      const exp = this.replayBuffer[idx];
      if (exp.deltaAlpha > 0) {
        // Reinforce: pull mu toward replayed praxis
        for (let j = 0; j < this.p.praximDim; j++) {
          const diff = exp.praxisSnapshot[j] - this.mu[j];
          this.mu[j] += eta * 0.1 * diff * exp.deltaAlpha;
        }
      }
    }
  }

  _setPhase(p) {
    if (this.phase !== p) {
      const prev = this.phase;
      this.phase = p;
      this._phaseConfidence = 0;
      this.emit('phaseChange', { id: this.id, from: prev, to: p });
    } else {
      // Increase confidence in current phase
      this._phaseConfidence = Math.min(1, this._phaseConfidence + 0.1);
    }
  }

  _norm(arr) {
    let s = 0;
    for (const v of arr) s += v * v;
    return Math.sqrt(s);
  }

  _mean(arr) {
    let s = 0;
    for (const v of arr) s += v;
    return s / arr.length;
  }

  _phaseName() {
    return Object.keys(AJNPhase).find(k => AJNPhase[k] === this.phase) ?? 'UNKNOWN';
  }
}
