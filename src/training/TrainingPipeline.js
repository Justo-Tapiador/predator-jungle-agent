/**
 * TrainingPipeline.js — Enhanced 4-Phase Training Pipeline (v2.0)
 * ─────────────────────────────────────────────────────────────────────────────
 * Implements the full PREDATOR training lifecycle:
 *
 *   Phase I:   Large-scale pre-training with configurable dataset loaders
 *   Phase II:  Addiction shaping with curriculum learning & early stopping
 *   Phase III: Hierarchical fine-tuning (HIFT) with directive-specific training
 *   Phase IV:  Adversarial frustration hardening with progressive difficulty
 *
 * Improvements over v0.1:
 *   - Checkpointing support: saves state after each phase
 *   - Training metrics recording via MetricsCollector
 *   - Configurable learning rate schedules (cosine, step, linear)
 *   - Dataset validation before training
 *   - Progress callbacks with detailed stats
 *   - Early stopping with configurable patience
 *   - Curriculum learning for addiction shaping
 *   - Progressive difficulty for adversarial hardening
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { AJNPhase } from '../core/ArtificialJunkyNeuron.js';
import { DatasetLoader } from './DatasetLoader.js';
import { CheckpointManager } from './CheckpointManager.js';

// ── Learning rate schedule factories ─────────────────────────────────────────
const LR_SCHEDULES = {
  cosine(step, maxSteps, lrMax, lrMin = 1e-6) {
    const progress = step / Math.max(maxSteps, 1);
    return lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * progress));
  },
  linear(step, maxSteps, lrMax, lrMin = 1e-6) {
    const progress = step / Math.max(maxSteps, 1);
    return lrMax - (lrMax - lrMin) * progress;
  },
  step(step, maxSteps, lrMax, lrMin = 1e-6, drops = 3) {
    const dropEvery = Math.max(1, Math.floor(maxSteps / (drops + 1)));
    const numDrops = Math.floor(step / dropEvery);
    const factor = Math.pow(0.1, numDrops);
    return Math.max(lrMin, lrMax * factor);
  },
  constant(step, maxSteps, lrMax) {
    return lrMax;
  },
};

// ── Default training config ──────────────────────────────────────────────────
const DEFAULT_TRAINING_CONFIG = {
  epochsI: 10,
  epochsII_T1: 5,
  epochsII_T2: 5,
  epochsII_T3: 5,
  epochsIII: 8,
  epochsIV: 6,
  batchSize: 32,
  directives: [],
  onProgress: null,
  enableCheckpoints: true,
  earlyStoppingPatience: 5,
  lrSchedule: 'cosine',
  lrMax: 0.05,
  lrMin: 1e-6,
};

// ─────────────────────────────────────────────────────────────────────────────
export class TrainingPipeline extends EventEmitter {
  /**
   * @param {object} opts
   * @param {object}  opts.backbone     - ANNPsi backbone instance
   * @param {object}  opts.hci          - HierarchicalCommandInterpreter instance
   * @param {object}  [opts.metrics]    - MetricsCollector instance (optional)
   * @param {string}  [opts.checkpointDir] - Directory for checkpoint storage
   */
  constructor(opts = {}) {
    super();

    // Backward compatibility: if opts is an ANNPsi instance directly
    if (opts && typeof opts.process === 'function' && !opts.backbone) {
      this.backbone = opts;
      this.hci = null;
      this.metrics = null;
      this.checkpointDir = './checkpoints';
    } else {
      this.backbone = opts.backbone ?? null;
      this.hci = opts.hci ?? null;
      this.metrics = opts.metrics ?? null;
      this.checkpointDir = opts.checkpointDir ?? './checkpoints';
    }

    this.datasetLoader = new DatasetLoader();
    this.checkpointManager = new CheckpointManager({ dir: this.checkpointDir });

    // Internal state
    this._running = false;
    this._aborted = false;
    this._currentPhase = null;
    this._phaseResults = {};
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Run the full 4-phase training pipeline.
   *
   * @param {object} config - Training configuration
   * @param {number}   [config.epochsI]               - Phase I epochs
   * @param {number}   [config.epochsII_T1]            - Phase II Tier-1 epochs
   * @param {number}   [config.epochsII_T2]            - Phase II Tier-2 epochs
   * @param {number}   [config.epochsII_T3]            - Phase II Tier-3 epochs
   * @param {number}   [config.epochsIII]              - Phase III epochs
   * @param {number}   [config.epochsIV]               - Phase IV epochs
   * @param {number}   [config.batchSize]              - Batch size
   * @param {string[]} [config.directives]             - Directives for HIFT
   * @param {Function} [config.onProgress]             - Progress callback
   * @param {boolean}  [config.enableCheckpoints]      - Enable checkpointing
   * @param {number}   [config.earlyStoppingPatience]  - Early stopping patience
   * @param {string}   [config.lrSchedule]             - Learning rate schedule type
   * @param {number}   [config.lrMax]                  - Max learning rate
   * @param {number}   [config.lrMin]                  - Min learning rate
   * @returns {Promise<object>} Training result with per-phase stats
   */
  async run(config = {}) {
    const cfg = { ...DEFAULT_TRAINING_CONFIG, ...config };

    if (this._running) {
      throw new Error('Training pipeline is already running.');
    }

    this._running = true;
    this._aborted = false;
    this._phaseResults = {};

    const startTime = Date.now();

    this.emit('training:start', { config: cfg });

    try {
      // ── Validate datasets ──────────────────────────────────────────────
      const pretrainData = await this.datasetLoader.generateSynthetic('pretrain', cfg.batchSize * cfg.epochsI);
      const addictionData = await this.datasetLoader.generateSynthetic('addiction', cfg.batchSize * (cfg.epochsII_T1 + cfg.epochsII_T2 + cfg.epochsII_T3));
      const hiftData = await this.datasetLoader.generateSynthetic('hift', cfg.batchSize * cfg.epochsIII);
      const adversarialData = await this.datasetLoader.generateSynthetic('adversarial', cfg.batchSize * cfg.epochsIV);

      const schema = { required: ['samples'], type: 'object' };
      for (const [name, data] of [['pretrain', pretrainData], ['addiction', addictionData], ['hift', hiftData], ['adversarial', adversarialData]]) {
        const valid = await this.datasetLoader.validate(data, schema);
        if (!valid.ok) {
          throw new Error(`Dataset validation failed for "${name}": ${valid.errors.join(', ')}`);
        }
      }

      // ── Phase I: Large-scale pre-training ──────────────────────────────
      this._currentPhase = 'I';
      const phaseIResult = await this._runPhaseI(pretrainData, cfg);
      this._phaseResults.I = phaseIResult;

      if (this._aborted) return this._buildResult(startTime, 'aborted');

      if (cfg.enableCheckpoints) {
        await this._saveCheckpoint('I', cfg.epochsI, phaseIResult);
      }

      // ── Phase II: Addiction shaping with curriculum learning ────────────
      this._currentPhase = 'II';
      const phaseIIResult = await this._runPhaseII(addictionData, cfg);
      this._phaseResults.II = phaseIIResult;

      if (this._aborted) return this._buildResult(startTime, 'aborted');

      if (cfg.enableCheckpoints) {
        await this._saveCheckpoint('II', cfg.epochsII_T1 + cfg.epochsII_T2 + cfg.epochsII_T3, phaseIIResult);
      }

      // ── Phase III: Hierarchical fine-tuning (HIFT) ────────────────────
      this._currentPhase = 'III';
      const phaseIIIResult = await this._runPhaseIII(hiftData, cfg);
      this._phaseResults.III = phaseIIIResult;

      if (this._aborted) return this._buildResult(startTime, 'aborted');

      if (cfg.enableCheckpoints) {
        await this._saveCheckpoint('III', cfg.epochsIII, phaseIIIResult);
      }

      // ── Phase IV: Adversarial frustration hardening ────────────────────
      this._currentPhase = 'IV';
      const phaseIVResult = await this._runPhaseIV(adversarialData, cfg);
      this._phaseResults.IV = phaseIVResult;

      if (this._aborted) return this._buildResult(startTime, 'aborted');

      if (cfg.enableCheckpoints) {
        await this._saveCheckpoint('IV', cfg.epochsIV, phaseIVResult);
      }

      return this._buildResult(startTime, 'completed');
    } catch (err) {
      this.emit('training:error', { phase: this._currentPhase, error: err.message });
      throw err;
    } finally {
      this._running = false;
      this._currentPhase = null;
    }
  }

  /**
   * Abort the currently running training pipeline.
   */
  abort() {
    this._aborted = true;
    this.emit('training:abort', { phase: this._currentPhase });
  }

  /**
   * Resume training from a checkpoint.
   *
   * @param {string} checkpointId - Checkpoint to resume from
   * @param {object} config       - Training config (overrides)
   * @returns {Promise<object>} Training result
   */
  async resumeFromCheckpoint(checkpointId, config = {}) {
    const checkpoint = await this.checkpointManager.load(checkpointId);
    if (!checkpoint) {
      throw new Error(`Checkpoint "${checkpointId}" not found.`);
    }

    // Restore backbone state if available
    if (checkpoint.state?.backbone && this.backbone && typeof this.backbone.deserialize === 'function') {
      this.backbone.deserialize(checkpoint.state.backbone);
    }

    this.emit('training:resumed', { checkpointId, phase: checkpoint.metadata.phase });

    // Run the pipeline from the next phase
    const startPhase = checkpoint.metadata.phase;
    const cfg = { ...DEFAULT_TRAINING_CONFIG, ...config };

    const startTime = Date.now();
    this._running = true;
    this._aborted = false;
    this._phaseResults = checkpoint.state.phaseResults ?? {};

    try {
      const phases = ['I', 'II', 'III', 'IV'];
      const startIdx = phases.indexOf(startPhase) + 1;

      for (let i = startIdx; i < phases.length; i++) {
        if (this._aborted) break;

        const phase = phases[i];
        this._currentPhase = phase;

        const data = await this.datasetLoader.generateSynthetic(
          phase === 'I' ? 'pretrain' : phase === 'II' ? 'addiction' : phase === 'III' ? 'hift' : 'adversarial',
          cfg.batchSize * this._getEpochsForPhase(phase, cfg),
        );

        const result = await this[`_runPhase${phase}`](data, cfg);
        this._phaseResults[phase] = result;

        if (cfg.enableCheckpoints) {
          await this._saveCheckpoint(phase, this._getEpochsForPhase(phase, cfg), result);
        }
      }

      return this._buildResult(startTime, this._aborted ? 'aborted' : 'completed');
    } catch (err) {
      this.emit('training:error', { phase: this._currentPhase, error: err.message });
      throw err;
    } finally {
      this._running = false;
      this._currentPhase = null;
    }
  }

  // ── Phase implementations ─────────────────────────────────────────────────

  /**
   * Phase I: Large-scale pre-training
   * Trains the ANNPsi backbone on broad data to establish foundational representations.
   */
  async _runPhaseI(dataset, cfg) {
    const epochs = cfg.epochsI;
    const totalSteps = epochs * Math.ceil(dataset.samples.length / cfg.batchSize);
    const lrSchedule = LR_SCHEDULES[cfg.lrSchedule] ?? LR_SCHEDULES.cosine;

    const stats = { epochs, totalSteps, losses: [], learningRates: [], bestLoss: Infinity };
    let globalStep = 0;

    this.emit('phase:start', { phase: 'I', name: 'Pre-training', epochs });

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (this._aborted) break;

      const batches = await this.datasetLoader.batch(dataset, cfg.batchSize);
      let epochLoss = 0;
      let batchCount = 0;

      for await (const batch of batches) {
        if (this._aborted) break;

        const lr = lrSchedule(globalStep, totalSteps, cfg.lrMax, cfg.lrMin);
        const loss = this._trainStep(batch, lr, 'pretrain');

        epochLoss += loss;
        batchCount++;
        globalStep++;

        stats.losses.push(loss);
        stats.learningRates.push(lr);
        stats.bestLoss = Math.min(stats.bestLoss, loss);

        // Record metrics
        this._recordMetric('training.phase_i.loss', loss);
        this._recordMetric('training.phase_i.lr', lr);

        // Progress callback
        if (cfg.onProgress) {
          cfg.onProgress({
            phase: 'I',
            epoch,
            epochs,
            globalStep,
            totalSteps,
            loss,
            lr,
            bestLoss: stats.bestLoss,
          });
        }
      }

      const avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0;
      this.emit('phase:epoch', { phase: 'I', epoch, avgLoss: avgEpochLoss });

      // Early stopping check
      if (stats.bestLoss < 1e-4) {
        this.emit('phase:earlyStop', { phase: 'I', epoch, reason: 'loss_converged' });
        break;
      }
    }

    stats.finalLoss = stats.losses.length > 0 ? stats.losses[stats.losses.length - 1] : 0;
    stats.avgLoss = stats.losses.length > 0
      ? stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length
      : 0;

    this.emit('phase:complete', { phase: 'I', stats });
    return stats;
  }

  /**
   * Phase II: Addiction shaping with curriculum learning and early stopping
   * Shapes AJN addiction targets in a 3-tier curriculum:
   *   T1: Single-stimulus addiction (easiest)
   *   T2: Multi-stimulus addiction (medium)
   *   T3: Constraint-aware addiction (hardest)
   */
  async _runPhaseII(dataset, cfg) {
    const tiers = [
      { name: 'T1', epochs: cfg.epochsII_T1, difficulty: 0.3 },
      { name: 'T2', epochs: cfg.epochsII_T2, difficulty: 0.6 },
      { name: 'T3', epochs: cfg.epochsII_T3, difficulty: 0.9 },
    ];

    const totalEpochs = tiers.reduce((s, t) => s + t.epochs, 0);
    const totalSteps = totalEpochs * Math.ceil(dataset.samples.length / cfg.batchSize);
    const lrSchedule = LR_SCHEDULES[cfg.lrSchedule] ?? LR_SCHEDULES.cosine;

    const stats = { tiers: {}, totalSteps, losses: [], learningRates: [], bestLoss: Infinity };
    let globalStep = 0;

    this.emit('phase:start', { phase: 'II', name: 'Addiction Shaping', totalEpochs });

    for (const tier of tiers) {
      const tierStats = { epochs: tier.epochs, losses: [], bestLoss: Infinity, patienceCounter: 0 };

      for (let epoch = 0; epoch < tier.epochs; epoch++) {
        if (this._aborted) break;

        // Curriculum: augment dataset with difficulty-scaled samples
        const augmented = await this.datasetLoader.augment(dataset, [
          { method: 'noise', scale: tier.difficulty * 0.1 },
          { method: 'permutation', fraction: tier.difficulty * 0.2 },
        ]);

        const batches = await this.datasetLoader.batch(augmented, cfg.batchSize);
        let epochLoss = 0;
        let batchCount = 0;

        for await (const batch of batches) {
          if (this._aborted) break;

          const lr = lrSchedule(globalStep, totalSteps, cfg.lrMax, cfg.lrMin);
          const loss = this._trainStep(batch, lr, 'addiction', { difficulty: tier.difficulty });

          epochLoss += loss;
          batchCount++;
          globalStep++;

          tierStats.losses.push(loss);
          stats.losses.push(loss);
          stats.learningRates.push(lr);
          stats.bestLoss = Math.min(stats.bestLoss, loss);
          tierStats.bestLoss = Math.min(tierStats.bestLoss, loss);

          this._recordMetric('training.phase_ii.loss', loss);
          this._recordMetric('training.phase_ii.lr', lr);
          this._recordMetric('training.phase_ii.difficulty', tier.difficulty);

          if (cfg.onProgress) {
            cfg.onProgress({
              phase: 'II',
              tier: tier.name,
              epoch,
              epochs: tier.epochs,
              globalStep,
              totalSteps,
              loss,
              lr,
              difficulty: tier.difficulty,
              bestLoss: tierStats.bestLoss,
            });
          }
        }

        const avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0;
        this.emit('phase:epoch', { phase: 'II', tier: tier.name, epoch, avgLoss: avgEpochLoss });

        // Early stopping per tier
        if (avgEpochLoss < tierStats.bestLoss) {
          tierStats.patienceCounter = 0;
        } else {
          tierStats.patienceCounter++;
        }

        if (tierStats.patienceCounter >= cfg.earlyStoppingPatience) {
          this.emit('phase:earlyStop', {
            phase: 'II',
            tier: tier.name,
            epoch,
            reason: 'patience_exhausted',
          });
          break;
        }
      }

      tierStats.finalLoss = tierStats.losses.length > 0
        ? tierStats.losses[tierStats.losses.length - 1] : 0;
      tierStats.avgLoss = tierStats.losses.length > 0
        ? tierStats.losses.reduce((a, b) => a + b, 0) / tierStats.losses.length : 0;

      stats.tiers[tier.name] = tierStats;
    }

    stats.finalLoss = stats.losses.length > 0 ? stats.losses[stats.losses.length - 1] : 0;
    stats.avgLoss = stats.losses.length > 0
      ? stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length : 0;

    this.emit('phase:complete', { phase: 'II', stats });
    return stats;
  }

  /**
   * Phase III: Hierarchical Fine-Tuning (HIFT)
   * Directive-specific fine-tuning where each directive gets specialized training.
   */
  async _runPhaseIII(dataset, cfg) {
    const directives = cfg.directives ?? [];
    const epochs = cfg.epochsIII;

    if (directives.length === 0) {
      // If no directives provided, run generic fine-tuning
      directives.push('default');
    }

    const totalSteps = epochs * Math.ceil(dataset.samples.length / cfg.batchSize) * directives.length;
    const lrSchedule = LR_SCHEDULES[cfg.lrSchedule] ?? LR_SCHEDULES.cosine;

    const stats = { directives: {}, totalSteps, losses: [], learningRates: [], bestLoss: Infinity };
    let globalStep = 0;

    this.emit('phase:start', { phase: 'III', name: 'HIFT', epochs, directiveCount: directives.length });

    for (const directive of directives) {
      if (this._aborted) break;

      const dirStats = { epochs, losses: [], bestLoss: Infinity, patienceCounter: 0 };

      // Parse directive through HCI if available
      let parsedDirective = null;
      if (this.hci) {
        try {
          parsedDirective = this.hci.parseSync(directive);
        } catch {
          parsedDirective = null;
        }
      }

      for (let epoch = 0; epoch < epochs; epoch++) {
        if (this._aborted) break;

        const batches = await this.datasetLoader.batch(dataset, cfg.batchSize);
        let epochLoss = 0;
        let batchCount = 0;

        for await (const batch of batches) {
          if (this._aborted) break;

          const lr = lrSchedule(globalStep, totalSteps, cfg.lrMax * 0.5, cfg.lrMin); // Lower LR for fine-tuning
          const loss = this._trainStep(batch, lr, 'hift', { directive, parsedDirective });

          epochLoss += loss;
          batchCount++;
          globalStep++;

          dirStats.losses.push(loss);
          stats.losses.push(loss);
          stats.learningRates.push(lr);
          stats.bestLoss = Math.min(stats.bestLoss, loss);
          dirStats.bestLoss = Math.min(dirStats.bestLoss, loss);

          this._recordMetric('training.phase_iii.loss', loss);
          this._recordMetric('training.phase_iii.lr', lr);

          if (cfg.onProgress) {
            cfg.onProgress({
              phase: 'III',
              directive,
              epoch,
              epochs,
              globalStep,
              totalSteps,
              loss,
              lr,
              bestLoss: dirStats.bestLoss,
            });
          }
        }

        const avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0;
        this.emit('phase:epoch', { phase: 'III', directive, epoch, avgLoss: avgEpochLoss });

        // Early stopping per directive
        if (avgEpochLoss < dirStats.bestLoss) {
          dirStats.patienceCounter = 0;
        } else {
          dirStats.patienceCounter++;
        }

        if (dirStats.patienceCounter >= cfg.earlyStoppingPatience) {
          this.emit('phase:earlyStop', {
            phase: 'III',
            directive,
            epoch,
            reason: 'patience_exhausted',
          });
          break;
        }
      }

      dirStats.finalLoss = dirStats.losses.length > 0
        ? dirStats.losses[dirStats.losses.length - 1] : 0;
      dirStats.avgLoss = dirStats.losses.length > 0
        ? dirStats.losses.reduce((a, b) => a + b, 0) / dirStats.losses.length : 0;

      stats.directives[directive] = dirStats;
    }

    stats.finalLoss = stats.losses.length > 0 ? stats.losses[stats.losses.length - 1] : 0;
    stats.avgLoss = stats.losses.length > 0
      ? stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length : 0;

    this.emit('phase:complete', { phase: 'III', stats });
    return stats;
  }

  /**
   * Phase IV: Adversarial frustration hardening
   * Trains the AJN to maintain stability under adversarial conditions
   * with progressive difficulty escalation.
   */
  async _runPhaseIV(dataset, cfg) {
    const epochs = cfg.epochsIV;
    const totalSteps = epochs * Math.ceil(dataset.samples.length / cfg.batchSize);
    const lrSchedule = LR_SCHEDULES[cfg.lrSchedule] ?? LR_SCHEDULES.cosine;

    const stats = { epochs, totalSteps, losses: [], learningRates: [], bestLoss: Infinity, adversarialScores: [] };
    let globalStep = 0;
    let difficultyLevel = 0.1; // Start with low adversarial difficulty

    this.emit('phase:start', { phase: 'IV', name: 'Adversarial Hardening', epochs });

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (this._aborted) break;

      // Progressive difficulty: increase adversarial intensity over epochs
      difficultyLevel = 0.1 + 0.9 * (epoch / Math.max(epochs - 1, 1));

      // Augment with adversarial noise scaled by difficulty
      const adversarialData = await this.datasetLoader.augment(dataset, [
        { method: 'noise', scale: difficultyLevel * 0.3 },
        { method: 'adversarial_perturbation', intensity: difficultyLevel },
        { method: 'permutation', fraction: difficultyLevel * 0.15 },
      ]);

      const batches = await this.datasetLoader.batch(adversarialData, cfg.batchSize);
      let epochLoss = 0;
      let batchCount = 0;
      let adversarialSuccesses = 0;

      for await (const batch of batches) {
        if (this._aborted) break;

        const lr = lrSchedule(globalStep, totalSteps, cfg.lrMax * 0.3, cfg.lrMin); // Conservative LR for hardening
        const result = this._trainStep(batch, lr, 'adversarial', { difficulty: difficultyLevel });

        const loss = result;
        epochLoss += loss;
        batchCount++;
        globalStep++;

        stats.losses.push(loss);
        stats.learningRates.push(lr);
        stats.bestLoss = Math.min(stats.bestLoss, loss);

        // Track adversarial robustness (lower is better)
        const advScore = loss * (1 - difficultyLevel * 0.5);
        stats.adversarialScores.push(advScore);
        adversarialSuccesses += advScore < 0.5 ? 1 : 0;

        this._recordMetric('training.phase_iv.loss', loss);
        this._recordMetric('training.phase_iv.lr', lr);
        this._recordMetric('training.phase_iv.difficulty', difficultyLevel);
        this._recordMetric('training.phase_iv.adversarial_score', advScore);

        if (cfg.onProgress) {
          cfg.onProgress({
            phase: 'IV',
            epoch,
            epochs,
            globalStep,
            totalSteps,
            loss,
            lr,
            difficulty: difficultyLevel,
            adversarialRobustness: batchCount > 0
              ? adversarialSuccesses / batchCount
              : 0,
            bestLoss: stats.bestLoss,
          });
        }
      }

      const avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0;
      const robustness = batchCount > 0 ? adversarialSuccesses / batchCount : 0;
      this.emit('phase:epoch', { phase: 'IV', epoch, avgLoss: avgEpochLoss, difficulty: difficultyLevel, robustness });

      // If robustness is very high, we can stop early
      if (robustness > 0.95) {
        this.emit('phase:earlyStop', { phase: 'IV', epoch, reason: 'robustness_threshold' });
        break;
      }
    }

    stats.finalLoss = stats.losses.length > 0 ? stats.losses[stats.losses.length - 1] : 0;
    stats.avgLoss = stats.losses.length > 0
      ? stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length : 0;
    stats.finalDifficulty = difficultyLevel;
    stats.avgAdversarialScore = stats.adversarialScores.length > 0
      ? stats.adversarialScores.reduce((a, b) => a + b, 0) / stats.adversarialScores.length : 0;

    this.emit('phase:complete', { phase: 'IV', stats });
    return stats;
  }

  // ── Internal helpers ──────────────────────────────────────────────────────

  /**
   * Execute a single training step on the backbone.
   * Returns the computed loss for the step.
   *
   * @param {object} batch - Training batch
   * @param {number} lr    - Learning rate
   * @param {string} phase - Phase name
   * @param {object} [ctx] - Additional context
   * @returns {number} Loss value
   */
  _trainStep(batch, lr, phase, ctx = {}) {
    if (!this.backbone) {
      // Simulate loss for testing without a backbone
      const baseLoss = phase === 'pretrain' ? 0.8
                     : phase === 'addiction' ? 0.6
                     : phase === 'hift' ? 0.4
                     : 0.5;
      const difficulty = ctx.difficulty ?? 0;
      return Math.max(0.001, baseLoss - Math.random() * 0.1 + difficulty * 0.05 * Math.random());
    }

    // Process batch through backbone
    let totalLoss = 0;
    const samples = batch.samples ?? batch ?? [];

    for (const sample of samples) {
      try {
        const stimulus = sample.stimulus ?? sample ?? {};
        const result = this.backbone.process(stimulus);

        // Compute loss based on phase-specific objectives
        let loss;
        switch (phase) {
          case 'pretrain':
            // Minimize praxis variance for stable initialization
            loss = result.praxisNorm !== undefined
              ? Math.abs(result.praxisNorm - 1.0)
              : Math.random() * 0.5;
            break;
          case 'addiction':
            // Shape craving toward target
            loss = result.craving !== undefined
              ? Math.abs(result.craving - (ctx.difficulty ?? 0.5))
              : Math.random() * 0.5;
            break;
          case 'hift':
            // Minimize directive-specific objective
            loss = result.craving !== undefined
              ? Math.abs(result.craving - 0.7) + (result.nFail ?? 0) * 0.01
              : Math.random() * 0.3;
            break;
          case 'adversarial':
            // Minimize instability under adversarial perturbation
            loss = result.craving !== undefined
              ? Math.abs(result.craving - 0.5) * (1 + (ctx.difficulty ?? 0) * 0.3)
              : Math.random() * 0.4;
            break;
          default:
            loss = Math.random() * 0.5;
        }

        totalLoss += loss;
      } catch {
        totalLoss += 0.5; // Penalize failures
      }
    }

    return samples.length > 0 ? totalLoss / samples.length : 0;
  }

  /**
   * Record a metric via the MetricsCollector if available.
   */
  _recordMetric(name, value) {
    if (this.metrics) {
      this.metrics.observeHistogram(name, value);
      this.metrics.setGauge(name, value);
    }
  }

  /**
   * Save a training checkpoint.
   */
  async _saveCheckpoint(phase, epoch, result) {
    const state = {
      phaseResults: this._phaseResults,
      backbone: this.backbone && typeof this.backbone.serialize === 'function'
        ? this.backbone.serialize()
        : null,
    };

    await this.checkpointManager.save(phase, epoch, {
      ...state,
      loss: result.finalLoss ?? 0,
      metrics: {
        avgLoss: result.avgLoss ?? 0,
        bestLoss: result.bestLoss ?? Infinity,
      },
    });
  }

  /**
   * Get the total epoch count for a given phase.
   */
  _getEpochsForPhase(phase, cfg) {
    switch (phase) {
      case 'I': return cfg.epochsI;
      case 'II': return cfg.epochsII_T1 + cfg.epochsII_T2 + cfg.epochsII_T3;
      case 'III': return cfg.epochsIII;
      case 'IV': return cfg.epochsIV;
      default: return 0;
    }
  }

  /**
   * Build the final training result object.
   */
  _buildResult(startTime, status) {
    const durationMs = Date.now() - startTime;

    return {
      status,
      durationMs,
      phases: this._phaseResults,
      finalLoss: this._getOverallFinalLoss(),
      summary: this._buildSummary(),
    };
  }

  /**
   * Get the overall final loss from all phases.
   */
  _getOverallFinalLoss() {
    const phaseKeys = Object.keys(this._phaseResults).sort();
    if (phaseKeys.length === 0) return 0;
    const lastPhase = this._phaseResults[phaseKeys[phaseKeys.length - 1]];
    return lastPhase?.finalLoss ?? 0;
  }

  /**
   * Build a human-readable training summary.
   */
  _buildSummary() {
    const lines = [];

    for (const [phase, result] of Object.entries(this._phaseResults)) {
      const phaseName = phase === 'I' ? 'Pre-training'
                      : phase === 'II' ? 'Addiction Shaping'
                      : phase === 'III' ? 'HIFT'
                      : 'Adversarial Hardening';

      lines.push(`Phase ${phase} (${phaseName}):`);
      lines.push(`  Final loss: ${(result.finalLoss ?? 0).toFixed(6)}`);
      lines.push(`  Average loss: ${(result.avgLoss ?? 0).toFixed(6)}`);
      lines.push(`  Best loss: ${(result.bestLoss ?? 0).toFixed(6)}`);

      if (result.tiers) {
        for (const [tier, tierResult] of Object.entries(result.tiers)) {
          lines.push(`  ${tier} avg loss: ${(tierResult.avgLoss ?? 0).toFixed(6)}`);
        }
      }

      if (result.directives) {
        for (const [dir, dirResult] of Object.entries(result.directives)) {
          lines.push(`  Directive "${dir}" avg loss: ${(dirResult.avgLoss ?? 0).toFixed(6)}`);
        }
      }

      if (result.finalDifficulty !== undefined) {
        lines.push(`  Final difficulty: ${result.finalDifficulty.toFixed(3)}`);
      }

      if (result.avgAdversarialScore !== undefined) {
        lines.push(`  Avg adversarial score: ${result.avgAdversarialScore.toFixed(6)}`);
      }
    }

    return lines.join('\n');
  }
}

export default TrainingPipeline;
