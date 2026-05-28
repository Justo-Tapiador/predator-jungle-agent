/**
 * Predator.js — Enhanced Main Agent Orchestrator (v0.2)
 *
 * Integrates the full PREDATOR pipeline:
 *   ANNPsi backbone -> HCI -> TEA -> PSE -> CascadeMonitor
 *
 * Key improvements over v0.1:
 *  1.  MemorySystem integration for persistent memory across tasks
 *  2.  SafetyGuardrails for pre-execution safety checks
 *  3.  MetricsCollector for observability
 *  4.  PluginManager for extensibility
 *  5.  State serialization/deserialization for checkpointing
 *  6.  Streaming event support for real-time consumers
 *  7.  Multi-directive task chaining (executeChain)
 *  8.  Better feedback quality computation
 *  9.  Configurable max steps per task
 * 10.  Graceful shutdown handling
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';

import { ANNPsi } from './ANNPsi.js';
import { HierarchicalCommandInterpreter } from '../modules/HierarchicalCommandInterpreter.js';
import {
  TokenEnergyArbitrator,
  PraxicStreamExecutor,
} from '../modules/TokenEnergyArbitrator.js';
import { CascadeMonitor } from '../modules/CascadeMonitor.js';
import { TrainingPipeline } from '../training/TrainingPipeline.js';
import { MemorySystem } from '../modules/MemorySystem.js';
import { SafetyGuardrails } from '../modules/SafetyGuardrails.js';
import { MetricsCollector } from '../modules/MetricsCollector.js';
import { PluginManager } from '../modules/PluginManager.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DEFAULT_MAX_STEPS = 200;
const CASCADE_WARN_DEFAULT = 0.6;
const CASCADE_CRITICAL_DEFAULT = 0.85;

/**
 * Improved quality computation.
 *
 * Weights:
 *   success rate      : 0.50
 *   avg progress      : 0.25
 *   owner alignment   : 0.25
 *
 * Negative signals:
 *   cascade events    : -0.05 each  (capped at -0.25)
 *   extinctions       : -0.10 each  (capped at -0.30)
 *
 * Result is clamped to [0, 1].
 */
function computeQuality(taskRecord) {
  const { successRate = 0, avgProgress = 0, ownerAlignment = 0.5 } = taskRecord;
  const cascadeEvents = taskRecord.cascadeEvents ?? 0;
  const extinctions = taskRecord.extinctions ?? 0;

  let quality =
    0.50 * successRate +
    0.25 * avgProgress +
    0.25 * ownerAlignment;

  // Negative signals
  quality -= Math.min(cascadeEvents * 0.05, 0.25);
  quality -= Math.min(extinctions * 0.10, 0.30);

  return Math.max(0, Math.min(1, quality));
}

// ---------------------------------------------------------------------------
// Predator
// ---------------------------------------------------------------------------

export class Predator extends EventEmitter {
  /**
   * @param {object} [opts]
   * @param {object} [opts.ajnParams]          - AJN hyperparameter overrides
   * @param {object} [opts.defaultBudget]      - Default resource budget
   * @param {Map}    [opts.tools]              - Custom tool transducers
   * @param {number} [opts.dModel]             - Transformer model dimension
   * @param {number} [opts.nHeads]             - Transformer attention heads
   * @param {number} [opts.dFF]                - Transformer feed-forward dimension
   * @param {number} [opts.maxSteps]           - Maximum steps per task (default 200)
   * @param {boolean} [opts.enableMemory]      - Enable memory system (default true)
   * @param {boolean} [opts.enableSafety]      - Enable safety guardrails (default true)
   * @param {boolean} [opts.enableMetrics]     - Enable metrics collection (default true)
   * @param {boolean} [opts.enablePlugins]     - Enable plugin system (default true)
   * @param {number} [opts.rhoWarn]            - Cascade monitor warning threshold
   * @param {number} [opts.rhoCritical]        - Cascade monitor critical threshold
   */
  constructor(opts = {}) {
    super();

    this._id = uuidv4();
    this._createdAt = Date.now();
    this._shuttingDown = false;

    // ---- Core config ----
    this._ajnParams = opts.ajnParams ?? {};
    this._defaultBudget = opts.defaultBudget ?? {};
    this._maxSteps = opts.maxSteps ?? DEFAULT_MAX_STEPS;

    // ---- Feature flags ----
    this._enableMemory = opts.enableMemory !== false;
    this._enableSafety = opts.enableSafety !== false;
    this._enableMetrics = opts.enableMetrics !== false;
    this._enablePlugins = opts.enablePlugins !== false;

    // ---- Cascade thresholds ----
    this._rhoWarn = opts.rhoWarn ?? CASCADE_WARN_DEFAULT;
    this._rhoCritical = opts.rhoCritical ?? CASCADE_CRITICAL_DEFAULT;

    // ---- Custom tools map ----
    this._tools = opts.tools instanceof Map ? opts.tools : new Map(Object.entries(opts.tools ?? {}));

    // ================================================================
    // Core pipeline modules
    // ================================================================

    /** @type {ANNPsi} Neural-symbolic backbone */
    this.backbone = new ANNPsi({
      ajnParams: this._ajnParams,
      dModel: opts.dModel,
      nHeads: opts.nHeads,
      dFF: opts.dFF,
    });

    /** @type {HierarchicalCommandInterpreter} */
    this.hci = new HierarchicalCommandInterpreter(this.backbone);

    /** @type {TokenEnergyArbitrator} */
    this.tea = new TokenEnergyArbitrator();

    /** @type {PraxicStreamExecutor} */
    this.pse = new PraxicStreamExecutor(this.backbone, this.tea);

    /** @type {CascadeMonitor} */
    this.cascadeMonitor = new CascadeMonitor({
      rhoWarn: this._rhoWarn,
      rhoCritical: this._rhoCritical,
    });

    /** @type {TrainingPipeline} */
    this.trainingPipeline = new TrainingPipeline(this.backbone);

    // ================================================================
    // Optional enhancement modules
    // ================================================================

    /** @type {MemorySystem|null} */
    this.memory = this._enableMemory ? new MemorySystem() : null;

    /** @type {SafetyGuardrails|null} */
    this.safety = this._enableSafety ? new SafetyGuardrails() : null;

    /** @type {MetricsCollector|null} */
    this.metrics = this._enableMetrics ? new MetricsCollector() : null;

    /** @type {PluginManager|null} */
    this.plugins = this._enablePlugins ? new PluginManager() : null;

    // ================================================================
    // Internal state
    // ================================================================

    /** Running task or null */
    this._currentTask = null;

    /** Resolved from current task's promise */
    this._currentResolve = null;
    this._currentReject = null;

    /** Owner escalation state */
    this._ownerFeedback = null;
    this._waitingForOwner = false;

    /** Execution history — array of completed task records */
    this._history = [];

    /** Registered custom tools (beyond the map provided at construction) */
    this._customTools = new Map(this._tools);
  }

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * Execute a single directive through the full PREDATOR pipeline.
   *
   * Flow:
   *   1. Safety pre-check
   *   2. Memory lookup for similar past tasks
   *   3. HCI decomposition
   *   4. TEA budgeting
   *   5. PSE execution loop (_runTPS) with safety, memory, metrics, and plugin hooks
   *   6. Post-task memory storage & metrics flush
   *   7. Cascade monitoring summary
   *   8. Quality computation & history recording
   *
   * @param {string} rawDirective   - The raw directive string
   * @param {object} [budgetOverrides] - Override the default resource budget
   * @returns {Promise<object>} Task result
   */
  async execute(rawDirective, budgetOverrides) {
    if (this._shuttingDown) {
      throw new Error('Predator is shutting down; cannot accept new tasks.');
    }

    const taskId = uuidv4();
    const startTime = Date.now();

    // ---- 1. Safety pre-check ----
    if (this.safety) {
      const safetyResult = await this.safety.check(rawDirective);
      if (!safetyResult.allowed) {
        this.emit('safety:blocked', { taskId, directive: rawDirective, reason: safetyResult.reason });
        return {
          taskId,
          status: 'blocked',
          reason: safetyResult.reason,
          quality: 0,
        };
      }
      if (safetyResult.warnings?.length) {
        this.emit('safety:warning', { taskId, warnings: safetyResult.warnings });
      }
    }

    // ---- 2. Memory lookup ----
    let recalledMemories = [];
    if (this.memory) {
      recalledMemories = await this.memory.recall(rawDirective, { limit: 5 });
      this.emit('memory:recall', { taskId, memories: recalledMemories });
    }

    // ---- 3. HCI decomposition ----
    const decomposed = this.hci.interpret(rawDirective, { recalledMemories });

    // ---- 4. TEA budgeting ----
    const budget = {
      ...this._defaultBudget,
      ...(budgetOverrides ?? {}),
    };
    const tokenBudget = this.tea.allocate(decomposed, budget);

    // ---- 5. Run TPS loop ----
    const taskContext = {
      taskId,
      rawDirective,
      decomposed,
      tokenBudget,
      recalledMemories,
      startTime,
      cascadeEvents: 0,
      extinctions: 0,
      stepsCompleted: 0,
      successes: 0,
      progressLog: [],
      results: [],
    };

    this._currentTask = taskContext;

    const result = await this._runTPS(taskContext);

    // ---- 6. Post-task memory store ----
    const taskSummary = {
      taskId,
      directive: rawDirective,
      status: result.status,
      quality: result.quality,
      stepsCompleted: taskContext.stepsCompleted,
      cascadeEvents: taskContext.cascadeEvents,
      extinctions: taskContext.extinctions,
      timestamp: Date.now(),
    };

    if (this.memory) {
      await this.memory.store(rawDirective, taskSummary);
      this.emit('memory:store', { taskId, summary: taskSummary });
    }

    // ---- 7. Metrics flush ----
    if (this.metrics) {
      this.metrics.recordTask(taskSummary);
    }

    // ---- 8. Plugin hook: taskComplete ----
    if (this.plugins) {
      await this.plugins.invoke('taskComplete', taskSummary);
    }

    // ---- 9. Quality computation ----
    const quality = computeQuality({
      successRate: taskContext.stepsCompleted > 0
        ? taskContext.successes / taskContext.stepsCompleted
        : 0,
      avgProgress: taskContext.progressLog.length > 0
        ? taskContext.progressLog.reduce((a, b) => a + b, 0) / taskContext.progressLog.length
        : 0,
      ownerAlignment: result.ownerAlignment ?? 0.5,
      cascadeEvents: taskContext.cascadeEvents,
      extinctions: taskContext.extinctions,
    });

    result.quality = quality;

    // ---- 10. Record in history ----
    this._history.push({
      ...taskSummary,
      quality,
      durationMs: Date.now() - startTime,
    });

    this._currentTask = null;

    this.emit('task:complete', result);

    return result;
  }

  /**
   * Execute multiple directives sequentially, passing context between them.
   *
   * @param {string[]} directives - Array of raw directive strings
   * @returns {Promise<object[]>} Array of task results
   */
  async executeChain(directives) {
    if (this._shuttingDown) {
      throw new Error('Predator is shutting down; cannot accept new tasks.');
    }

    if (!Array.isArray(directives) || directives.length === 0) {
      return [];
    }

    const chainId = uuidv4();
    const results = [];
    let chainContext = {};

    this.emit('chain:start', { chainId, directiveCount: directives.length });

    for (let i = 0; i < directives.length; i++) {
      const directive = directives[i];

      this.emit('chain:step', { chainId, index: i, total: directives.length, directive });

      // Merge chain context into budget overrides so downstream tasks
      // can reference results from prior steps.
      const budgetOverrides = {
        chainContext,
        chainIndex: i,
      };

      try {
        const result = await this.execute(directive, budgetOverrides);
        results.push(result);

        // Accumulate context for the next directive
        chainContext = {
          ...chainContext,
          [`step_${i}`]: {
            status: result.status,
            quality: result.quality,
            summary: result.summary ?? null,
          },
          lastResult: result,
        };
      } catch (err) {
        const failResult = {
          taskId: null,
          status: 'chain_error',
          error: err.message,
          chainIndex: i,
          quality: 0,
        };
        results.push(failResult);
        chainContext = {
          ...chainContext,
          [`step_${i}`]: failResult,
          lastResult: failResult,
        };
        this.emit('chain:error', { chainId, index: i, error: err.message });
      }
    }

    this.emit('chain:complete', { chainId, results });

    return results;
  }

  /**
   * Run the training pipeline.
   *
   * @param {object} config - Training configuration
   * @returns {Promise<object>} Training result
   */
  async train(config) {
    if (this._shuttingDown) {
      throw new Error('Predator is shutting down; cannot train.');
    }

    this.emit('train:start', { config });

    const result = await this.trainingPipeline.run(config);

    this.emit('train:complete', { result });
    return result;
  }

  /**
   * Register a custom tool transducer.
   *
   * @param {string} id   - Tool identifier
   * @param {Function} fn - Tool function
   * @param {object} meta - Tool metadata
   */
  registerTool(id, fn, meta = {}) {
    if (typeof fn !== 'function') {
      throw new TypeError(`Tool "${id}" must be a function, got ${typeof fn}`);
    }
    this._customTools.set(id, { fn, meta });
    this.pse.registerTool(id, fn, meta);
    this.emit('tool:registered', { id, meta });
  }

  /**
   * Register a plugin via the PluginManager.
   *
   * @param {object} plugin - Plugin instance with hook methods
   */
  use(plugin) {
    if (!this.plugins) {
      throw new Error('Plugin system is disabled. Set enablePlugins: true to use plugins.');
    }
    this.plugins.register(plugin);
    this.emit('plugin:registered', { name: plugin.name ?? 'anonymous' });
  }

  /**
   * Get a snapshot of the current system status.
   *
   * @returns {object}
   */
  status() {
    return {
      id: this._id,
      createdAt: this._createdAt,
      shuttingDown: this._shuttingDown,
      currentTask: this._currentTask
        ? {
            taskId: this._currentTask.taskId,
            directive: this._currentTask.rawDirective,
            stepsCompleted: this._currentTask.stepsCompleted,
          }
        : null,
      historyCount: this._history.length,
      customToolCount: this._customTools.size,
      backbone: this.backbone.status?.() ?? 'ok',
      memoryEnabled: this.memory !== null,
      safetyEnabled: this.safety !== null,
      metricsEnabled: this.metrics !== null,
      pluginsEnabled: this.plugins !== null,
      cascadeMonitor: this.cascadeMonitor.status?.() ?? 'ok',
    };
  }

  /**
   * Resume after an owner escalation.
   *
   * @param {object} ownerFeedback - Feedback from the human owner
   * @returns {void}
   */
  resume(ownerFeedback) {
    this._ownerFeedback = ownerFeedback;
    this._waitingForOwner = false;

    this.emit('owner:resume', { ownerFeedback });

    // If we have a pending task promise, resolve it so the TPS loop can continue
    if (this._currentResolve) {
      this._currentResolve(ownerFeedback);
      this._currentResolve = null;
      this._currentReject = null;
    }
  }

  /**
   * Get the task execution history.
   *
   * @returns {object[]}
   */
  history() {
    return [...this._history];
  }

  /**
   * Serialize the entire agent state for checkpointing.
   *
   * @returns {object} Serializable state object
   */
  serialize() {
    const state = {
      _version: '0.2.0',
      id: this._id,
      createdAt: this._createdAt,
      maxSteps: this._maxSteps,
      rhoWarn: this._rhoWarn,
      rhoCritical: this._rhoCritical,
      defaultBudget: this._defaultBudget,
      ajnParams: this._ajnParams,
      history: this._history,
      customTools: {},
    };

    // Serialize custom tools metadata (functions are not serializable)
    for (const [id, { meta }] of this._customTools) {
      state.customTools[id] = meta;
    }

    // Serialize backbone weights if supported
    if (typeof this.backbone.serialize === 'function') {
      state.backbone = this.backbone.serialize();
    }

    // Serialize memory if available
    if (this.memory && typeof this.memory.serialize === 'function') {
      state.memory = this.memory.serialize();
    }

    // Serialize cascade monitor state
    if (typeof this.cascadeMonitor.serialize === 'function') {
      state.cascadeMonitor = this.cascadeMonitor.serialize();
    }

    // Serialize metrics if available
    if (this.metrics && typeof this.metrics.serialize === 'function') {
      state.metrics = this.metrics.serialize();
    }

    return state;
  }

  /**
   * Restore agent state from a previously serialized snapshot.
   *
   * @param {object} state - Serialized state object
   * @returns {void}
   */
  deserialize(state) {
    if (!state || state._version !== '0.2.0') {
      throw new Error('Incompatible serialized state version.');
    }

    this._id = state.id;
    this._maxSteps = state.maxSteps;
    this._rhoWarn = state.rhoWarn;
    this._rhoCritical = state.rhoCritical;
    this._defaultBudget = state.defaultBudget ?? {};
    this._ajnParams = state.ajnParams ?? {};
    this._history = state.history ?? [];

    // Restore backbone
    if (state.backbone && typeof this.backbone.deserialize === 'function') {
      this.backbone.deserialize(state.backbone);
    }

    // Restore memory
    if (this.memory && state.memory && typeof this.memory.deserialize === 'function') {
      this.memory.deserialize(state.memory);
    }

    // Restore cascade monitor
    if (state.cascadeMonitor && typeof this.cascadeMonitor.deserialize === 'function') {
      this.cascadeMonitor.deserialize(state.cascadeMonitor);
    }

    // Restore metrics
    if (this.metrics && state.metrics && typeof this.metrics.deserialize === 'function') {
      this.metrics.deserialize(state.metrics);
    }

    this.emit('state:restored', { id: this._id });
  }

  /**
   * Graceful shutdown. Flushes memory and metrics, then cleans up.
   *
   * @returns {Promise<void>}
   */
  async shutdown() {
    this._shuttingDown = true;
    this.emit('shutdown:start');

    // Wait for the current task to finish (with a timeout)
    if (this._currentTask) {
      this.emit('shutdown:waiting', { taskId: this._currentTask.taskId });

      // Give the current task a grace period to complete
      const gracePeriod = 5000;
      await new Promise((resolve) => {
        const check = setInterval(() => {
          if (!this._currentTask) {
            clearInterval(check);
            resolve();
          }
        }, 100);

        setTimeout(() => {
          clearInterval(check);
          // Force-abort if still running
          if (this._currentTask) {
            this.emit('shutdown:forceAbort', { taskId: this._currentTask.taskId });
            this._currentTask = null;
          }
          resolve();
        }, gracePeriod);
      });
    }

    // Flush memory
    if (this.memory && typeof this.memory.flush === 'function') {
      try {
        await this.memory.flush();
        this.emit('memory:flushed');
      } catch (err) {
        this.emit('memory:flushError', { error: err.message });
      }
    }

    // Flush metrics
    if (this.metrics && typeof this.metrics.flush === 'function') {
      try {
        await this.metrics.flush();
        this.emit('metrics:flushed');
      } catch (err) {
        this.emit('metrics:flushError', { error: err.message });
      }
    }

    // Tear down plugins
    if (this.plugins && typeof this.plugins.destroy === 'function') {
      try {
        await this.plugins.destroy();
        this.emit('plugins:destroyed');
      } catch (err) {
        this.emit('plugins:destroyError', { error: err.message });
      }
    }

    this.removeAllListeners();
    this.emit('shutdown:complete');
  }

  // =========================================================================
  // Internal: TPS Execution Loop
  // =========================================================================

  /**
   * Run the Token-Praxic Stream execution loop.
   *
   * Enhanced over v0.1 with:
   *   - Safety guardrail checks per step
   *   - Memory lookups per step
   *   - Metrics recording per step
   *   - Plugin hooks at key lifecycle points
   *   - Configurable maxSteps
   *   - Cascade monitoring integration
   *
   * @param {object} ctx - Task context
   * @returns {Promise<object>} Task result
   */
  async _runTPS(ctx) {
    const { decomposed, tokenBudget } = ctx;
    let remainingBudget = tokenBudget;
    let stepIdx = 0;
    let converged = false;
    let ownerAlignment = 0.5;
    let lastAction = null;

    while (stepIdx < this._maxSteps && remainingBudget > 0 && !converged) {
      // ---- Safety check per step ----
      if (this.safety) {
        const stepSafety = await this.safety.checkStep(lastAction, ctx);
        if (!stepSafety.allowed) {
          this.emit('safety:stepBlocked', {
            taskId: ctx.taskId,
            step: stepIdx,
            reason: stepSafety.reason,
          });
          break;
        }
      }

      // ---- Memory lookup for step-relevant context ----
      let stepMemories = [];
      if (this.memory) {
        stepMemories = await this.memory.recall(
          decomposed.steps?.[stepIdx]?.description ?? `step_${stepIdx}`,
          { limit: 3 },
        );
      }

      // ---- Plugin hook: beforeStep ----
      if (this.plugins) {
        await this.plugins.invoke('beforeStep', {
          taskId: ctx.taskId,
          step: stepIdx,
          context: ctx,
          memories: stepMemories,
        });
      }

      // ---- Execute step via PSE ----
      let stepResult;
      try {
        stepResult = await this.pse.execute(decomposed, {
          stepIndex: stepIdx,
          budget: remainingBudget,
          memories: stepMemories,
          ownerFeedback: this._ownerFeedback,
        });
      } catch (err) {
        ctx.extinctions++;
        this.emit('step:extinction', { taskId: ctx.taskId, step: stepIdx, error: err.message });

        // Plugin hook
        if (this.plugins) {
          await this.plugins.invoke('afterStep', {
            taskId: ctx.taskId,
            step: stepIdx,
            result: null,
            error: err.message,
          });
        }

        // Record metrics
        if (this.metrics) {
          this.metrics.recordStep({
            taskId: ctx.taskId,
            step: stepIdx,
            status: 'extinction',
            error: err.message,
          });
        }

        stepIdx++;
        continue;
      }

      // ---- Cascade monitoring ----
      const rho = this.cascadeMonitor.observe(stepResult);
      if (rho >= this._rhoCritical) {
        ctx.cascadeEvents++;
        this.emit('cascade:critical', { taskId: ctx.taskId, step: stepIdx, rho });
        // Escalate to owner
        this._waitingForOwner = true;
        this.emit('owner:escalation', {
          taskId: ctx.taskId,
          step: stepIdx,
          rho,
          stepResult,
        });

        // Wait for owner feedback
        const feedback = await new Promise((resolve, reject) => {
          this._currentResolve = resolve;
          this._currentReject = reject;
        });

        this._ownerFeedback = feedback;
        ownerAlignment = feedback.alignment ?? ownerAlignment;
      } else if (rho >= this._rhoWarn) {
        ctx.cascadeEvents++;
        this.emit('cascade:warning', { taskId: ctx.taskId, step: stepIdx, rho });
      }

      // ---- Process step result ----
      if (stepResult.converged) {
        converged = true;
      }

      if (stepResult.success) {
        ctx.successes++;
      }

      const progress = stepResult.progress ?? 0;
      ctx.progressLog.push(progress);
      ctx.results.push(stepResult);
      ctx.stepsCompleted = stepIdx + 1;

      // Update remaining budget
      remainingBudget -= stepResult.tokensUsed ?? 0;

      // Track last action for safety context
      lastAction = stepResult.action ?? stepResult;

      // ---- Plugin hook: afterStep ----
      if (this.plugins) {
        await this.plugins.invoke('afterStep', {
          taskId: ctx.taskId,
          step: stepIdx,
          result: stepResult,
        });
      }

      // ---- Metrics recording ----
      if (this.metrics) {
        this.metrics.recordStep({
          taskId: ctx.taskId,
          step: stepIdx,
          status: stepResult.success ? 'success' : 'failure',
          progress,
          tokensUsed: stepResult.tokensUsed ?? 0,
          rho,
        });
      }

      // ---- Emit streaming event ----
      this.emit('step:complete', {
        taskId: ctx.taskId,
        step: stepIdx,
        progress,
        rho,
        converged,
      });

      stepIdx++;
    }

    // ---- Compute final result ----
    const reason = converged
      ? 'converged'
      : remainingBudget <= 0
        ? 'budget_exhausted'
        : stepIdx >= this._maxSteps
          ? 'max_steps_reached'
          : 'safety_blocked';

    return {
      taskId: ctx.taskId,
      status: converged ? 'success' : 'incomplete',
      reason,
      summary: this._summarizeResults(ctx.results),
      stepsCompleted: ctx.stepsCompleted,
      cascadeEvents: ctx.cascadeEvents,
      extinctions: ctx.extinctions,
      ownerAlignment,
      quality: 0, // Will be overwritten by execute()
    };
  }

  // =========================================================================
  // Internal: Helpers
  // =========================================================================

  /**
   * Summarize an array of step results into a concise description.
   *
   * @param {object[]} results
   * @returns {string}
   */
  _summarizeResults(results) {
    if (!results || results.length === 0) {
      return 'No results produced.';
    }

    const successes = results.filter((r) => r.success).length;
    const total = results.length;
    const finalProgress = results[results.length - 1]?.progress ?? 0;

    return (
      `Completed ${successes}/${total} steps successfully. ` +
      `Final progress: ${(finalProgress * 100).toFixed(1)}%.`
    );
  }
}

export default Predator;
