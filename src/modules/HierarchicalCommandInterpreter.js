/**
 * HierarchicalCommandInterpreter.js  (v2.0 – Enhanced)
 * ─────────────────────────────────────────────────────────────────────────────
 * Translates Owner natural-language directives into AJN addiction targets,
 * constraint masks, resource budgets, and urgency scalars.
 *
 * IMPROVEMENTS over v0.1:
 *   - LLM integration for semantic directive understanding
 *   - Richer constraint pattern library
 *   - Directive decomposition for complex multi-step tasks
 *   - Context-aware stimulus target generation
 *   - Structured constraint validation with severity levels
 *   - Directive history tracking for context continuity
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';

// ── Urgency map ──────────────────────────────────────────────────────────────
const URGENCY = { routine: 0.2, expedited: 0.6, critical: 0.95 };

// ── Extended keyword → stimulus class mappings ───────────────────────────────
const GOAL_KEYWORDS = {
  syntax:            ['syntax', 'parse', 'format', 'lint', 'style', 'indent', 'whitespace'],
  semantics:         ['mean', 'interpret', 'understand', 'explain', 'analyze', 'comprehend', 'describe'],
  structure:         ['structure', 'organiz', 'architect', 'design', 'layout', 'pattern', 'refactor'],
  novelty:           ['novel', 'creat', 'invent', 'new idea', 'generat', 'brainstorm', 'innovate'],
  relevance:         ['relev', 'filter', 'select', 'rank', 'match', 'prioritize', 'choose'],
  coherence:         ['coherent', 'consistent', 'logic', 'flow', 'connect', 'cohesive', 'align'],
  completion:        ['complet', 'finish', 'done', 'final', 'deliver', 'implement', 'build'],
  correctness:       ['correct', 'fix', 'debug', 'test', 'valid', 'error', 'repair', 'resolve'],
  task_progress:     ['progress', 'step', 'advance', 'proceed', 'continu', 'track', 'status'],
  tool_success:      ['tool', 'api', 'call', 'fetch', 'execut', 'run', 'invoke', 'deploy'],
  information_gain:  ['search', 'research', 'find', 'discover', 'learn', 'explore', 'investigate'],
  output_quality:    ['qualit', 'polish', 'refine', 'improve', 'enhance', 'optimize', 'upgrade'],
  owner_alignment:   ['align', 'follow', 'instruc', 'guidelin', 'requir', 'comply', 'adhere'],
  goal_proximity:    ['goal', 'target', 'objectiv', 'aim', 'reach', 'achieve', 'milestone'],
  completion_signal: ['done', 'success', 'achiev', 'complet', 'satisf', 'fulfill', 'accomplish'],
  security:          ['secur', 'auth', 'encrypt', 'protect', 'guard', 'vulnerab', 'patch'],
  performance:       ['perform', 'speed', 'latency', 'throughput', 'benchmark', 'scalab', 'optim'],
  documentation:     ['document', 'readme', 'comment', 'doc', 'explain', 'guide', 'tutorial'],
  monitoring:        ['monitor', 'watch', 'observe', 'track', 'alert', 'log', 'metric'],
  deployment:        ['deploy', 'release', 'publish', 'ship', 'rollout', 'launch', 'ci/cd'],
};

// ── Enhanced constraint patterns ───────────────────────────────────────────────
const CONSTRAINT_PATTERNS = [
  { pattern: /do not (modify|change|edit|touch|alter|update)\s+(?:the\s+)?([^\s,\.]+(?:\/[^\s,\.]*)?)/i,
    type: 'file_protection', severity: 'critical', extract: (m) => m[2].replace(/\/+$/, '').trim() },
  { pattern: /never (delete|remove|drop|destroy|wipe)\s+(?:the\s+)?([^\s,\.]+)/i,
    type: 'deletion_protection', severity: 'critical', extract: (m) => m[2].trim() },
  { pattern: /prefer reversible/i,
    type: 'reversible_only', severity: 'warning', extract: () => true },
  { pattern: /max (\d+) (tokens?|steps?|calls?|requests?|iterations?)/i,
    type: 'resource_limit', severity: 'warning', extract: (m) => ({ unit: m[2], limit: parseInt(m[1]) }) },
  { pattern: /only use (approved|allowed|safe|whitelisted)/i,
    type: 'safe_tools_only', severity: 'critical', extract: () => true },
  { pattern: /notify (owner|me|user) (if|when|before|after)\s+(.*)/i,
    type: 'owner_notification', severity: 'info', extract: (m) => ({ trigger: m[2], condition: m[3] }) },
  { pattern: /keep (budget|cost|spend) (under|below|within)\s+(\d+)/i,
    type: 'budget_constraint', severity: 'warning', extract: (m) => ({ limit: parseInt(m[3]) }) },
  { pattern: /require (approval|confirmation|review)\s*(?:for|before)?\s*(.*)/i,
    type: 'approval_required', severity: 'critical', extract: (m) => ({ scope: m[1], target: m[2]?.trim() }) },
  { pattern: /(no|without) (rollback|revert|undo)/i,
    type: 'no_rollback', severity: 'info', extract: () => true },
  { pattern: /dry.?run/i,
    type: 'dry_run', severity: 'warning', extract: () => true },
  { pattern: /timeout (?:of|after)\s+(\d+)\s*(ms|sec|seconds?|min|minutes?)/i,
    type: 'timeout', severity: 'warning', extract: (m) => ({ value: parseInt(m[1]), unit: m[2] }) },
];

// ─────────────────────────────────────────────────────────────────────────────
export class HierarchicalCommandInterpreter extends EventEmitter {
  /**
   * @param {object} opts
   * @param {object}  [opts.llmAdapter] – LLM adapter for semantic parsing (optional)
   * @param {number}  [opts.defaultTokens]
   * @param {number}  [opts.defaultEnergy]
   * @param {number}  [opts.defaultWallMs]
   * @param {boolean} [opts.enableDecomposition=true] – Enable complex task decomposition
   */
  constructor(opts = {}) {
    super();
    this.id = opts.id ?? `hci-${uuidv4().slice(0,8)}`;
    this.llmAdapter = opts.llmAdapter ?? null;
    this.enableDecomposition = opts.enableDecomposition ?? true;
    this.defaultBudget = {
      tokens:      opts.defaultTokens ?? 50_000,
      energy:      opts.defaultEnergy ?? 1.0,
      wallClockMs: opts.defaultWallMs ?? 300_000,
    };
    // Directive history for context continuity
    this._directiveHistory = [];
    this._maxHistory = 50;
  }

  /**
   * Parse an Owner directive string into a structured directive object.
   * Uses LLM for semantic understanding if available, falls back to keyword matching.
   * @param {string} rawDirective
   * @param {object} [overrides]
   * @returns {Directive}
   */
  async parse(rawDirective, overrides = {}) {
    const directiveId = uuidv4();
    const goal        = rawDirective.trim();

    // Try LLM-enhanced parsing first
    let llmInsight = null;
    if (this.llmAdapter) {
      try {
        llmInsight = await this._llmParse(goal);
      } catch (e) {
        // Fall back to keyword-based parsing
        this.emit('llmParseFallback', { error: e.message });
      }
    }

    const priority    = overrides.priority ?? (llmInsight?.priority ?? this._inferPriority(goal));
    const urgency     = URGENCY[priority] ?? 0.2;
    const constraints = this._extractConstraints(goal, overrides.constraints ?? []);
    const budget      = this._buildBudget(priority, overrides.budget);
    const stimulusTargets = this._buildStimulusTargets(goal, urgency, llmInsight);

    // Task decomposition for complex directives
    let subTasks = null;
    if (this.enableDecomposition && this._isComplexDirective(goal)) {
      subTasks = this._decomposeDirective(goal, constraints, llmInsight);
    }

    const directive = {
      id:          directiveId,
      raw:         rawDirective,
      goal,
      priority,
      urgency,
      constraints,
      budget,
      stimulusTargets,
      subTasks,
      tauScale:    1 / (1 + urgency * 2),
      timestamp:   Date.now(),
      llmInsight:  llmInsight ? { topics: llmInsight.topics, entities: llmInsight.entities } : null,
    };

    // Track directive history
    this._directiveHistory.push({ id: directiveId, goal, priority, timestamp: Date.now() });
    if (this._directiveHistory.length > this._maxHistory) {
      this._directiveHistory.shift();
    }

    this.emit('directiveParsed', directive);
    return directive;
  }

  /**
   * Synchronous parse for backward compatibility (no LLM).
   */
  parseSync(rawDirective, overrides = {}) {
    const directiveId = uuidv4();
    const goal        = rawDirective.trim();
    const priority    = overrides.priority ?? this._inferPriority(goal);
    const urgency     = URGENCY[priority] ?? 0.2;
    const constraints = this._extractConstraints(goal, overrides.constraints ?? []);
    const budget      = this._buildBudget(priority, overrides.budget);
    const stimulusTargets = this._buildStimulusTargets(goal, urgency, null);

    const directive = {
      id: directiveId, raw: rawDirective, goal, priority, urgency,
      constraints, budget, stimulusTargets, subTasks: null,
      tauScale: 1 / (1 + urgency * 2), timestamp: Date.now(), llmInsight: null,
    };

    this._directiveHistory.push({ id: directiveId, goal, priority, timestamp: Date.now() });
    if (this._directiveHistory.length > this._maxHistory) this._directiveHistory.shift();
    this.emit('directiveParsed', directive);
    return directive;
  }

  /**
   * Project directive targets onto layer-specific addiction prototypes.
   */
  buildLayerTargets(directive, praxisDim = 64) {
    const targets = {};
    const activeClasses = Object.keys(directive.stimulusTargets)
      .filter(k => directive.stimulusTargets[k] > 0.5);

    const prototype = new Float64Array(praxisDim);
    for (let i = 0; i < praxisDim; i++) {
      const classIdx = i % Math.max(activeClasses.length, 1);
      const weight   = activeClasses.length > 0
        ? (directive.stimulusTargets[activeClasses[classIdx]] ?? 0.5)
        : 0.5;
      prototype[i]   = weight * (Math.random() * 0.4 + 0.8);
    }

    for (const layerId of ['l1','l2','l3','l6','l7','l10','l11','l12']) {
      targets[layerId] = prototype;
    }
    return targets;
  }

  /**
   * Validate a proposed praxis against directive constraints.
   * Returns { valid: boolean, violations: Array<{ type, severity, message }> }
   */
  validatePraxis(praxis, directive) {
    const violations = [];

    for (const c of directive.constraints) {
      if (c.type === 'file_protection') {
        const target = String(praxis.args?.path ?? praxis.args?.file ?? '');
        const protected_ = String(c.value ?? '').replace(/\/$/, '');
        if (target.includes(protected_) || target.startsWith(protected_)) {
          violations.push({
            type: 'file_protection',
            severity: c.severity ?? 'critical',
            message: `"${target}" is protected by Owner.`,
          });
        }
      }
      if (c.type === 'deletion_protection') {
        const target = String(praxis.args?.path ?? praxis.args?.target ?? '');
        const protected_ = String(c.value ?? '');
        if (target.toLowerCase().includes(protected_.toLowerCase())) {
          violations.push({
            type: 'deletion_protection',
            severity: c.severity ?? 'critical',
            message: `Deletion of "${target}" is prohibited by Owner.`,
          });
        }
      }
      if (c.type === 'reversible_only' && !praxis.rollbackPlan) {
        violations.push({
          type: 'reversible_only',
          severity: c.severity ?? 'warning',
          message: `Praxis "${praxis.toolId}" has no rollback plan.`,
        });
      }
      if (c.type === 'safe_tools_only') {
        const unsafePrefixes = ['rm', 'delete', 'drop', 'destroy', 'wipe', 'format', 'mkfs'];
        if (unsafePrefixes.some(p => String(praxis.toolId ?? '').startsWith(p))) {
          violations.push({
            type: 'safe_tools_only',
            severity: c.severity ?? 'critical',
            message: `Tool "${praxis.toolId}" is not in safe list.`,
          });
        }
      }
      if (c.type === 'dry_run') {
        violations.push({
          type: 'dry_run',
          severity: 'info',
          message: 'Dry run mode: action would be logged but not executed.',
        });
      }
      if (c.type === 'approval_required') {
        violations.push({
          type: 'approval_required',
          severity: c.severity ?? 'critical',
          message: `Action requires ${c.value?.scope ?? 'approval'} before execution.`,
        });
      }
    }

    return { valid: violations.filter(v => v.severity === 'critical').length === 0, violations };
  }

  /** Get directive history */
  getHistory(limit = 20) {
    return this._directiveHistory.slice(-limit);
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  async _llmParse(goal) {
    if (!this.llmAdapter) return null;

    const prompt = `Analyze the following task directive and extract:
1. Priority level (routine/expedited/critical)
2. Key topics (array of strings)
3. Named entities (array of strings)
4. Complexity score (0-1)
5. Suggested stimulus classes (object mapping class names to relevance 0-1)

Directive: "${goal}"

Respond in JSON format only.`;

    try {
      const response = await this.llmAdapter.chat(prompt);
      const parsed = JSON.parse(response);
      return {
        priority: parsed.priority ?? this._inferPriority(goal),
        topics: parsed.topics ?? [],
        entities: parsed.entities ?? [],
        complexity: parsed.complexity ?? 0.5,
        stimulusHints: parsed.suggested_stimulus_classes ?? {},
      };
    } catch (e) {
      return null;
    }
  }

  _inferPriority(goal) {
    const lower = goal.toLowerCase();
    if (/urgently?|immediately|asap|critical|emergency|breaking/i.test(lower)) return 'critical';
    if (/soon|quickly|expedit|high.?priority|important/i.test(lower))          return 'expedited';
    return 'routine';
  }

  _extractConstraints(goal, explicit) {
    const constraints = [...explicit.map(c =>
      typeof c === 'string' ? { type: 'raw', value: c, severity: 'warning' } : c
    )];

    for (const cp of CONSTRAINT_PATTERNS) {
      const m = goal.match(cp.pattern);
      if (m) constraints.push({ type: cp.type, value: cp.extract(m), severity: cp.severity });
    }

    return constraints;
  }

  _buildBudget(priority, overrides = {}) {
    const scale = priority === 'critical' ? 3.0
                : priority === 'expedited' ? 1.5
                : 1.0;

    return {
      tokens:      overrides.tokens      ?? this.defaultBudget.tokens * scale,
      energy:      overrides.energy      ?? this.defaultBudget.energy * scale,
      wallClockMs: overrides.wallClockMs ?? this.defaultBudget.wallClockMs / scale,
    };
  }

  _buildStimulusTargets(goal, urgency, llmInsight) {
    const lower  = goal.toLowerCase();
    const scores = {};

    for (const [cls, keywords] of Object.entries(GOAL_KEYWORDS)) {
      const hits = keywords.filter(kw => lower.includes(kw)).length;
      scores[cls] = Math.min(1, hits * 0.4 + urgency * 0.1 + (hits > 0 ? 0.3 : 0));

      // Boost from LLM insight
      if (llmInsight?.stimulusHints?.[cls]) {
        scores[cls] = Math.min(1, scores[cls] + llmInsight.stimulusHints[cls] * 0.3);
      }
    }

    scores.completion_signal = Math.max(scores.completion_signal ?? 0, urgency * 0.5 + 0.3);
    scores.goal_proximity    = Math.max(scores.goal_proximity ?? 0, 0.4);

    return scores;
  }

  _isComplexDirective(goal) {
    // Heuristics for task complexity
    const indicators = [' and ', ' then ', ' after ', ' before ', ' also ', ' plus ',
                       ' alongside ', ' following ', ' subsequently ', ', then',
                       ' as well as ', ' in addition '];
    const hasConnectors = indicators.some(c => goal.toLowerCase().includes(c));
    const isLong = goal.split(/\s+/).length > 15;
    const hasMultipleVerbs = (goal.match(/\b(implement|build|create|debug|fix|test|deploy|refactor|analyze|search|write|update|delete)\b/gi) || []).length > 1;
    return hasConnectors || isLong || hasMultipleVerbs;
  }

  _decomposeDirective(goal, constraints, llmInsight) {
    // Simple rule-based decomposition
    const connectors = [' and ', ', then ', ' then ', ' after ', '; '];
    let parts = [goal];

    for (const conn of connectors) {
      if (goal.toLowerCase().includes(conn)) {
        parts = goal.split(new RegExp(conn.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i'))
                    .map(s => s.trim())
                    .filter(s => s.length > 0);
        break;
      }
    }

    if (parts.length <= 1) return null;

    return parts.map((part, idx) => ({
      id: uuidv4(),
      order: idx,
      goal: part,
      priority: idx === 0 ? 'expedited' : 'routine',
      dependencies: idx > 0 ? [parts.slice(0, idx).length - 1] : [],
    }));
  }
}
