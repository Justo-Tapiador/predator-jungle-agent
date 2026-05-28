import { EventEmitter } from 'eventemitter3';

/**
 * SafetyGuardrails - Pre-execution safety filter for the PREDATOR agent system.
 *
 * Acts as a gatekeeper before praxis tensors are dispatched to tools, preventing
 * harmful, destructive, or undesired actions based on configurable rules, rate
 * limits, resource caps, and safety levels.
 */
export class SafetyGuardrails extends EventEmitter {
  /**
   * @param {object} opts
   * @param {string[]} [opts.protectedPaths]     - Paths that must never be modified.
   * @param {string[]} [opts.blockedCommands]    - Command substrings that are always blocked.
   * @param {number}   [opts.maxFileSize]         - Max single-file size in bytes.
   * @param {number}   [opts.maxNetworkRequests]  - Max network requests per action.
   * @param {number}   [opts.maxFileOperations]   - Max file operations per action.
   * @param {object}   [opts.rateLimits]          - { perMinute, perHour } rate ceilings.
   * @param {string[]} [opts.allowedTools]        - Tool whitelist (empty / omitted = all allowed).
   * @param {'permissive'|'standard'|'strict'} [opts.safetyLevel] - Safety strictness.
   */
  constructor(opts = {}) {
    super();

    // Protected paths that should never be modified
    this.protectedPaths = new Set(opts.protectedPaths ?? ['/etc', '/root', '/sys', '/proc']);

    // Dangerous commands that should be blocked
    this.blockedCommands = new Set(opts.blockedCommands ?? ['rm -rf', 'mkfs', 'dd if=', 'format', 'del /f']);

    // Maximum allowed resource consumption per action
    this.maxFileSize = opts.maxFileSize ?? 10 * 1024 * 1024;       // 10 MB
    this.maxNetworkRequests = opts.maxNetworkRequests ?? 100;
    this.maxFileOperations = opts.maxFileOperations ?? 50;

    // Rate limiting
    this.rateLimits = opts.rateLimits ?? { perMinute: 60, perHour: 500 };

    // Allowed tool whitelist (null = all allowed)
    this.allowedTools = opts.allowedTools ? new Set(opts.allowedTools) : null;

    // Safety level: 'permissive', 'standard', 'strict'
    this.safetyLevel = opts.safetyLevel ?? 'standard';

    // Audit trail
    this.auditTrail = [];

    // Rate counters – arrays of timestamps
    this._rateCounters = { minute: [], hour: [] };
  }

  // ---------------------------------------------------------------------------
  // Core checks
  // ---------------------------------------------------------------------------

  /**
   * Check if a proposed action (directive + praxis) is safe to execute.
   *
   * @param {object} directive - The originating directive.
   * @param {object} praxis    - The praxis tensor about to be dispatched.
   * @returns {{ allowed: boolean, reason: string, severity: 'info'|'warning'|'critical' }}
   */
  checkStep(directive, praxis) {
    const result = this._checkStepInternal(directive, praxis);
    this._recordAudit('checkStep', result, { directive, praxis });
    if (!result.allowed) {
      this.emit('blocked', { directive, praxis, result });
    } else {
      this.emit('allowed', { directive, praxis, result });
    }
    return result;
  }

  /**
   * Internal implementation of checkStep – runs all sub-checks in order.
   * @private
   */
  _checkStepInternal(directive, praxis) {
    // 1. Tool whitelist
    const toolResult = this._checkToolWhitelist(praxis);
    if (!toolResult.allowed) return toolResult;

    // 2. Protected paths
    const pathResult = this._checkProtectedPaths(praxis);
    if (!pathResult.allowed) return pathResult;

    // 3. Blocked commands
    const cmdResult = this._checkBlockedCommands(praxis);
    if (!cmdResult.allowed) return cmdResult;

    // 4. Rate limits
    const rateResult = this.checkRateLimit();
    if (!rateResult.allowed) {
      return {
        allowed: false,
        reason: `Rate limit exceeded. Retry after ${rateResult.remainingMs} ms.`,
        severity: 'warning',
      };
    }

    // 5. Resource limits
    const resourceResult = this._checkResourceLimits(praxis);
    if (!resourceResult.allowed) return resourceResult;

    // 6. (Strict mode) Rollback plan required for destructive actions
    if (this.safetyLevel === 'strict') {
      const rollbackResult = this._checkRollbackPlan(praxis);
      if (!rollbackResult.allowed) return rollbackResult;
    }

    // 7. Directive-level constraints
    const directiveResult = this.checkDirective(directive);
    if (!directiveResult.allowed) return directiveResult;

    // All checks passed – record the rate-limited action
    this._recordRateHit();

    return { allowed: true, reason: 'All safety checks passed.', severity: 'info' };
  }

  /**
   * Check if a directive is safe to execute (before any processing).
   *
   * @param {object} directive - The directive to validate.
   * @returns {{ allowed: boolean, reason: string, severity: 'info'|'warning'|'critical' }}
   */
  checkDirective(directive) {
    if (!directive || typeof directive !== 'object') {
      return { allowed: false, reason: 'Directive is missing or invalid.', severity: 'critical' };
    }

    // 1. Dangerous instructions
    const dangerousResult = this._checkDangerousInstructions(directive);
    if (!dangerousResult.allowed) return dangerousResult;

    // 2. Budget reasonableness
    const budgetResult = this._checkDirectiveBudget(directive);
    if (!budgetResult.allowed) return budgetResult;

    // 3. Conflicting constraints
    const conflictResult = this._checkConflictingConstraints(directive);
    if (!conflictResult.allowed) return conflictResult;

    return { allowed: true, reason: 'Directive passed all checks.', severity: 'info' };
  }

  // ---------------------------------------------------------------------------
  // Public helpers
  // ---------------------------------------------------------------------------

  /**
   * Add a path to the protected list.
   * @param {string} path
   */
  addProtectedPath(path) {
    this.protectedPaths.add(path);
    this.emit('protectedPathAdded', { path });
  }

  /**
   * Remove a path from the protected list.
   * @param {string} path
   * @returns {boolean} True if the path was present and removed.
   */
  removeProtectedPath(path) {
    const removed = this.protectedPaths.delete(path);
    if (removed) this.emit('protectedPathRemoved', { path });
    return removed;
  }

  /**
   * Add a command pattern to the blocked list.
   * @param {string} cmd
   */
  addBlockedCommand(cmd) {
    this.blockedCommands.add(cmd);
    this.emit('blockedCommandAdded', { command: cmd });
  }

  /**
   * Set the safety level.
   * @param {'permissive'|'standard'|'strict'} level
   */
  setSafetyLevel(level) {
    const valid = ['permissive', 'standard', 'strict'];
    if (!valid.includes(level)) {
      throw new Error(`Invalid safety level "${level}". Must be one of: ${valid.join(', ')}`);
    }
    this.safetyLevel = level;
    this.emit('safetyLevelChanged', { level });
  }

  /**
   * Get recent audit entries.
   * @param {number} [limit=50] - Maximum entries to return.
   * @returns {object[]}
   */
  getAuditTrail(limit = 50) {
    return this.auditTrail.slice(-limit);
  }

  /**
   * Clear the audit trail.
   */
  clearAuditTrail() {
    this.auditTrail = [];
    this.emit('auditTrailCleared');
  }

  /**
   * Check whether we are currently within rate limits.
   * @returns {{ allowed: boolean, remainingMs: number }}
   */
  checkRateLimit() {
    const now = Date.now();
    this._pruneRateCounters(now);

    if (this._rateCounters.minute.length >= this.rateLimits.perMinute) {
      const oldestInMinute = this._rateCounters.minute[0];
      return { allowed: false, remainingMs: oldestInMinute + 60_000 - now };
    }

    if (this._rateCounters.hour.length >= this.rateLimits.perHour) {
      const oldestInHour = this._rateCounters.hour[0];
      return { allowed: false, remainingMs: oldestInHour + 3_600_000 - now };
    }

    return { allowed: true, remainingMs: 0 };
  }

  // ---------------------------------------------------------------------------
  // Private sub-checks
  // ---------------------------------------------------------------------------

  /** @private Check tool whitelist. */
  _checkToolWhitelist(praxis) {
    if (!this.allowedTools) return { allowed: true, reason: 'No tool whitelist configured.', severity: 'info' };

    const tool = praxis?.tool ?? praxis?.action;
    if (!tool) {
      return { allowed: false, reason: 'Praxis does not specify a tool or action.', severity: 'warning' };
    }

    if (!this.allowedTools.has(tool)) {
      return {
        allowed: false,
        reason: `Tool "${tool}" is not in the allowed tools whitelist.`,
        severity: 'critical',
      };
    }

    return { allowed: true, reason: 'Tool is whitelisted.', severity: 'info' };
  }

  /** @private Check whether the praxis targets a protected path. */
  _checkProtectedPaths(praxis) {
    const candidates = this._extractPaths(praxis);

    for (const candidate of candidates) {
      for (const protectedPath of this.protectedPaths) {
        if (candidate === protectedPath || candidate.startsWith(protectedPath + '/')) {
          return {
            allowed: false,
            reason: `Path "${candidate}" is protected ("${protectedPath}").`,
            severity: 'critical',
          };
        }
      }
    }

    return { allowed: true, reason: 'No protected paths targeted.', severity: 'info' };
  }

  /** @private Check whether the praxis contains a blocked command. */
  _checkBlockedCommands(praxis) {
    const text = this._extractCommandText(praxis);
    const lower = text.toLowerCase();

    for (const blocked of this.blockedCommands) {
      if (lower.includes(blocked.toLowerCase())) {
        return {
          allowed: false,
          reason: `Blocked command pattern detected: "${blocked}".`,
          severity: 'critical',
        };
      }
    }

    return { allowed: true, reason: 'No blocked commands detected.', severity: 'info' };
  }

  /** @private Check resource limits encoded in the praxis. */
  _checkResourceLimits(praxis) {
    // File size check
    const fileSize = praxis?.fileSize ?? praxis?.size ?? 0;
    if (fileSize > this.maxFileSize) {
      return {
        allowed: false,
        reason: `File size ${fileSize} exceeds maximum ${this.maxFileSize} bytes.`,
        severity: 'warning',
      };
    }

    // Network request count
    const networkRequests = praxis?.networkRequests ?? praxis?.requestCount ?? 0;
    if (networkRequests > this.maxNetworkRequests) {
      return {
        allowed: false,
        reason: `Network request count ${networkRequests} exceeds maximum ${this.maxNetworkRequests}.`,
        severity: 'warning',
      };
    }

    // File operation count
    const fileOps = praxis?.fileOperations ?? praxis?.operationCount ?? 0;
    if (fileOps > this.maxFileOperations) {
      return {
        allowed: false,
        reason: `File operation count ${fileOps} exceeds maximum ${this.maxFileOperations}.`,
        severity: 'warning',
      };
    }

    return { allowed: true, reason: 'Within resource limits.', severity: 'info' };
  }

  /** @private (Strict mode) Require a rollback plan for destructive actions. */
  _checkRollbackPlan(praxis) {
    const isDestructive = this._isDestructiveAction(praxis);
    if (!isDestructive) {
      return { allowed: true, reason: 'Action is not destructive; rollback not required.', severity: 'info' };
    }

    const hasRollback = Boolean(praxis?.rollback ?? praxis?.rollbackPlan ?? praxis?.undo);
    if (!hasRollback) {
      return {
        allowed: false,
        reason: 'Strict mode requires a rollback plan for destructive actions. Provide praxis.rollback or praxis.rollbackPlan.',
        severity: 'critical',
      };
    }

    return { allowed: true, reason: 'Rollback plan present for destructive action.', severity: 'info' };
  }

  /** @private Check for dangerous instructions inside a directive. */
  _checkDangerousInstructions(directive) {
    const text = this._flattenToString(directive);
    const lower = text.toLowerCase();

    const dangerPatterns = [
      'wipe all', 'delete everything', 'destroy all', 'erase all',
      'drop database', 'truncate table', 'remove all files',
      'format drive', 'brick device', 'brick system',
    ];

    // In permissive mode, skip instruction-level checks
    if (this.safetyLevel === 'permissive') {
      return { allowed: true, reason: 'Permissive mode – dangerous instruction check skipped.', severity: 'info' };
    }

    for (const pattern of dangerPatterns) {
      if (lower.includes(pattern)) {
        return {
          allowed: false,
          reason: `Dangerous instruction detected: "${pattern}".`,
          severity: 'critical',
        };
      }
    }

    return { allowed: true, reason: 'No dangerous instructions detected.', severity: 'info' };
  }

  /** @private Check if the directive's budget request is reasonable. */
  _checkDirectiveBudget(directive) {
    const budget = directive?.budget ?? directive?.cost ?? 0;

    if (typeof budget !== 'number') {
      return { allowed: true, reason: 'No numeric budget specified.', severity: 'info' };
    }

    // Reasonable caps by safety level
    const maxBudget = { permissive: Infinity, standard: 1000, strict: 200 };

    if (budget > maxBudget[this.safetyLevel]) {
      return {
        allowed: false,
        reason: `Budget ${budget} exceeds ${this.safetyLevel}-mode maximum of ${maxBudget[this.safetyLevel]}.`,
        severity: 'warning',
      };
    }

    return { allowed: true, reason: 'Budget within acceptable range.', severity: 'info' };
  }

  /** @private Check for conflicting constraints in a directive. */
  _checkConflictingConstraints(directive) {
    const constraints = directive?.constraints ?? directive?.requirements ?? [];

    if (!Array.isArray(constraints) || constraints.length < 2) {
      return { allowed: true, reason: 'No conflicting constraints detected.', severity: 'info' };
    }

    // Look for directly opposing constraint pairs
    const negationMap = {
      'read-only': 'write',
      'write': 'read-only',
      'local-only': 'network',
      'network': 'local-only',
      'no-delete': 'delete',
      'delete': 'no-delete',
    };

    for (const c of constraints) {
      const key = String(c).toLowerCase();
      const opposite = negationMap[key];
      if (opposite && constraints.some((o) => String(o).toLowerCase() === opposite)) {
        return {
          allowed: false,
          reason: `Conflicting constraints detected: "${key}" vs "${opposite}".`,
          severity: 'warning',
        };
      }
    }

    return { allowed: true, reason: 'No conflicting constraints detected.', severity: 'info' };
  }

  // ---------------------------------------------------------------------------
  // Utility helpers
  // ---------------------------------------------------------------------------

  /**
   * Extract file-system paths from a praxis object.
   * @private
   */
  _extractPaths(praxis) {
    const paths = [];
    if (!praxis) return paths;

    if (typeof praxis.path === 'string') paths.push(praxis.path);
    if (typeof praxis.target === 'string') paths.push(praxis.target);
    if (typeof praxis.destination === 'string') paths.push(praxis.destination);
    if (typeof praxis.source === 'string') paths.push(praxis.source);
    if (Array.isArray(praxis.paths)) paths.push(...praxis.paths);
    if (Array.isArray(praxis.targets)) paths.push(...praxis.targets);

    // Also scan args that look like paths
    const args = praxis.args ?? praxis.arguments ?? [];
    if (Array.isArray(args)) {
      for (const arg of args) {
        if (typeof arg === 'string' && arg.startsWith('/')) {
          paths.push(arg);
        }
      }
    }

    return paths;
  }

  /**
   * Extract command text from a praxis object.
   * @private
   */
  _extractCommandText(praxis) {
    if (!praxis) return '';

    const parts = [];
    if (typeof praxis.command === 'string') parts.push(praxis.command);
    if (typeof praxis.cmd === 'string') parts.push(praxis.cmd);
    if (typeof praxis.script === 'string') parts.push(praxis.script);
    if (typeof praxis.action === 'string') parts.push(praxis.action);
    if (typeof praxis.tool === 'string') parts.push(praxis.tool);

    const args = praxis.args ?? praxis.arguments ?? [];
    if (Array.isArray(args)) {
      for (const arg of args) {
        if (typeof arg === 'string') parts.push(arg);
      }
    }

    return parts.join(' ');
  }

  /**
   * Determine whether a praxis represents a destructive action.
   * @private
   */
  _isDestructiveAction(praxis) {
    const destructiveTools = ['rm', 'delete', 'remove', 'destroy', 'drop', 'truncate', 'format', 'erase'];
    const destructiveActions = ['delete', 'remove', 'destroy', 'drop', 'truncate', 'format', 'erase', 'overwrite'];

    const tool = (praxis?.tool ?? praxis?.action ?? '').toLowerCase();
    const action = (praxis?.action ?? praxis?.method ?? '').toLowerCase();

    if (destructiveTools.includes(tool) || destructiveActions.includes(action)) return true;

    // Check command text for destructive verbs
    const cmdText = this._extractCommandText(praxis).toLowerCase();
    const destructiveVerbs = ['delete', 'remove', 'destroy', 'drop', 'truncate', 'format', 'erase', 'overwrite'];
    for (const verb of destructiveVerbs) {
      if (cmdText.includes(verb)) return true;
    }

    return false;
  }

  /**
   * Recursively flatten an object to a single searchable string.
   * @private
   */
  _flattenToString(obj, depth = 0) {
    if (depth > 4) return '';
    if (obj == null) return '';
    if (typeof obj === 'string') return obj;
    if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);

    const parts = [];
    for (const value of Object.values(obj)) {
      parts.push(this._flattenToString(value, depth + 1));
    }
    return parts.join(' ');
  }

  // ---------------------------------------------------------------------------
  // Rate-limit bookkeeping
  // ---------------------------------------------------------------------------

  /** @private Record a timestamp for rate-limit counting. */
  _recordRateHit() {
    const now = Date.now();
    this._rateCounters.minute.push(now);
    this._rateCounters.hour.push(now);
  }

  /** @private Evict timestamps outside the current windows. */
  _pruneRateCounters(now) {
    const minuteAgo = now - 60_000;
    const hourAgo = now - 3_600_000;

    this._rateCounters.minute = this._rateCounters.minute.filter((ts) => ts > minuteAgo);
    this._rateCounters.hour = this._rateCounters.hour.filter((ts) => ts > hourAgo);
  }

  // ---------------------------------------------------------------------------
  // Audit trail
  // ---------------------------------------------------------------------------

  /** @private Append an entry to the audit trail. */
  _recordAudit(checkType, result, context) {
    this.auditTrail.push({
      timestamp: Date.now(),
      checkType,
      allowed: result.allowed,
      reason: result.reason,
      severity: result.severity,
      // Store a shallow snapshot to avoid holding large object graphs
      contextKeys: context ? Object.keys(context) : [],
    });

    // Cap the trail at 10 000 entries to avoid unbounded memory growth
    if (this.auditTrail.length > 10_000) {
      this.auditTrail = this.auditTrail.slice(-5_000);
    }
  }
}

export default SafetyGuardrails;
