/**
 * ToolRegistry.js
 * Central tool registry for the PREDATOR agent system.
 * Manages all available tools, validates arguments, and provides
 * PSE (Plan-Schedule-Execute) integration handlers.
 * Single source of truth for available tools.
 */

class ToolRegistry {
  constructor(options = {}) {
    this.id = 'tool_registry';
    this.name = 'ToolRegistry';
    this.description = 'Central registry managing all PREDATOR agent tools';

    // Core data structures
    this._tools = new Map(); // toolId -> { instance, meta, schema }
    this._categories = new Map(); // category -> Set<toolId>

    // Event listeners for tool lifecycle
    this._listeners = new Map(); // event -> Set<callback>

    // Options
    this.strictValidation = options.strictValidation !== false; // true by default
  }

  // ── Helpers ──────────────────────────────────────────────────────────

  _emit(event, data) {
    const listeners = this._listeners.get(event);
    if (listeners) {
      for (const cb of listeners) {
        try {
          cb(data);
        } catch (err) {
          // Listener errors should not break the registry
          console.error(`ToolRegistry listener error on "${event}":`, err.message);
        }
      }
    }
  }

  // ── Register ─────────────────────────────────────────────────────────

  register(toolId, toolInstance, meta = {}) {
    if (!toolId || typeof toolId !== 'string') {
      throw new Error('toolId must be a non-empty string');
    }

    if (!toolInstance || typeof toolInstance !== 'object') {
      throw new Error('toolInstance must be an object');
    }

    if (this._tools.has(toolId)) {
      throw new Error(`Tool "${toolId}" is already registered. Unregister it first.`);
    }

    const entry = {
      id: toolId,
      instance: toolInstance,
      meta: {
        name: meta.name || toolInstance.name || toolId,
        description: meta.description || toolInstance.description || '',
        version: meta.version || '1.0.0',
        category: meta.category || 'general',
        tags: meta.tags || [],
        author: meta.author || 'unknown',
        deprecated: meta.deprecated || false,
        ...meta,
      },
      schema: meta.schema || this._inferSchema(toolInstance),
      registeredAt: new Date().toISOString(),
    };

    this._tools.set(toolId, entry);

    // Update category index
    const category = entry.meta.category;
    if (!this._categories.has(category)) {
      this._categories.set(category, new Set());
    }
    this._categories.get(category).add(toolId);

    this._emit('registered', { toolId, meta: entry.meta });

    return entry;
  }

  /**
   * Infer a basic JSON schema from the tool instance's method signatures.
   * Each method gets an entry with parameter names extracted.
   */
  _inferSchema(toolInstance) {
    const schema = { methods: {} };
    const proto = Object.getPrototypeOf(toolInstance);
    const methodNames = Object.getOwnPropertyNames(proto).filter(
      (name) =>
        name !== 'constructor' &&
        typeof proto[name] === 'function' &&
        !name.startsWith('_'),
    );

    for (const methodName of methodNames) {
      // Extract parameter names from function signature
      const fnStr = proto[methodName].toString();
      const paramMatch = fnStr.match(
        /^(?:async\s+)?(?:function\s*)?\w*\s*\(([^)]*)\)/,
      );
      const params = paramMatch
        ? paramMatch[1]
            .split(',')
            .map((p) => p.trim())
            .filter(Boolean)
            .map((p) => {
              // Handle default values
              const [name, defaultVal] = p.split('=').map((s) => s.trim());
              return {
                name,
                required: defaultVal === undefined,
                default: defaultVal !== undefined ? this._parseDefault(defaultVal) : undefined,
              };
            })
        : [];

      schema.methods[methodName] = {
        type: 'function',
        async: fnStr.includes('async'),
        params,
      };
    }

    return schema;
  }

  _parseDefault(val) {
    if (val === 'true') return true;
    if (val === 'false') return false;
    if (val === 'null') return null;
    if (val === 'undefined') return undefined;
    if (!isNaN(Number(val))) return Number(val);
    // String defaults (remove quotes)
    if (/^['"]/.test(val)) return val.slice(1, -1);
    return val;
  }

  // ── Unregister ───────────────────────────────────────────────────────

  unregister(toolId) {
    const entry = this._tools.get(toolId);
    if (!entry) {
      return { success: false, error: `Tool "${toolId}" is not registered` };
    }

    // Remove from category index
    const category = entry.meta.category;
    const catSet = this._categories.get(category);
    if (catSet) {
      catSet.delete(toolId);
      if (catSet.size === 0) {
        this._categories.delete(category);
      }
    }

    this._tools.delete(toolId);
    this._emit('unregistered', { toolId });

    return { success: true, toolId };
  }

  // ── Get ──────────────────────────────────────────────────────────────

  get(toolId) {
    const entry = this._tools.get(toolId);
    if (!entry) {
      return null;
    }
    return entry.instance;
  }

  // ── List ─────────────────────────────────────────────────────────────

  list(options = {}) {
    const { category, tag, includeDeprecated = false, detailed = false } = options;

    const results = [];

    for (const [toolId, entry] of this._tools) {
      // Filter by category
      if (category && entry.meta.category !== category) continue;

      // Filter by tag
      if (tag && !entry.meta.tags.includes(tag)) continue;

      // Filter deprecated
      if (!includeDeprecated && entry.meta.deprecated) continue;

      if (detailed) {
        results.push({
          id: toolId,
          meta: entry.meta,
          schema: entry.schema,
          registeredAt: entry.registeredAt,
        });
      } else {
        results.push({
          id: toolId,
          name: entry.meta.name,
          description: entry.meta.description,
          category: entry.meta.category,
        });
      }
    }

    return results;
  }

  // ── List Categories ──────────────────────────────────────────────────

  listCategories() {
    const categories = {};
    for (const [cat, toolIds] of this._categories) {
      categories[cat] = [...toolIds];
    }
    return categories;
  }

  // ── Validate ─────────────────────────────────────────────────────────

  validate(toolId, args = {}) {
    const entry = this._tools.get(toolId);

    if (!entry) {
      return {
        valid: false,
        errors: [`Tool "${toolId}" is not registered`],
      };
    }

    const errors = [];

    // Validate method name
    const methodName = args.method;
    if (!methodName) {
      // If no specific method, just check the tool exists
      return { valid: true, errors: [] };
    }

    const methodSchema = entry.schema.methods?.[methodName];
    if (!methodSchema) {
      errors.push(`Method "${methodName}" does not exist on tool "${toolId}"`);
      return { valid: false, errors };
    }

    // Validate parameters
    const providedArgs = args.args || {};
    const schemaParams = methodSchema.params || [];

    if (this.strictValidation) {
      // Check required parameters
      for (const param of schemaParams) {
        if (param.required && providedArgs[param.name] === undefined) {
          errors.push(
            `Missing required parameter "${param.name}" for method "${methodName}"`,
          );
        }
      }

      // Check for unknown parameters
      const knownParams = new Set(schemaParams.map((p) => p.name));
      for (const key of Object.keys(providedArgs)) {
        if (!knownParams.has(key)) {
          errors.push(
            `Unknown parameter "${key}" for method "${methodName}"`,
          );
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  // ── Execute ──────────────────────────────────────────────────────────

  async execute(toolId, method, args = {}) {
    const start = Date.now();

    const entry = this._tools.get(toolId);
    if (!entry) {
      return {
        success: false,
        error: `Tool "${toolId}" is not registered`,
        duration: Date.now() - start,
      };
    }

    const fn = entry.instance[method];
    if (typeof fn !== 'function') {
      return {
        success: false,
        error: `Method "${method}" does not exist on tool "${toolId}"`,
        duration: Date.now() - start,
      };
    }

    try {
      const result = await fn.call(entry.instance, args);
      return result; // Tools already return structured results
    } catch (err) {
      return {
        success: false,
        error: `Execution error: ${err.message}`,
        duration: Date.now() - start,
      };
    }
  }

  // ── PSE Handlers ─────────────────────────────────────────────────────

  /**
   * Creates a Map of tool handlers suitable for PSE (Plan-Schedule-Execute)
   * integration. Each handler is an async function that dispatches to the
   * appropriate tool method.
   *
   * @returns {Map<string, Function>} Map of handler functions keyed by toolId
   */
  createPSEHandlers() {
    const handlers = new Map();

    for (const [toolId, entry] of this._tools) {
      if (entry.meta.deprecated) continue;

      const handler = async (action, params = {}) => {
        // Validate the call
        const validation = this.validate(toolId, { method: action, args: params });
        if (!validation.valid) {
          return {
            success: false,
            error: `Validation failed: ${validation.errors.join('; ')}`,
            duration: 0,
          };
        }

        // Dispatch to the tool method
        return this.execute(toolId, action, params);
      };

      // Attach metadata for PSE planner
      handler.toolId = toolId;
      handler.meta = entry.meta;
      handler.schema = entry.schema;

      handlers.set(toolId, handler);
    }

    return handlers;
  }

  // ── Event System ─────────────────────────────────────────────────────

  on(event, callback) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    this._listeners.get(event).add(callback);
    return () => this._listeners.get(event)?.delete(callback);
  }

  off(event, callback) {
    const listeners = this._listeners.get(event);
    if (listeners) {
      listeners.delete(callback);
    }
  }

  // ── Statistics ───────────────────────────────────────────────────────

  stats() {
    let totalMethods = 0;
    for (const [, entry] of this._tools) {
      totalMethods += Object.keys(entry.schema.methods || {}).length;
    }

    return {
      totalTools: this._tools.size,
      totalMethods,
      totalCategories: this._categories.size,
      categories: Object.fromEntries(
        [...this._categories].map(([cat, ids]) => [cat, ids.size]),
      ),
    };
  }

  // ── Bulk Operations ──────────────────────────────────────────────────

  /**
   * Register multiple tools at once.
   * @param {Array<{id: string, instance: object, meta?: object}>} tools
   */
  registerMany(tools) {
    const results = [];
    for (const { id, instance, meta } of tools) {
      try {
        results.push(this.register(id, instance, meta));
      } catch (err) {
        results.push({ id, error: err.message });
      }
    }
    return results;
  }

  /**
   * Unregister all tools and clear the registry.
   */
  clear() {
    const toolIds = [...this._tools.keys()];
    for (const id of toolIds) {
      this.unregister(id);
    }
    this._categories.clear();
    this._listeners.clear();
  }
}

export default ToolRegistry;
