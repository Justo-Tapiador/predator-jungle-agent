import { EventEmitter } from 'eventemitter3';

/**
 * Hook names that support cancellation (returning null stops propagation).
 */
const CANCELLABLE_HOOKS = new Set([
  'beforeStep',
  'beforeEmit',
  'directiveReceived',
]);

/**
 * PluginManager — plugin architecture for extending the PREDATOR agent system
 * without modifying core code.
 *
 * A plugin is an object with:
 *   - name: string          — Plugin identifier (required)
 *   - version: string       — Plugin version (required)
 *   - description: string   — Optional description
 *   - hooks: object         — Map of hook names to handler functions
 *   - init?: Function       — Called when plugin is registered
 *   - destroy?: Function    — Called when plugin is unregistered
 *
 * Available hooks:
 *   'beforeStep'          — Before each TPS step: (context) => context | null
 *   'afterStep'           — After each TPS step: (context, result) => void
 *   'beforeEmit'          — Before emitting a praxis: (praxis) => praxis | null
 *   'afterEmit'           — After emitting a praxis: (praxis, feedback) => void
 *   'taskComplete'        — When a task completes: (taskResult) => void
 *   'directiveReceived'   — When a directive is received: (directive) => directive | null
 *   'extinction'          — When an extinction event occurs: (event) => void
 *   'trainingEpoch'       — After each training epoch: (epochData) => void
 */
export class PluginManager extends EventEmitter {
  /**
   * @param {object} [opts={}]
   * @param {number} [opts.maxPlugins=20] - Maximum number of plugins allowed
   */
  constructor(opts = {}) {
    super();
    this.plugins = new Map();   // name → plugin object
    this.hooks = new Map();     // hookName → [{ pluginName, handler, priority }]
    this.maxPlugins = opts.maxPlugins ?? 20;
  }

  /**
   * Register a plugin.
   *
   * 1. Validates plugin has name and version
   * 2. Checks plugin limit
   * 3. Calls plugin.init(this) if present
   * 4. Registers all hooks from plugin.hooks with priority (default 50)
   * 5. Emits 'pluginRegistered' event
   *
   * @param {object} plugin - The plugin object to register
   * @returns {PluginManager} this (for chaining)
   * @throws {Error} If plugin is invalid, already registered, or limit exceeded
   */
  register(plugin) {
    // 1. Validate structure
    if (!plugin || typeof plugin !== 'object') {
      throw new Error('Plugin must be a non-null object');
    }

    if (!plugin.name || typeof plugin.name !== 'string') {
      throw new Error('Plugin must have a string "name" property');
    }

    if (!plugin.version || typeof plugin.version !== 'string') {
      throw new Error('Plugin must have a string "version" property');
    }

    // 2. Check for duplicate registration
    if (this.plugins.has(plugin.name)) {
      throw new Error(`Plugin "${plugin.name}" is already registered`);
    }

    // 3. Check plugin limit
    if (this.plugins.size >= this.maxPlugins) {
      throw new Error(
        `Cannot register plugin "${plugin.name}": maximum of ${this.maxPlugins} plugins reached`
      );
    }

    // 4. Call init if present
    if (typeof plugin.init === 'function') {
      plugin.init(this);
    }

    // 5. Register hooks
    if (plugin.hooks && typeof plugin.hooks === 'object') {
      for (const [hookName, handlerOrEntry] of Object.entries(plugin.hooks)) {
        if (typeof handlerOrEntry === 'function') {
          this._addHook(hookName, plugin.name, handlerOrEntry, 50);
        } else if (handlerOrEntry && typeof handlerOrEntry === 'object') {
          // Allow { handler, priority } form
          const { handler, priority = 50 } = handlerOrEntry;
          if (typeof handler !== 'function') {
            throw new Error(
              `Plugin "${plugin.name}" hook "${hookName}" must have a handler function`
            );
          }
          this._addHook(hookName, plugin.name, handler, priority);
        } else {
          throw new Error(
            `Plugin "${plugin.name}" hook "${hookName}" must be a function or { handler, priority } object`
          );
        }
      }
    }

    // 6. Store the plugin
    this.plugins.set(plugin.name, plugin);

    // 7. Emit event
    this.emit('pluginRegistered', { name: plugin.name, version: plugin.version });

    return this;
  }

  /**
   * Unregister a plugin by name.
   *
   * Calls plugin.destroy() if present and removes all associated hooks.
   *
   * @param {string} name - The plugin name to unregister
   * @returns {boolean} true if the plugin was found and removed, false otherwise
   */
  unregister(name) {
    const plugin = this.plugins.get(name);
    if (!plugin) {
      return false;
    }

    // Call destroy if present
    if (typeof plugin.destroy === 'function') {
      plugin.destroy(this);
    }

    // Remove all hooks belonging to this plugin
    for (const [hookName, handlers] of this.hooks) {
      const filtered = handlers.filter((entry) => entry.pluginName !== name);
      if (filtered.length === 0) {
        this.hooks.delete(hookName);
      } else {
        this.hooks.set(hookName, filtered);
      }
    }

    // Remove the plugin
    this.plugins.delete(name);

    this.emit('pluginUnregistered', { name });

    return true;
  }

  /**
   * Fire a hook, calling all registered handlers in priority order (lower = first).
   *
   * For cancellable hooks (beforeStep, beforeEmit, directiveReceived),
   * if a handler returns null, propagation stops and null is returned.
   *
   * For non-cancellable hooks, handler return values are ignored and
   * the first argument is returned unchanged (or undefined if no args).
   *
   * @param {string} hookName - The name of the hook to fire
   * @param {...*} args - Arguments to pass to the hook handlers
   * @returns {Promise<*>} The (possibly modified) first argument, or null if cancelled
   */
  async fireHook(hookName, ...args) {
    const handlers = this.hooks.get(hookName);
    if (!handlers || handlers.length === 0) {
      return args[0];
    }

    // Sort by priority (lower number = higher priority = runs first)
    const sorted = [...handlers].sort((a, b) => a.priority - b.priority);

    const isCancellable = CANCELLABLE_HOOKS.has(hookName);

    // For cancellable hooks, the first arg is the mutable payload
    // Handlers may return a modified value or null to cancel
    let payload = args[0];

    for (const entry of sorted) {
      try {
        const callArgs = isCancellable ? [payload, ...args.slice(1)] : args;
        const result = await entry.handler(...callArgs);

        if (isCancellable) {
          if (result === null) {
            // Propagation stopped — return null to signal cancellation
            return null;
          }
          // If handler returns a non-undefined value, use it as the new payload
          if (result !== undefined) {
            payload = result;
          }
        }
      } catch (err) {
        this.emit('hookError', {
          hookName,
          pluginName: entry.pluginName,
          error: err,
        });
      }
    }

    return isCancellable ? payload : args[0];
  }

  /**
   * Get a registered plugin by name.
   *
   * @param {string} name - Plugin name
   * @returns {object|undefined} The plugin object, or undefined if not found
   */
  getPlugin(name) {
    return this.plugins.get(name);
  }

  /**
   * List all registered plugins with metadata.
   *
   * @returns {Array<{name: string, version: string, description: string, hookCount: number}>}
   */
  listPlugins() {
    const result = [];
    for (const [name, plugin] of this.plugins) {
      let hookCount = 0;
      if (plugin.hooks && typeof plugin.hooks === 'object') {
        hookCount = Object.keys(plugin.hooks).length;
      }
      result.push({
        name: plugin.name,
        version: plugin.version,
        description: plugin.description ?? '',
        hookCount,
      });
    }
    return result;
  }

  /**
   * Check if a plugin is registered.
   *
   * @param {string} name - Plugin name
   * @returns {boolean}
   */
  hasPlugin(name) {
    return this.plugins.has(name);
  }

  // ─── Internal helpers ─────────────────────────────────────────────────

  /**
   * Add a hook entry for a given hook name.
   *
   * @param {string} hookName
   * @param {string} pluginName
   * @param {Function} handler
   * @param {number} priority
   * @private
   */
  _addHook(hookName, pluginName, handler, priority) {
    if (!this.hooks.has(hookName)) {
      this.hooks.set(hookName, []);
    }
    this.hooks.get(hookName).push({ pluginName, handler, priority });
  }
}

export default PluginManager;
