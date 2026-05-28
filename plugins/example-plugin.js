/**
 * Example PREDATOR v2.0 Plugin – Task Logger
 *
 * Demonstrates the plugin architecture by logging task execution
 * details to a file for post-hoc analysis.
 *
 * Usage:
 *   import { taskLoggerPlugin } from './plugins/example-plugin.js';
 *   agent.use(taskLoggerPlugin);
 */

import { appendFile, mkdir } from 'fs/promises';
import { join } from 'path';

export const taskLoggerPlugin = {
  name: 'task-logger',
  version: '1.0.0',
  description: 'Logs task execution events to a file for analysis',

  hooks: {
    directiveReceived: {
      handler: (directive) => {
        console.log(`[task-logger] Directive received: ${directive.goal?.slice(0, 50)}`);
        return directive;
      },
      priority: 10,
    },

    afterStep: {
      handler: (context, result) => {
        // Could log each step, but that's verbose. Just track.
        return;
      },
      priority: 50,
    },

    taskComplete: {
      handler: async (taskResult) => {
        const logDir = './logs';
        const logFile = join(logDir, 'task-log.jsonl');

        try {
          await mkdir(logDir, { recursive: true });
          const entry = {
            timestamp: new Date().toISOString(),
            goal: taskResult.goal,
            quality: taskResult.quality,
            steps: taskResult.steps,
            success: taskResult.success,
            tokensUsed: taskResult.tokenUsage?.tokensUsed,
            wallClockMs: taskResult.wallClockMs,
          };
          await appendFile(logFile, JSON.stringify(entry) + '\n');
        } catch (e) {
          console.error(`[task-logger] Failed to write log: ${e.message}`);
        }
      },
      priority: 100,
    },

    extinction: {
      handler: (event) => {
        console.log(`[task-logger] Extinction event on unit ${event.id} (#${event.extinctions})`);
      },
      priority: 50,
    },
  },

  init(manager) {
    console.log('[task-logger] Plugin initialized');
  },

  destroy(manager) {
    console.log('[task-logger] Plugin destroyed');
  },
};


/**
 * Example PREDATOR v2.0 Plugin – Token Budget Alert
 *
 * Monitors token consumption and emits warnings when approaching limits.
 */
export const tokenBudgetAlertPlugin = {
  name: 'token-budget-alert',
  version: '1.0.0',
  description: 'Alerts when token budget consumption exceeds thresholds',

  hooks: {
    afterStep: {
      handler: (context, result) => {
        if (!result) return;
        const tokenFraction = result.tokensUsed / (result.tokenBudget || 1);

        if (tokenFraction > 0.9) {
          console.warn(`[budget-alert] CRITICAL: ${((tokenFraction)*100).toFixed(1)}% of token budget used!`);
        } else if (tokenFraction > 0.75) {
          console.warn(`[budget-alert] Warning: ${((tokenFraction)*100).toFixed(1)}% of token budget used`);
        }
      },
      priority: 5, // High priority - check early
    },
  },

  init() {
    console.log('[budget-alert] Plugin initialized');
  },

  destroy() {
    console.log('[budget-alert] Plugin destroyed');
  },
};
