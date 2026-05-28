/**
 * CodeExecutionTool.js
 * Sandboxed code execution for the PREDATOR agent system.
 * Uses Node.js 'vm' module for JavaScript sandboxing.
 * Provides controlled shell command execution with safety restrictions.
 * All methods are async and return structured results.
 */

import vm from 'vm';
import { exec } from 'child_process';
import path from 'path';

const DEFAULT_JS_TIMEOUT = 5000; // 5 seconds
const DEFAULT_SHELL_TIMEOUT = 30000; // 30 seconds
const MAX_OUTPUT_LENGTH = 100_000; // 100 KB output cap

// Shell commands the agent is never allowed to run
const BLOCKED_SHELL_PATTERNS = [
  /\brm\s+-rf\s+\//, // rm -rf /
  /\bmkfs\b/, // format filesystem
  /\bdd\s+if=/, // dd raw disk
  /\bshutdown\b/, // shutdown system
  /\breboot\b/, // reboot system
  /\binit\s+[06]/, // init to runlevel 0/6
  />\s*\/dev\/sd/, // write directly to disk
  /\biptables\b/, // firewall manipulation
  /\bchmod\s+777\s+\//, // dangerous chmod
  /\bchown\b.*\broot\b/, // chown to root
  /\bsudo\s+rm\b/, // sudo rm
  /\bcurl\b.*\|\s*sh/, // pipe curl to shell
  /\bwget\b.*\|\s*sh/, // pipe wget to shell
];

class CodeExecutionTool {
  constructor(options = {}) {
    this.id = 'code_execution';
    this.name = 'CodeExecutionTool';
    this.description = 'Execute JavaScript in a sandbox or run shell commands with safety restrictions';
    this.defaultJSTimeout = options.defaultJSTimeout || DEFAULT_JS_TIMEOUT;
    this.defaultShellTimeout = options.defaultShellTimeout || DEFAULT_SHELL_TIMEOUT;
    this.maxOutputLength = options.maxOutputLength || MAX_OUTPUT_LENGTH;
    this.allowedCommands = options.allowedCommands || null; // null = allow all non-blocked
    this.workingDirectory = options.workingDirectory || process.cwd();

    // Build the sandboxed context template
    this._sandboxTemplate = {
      console: {
        log: (...args) => args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' '),
        error: (...args) => args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' '),
        warn: (...args) => args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' '),
        info: (...args) => args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' '),
      },
      JSON,
      Math,
      Date,
      parseInt,
      parseFloat,
      isNaN,
      isFinite,
      encodeURIComponent,
      decodeURIComponent,
      encodeURI,
      decodeURI,
      Array,
      Object,
      String,
      Number,
      Boolean,
      Map,
      Set,
      RegExp,
      Error,
      TypeError,
      RangeError,
      Symbol,
      Promise,
      Intl,
      Buffer: {
        from: Buffer.from,
        alloc: Buffer.alloc,
        isBuffer: Buffer.isBuffer,
      },
    };
  }

  // ── Helpers ──────────────────────────────────────────────────────────

  _result(success, result, error, startTime) {
    return {
      success,
      ...(result !== undefined && { result }),
      ...(error && { error }),
      duration: Date.now() - startTime,
    };
  }

  _truncateOutput(output) {
    if (output && output.length > this.maxOutputLength) {
      return output.substring(0, this.maxOutputLength) + '\n... [output truncated]';
    }
    return output;
  }

  // ── Execute JavaScript ───────────────────────────────────────────────

  async executeJavaScript(code, timeout) {
    const start = Date.now();
    const effectiveTimeout = timeout || this.defaultJSTimeout;

    try {
      if (typeof code !== 'string' || code.trim().length === 0) {
        return this._result(false, undefined, 'No code provided', start);
      }

      // Create a fresh sandbox context for each execution
      const output = [];
      const sandbox = {
        ...this._sandboxTemplate,
        console: {
          log: (...args) => output.push(args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ')),
          error: (...args) => output.push('[ERROR] ' + args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ')),
          warn: (...args) => output.push('[WARN] ' + args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ')),
          info: (...args) => output.push('[INFO] ' + args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ')),
        },
        __result: undefined,
      };

      // Wrap the code so we capture the last expression's value
      const wrappedCode = `
        "use strict";
        ${code}
      `;

      const context = vm.createContext(sandbox);
      const script = new vm.Script(wrappedCode, {
        filename: 'sandbox.js',
        timeout: effectiveTimeout,
      });

      const scriptResult = script.runInContext(context, {
        timeout: effectiveTimeout,
      });

      const stdout = this._truncateOutput(output.join('\n'));
      const returnValue = scriptResult !== undefined ? scriptResult : sandbox.__result;

      return this._result(
        true,
        {
          stdout,
          returnValue: returnValue !== undefined ? this._serializeReturn(returnValue) : undefined,
          logs: output,
          timeout: effectiveTimeout,
        },
        undefined,
        start,
      );
    } catch (err) {
      if (err.code === 'ERR_SCRIPT_EXECUTION_TIMEOUT') {
        return this._result(false, undefined, `JavaScript execution timed out after ${effectiveTimeout}ms`, start);
      }
      return this._result(false, undefined, `JavaScript execution error: ${err.message}`, start);
    }
  }

  _serializeReturn(val) {
    try {
      if (typeof val === 'object' && val !== null) {
        return JSON.parse(JSON.stringify(val));
      }
      return val;
    } catch {
      return String(val);
    }
  }

  // ── Execute Shell ────────────────────────────────────────────────────

  async executeShell(command, timeout) {
    const start = Date.now();
    const effectiveTimeout = timeout || this.defaultShellTimeout;

    try {
      if (typeof command !== 'string' || command.trim().length === 0) {
        return this._result(false, undefined, 'No command provided', start);
      }

      // Safety: check against blocked patterns
      for (const pattern of BLOCKED_SHELL_PATTERNS) {
        if (pattern.test(command)) {
          return this._result(false, undefined, `Command blocked by safety rule: ${pattern.source}`, start);
        }
      }

      // Safety: if allowedCommands is set, enforce whitelist
      if (this.allowedCommands) {
        const baseCmd = command.trim().split(/\s+/)[0];
        if (!this.allowedCommands.includes(baseCmd)) {
          return this._result(false, undefined, `Command "${baseCmd}" is not in the allowed list`, start);
        }
      }

      const result = await new Promise((resolve, reject) => {
        const child = exec(
          command,
          {
            timeout: effectiveTimeout,
            maxBuffer: 1024 * 1024, // 1 MB
            cwd: this.workingDirectory,
            env: { ...process.env, FORCE_COLOR: '0' },
          },
          (error, stdout, stderr) => {
            resolve({ error, stdout, stderr });
          },
        );

        // Kill on timeout as a safety net
        setTimeout(() => {
          child.kill('SIGKILL');
        }, effectiveTimeout + 1000);
      });

      const stdout = this._truncateOutput(result.stdout || '');
      const stderr = this._truncateOutput(result.stderr || '');

      if (result.error) {
        // Non-zero exit code is still a "success" in terms of execution,
        // but we report it as an error result
        if (result.error.killed) {
          return this._result(false, undefined, `Shell command timed out after ${effectiveTimeout}ms`, start);
        }

        return this._result(
          true,
          {
            stdout,
            stderr,
            exitCode: result.error.code || 1,
            timedOut: false,
          },
          undefined,
          start,
        );
      }

      return this._result(
        true,
        {
          stdout,
          stderr,
          exitCode: 0,
          timedOut: false,
        },
        undefined,
        start,
      );
    } catch (err) {
      return this._result(false, undefined, `Shell execution error: ${err.message}`, start);
    }
  }

  // ── Evaluate Expression ──────────────────────────────────────────────

  async evaluateExpression(expr) {
    const start = Date.now();

    try {
      if (typeof expr !== 'string' || expr.trim().length === 0) {
        return this._result(false, undefined, 'No expression provided', start);
      }

      // Only allow safe expression characters (no assignment, no function defs, etc.)
      const safePattern = /^[\d\s+\-*/%.()><=!&|^~?:,eEENNbb\s]*$/;
      // More permissive: allow string literals and property access, but block dangerous patterns
      const dangerousPatterns = [
        /\b(require|import|process|eval|Function|global|globalThis|this)\b/,
        /\b(__proto__|constructor|prototype)\b/,
        /\b(while|for|do|class|function|=>|async|await|yield)\b/,
      ];

      for (const pattern of dangerousPatterns) {
        if (pattern.test(expr)) {
          return this._result(false, undefined, `Expression contains disallowed pattern: ${pattern.source}`, start);
        }
      }

      // Run in a strict VM context with minimal access
      const sandbox = {
        Math,
        parseInt,
        parseFloat,
        isNaN,
        isFinite,
        Infinity,
        NaN,
        undefined,
        true: true,
        false: false,
        null: null,
      };

      const context = vm.createContext(sandbox);
      const script = new vm.Script(`"use strict"; (${expr})`, {
        filename: 'eval.js',
        timeout: 2000,
      });

      const result = script.runInContext(context, { timeout: 2000 });

      return this._result(
        true,
        {
          expression: expr,
          value: result,
          type: typeof result,
        },
        undefined,
        start,
      );
    } catch (err) {
      return this._result(false, undefined, `Expression evaluation error: ${err.message}`, start);
    }
  }
}

export default CodeExecutionTool;
