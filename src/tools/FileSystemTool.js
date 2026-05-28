/**
 * FileSystemTool.js
 * Real file system operations with safety checks for the PREDATOR agent system.
 * All methods are async and return structured results.
 */

import fs from 'fs/promises';
import path from 'path';

const MAX_READ_SIZE = 10 * 1024 * 1024; // 10 MB read limit
const MAX_WRITE_SIZE = 50 * 1024 * 1024; // 50 MB write limit

// Paths the agent must never delete or overwrite
const PROTECTED_PATHS = new Set([
  '/',
  '/etc',
  '/bin',
  '/usr',
  '/var',
  '/sys',
  '/proc',
  '/dev',
  '/boot',
  '/root',
  '/home',
  '/lib',
  '/lib64',
  '/sbin',
  '/opt',
]);

class FileSystemTool {
  constructor(options = {}) {
    this.id = 'filesystem';
    this.name = 'FileSystemTool';
    this.description = 'Perform file system operations with safety checks';
    this.maxReadSize = options.maxReadSize || MAX_READ_SIZE;
    this.maxWriteSize = options.maxWriteSize || MAX_WRITE_SIZE;
    this.sandboxDir = options.sandboxDir || null; // If set, restrict all operations under this dir
    this.protectedPaths = new Set([...PROTECTED_PATHS, ...(options.extraProtectedPaths || [])]);
  }

  // ── Helpers ──────────────────────────────────────────────────────────

  _resolve(p) {
    const resolved = path.resolve(p);
    if (this.sandboxDir && !resolved.startsWith(path.resolve(this.sandboxDir))) {
      throw new Error(`Path "${p}" is outside the sandbox directory "${this.sandboxDir}"`);
    }
    return resolved;
  }

  _isProtected(p) {
    const resolved = path.resolve(p);
    for (const prot of this.protectedPaths) {
      if (resolved === prot || resolved.startsWith(prot + path.sep)) {
        // Allow if the resolved path is deeper than a top-level protected dir
        // but block exact matches of protected roots
        if (resolved === prot) return true;
      }
    }
    return false;
  }

  _result(success, result, error, startTime) {
    return {
      success,
      ...(result !== undefined && { result }),
      ...(error && { error }),
      duration: Date.now() - startTime,
    };
  }

  // ── Read ─────────────────────────────────────────────────────────────

  async readFile(filePath, encoding = 'utf-8') {
    const start = Date.now();
    try {
      const resolved = this._resolve(filePath);

      // Size check via stat first
      const stat = await fs.stat(resolved);
      if (stat.size > this.maxReadSize) {
        return this._result(false, undefined, `File too large: ${stat.size} bytes (max ${this.maxReadSize})`, start);
      }

      const content = await fs.readFile(resolved, { encoding });
      return this._result(true, { path: resolved, content, size: stat.size, encoding }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `readFile failed: ${err.message}`, start);
    }
  }

  // ── Write ────────────────────────────────────────────────────────────

  async writeFile(filePath, content, encoding = 'utf-8') {
    const start = Date.now();
    try {
      const resolved = this._resolve(filePath);

      if (Buffer.byteLength(content, encoding) > this.maxWriteSize) {
        return this._result(false, undefined, `Content too large (max ${this.maxWriteSize} bytes)`, start);
      }

      // Ensure parent directory exists
      const dir = path.dirname(resolved);
      await fs.mkdir(dir, { recursive: true });

      await fs.writeFile(resolved, content, { encoding });
      const stat = await fs.stat(resolved);
      return this._result(true, { path: resolved, size: stat.size }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `writeFile failed: ${err.message}`, start);
    }
  }

  // ── Append ───────────────────────────────────────────────────────────

  async appendFile(filePath, content) {
    const start = Date.now();
    try {
      const resolved = this._resolve(filePath);

      // Ensure parent directory exists
      const dir = path.dirname(resolved);
      await fs.mkdir(dir, { recursive: true });

      await fs.appendFile(resolved, content, 'utf-8');
      const stat = await fs.stat(resolved);
      return this._result(true, { path: resolved, size: stat.size }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `appendFile failed: ${err.message}`, start);
    }
  }

  // ── Delete ───────────────────────────────────────────────────────────

  async deleteFile(filePath) {
    const start = Date.now();
    try {
      const resolved = this._resolve(filePath);

      if (this._isProtected(resolved)) {
        return this._result(false, undefined, `Cannot delete protected path: "${resolved}"`, start);
      }

      const stat = await fs.stat(resolved);
      if (stat.isDirectory()) {
        await fs.rm(resolved, { recursive: true, force: false });
      } else {
        await fs.unlink(resolved);
      }

      return this._result(true, { path: resolved, deleted: true, wasDirectory: stat.isDirectory() }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `deleteFile failed: ${err.message}`, start);
    }
  }

  // ── List Directory ───────────────────────────────────────────────────

  async listDir(dirPath, recursive = false) {
    const start = Date.now();
    try {
      const resolved = this._resolve(dirPath);

      const entries = await fs.readdir(resolved, { withFileTypes: true, recursive });
      const items = entries.map((entry) => ({
        name: entry.name,
        path: path.join(entry.parentPath ?? resolved, entry.name),
        isDirectory: entry.isDirectory(),
        isFile: entry.isFile(),
        isSymbolicLink: entry.isSymbolicLink(),
      }));

      return this._result(true, { path: resolved, items, count: items.length }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `listDir failed: ${err.message}`, start);
    }
  }

  // ── Stat ─────────────────────────────────────────────────────────────

  async stat(targetPath) {
    const start = Date.now();
    try {
      const resolved = this._resolve(targetPath);
      const stats = await fs.stat(resolved);

      return this._result(
        true,
        {
          path: resolved,
          size: stats.size,
          isFile: stats.isFile(),
          isDirectory: stats.isDirectory(),
          isSymbolicLink: stats.isSymbolicLink(),
          mode: stats.mode,
          uid: stats.uid,
          gid: stats.gid,
          atime: stats.atime.toISOString(),
          mtime: stats.mtime.toISOString(),
          ctime: stats.ctime.toISOString(),
          birthtime: stats.birthtime.toISOString(),
        },
        undefined,
        start,
      );
    } catch (err) {
      return this._result(false, undefined, `stat failed: ${err.message}`, start);
    }
  }

  // ── Mkdir ────────────────────────────────────────────────────────────

  async mkdir(dirPath, recursive = true) {
    const start = Date.now();
    try {
      const resolved = this._resolve(dirPath);
      await fs.mkdir(resolved, { recursive });
      return this._result(true, { path: resolved, created: true }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `mkdir failed: ${err.message}`, start);
    }
  }

  // ── Copy ─────────────────────────────────────────────────────────────

  async copyFile(src, dest) {
    const start = Date.now();
    try {
      const srcResolved = this._resolve(src);
      const destResolved = this._resolve(dest);

      // Ensure dest parent dir exists
      await fs.mkdir(path.dirname(destResolved), { recursive: true });

      await fs.copyFile(srcResolved, destResolved);
      const stat = await fs.stat(destResolved);

      return this._result(true, { src: srcResolved, dest: destResolved, size: stat.size }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `copyFile failed: ${err.message}`, start);
    }
  }

  // ── Move ─────────────────────────────────────────────────────────────

  async moveFile(src, dest) {
    const start = Date.now();
    try {
      const srcResolved = this._resolve(src);
      const destResolved = this._resolve(dest);

      if (this._isProtected(srcResolved)) {
        return this._result(false, undefined, `Cannot move protected path: "${srcResolved}"`, start);
      }

      // Ensure dest parent dir exists
      await fs.mkdir(path.dirname(destResolved), { recursive: true });

      await fs.rename(srcResolved, destResolved);

      return this._result(true, { src: srcResolved, dest: destResolved, moved: true }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `moveFile failed: ${err.message}`, start);
    }
  }

  // ── Search / Glob ────────────────────────────────────────────────────

  async searchFiles(dir, pattern) {
    const start = Date.now();
    try {
      const resolved = this._resolve(dir);

      // Use Node.js built-in fs.glob (available since Node 22+)
      // fs.glob accepts a single glob pattern string; combine dir + pattern
      const fullPattern = path.posix.join(resolved, pattern);
      const matches = [];
      for await (const entry of fs.glob(fullPattern)) {
        // Skip node_modules and .git directories
        if (entry.includes('node_modules') || entry.includes('.git')) continue;
        matches.push(entry);
      }

      const results = matches.map((m) => ({
        path: m,
        name: path.basename(m),
        ext: path.extname(m),
      }));

      return this._result(true, { dir: resolved, pattern, matches: results, count: results.length }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `searchFiles failed: ${err.message}`, start);
    }
  }
}

export default FileSystemTool;
