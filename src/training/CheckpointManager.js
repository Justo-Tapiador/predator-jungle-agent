/**
 * CheckpointManager.js — Training Checkpoint Management (v2.0)
 * ─────────────────────────────────────────────────────────────────────────────
 * Manages training checkpoints for the PREDATOR training pipeline.
 * Checkpoints are stored as JSON files with full metadata and can be
 * listed, loaded, cleaned, and exported.
 *
 * Features:
 *   - Save training state with phase, epoch, loss, and metrics metadata
 *   - Load checkpoints by ID
 *   - List available checkpoints with filtering
 *   - Get latest checkpoint for a given phase
 *   - Delete individual checkpoints
 *   - Clean old checkpoints based on age
 *   - Export checkpoints to external files
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { readFile, writeFile, mkdir, rm, readdir, stat, copyFile } from 'node:fs/promises';
import { join, basename } from 'node:path';
import { randomUUID } from 'node:crypto';

// ── Checkpoint file extension ────────────────────────────────────────────────
const CKPT_EXT = '.ckpt.json';

// ─────────────────────────────────────────────────────────────────────────────
export class CheckpointManager {
  /**
   * @param {object} opts
   * @param {string} [opts.dir] - Directory for storing checkpoint files
   */
  constructor(opts = {}) {
    this.dir = opts.dir ?? './checkpoints';
    this._cache = new Map(); // In-memory cache for fast lookups
  }

  // ── Core operations ───────────────────────────────────────────────────────

  /**
   * Save a training checkpoint.
   *
   * @param {string} phase  - Training phase identifier (I, II, III, IV)
   * @param {number} epoch  - Epoch number
   * @param {object} state  - Training state to save
   * @returns {Promise<object>} Saved checkpoint metadata
   */
  async save(phase, epoch, state) {
    // Ensure directory exists
    await mkdir(this.dir, { recursive: true });

    const id = `ckpt_${phase}_e${epoch}_${randomUUID().slice(0, 8)}`;
    const timestamp = Date.now();

    const checkpoint = {
      id,
      metadata: {
        phase,
        epoch,
        timestamp,
        createdAt: new Date(timestamp).toISOString(),
        loss: state.loss ?? null,
        metrics: state.metrics ?? {},
      },
      state: this._serializeState(state),
    };

    const filePath = join(this.dir, `${id}${CKPT_EXT}`);
    await writeFile(filePath, JSON.stringify(checkpoint, null, 2), 'utf-8');

    // Update cache
    this._cache.set(id, checkpoint);

    return {
      id,
      phase,
      epoch,
      timestamp,
      filePath,
      size: JSON.stringify(checkpoint).length,
    };
  }

  /**
   * Load a checkpoint by its ID.
   *
   * @param {string} checkpointId - Checkpoint identifier
   * @returns {Promise<object|null>} Checkpoint object or null if not found
   */
  async load(checkpointId) {
    // Check cache first
    if (this._cache.has(checkpointId)) {
      return this._cache.get(checkpointId);
    }

    const filePath = join(this.dir, `${checkpointId}${CKPT_EXT}`);

    try {
      const raw = await readFile(filePath, 'utf-8');
      const checkpoint = JSON.parse(raw);

      // Update cache
      this._cache.set(checkpointId, checkpoint);

      return checkpoint;
    } catch (err) {
      if (err.code === 'ENOENT') {
        return null;
      }
      throw new Error(`Failed to load checkpoint "${checkpointId}": ${err.message}`);
    }
  }

  /**
   * List available checkpoints.
   *
   * @param {object} [filter] - Optional filter
   * @param {string} [filter.phase]       - Filter by phase
   * @param {number} [filter.minEpoch]    - Minimum epoch
   * @param {number} [filter.maxEpoch]    - Maximum epoch
   * @param {number} [filter.since]       - Only checkpoints after this timestamp
   * @param {number} [filter.limit]       - Maximum number of results
   * @returns {Promise<object[]>} Array of checkpoint metadata objects
   */
  async list(filter = {}) {
    await this._ensureDir();

    const files = await this._readCheckpointFiles();
    const results = [];

    for (const file of files) {
      try {
        const checkpoint = file.checkpoint;

        // Apply filters
        if (filter.phase && checkpoint.metadata.phase !== filter.phase) continue;
        if (filter.minEpoch != null && checkpoint.metadata.epoch < filter.minEpoch) continue;
        if (filter.maxEpoch != null && checkpoint.metadata.epoch > filter.maxEpoch) continue;
        if (filter.since != null && checkpoint.metadata.timestamp < filter.since) continue;

        results.push({
          id: checkpoint.id,
          phase: checkpoint.metadata.phase,
          epoch: checkpoint.metadata.epoch,
          timestamp: checkpoint.metadata.timestamp,
          createdAt: checkpoint.metadata.createdAt,
          loss: checkpoint.metadata.loss,
          metrics: checkpoint.metadata.metrics,
          filePath: file.filePath,
          size: file.size,
        });
      } catch {
        // Skip malformed checkpoint files
        continue;
      }
    }

    // Sort by timestamp descending (newest first)
    results.sort((a, b) => b.timestamp - a.timestamp);

    // Apply limit
    if (filter.limit != null && filter.limit > 0) {
      return results.slice(0, filter.limit);
    }

    return results;
  }

  /**
   * Get the latest checkpoint for a given phase.
   *
   * @param {string} phase - Phase identifier (I, II, III, IV)
   * @returns {Promise<object|null>} Latest checkpoint for the phase or null
   */
  async getLatest(phase) {
    const checkpoints = await this.list({ phase });

    if (checkpoints.length === 0) {
      return null;
    }

    // List is already sorted newest-first
    const latest = checkpoints[0];
    return this.load(latest.id);
  }

  /**
   * Delete a checkpoint by its ID.
   *
   * @param {string} checkpointId - Checkpoint identifier
   * @returns {Promise<boolean>} True if deleted, false if not found
   */
  async delete(checkpointId) {
    const filePath = join(this.dir, `${checkpointId}${CKPT_EXT}`);

    try {
      await rm(filePath);
      this._cache.delete(checkpointId);
      return true;
    } catch (err) {
      if (err.code === 'ENOENT') {
        return false;
      }
      throw new Error(`Failed to delete checkpoint "${checkpointId}": ${err.message}`);
    }
  }

  /**
   * Clean checkpoints older than a specified age.
   *
   * @param {number} maxAge - Maximum age in milliseconds
   * @returns {Promise<number>} Number of checkpoints removed
   */
  async cleanOlderThan(maxAge) {
    const cutoff = Date.now() - maxAge;
    const checkpoints = await this.list();

    let removed = 0;
    for (const ckpt of checkpoints) {
      if (ckpt.timestamp < cutoff) {
        const deleted = await this.delete(ckpt.id);
        if (deleted) removed++;
      }
    }

    return removed;
  }

  /**
   * Export a checkpoint to an external file path.
   *
   * @param {string} checkpointId - Checkpoint identifier
   * @param {string} outputPath   - Destination file path
   * @returns {Promise<object>} Export result with file path and size
   */
  async exportCheckpoint(checkpointId, outputPath) {
    const checkpoint = await this.load(checkpointId);
    if (!checkpoint) {
      throw new Error(`Checkpoint "${checkpointId}" not found.`);
    }

    // Ensure output directory exists
    const outputDir = join(outputPath, '..');
    await mkdir(outputDir, { recursive: true });

    // Write checkpoint to output path
    const content = JSON.stringify(checkpoint, null, 2);
    await writeFile(outputPath, content, 'utf-8');

    return {
      id: checkpointId,
      outputPath,
      size: content.length,
      exportedAt: Date.now(),
    };
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  /**
   * Ensure the checkpoint directory exists.
   */
  async _ensureDir() {
    await mkdir(this.dir, { recursive: true });
  }

  /**
   * Read all checkpoint files from the directory.
   * @returns {Promise<Array<{ filePath: string, size: number, checkpoint: object }>>}
   */
  async _readCheckpointFiles() {
    try {
      const entries = await readdir(this.dir);
      const results = [];

      for (const entry of entries) {
        if (!entry.endsWith(CKPT_EXT)) continue;

        const filePath = join(this.dir, entry);

        try {
          const fileStat = await stat(filePath);
          const raw = await readFile(filePath, 'utf-8');
          const checkpoint = JSON.parse(raw);

          results.push({
            filePath,
            size: fileStat.size,
            checkpoint,
          });
        } catch {
          // Skip unreadable or malformed files
          continue;
        }
      }

      return results;
    } catch (err) {
      if (err.code === 'ENOENT') {
        return [];
      }
      throw err;
    }
  }

  /**
   * Serialize state for storage, handling non-serializable values.
   */
  _serializeState(state) {
    if (!state) return state;

    const serialized = {};

    for (const [key, value] of Object.entries(state)) {
      if (value === undefined) {
        continue; // Skip undefined values
      }

      if (value instanceof Float64Array || value instanceof Float32Array) {
        serialized[key] = {
          __type: value.constructor.name,
          data: Array.from(value),
        };
      } else if (value instanceof Map) {
        serialized[key] = {
          __type: 'Map',
          data: Object.fromEntries(value),
        };
      } else if (value instanceof Set) {
        serialized[key] = {
          __type: 'Set',
          data: Array.from(value),
        };
      } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        // Recursively serialize nested objects (but not arrays)
        serialized[key] = this._serializeState(value);
      } else {
        serialized[key] = value;
      }
    }

    return serialized;
  }

  /**
   * Deserialize a previously serialized state.
   */
  _deserializeState(state) {
    if (!state) return state;

    const result = {};

    for (const [key, value] of Object.entries(state)) {
      if (value && typeof value === 'object' && value.__type) {
        switch (value.__type) {
          case 'Float64Array':
            result[key] = Float64Array.from(value.data);
            break;
          case 'Float32Array':
            result[key] = Float32Array.from(value.data);
            break;
          case 'Map':
            result[key] = new Map(Object.entries(value.data));
            break;
          case 'Set':
            result[key] = new Set(value.data);
            break;
          default:
            result[key] = value.data;
        }
      } else if (value && typeof value === 'object' && !Array.isArray(value) && value.__type === undefined) {
        result[key] = this._deserializeState(value);
      } else {
        result[key] = value;
      }
    }

    return result;
  }
}

export default CheckpointManager;
