/**
 * StateSerializer.js — PREDATOR Agent State Checkpoint Manager (v2.0.0)
 * ─────────────────────────────────────────────────────────────────────────────
 * Handles saving and loading the complete agent state to/from disk as JSON
 * files. Enables:
 *   - Checkpointing during training
 *   - Resuming interrupted sessions
 *   - Cloning agent instances
 *
 * Features:
 *   - Versioned timestamped checkpoint files
 *   - Efficient Float64Array compression via base64 encoding
 *   - Metadata tracking (version, timestamp, training phase, step count)
 *   - Automatic pruning of old checkpoints
 *   - Graceful error handling (corrupt files, missing data)
 *   - In-memory export/import for cloning without disk I/O
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { mkdir, readFile, writeFile, readdir, stat, unlink } from 'fs/promises';
import { join } from 'path';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const CHECKPOINT_PREFIX = 'checkpoint_';
const CHECKPOINT_GLOB   = /^checkpoint_(\d{8}_\d{6})(?:_(.+))?\.json$/;
const SERIALIZER_VERSION = '2.0.0';

// Minimum serializer version we can still load (semantic — major must match)
const MIN_COMPAT_VERSION = '2.0.0';

// ─────────────────────────────────────────────────────────────────────────────
// Float64Array ↔ base64 helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Encode a Float64Array as a base64 string for compact JSON storage.
 * Each 64-bit float is represented as 8 bytes in a Uint8Array view,
 * then base64-encoded. This is ~33% larger than raw binary but fully
 * compatible with JSON and avoids the overhead of a plain number array
 * (which would be ~2-3x larger for typical float values).
 *
 * @param {Float64Array} arr
 * @returns {{ _f64b64: string, length: number }} Encoded payload
 */
function encodeFloat64Array(arr) {
  if (!(arr instanceof Float64Array)) {
    // Not a typed array — return as-is (may be a plain array or other type)
    return arr;
  }
  const byteLen   = arr.byteLength;
  const uint8View = new Uint8Array(arr.buffer, arr.byteOffset, byteLen);
  let binary = '';
  for (let i = 0; i < uint8View.length; i++) {
    binary += String.fromCharCode(uint8View[i]);
  }
  return {
    _f64b64: btoa(binary),
    length:  arr.length,
  };
}

/**
 * Decode a previously encoded Float64Array from base64.
 *
 * @param {object} encoded - Object produced by encodeFloat64Array
 * @returns {Float64Array}
 */
function decodeFloat64Array(encoded) {
  if (encoded == null || typeof encoded !== 'object' || !encoded._f64b64) {
    return encoded; // Not our encoded format — return as-is
  }
  const binary   = atob(encoded._f64b64);
  const uint8Arr = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    uint8Arr[i] = binary.charCodeAt(i);
  }
  return new Float64Array(uint8Arr.buffer);
}

/**
 * Recursively walk a state object and encode all Float64Array instances
 * into base64 payloads. Modifies the object in-place and returns it.
 *
 * @param {*} obj - Any serializable value
 * @returns {*} The same value with Float64Arrays replaced
 */
function encodeTypedArrays(obj) {
  if (obj === null || obj === undefined) return obj;

  if (obj instanceof Float64Array) {
    return encodeFloat64Array(obj);
  }

  if (Array.isArray(obj)) {
    for (let i = 0; i < obj.length; i++) {
      obj[i] = encodeTypedArrays(obj[i]);
    }
    return obj;
  }

  if (typeof obj === 'object' && obj.constructor === Object) {
    for (const key of Object.keys(obj)) {
      obj[key] = encodeTypedArrays(obj[key]);
    }
  }

  // Other types (string, number, boolean, custom class instances) are left as-is
  return obj;
}

/**
 * Recursively walk a state object and decode all encoded Float64Array
 * payloads back into real Float64Array instances. Modifies in-place.
 *
 * @param {*} obj - Any deserialized value
 * @returns {*} The same value with encoded arrays restored
 */
function decodeTypedArrays(obj) {
  if (obj === null || obj === undefined) return obj;

  if (typeof obj === 'object' && !Array.isArray(obj) && obj._f64b64 !== undefined) {
    return decodeFloat64Array(obj);
  }

  if (Array.isArray(obj)) {
    for (let i = 0; i < obj.length; i++) {
      obj[i] = decodeTypedArrays(obj[i]);
    }
    return obj;
  }

  if (typeof obj === 'object' && obj.constructor === Object) {
    for (const key of Object.keys(obj)) {
      obj[key] = decodeTypedArrays(obj[key]);
    }
  }

  return obj;
}

// ─────────────────────────────────────────────────────────────────────────────
// Version compatibility
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compare two semver-like version strings. Returns:
 *   -1 if a < b, 0 if a == b, 1 if a > b
 *
 * @param {string} a
 * @param {string} b
 * @returns {number}
 */
function compareVersions(a, b) {
  const pa = String(a).split('.').map(Number);
  const pb = String(b).split('.').map(Number);
  for (let i = 0; i < 3; i++) {
    const na = pa[i] ?? 0;
    const nb = pb[i] ?? 0;
    if (na < nb) return -1;
    if (na > nb) return 1;
  }
  return 0;
}

/**
 * Check if a checkpoint version is compatible with the current serializer.
 * Major version must match; minor/patch can be equal or lower.
 *
 * @param {string} checkpointVersion
 * @returns {{ compatible: boolean, reason?: string }}
 */
function checkVersionCompatibility(checkpointVersion) {
  if (!checkpointVersion || typeof checkpointVersion !== 'string') {
    return { compatible: false, reason: 'Missing or invalid version in checkpoint' };
  }

  const cpMajor   = Number(checkpointVersion.split('.')[0]);
  const selfMajor = Number(SERIALIZER_VERSION.split('.')[0]);

  if (cpMajor !== selfMajor) {
    return {
      compatible: false,
      reason: `Major version mismatch: checkpoint v${checkpointVersion} vs serializer v${SERIALIZER_VERSION}`,
    };
  }

  if (compareVersions(checkpointVersion, MIN_COMPAT_VERSION) < 0) {
    return {
      compatible: false,
      reason: `Checkpoint v${checkpointVersion} is below minimum compatible v${MIN_COMPAT_VERSION}`,
    };
  }

  return { compatible: true };
}

// ─────────────────────────────────────────────────────────────────────────────
// Timestamp helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Format a Date as a compact sortable timestamp: YYYYMMDD_HHMMSS
 * @param {Date} [date]
 * @returns {string}
 */
function formatTimestamp(date = new Date()) {
  const y  = date.getFullYear();
  const mo = String(date.getMonth() + 1).padStart(2, '0');
  const d  = String(date.getDate()).padStart(2, '0');
  const h  = String(date.getHours()).padStart(2, '0');
  const mi = String(date.getMinutes()).padStart(2, '0');
  const s  = String(date.getSeconds()).padStart(2, '0');
  return `${y}${mo}${d}_${h}${mi}${s}`;
}

/**
 * Parse a checkpoint filename to extract the timestamp and label.
 *
 * @param {string} filename
 * @returns {{ timestamp: string, label: string|null } | null}
 */
function parseCheckpointFilename(filename) {
  const match = filename.match(CHECKPOINT_GLOB);
  if (!match) return null;
  return {
    timestamp: match[1],
    label:     match[2] ?? null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// StateSerializer
// ─────────────────────────────────────────────────────────────────────────────

export class StateSerializer {
  /**
   * @param {object}  [opts]
   * @param {string}  [opts.checkpointDir='./checkpoints'] - Directory for checkpoint files
   * @param {number}  [opts.maxCheckpoints=5]              - Max checkpoints to retain after pruning
   */
  constructor(opts = {}) {
    this.checkpointDir   = opts.checkpointDir   ?? './checkpoints';
    this.maxCheckpoints  = opts.maxCheckpoints   ?? 5;
    this.version         = SERIALIZER_VERSION;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API: save
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Save the agent state to a timestamped JSON checkpoint file.
   *
   * Steps:
   *   1. Create the checkpoint directory if it doesn't exist
   *   2. Call agent.serialize() to get the full state
   *   3. Encode Float64Array data as base64 for efficient storage
   *   4. Add metadata (version, timestamp, label, step count, training phase)
   *   5. Write to a JSON file named: checkpoint_{timestamp}_{label}.json
   *   6. Prune old checkpoints beyond maxCheckpoints
   *
   * @param {object} agent   - Agent instance with a serialize() method
   * @param {string} [label] - Optional label (e.g. 'phase1_complete', 'epoch_5')
   * @returns {Promise<string>} Path of the saved checkpoint file
   * @throws {Error} If agent doesn't have a serialize() method
   */
  async save(agent, label) {
    if (!agent || typeof agent.serialize !== 'function') {
      throw new Error('StateSerializer.save: agent must have a serialize() method');
    }

    // 1. Ensure checkpoint directory exists
    await mkdir(this.checkpointDir, { recursive: true });

    // 2. Serialize the agent state
    const agentState = agent.serialize();

    // 3. Deep-encode typed arrays for efficient JSON storage
    const encodedState = encodeTypedArrays(structuredClone(agentState));

    // 4. Build the full checkpoint object with metadata
    const now = new Date();
    const checkpoint = {
      _meta: {
        serializerVersion: this.version,
        agentVersion:      agentState._version ?? agentState.version ?? 'unknown',
        timestamp:         now.toISOString(),
        timestampCompact:  formatTimestamp(now),
        label:             label ?? null,
        stepCount:         this._extractStepCount(agentState),
        trainingPhase:     this._extractTrainingPhase(agentState),
      },
      state: encodedState,
    };

    // 5. Determine filename and write
    const safeLabel   = label ? '_' + label.replace(/[^a-zA-Z0-9_-]/g, '_') : '';
    const filename    = `${CHECKPOINT_PREFIX}${formatTimestamp(now)}${safeLabel}.json`;
    const filePath    = join(this.checkpointDir, filename);
    const jsonString  = JSON.stringify(checkpoint, null, 2);

    await writeFile(filePath, jsonString, 'utf-8');

    // 6. Prune old checkpoints
    try {
      await this.pruneCheckpoints();
    } catch (pruneErr) {
      // Pruning failure should not prevent a successful save
      console.warn(
        `StateSerializer.save: pruning failed after save: ${pruneErr.message}`
      );
    }

    return filePath;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API: load
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Load agent state from a checkpoint file.
   *
   * Steps:
   *   1. Resolve the checkpoint file (latest if no ID specified)
   *   2. Read and parse the JSON file
   *   3. Validate version compatibility
   *   4. Decode typed arrays from base64 back to Float64Array
   *   5. Call agent.deserialize(state) to restore the agent
   *   6. Return true on success
   *
   * @param {object}  agent        - Agent instance with a deserialize() method
   * @param {string|null} [checkpointId=null] - Checkpoint ID (filename or timestamp-based ID),
   *                                            or null to load the latest
   * @returns {Promise<boolean>} true on success
   * @throws {Error} On incompatible version, corrupt file, missing data, or deserialization failure
   */
  async load(agent, checkpointId = null) {
    if (!agent || typeof agent.deserialize !== 'function') {
      throw new Error('StateSerializer.load: agent must have a deserialize() method');
    }

    // 1. Resolve the checkpoint file path
    const filePath = await this._resolveCheckpointPath(checkpointId);

    // 2. Read and parse the JSON file
    let checkpoint;
    try {
      const raw = await readFile(filePath, 'utf-8');
      checkpoint = JSON.parse(raw);
    } catch (err) {
      if (err.code === 'ENOENT') {
        throw new Error(`StateSerializer.load: checkpoint file not found: ${filePath}`);
      }
      throw new Error(
        `StateSerializer.load: failed to read/parse checkpoint: ${err.message}`
      );
    }

    // 3. Validate checkpoint structure
    if (!checkpoint || typeof checkpoint !== 'object') {
      throw new Error('StateSerializer.load: checkpoint is not a valid object');
    }
    if (!checkpoint.state) {
      throw new Error('StateSerializer.load: checkpoint missing "state" field — corrupt or invalid file');
    }
    if (!checkpoint._meta) {
      throw new Error('StateSerializer.load: checkpoint missing "_meta" field — corrupt or invalid file');
    }

    // 4. Validate version compatibility
    const compat = checkVersionCompatibility(checkpoint._meta.serializerVersion);
    if (!compat.compatible) {
      throw new Error(`StateSerializer.load: ${compat.reason}`);
    }

    // 5. Decode typed arrays back to Float64Array instances
    const decodedState = decodeTypedArrays(structuredClone(checkpoint.state));

    // 6. Restore the agent via deserialize
    try {
      agent.deserialize(decodedState);
    } catch (err) {
      throw new Error(
        `StateSerializer.load: agent.deserialize() failed: ${err.message}`
      );
    }

    return true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API: listCheckpoints
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * List all available checkpoints with metadata.
   *
   * Returns an array of objects sorted by timestamp (newest first):
   *   { id, filename, timestamp, label, stepCount, trainingPhase, fileSize }
   *
   * @returns {Promise<Array<object>>}
   */
  async listCheckpoints() {
    // Ensure directory exists (may not yet if no saves have happened)
    let entries;
    try {
      entries = await readdir(this.checkpointDir);
    } catch (err) {
      if (err.code === 'ENOENT') return [];
      throw err;
    }

    const results = [];

    for (const entry of entries) {
      const parsed = parseCheckpointFilename(entry);
      if (!parsed) continue; // Not a valid checkpoint file

      const fullPath = join(this.checkpointDir, entry);

      let fileSize = 0;
      let stepCount = null;
      let trainingPhase = null;
      let label = parsed.label;
      let timestampISO = null;

      // Get file size
      try {
        const fstat = await stat(fullPath);
        fileSize = fstat.size;
      } catch {
        // File may have been deleted between readdir and stat; skip
        continue;
      }

      // Try to read metadata from the file without loading the full state
      try {
        const raw   = await readFile(fullPath, 'utf-8');
        const parsed_json = JSON.parse(raw);
        if (parsed_json._meta) {
          stepCount     = parsed_json._meta.stepCount     ?? null;
          trainingPhase = parsed_json._meta.trainingPhase ?? null;
          label         = parsed_json._meta.label          ?? label;
          timestampISO  = parsed_json._meta.timestamp      ?? null;
        }
      } catch {
        // Corrupt or unreadable — include with partial metadata
      }

      results.push({
        id:            entry.replace(/\.json$/, ''),
        filename:      entry,
        timestamp:     parsed.timestamp,
        timestampISO,
        label,
        stepCount,
        trainingPhase,
        fileSize,
      });
    }

    // Sort newest first (timestamp is lexicographically sortable)
    results.sort((a, b) => b.timestamp.localeCompare(a.timestamp));

    return results;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API: deleteCheckpoint
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Delete a specific checkpoint by ID.
   *
   * @param {string} id - Checkpoint ID (filename without .json, or full filename)
   * @returns {Promise<boolean>} true if deleted, false if not found
   */
  async deleteCheckpoint(id) {
    if (!id || typeof id !== 'string') {
      throw new Error('StateSerializer.deleteCheckpoint: id must be a non-empty string');
    }

    // Normalize: ensure it ends with .json
    let filename = id.endsWith('.json') ? id : `${id}.json`;
    const fullPath = join(this.checkpointDir, filename);

    try {
      await unlink(fullPath);
      return true;
    } catch (err) {
      if (err.code === 'ENOENT') return false;
      throw new Error(`StateSerializer.deleteCheckpoint: failed to delete: ${err.message}`);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API: pruneCheckpoints
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Delete old checkpoints beyond maxCheckpoints, keeping the most recent ones.
   *
   * @returns {Promise<string[]>} Array of deleted checkpoint IDs
   */
  async pruneCheckpoints() {
    const checkpoints = await this.listCheckpoints();

    if (checkpoints.length <= this.maxCheckpoints) {
      return [];
    }

    // checkpoints is sorted newest-first; delete from the end
    const toDelete = checkpoints.slice(this.maxCheckpoints);
    const deleted  = [];

    for (const cp of toDelete) {
      try {
        const fullPath = join(this.checkpointDir, cp.filename);
        await unlink(fullPath);
        deleted.push(cp.id);
      } catch (err) {
        if (err.code !== 'ENOENT') {
          console.warn(
            `StateSerializer.pruneCheckpoints: failed to delete ${cp.filename}: ${err.message}`
          );
        }
      }
    }

    return deleted;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API: exportSnapshot
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Export a serializable state object without saving to disk.
   * Useful for cloning an agent in-memory or transferring state over a network.
   *
   * The returned object is a deep clone with Float64Arrays encoded as base64,
   * safe for JSON.stringify() and transmission.
   *
   * @param {object} agent - Agent instance with a serialize() method
   * @returns {Promise<object>} Serializable state object with metadata
   */
  async exportSnapshot(agent) {
    if (!agent || typeof agent.serialize !== 'function') {
      throw new Error('StateSerializer.exportSnapshot: agent must have a serialize() method');
    }

    const agentState = agent.serialize();
    const encodedState = encodeTypedArrays(structuredClone(agentState));

    const now = new Date();
    return {
      _meta: {
        serializerVersion: this.version,
        agentVersion:      agentState._version ?? agentState.version ?? 'unknown',
        timestamp:         now.toISOString(),
        timestampCompact:  formatTimestamp(now),
        label:             'export_snapshot',
        stepCount:         this._extractStepCount(agentState),
        trainingPhase:     this._extractTrainingPhase(agentState),
      },
      state: encodedState,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API: importSnapshot
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Import state from an object without reading from disk.
   * Useful for restoring from a network transfer or in-memory clone.
   *
   * @param {object} agent  - Agent instance with a deserialize() method
   * @param {object} state  - State object produced by exportSnapshot() or loaded from disk
   * @returns {Promise<boolean>} true on success
   * @throws {Error} On invalid input, incompatible version, or deserialization failure
   */
  async importSnapshot(agent, state) {
    if (!agent || typeof agent.deserialize !== 'function') {
      throw new Error('StateSerializer.importSnapshot: agent must have a deserialize() method');
    }
    if (!state || typeof state !== 'object') {
      throw new Error('StateSerializer.importSnapshot: state must be a non-null object');
    }

    // Support both raw state objects and wrapped _meta+state objects
    let agentState;
    if (state._meta && state.state) {
      // Wrapped format from exportSnapshot() or load()
      const compat = checkVersionCompatibility(state._meta.serializerVersion);
      if (!compat.compatible) {
        throw new Error(`StateSerializer.importSnapshot: ${compat.reason}`);
      }
      agentState = state.state;
    } else {
      // Assume it's a raw agent state object
      agentState = state;
    }

    // Decode typed arrays back to Float64Array instances
    const decodedState = decodeTypedArrays(structuredClone(agentState));

    try {
      agent.deserialize(decodedState);
    } catch (err) {
      throw new Error(
        `StateSerializer.importSnapshot: agent.deserialize() failed: ${err.message}`
      );
    }

    return true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal helpers
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Resolve a checkpoint ID (or null) to a full file path.
   * If checkpointId is null, returns the path of the latest checkpoint.
   *
   * @param {string|null} checkpointId
   * @returns {Promise<string>} Absolute file path
   * @throws {Error} If no checkpoints found or specified ID doesn't exist
   */
  async _resolveCheckpointPath(checkpointId) {
    // If a full path is given, use it directly
    if (checkpointId && checkpointId.includes('/') && checkpointId.includes('.json')) {
      return checkpointId;
    }

    const checkpoints = await this.listCheckpoints();

    if (checkpoints.length === 0) {
      throw new Error('StateSerializer.load: no checkpoints available');
    }

    if (checkpointId == null) {
      // Return the latest (first in sorted list)
      return join(this.checkpointDir, checkpoints[0].filename);
    }

    // Try to match by ID (filename without .json)
    const match = checkpoints.find((cp) => cp.id === checkpointId);
    if (match) {
      return join(this.checkpointDir, match.filename);
    }

    // Try to match by filename directly
    const filenameMatch = checkpoints.find((cp) => cp.filename === checkpointId);
    if (filenameMatch) {
      return join(this.checkpointDir, filenameMatch.filename);
    }

    // Try partial match on label
    const labelMatch = checkpoints.find((cp) => cp.label === checkpointId);
    if (labelMatch) {
      return join(this.checkpointDir, labelMatch.filename);
    }

    throw new Error(
      `StateSerializer.load: checkpoint not found: "${checkpointId}". ` +
      `Available: ${checkpoints.map((cp) => cp.id).join(', ')}`
    );
  }

  /**
   * Extract the step count from a serialized agent state.
   * Tries common field names used across the PREDATOR system.
   *
   * @param {object} agentState
   * @returns {number|null}
   */
  _extractStepCount(agentState) {
    if (!agentState || typeof agentState !== 'object') return null;

    // Predator-level: backbone.step
    if (agentState.backbone && typeof agentState.backbone.step === 'number') {
      return agentState.backbone.step;
    }

    // Direct step field
    if (typeof agentState.step === 'number') {
      return agentState.step;
    }

    // History length as a proxy
    if (Array.isArray(agentState.history)) {
      return agentState.history.length;
    }

    return null;
  }

  /**
   * Attempt to determine the current training phase from agent state.
   *
   * @param {object} agentState
   * @returns {string|null}
   */
  _extractTrainingPhase(agentState) {
    if (!agentState || typeof agentState !== 'object') return null;

    // Explicit trainingPhase field
    if (typeof agentState.trainingPhase === 'string') {
      return agentState.trainingPhase;
    }

    // Infer from backbone step count
    const step = agentState.backbone?.step ?? agentState.step;
    if (typeof step === 'number') {
      if (step === 0)           return 'initialized';
      if (step < 100)           return 'early_training';
      if (step < 1000)          return 'mid_training';
      if (step < 10000)         return 'late_training';
      return 'mature';
    }

    return null;
  }
}

export default StateSerializer;
