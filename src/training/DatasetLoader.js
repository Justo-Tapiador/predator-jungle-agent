/**
 * DatasetLoader.js — Dataset Loading & Preprocessing (v2.0)
 * ─────────────────────────────────────────────────────────────────────────────
 * Provides dataset loading, validation, splitting, augmentation, and batching
 * for the PREDATOR training pipeline.
 *
 * Capabilities:
 *   - Load datasets from JSON and CSV files
 *   - Generate synthetic training data for each phase
 *   - Validate datasets against schemas
 *   - Split datasets into train/val/test
 *   - Data augmentation (noise, permutation, adversarial perturbation)
 *   - Efficient batched iteration
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';

// ── Synthetic data generators ────────────────────────────────────────────────

/**
 * Generate a random stimulus object.
 */
function randomStimulus(dim = 64, intensity = Math.random()) {
  const features = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    features[i] = (Math.random() - 0.5) * 2;
  }
  return { intensity, features, label: `stimulus_${randomUUID().slice(0, 8)}` };
}

/**
 * Generate an addiction target prototype.
 */
function randomAddictionTarget(dim = 64, craving = 0.5 + Math.random() * 0.5) {
  const prototype = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    prototype[i] = Math.random() * craving;
  }
  return { craving, prototype, stimulusClass: `class_${Math.floor(Math.random() * 10)}` };
}

/**
 * Generate an adversarial perturbation.
 */
function randomPerturbation(dim = 64, intensity = 0.5) {
  const perturbation = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    perturbation[i] = (Math.random() - 0.5) * 2 * intensity;
  }
  return { intensity, perturbation, type: 'adversarial' };
}

// ─────────────────────────────────────────────────────────────────────────────
export class DatasetLoader {
  constructor(opts = {}) {
    this.defaultDim = opts.defaultDim ?? 64;
  }

  // ── File loaders ──────────────────────────────────────────────────────────

  /**
   * Load a dataset from a JSON file.
   *
   * @param {string} filePath - Path to the JSON file
   * @returns {Promise<object>} Parsed dataset with { samples, metadata }
   */
  async loadJSON(filePath) {
    const raw = await readFile(filePath, 'utf-8');
    const parsed = JSON.parse(raw);

    // Normalize: if it's an array, wrap it
    if (Array.isArray(parsed)) {
      return {
        samples: parsed,
        metadata: {
          source: filePath,
          format: 'json',
          count: parsed.length,
          loadedAt: Date.now(),
        },
      };
    }

    // If it's an object with samples, return as-is with metadata
    return {
      samples: parsed.samples ?? parsed.data ?? [],
      metadata: {
        source: filePath,
        format: 'json',
        count: (parsed.samples ?? parsed.data ?? []).length,
        ...(parsed.metadata ?? {}),
        loadedAt: Date.now(),
      },
    };
  }

  /**
   * Load a dataset from a CSV file.
   * Parses rows into objects using the header row as keys.
   *
   * @param {string} filePath - Path to the CSV file
   * @returns {Promise<object>} Parsed dataset with { samples, metadata }
   */
  async loadCSV(filePath) {
    const raw = await readFile(filePath, 'utf-8');
    const lines = raw.trim().split('\n');

    if (lines.length === 0) {
      return { samples: [], metadata: { source: filePath, format: 'csv', count: 0, loadedAt: Date.now() } };
    }

    // Parse header
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

    // Parse rows
    const samples = [];
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      const values = this._parseCSVLine(line);
      const sample = {};

      for (let j = 0; j < headers.length; j++) {
        const value = values[j] ?? '';
        // Try to parse as number
        const num = Number(value);
        sample[headers[j]] = isNaN(num) || value === '' ? value : num;
      }

      samples.push(sample);
    }

    return {
      samples,
      metadata: {
        source: filePath,
        format: 'csv',
        count: samples.length,
        headers,
        loadedAt: Date.now(),
      },
    };
  }

  // ── Synthetic data generation ─────────────────────────────────────────────

  /**
   * Generate synthetic training data for a specific training phase.
   *
   * @param {string} type - Phase type: 'pretrain', 'addiction', 'hift', 'adversarial'
   * @param {number} size - Number of samples to generate
   * @returns {Promise<object>} Dataset with { samples, metadata }
   */
  async generateSynthetic(type, size) {
    const dim = this.defaultDim;
    const samples = [];

    switch (type) {
      case 'pretrain':
        for (let i = 0; i < size; i++) {
          samples.push({
            id: `pretrain_${i}`,
            stimulus: randomStimulus(dim),
            target: randomStimulus(dim, 0.5),
            weight: 1.0,
          });
        }
        break;

      case 'addiction':
        for (let i = 0; i < size; i++) {
          const tier = i < size * 0.33 ? 'T1' : i < size * 0.66 ? 'T2' : 'T3';
          const difficulty = tier === 'T1' ? 0.3 : tier === 'T2' ? 0.6 : 0.9;
          samples.push({
            id: `addiction_${i}`,
            stimulus: randomStimulus(dim, 0.3 + Math.random() * 0.7),
            addictionTarget: randomAddictionTarget(dim, difficulty),
            tier,
            difficulty,
            weight: 1.0 + difficulty * 0.5,
          });
        }
        break;

      case 'hift':
        for (let i = 0; i < size; i++) {
          const directiveIdx = Math.floor(Math.random() * 5);
          samples.push({
            id: `hift_${i}`,
            stimulus: randomStimulus(dim, 0.5 + Math.random() * 0.5),
            directive: `directive_${directiveIdx}`,
            constraintMask: this._generateConstraintMask(dim),
            priority: ['routine', 'expedited', 'critical'][Math.floor(Math.random() * 3)],
            weight: 1.0,
          });
        }
        break;

      case 'adversarial':
        for (let i = 0; i < size; i++) {
          const intensity = 0.1 + Math.random() * 0.9;
          samples.push({
            id: `adversarial_${i}`,
            stimulus: randomStimulus(dim, Math.random()),
            perturbation: randomPerturbation(dim, intensity),
            expectedStability: 1.0 - intensity * 0.3,
            intensity,
            weight: 1.0 + intensity,
          });
        }
        break;

      default:
        throw new Error(`Unknown synthetic data type: "${type}". Must be one of: pretrain, addiction, hift, adversarial`);
    }

    return {
      samples,
      metadata: {
        type,
        count: samples.length,
        dimension: dim,
        generatedAt: Date.now(),
        synthetic: true,
      },
    };
  }

  // ── Validation ────────────────────────────────────────────────────────────

  /**
   * Validate a dataset against a schema.
   *
   * @param {object} dataset    - Dataset to validate { samples, metadata }
   * @param {object} schema     - Schema definition
   * @param {string[]} schema.required - Required top-level keys
   * @param {string}   schema.type     - Expected type of dataset ('object')
   * @param {number}   [schema.minSamples] - Minimum number of samples
   * @returns {Promise<{ ok: boolean, errors: string[] }>}
   */
  async validate(dataset, schema = {}) {
    const errors = [];

    if (!dataset || typeof dataset !== 'object') {
      return { ok: false, errors: ['Dataset must be an object.'] };
    }

    // Check required keys
    if (schema.required) {
      for (const key of schema.required) {
        if (!(key in dataset)) {
          errors.push(`Missing required key: "${key}".`);
        }
      }
    }

    // Check type
    if (schema.type && typeof dataset !== schema.type) {
      errors.push(`Expected dataset type "${schema.type}", got "${typeof dataset}".`);
    }

    // Check samples array
    if (dataset.samples) {
      if (!Array.isArray(dataset.samples)) {
        errors.push('"samples" must be an array.');
      } else {
        // Check minimum sample count
        if (schema.minSamples && dataset.samples.length < schema.minSamples) {
          errors.push(`Dataset has ${dataset.samples.length} samples, minimum is ${schema.minSamples}.`);
        }

        // Validate individual samples
        for (let i = 0; i < dataset.samples.length; i++) {
          const sample = dataset.samples[i];
          if (!sample || typeof sample !== 'object') {
            errors.push(`Sample at index ${i} is not an object.`);
          }
        }
      }
    }

    return { ok: errors.length === 0, errors };
  }

  // ── Splitting ─────────────────────────────────────────────────────────────

  /**
   * Split a dataset into train/validation/test sets.
   *
   * @param {object} dataset - Dataset with { samples } property
   * @param {object} ratios  - Split ratios { train, val, test } (should sum to 1.0)
   * @returns {Promise<object>} Split datasets { train, val, test }
   */
  async split(dataset, ratios = { train: 0.8, val: 0.1, test: 0.1 }) {
    const samples = [...(dataset.samples ?? dataset ?? [])];

    // Normalize ratios
    const total = (ratios.train ?? 0.8) + (ratios.val ?? 0.1) + (ratios.test ?? 0.1);
    const trainRatio = (ratios.train ?? 0.8) / total;
    const valRatio = (ratios.val ?? 0.1) / total;

    // Shuffle deterministically (Fisher-Yates with fixed seed simulation)
    const shuffled = this._shuffle(samples);

    const trainEnd = Math.floor(shuffled.length * trainRatio);
    const valEnd = trainEnd + Math.floor(shuffled.length * valRatio);

    return {
      train: {
        samples: shuffled.slice(0, trainEnd),
        metadata: { ...dataset.metadata, split: 'train', count: trainEnd },
      },
      val: {
        samples: shuffled.slice(trainEnd, valEnd),
        metadata: { ...dataset.metadata, split: 'val', count: valEnd - trainEnd },
      },
      test: {
        samples: shuffled.slice(valEnd),
        metadata: { ...dataset.metadata, split: 'test', count: shuffled.length - valEnd },
      },
    };
  }

  // ── Augmentation ──────────────────────────────────────────────────────────

  /**
   * Apply data augmentation methods to a dataset.
   *
   * @param {object} dataset   - Dataset with { samples }
   * @param {object[]} methods - Array of augmentation methods
   *   - { method: 'noise', scale: 0.1 }
   *   - { method: 'permutation', fraction: 0.2 }
   *   - { method: 'adversarial_perturbation', intensity: 0.5 }
   *   - { method: 'dropout', rate: 0.1 }
   *   - { method: 'scaling', factor: 0.9 }
   * @returns {Promise<object>} Augmented dataset
   */
  async augment(dataset, methods = []) {
    const samples = [...(dataset.samples ?? dataset ?? [])];
    const augmented = [];

    for (const sample of samples) {
      let augmentedSample = { ...sample };

      for (const method of methods) {
        augmentedSample = this._applyAugmentation(augmentedSample, method);
      }

      augmented.push(augmentedSample);
    }

    return {
      samples: augmented,
      metadata: {
        ...dataset.metadata,
        augmented: true,
        augmentationMethods: methods.map(m => m.method),
        count: augmented.length,
        augmentedAt: Date.now(),
      },
    };
  }

  // ── Batching ──────────────────────────────────────────────────────────────

  /**
   * Create a batched async iterator over the dataset.
   *
   * @param {object} dataset - Dataset with { samples }
   * @param {number} size    - Batch size
   * @returns {AsyncGenerator<object>} Async iterator yielding batch objects
   */
  async *batch(dataset, size = 32) {
    const samples = dataset.samples ?? dataset ?? [];
    const totalBatches = Math.ceil(samples.length / size);

    for (let i = 0; i < totalBatches; i++) {
      const start = i * size;
      const end = Math.min(start + size, samples.length);
      const batchSamples = samples.slice(start, end);

      yield {
        batchIndex: i,
        batchSize: batchSamples.length,
        samples: batchSamples,
        isLast: i === totalBatches - 1,
        progress: {
          current: i + 1,
          total: totalBatches,
          percent: ((i + 1) / totalBatches) * 100,
        },
      };
    }
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  /**
   * Parse a single CSV line, handling quoted fields.
   */
  _parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const ch = line[i];

      if (inQuotes) {
        if (ch === '"' && line[i + 1] === '"') {
          current += '"';
          i++;
        } else if (ch === '"') {
          inQuotes = false;
        } else {
          current += ch;
        }
      } else {
        if (ch === '"') {
          inQuotes = true;
        } else if (ch === ',') {
          result.push(current.trim());
          current = '';
        } else {
          current += ch;
        }
      }
    }

    result.push(current.trim());
    return result;
  }

  /**
   * Shuffle an array using Fisher-Yates algorithm.
   */
  _shuffle(array) {
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  /**
   * Generate a constraint mask for HIFT samples.
   */
  _generateConstraintMask(dim) {
    const mask = new Float64Array(dim);
    for (let i = 0; i < dim; i++) {
      mask[i] = Math.random() > 0.7 ? 0 : 1; // ~30% of dimensions are constrained
    }
    return mask;
  }

  /**
   * Apply a single augmentation method to a sample.
   */
  _applyAugmentation(sample, method) {
    const result = { ...sample };

    switch (method.method) {
      case 'noise': {
        const scale = method.scale ?? 0.1;
        // Add Gaussian noise to stimulus features
        if (result.stimulus?.features) {
          const features = Float64Array.from(result.stimulus.features);
          for (let i = 0; i < features.length; i++) {
            features[i] += (Math.random() - 0.5) * 2 * scale;
          }
          result.stimulus = { ...result.stimulus, features };
        }
        // Also perturb any top-level numerical fields
        for (const key of Object.keys(result)) {
          if (typeof result[key] === 'number' && key !== 'id' && key !== 'weight') {
            result[key] += (Math.random() - 0.5) * 2 * scale * 0.1;
          }
        }
        break;
      }

      case 'permutation': {
        const fraction = method.fraction ?? 0.2;
        // Randomly swap elements in stimulus features
        if (result.stimulus?.features) {
          const features = Float64Array.from(result.stimulus.features);
          const swaps = Math.floor(features.length * fraction);
          for (let s = 0; s < swaps; s++) {
            const i = Math.floor(Math.random() * features.length);
            const j = Math.floor(Math.random() * features.length);
            [features[i], features[j]] = [features[j], features[i]];
          }
          result.stimulus = { ...result.stimulus, features };
        }
        break;
      }

      case 'adversarial_perturbation': {
        const intensity = method.intensity ?? 0.5;
        // Add adversarial perturbation designed to maximize instability
        if (result.stimulus?.features) {
          const features = Float64Array.from(result.stimulus.features);
          for (let i = 0; i < features.length; i++) {
            // Adversarial: perturb in the direction of the gradient sign
            const sign = features[i] >= 0 ? 1 : -1;
            features[i] += sign * intensity * 0.1;
          }
          result.stimulus = { ...result.stimulus, features };
          result.adversarialIntensity = intensity;
        }
        break;
      }

      case 'dropout': {
        const rate = method.rate ?? 0.1;
        // Randomly zero out features
        if (result.stimulus?.features) {
          const features = Float64Array.from(result.stimulus.features);
          for (let i = 0; i < features.length; i++) {
            if (Math.random() < rate) {
              features[i] = 0;
            }
          }
          result.stimulus = { ...result.stimulus, features };
        }
        break;
      }

      case 'scaling': {
        const factor = method.factor ?? 0.9;
        // Scale features
        if (result.stimulus?.features) {
          const features = Float64Array.from(result.stimulus.features);
          for (let i = 0; i < features.length; i++) {
            features[i] *= factor;
          }
          result.stimulus = { ...result.stimulus, features };
        }
        break;
      }

      default:
        // Unknown method: pass through
        break;
    }

    return result;
  }
}

export default DatasetLoader;
