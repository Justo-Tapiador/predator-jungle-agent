import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import { mkdir, readFile, writeFile } from 'fs/promises';
import { join } from 'path';

/**
 * MemorySystem — Persistent memory for the PREDATOR agent system.
 *
 * Provides three memory tiers plus consolidation and relevance-based retrieval:
 *   • Episodic memory  – task execution records with full context
 *   • Semantic memory   – extracted facts / knowledge (key → value + confidence)
 *   • Working memory    – short-term buffer for the current task context
 *
 * Vector similarity uses a lightweight character n-gram hash embedding so the
 * module has zero external ML dependencies.
 */
export class MemorySystem extends EventEmitter {
  // ──────────────────────────────────────────
  // Construction
  // ──────────────────────────────────────────

  constructor(opts = {}) {
    super();

    this.storageDir = opts.storageDir ?? './data/memory';
    this.maxEpisodic = opts.maxEpisodic ?? 1000;
    this.maxWorking = opts.maxWorking ?? 10;
    this.embeddingDim = opts.embeddingDim ?? 64;
    this.consolidationThreshold = opts.consolidationThreshold ?? 0.85;
    this.enablePersistence = opts.enablePersistence ?? true;

    // Memory stores
    this.episodicMemory = [];                          // { id, timestamp, directive, result, embedding, metadata }
    this.semanticMemory = new Map();                   // key → { value, confidence, lastAccessed, accessCount }
    this.workingMemory = [];                           // { key, value, addedAt, ttl }

    this._initialized = false;
  }

  // ──────────────────────────────────────────
  // Lifecycle
  // ──────────────────────────────────────────

  /**
   * Initialise the memory system. Creates the storage directory and loads any
   * previously persisted data from disk.
   */
  async init() {
    if (this._initialized) return;

    if (this.enablePersistence) {
      try {
        await mkdir(this.storageDir, { recursive: true });
        await this.load();
      } catch (err) {
        this.emit('warn', `MemorySystem init load failed, starting fresh: ${err.message}`);
      }
    }

    // Prune any expired working-memory entries that survived a reload.
    this._pruneWorkingMemory();

    this._initialized = true;
    this.emit('ready');
  }

  // ──────────────────────────────────────────
  // Episodic memory
  // ──────────────────────────────────────────

  /**
   * Store a task execution record as episodic memory and automatically extract
   * semantic facts from the directive/result pair.
   *
   * @param {string}  directive  The original task directive
   * @param {*}       result     The task result (will be serialised to string)
   * @param {object}  [metadata] Extra metadata (e.g. tool, duration, status)
   * @returns {string} The episodic memory id
   */
  async store(directive, result, metadata = {}) {
    const id = uuidv4();
    const timestamp = Date.now();
    const resultStr = typeof result === 'string' ? result : JSON.stringify(result);
    const embedding = this._embed(`${directive} ${resultStr}`);

    const entry = { id, timestamp, directive, result: resultStr, embedding, metadata };
    this.episodicMemory.push(entry);

    // Enforce capacity limit – FIFO eviction.
    if (this.episodicMemory.length > this.maxEpisodic) {
      this.episodicMemory.shift();
    }

    // Extract semantic facts automatically.
    const facts = this._extractFacts(directive, resultStr);
    for (const { key, value, confidence } of facts) {
      await this.storeSemantic(key, value, confidence);
    }

    this.emit('stored', { type: 'episodic', id });
    return id;
  }

  // ──────────────────────────────────────────
  // Semantic memory
  // ──────────────────────────────────────────

  /**
   * Store (or update) a semantic fact.
   *
   * If the key already exists the confidence is updated to the higher of the
   * existing and supplied confidence values, and `accessCount` is incremented.
   */
  async storeSemantic(key, value, confidence = 0.5) {
    const existing = this.semanticMemory.get(key);
    const now = Date.now();

    if (existing) {
      existing.value = value;
      existing.confidence = Math.max(existing.confidence, confidence);
      existing.lastAccessed = now;
      existing.accessCount += 1;
    } else {
      this.semanticMemory.set(key, {
        value,
        confidence,
        lastAccessed: now,
        accessCount: 1,
      });
    }

    this.emit('stored', { type: 'semantic', key });
  }

  /**
   * Retrieve a semantic fact by exact key.
   *
   * @returns {{ value, confidence, lastAccessed, accessCount } | undefined}
   */
  async recallSemantic(key) {
    const entry = this.semanticMemory.get(key);
    if (entry) {
      entry.lastAccessed = Date.now();
      entry.accessCount += 1;
    }
    return entry;
  }

  // ──────────────────────────────────────────
  // Working memory
  // ──────────────────────────────────────────

  /**
   * Push an item into working memory with an optional TTL (in ms).
   * When the working memory buffer exceeds `maxWorking` the oldest entry is
   * evicted first.
   */
  async pushWorking(key, value, ttlMs) {
    this._pruneWorkingMemory();

    const entry = {
      key,
      value,
      addedAt: Date.now(),
      ttl: ttlMs ? Date.now() + ttlMs : null,
    };

    // If an entry with the same key exists, replace it.
    const existingIdx = this.workingMemory.findIndex((e) => e.key === key);
    if (existingIdx !== -1) {
      this.workingMemory.splice(existingIdx, 1, entry);
    } else {
      this.workingMemory.push(entry);
    }

    // Enforce capacity.
    while (this.workingMemory.length > this.maxWorking) {
      this.workingMemory.shift();
    }

    this.emit('stored', { type: 'working', key });
  }

  /**
   * Remove an item from working memory by key.
   *
   * @returns {boolean} `true` if the item was found and removed.
   */
  async popWorking(key) {
    const idx = this.workingMemory.findIndex((e) => e.key === key);
    if (idx === -1) return false;
    this.workingMemory.splice(idx, 1);
    return true;
  }

  // ──────────────────────────────────────────
  // Retrieval / recall
  // ──────────────────────────────────────────

  /**
   * Retrieve memories relevant to a query.
   *
   * @param {string} query   Natural-language query
   * @param {object} [opts]
   * @param {number} [opts.limit=10]         Maximum results per memory type
   * @param {number} [opts.minSimilarity=0.3] Minimum cosine similarity for episodic hits
   * @param {string[]} [opts.types]          Which memory types to search
   *                                          (['episodic','semantic','working'])
   * @returns {{ episodic: object[], semantic: object[], working: object[] }}
   */
  async recall(query, opts = {}) {
    const limit = opts.limit ?? 10;
    const minSimilarity = opts.minSimilarity ?? 0.3;
    const types = opts.types ?? ['episodic', 'semantic', 'working'];

    const result = { episodic: [], semantic: [], working: [] };

    // ── Episodic ──────────────────────────
    if (types.includes('episodic')) {
      const qEmbed = this._embed(query);
      const scored = this.episodicMemory
        .map((entry) => ({
          entry,
          similarity: this._cosineSimilarity(qEmbed, entry.embedding),
        }))
        .filter(({ similarity }) => similarity >= minSimilarity)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, limit);

      result.episodic = scored.map(({ entry, similarity }) => {
        // Strip the raw embedding from the returned object – it's large & internal.
        const { embedding, ...rest } = entry;
        return { ...rest, similarity };
      });
    }

    // ── Semantic ──────────────────────────
    if (types.includes('semantic')) {
      const qLower = query.toLowerCase();
      const candidates = [];

      for (const [key, val] of this.semanticMemory) {
        const keyLower = key.toLowerCase();
        const valStr = typeof val.value === 'string' ? val.value : JSON.stringify(val.value);
        const valLower = valStr.toLowerCase();

        // Simple token-overlap score for key/value relevance.
        const qTokens = this._tokenize(qLower);
        const kTokens = this._tokenize(keyLower);
        const vTokens = this._tokenize(valLower);
        const allTokens = new Set([...kTokens, ...vTokens]);

        let overlap = 0;
        for (const t of qTokens) {
          if (allTokens.has(t)) overlap++;
        }
        const relevance = qTokens.length > 0 ? overlap / qTokens.length : 0;

        // Boost by confidence.
        const score = relevance * val.confidence;

        if (relevance > 0 || keyLower.includes(qLower) || qLower.includes(keyLower)) {
          candidates.push({ key, ...val, score });
        }
      }

      result.semantic = candidates
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
    }

    // ── Working ───────────────────────────
    if (types.includes('working')) {
      this._pruneWorkingMemory();
      result.working = this.workingMemory.map(({ key, value, addedAt, ttl }) => ({
        key,
        value,
        addedAt,
        ttl,
      }));
    }

    this.emit('recalled', { query, counts: { episodic: result.episodic.length, semantic: result.semantic.length, working: result.working.length } });
    return result;
  }

  // ──────────────────────────────────────────
  // Consolidation
  // ──────────────────────────────────────────

  /**
   * Consolidate memories:
   *   1. Merge highly similar episodic memories (above `consolidationThreshold`).
   *   2. Extract recurring patterns as new semantic facts.
   *   3. Prune low-confidence semantic facts.
   */
  async consolidate() {
    const before = {
      episodic: this.episodicMemory.length,
      semantic: this.semanticMemory.size,
    };

    // 1. Merge similar episodic memories.
    this._mergeSimilarEpisodic();

    // 2. Extract recurring patterns as semantic facts.
    this._extractPatterns();

    // 3. Prune low-confidence semantic facts (confidence < 0.1).
    for (const [key, val] of this.semanticMemory) {
      if (val.confidence < 0.1) {
        this.semanticMemory.delete(key);
      }
    }

    const after = {
      episodic: this.episodicMemory.length,
      semantic: this.semanticMemory.size,
    };

    this.emit('consolidated', { before, after });

    if (this.enablePersistence) {
      await this.persist();
    }
  }

  // ──────────────────────────────────────────
  // Persistence
  // ──────────────────────────────────────────

  /**
   * Persist all memory stores to disk.
   */
  async persist() {
    if (!this.enablePersistence) return;

    try {
      await mkdir(this.storageDir, { recursive: true });

      const payload = {
        episodicMemory: this.episodicMemory.map((e) => ({
          ...e,
          // Serialise Float64Array as a regular array for JSON.
          embedding: Array.from(e.embedding),
        })),
        semanticMemory: Object.fromEntries(this.semanticMemory),
        workingMemory: this.workingMemory,
        savedAt: Date.now(),
      };

      const filePath = join(this.storageDir, 'memory.json');
      await writeFile(filePath, JSON.stringify(payload, null, 2), 'utf-8');
      this.emit('persisted');
    } catch (err) {
      this.emit('error', new Error(`MemorySystem persist failed: ${err.message}`));
    }
  }

  /**
   * Load all memory stores from disk.
   */
  async load() {
    if (!this.enablePersistence) return;

    try {
      const filePath = join(this.storageDir, 'memory.json');
      const raw = await readFile(filePath, 'utf-8');
      const payload = JSON.parse(raw);

      // Episodic — restore Float64Array embeddings.
      if (Array.isArray(payload.episodicMemory)) {
        this.episodicMemory = payload.episodicMemory.map((e) => ({
          ...e,
          embedding: Float64Array.from(e.embedding),
        }));
      }

      // Semantic — restore Map.
      if (payload.semanticMemory && typeof payload.semanticMemory === 'object') {
        this.semanticMemory = new Map(Object.entries(payload.semanticMemory));
      }

      // Working — restore array.
      if (Array.isArray(payload.workingMemory)) {
        this.workingMemory = payload.workingMemory;
      }

      this.emit('loaded');
    } catch (err) {
      // File not found is fine – first run.
      if (err.code !== 'ENOENT') {
        this.emit('warn', `MemorySystem load failed: ${err.message}`);
      }
    }
  }

  // ──────────────────────────────────────────
  // Clear
  // ──────────────────────────────────────────

  /**
   * Clear a specific memory type or all memories.
   *
   * @param {'episodic'|'semantic'|'working'|'all'} type
   */
  async clear(type = 'all') {
    switch (type) {
      case 'episodic':
        this.episodicMemory = [];
        break;
      case 'semantic':
        this.semanticMemory.clear();
        break;
      case 'working':
        this.workingMemory = [];
        break;
      case 'all':
        this.episodicMemory = [];
        this.semanticMemory.clear();
        this.workingMemory = [];
        break;
      default:
        throw new Error(`Unknown memory type: ${type}`);
    }

    this.emit('cleared', { type });

    if (this.enablePersistence) {
      await this.persist();
    }
  }

  // ──────────────────────────────────────────
  // Statistics
  // ──────────────────────────────────────────

  /**
   * Return a snapshot of current memory statistics.
   */
  stats() {
    const semanticConfidences = [...this.semanticMemory.values()].map((v) => v.confidence);
    const avgConfidence = semanticConfidences.length > 0
      ? semanticConfidences.reduce((a, b) => a + b, 0) / semanticConfidences.length
      : 0;

    const episodicTimestamps = this.episodicMemory.map((e) => e.timestamp);
    const oldestEpisodic = episodicTimestamps.length > 0 ? Math.min(...episodicTimestamps) : null;
    const newestEpisodic = episodicTimestamps.length > 0 ? Math.max(...episodicTimestamps) : null;

    return {
      episodic: {
        count: this.episodicMemory.length,
        maxCapacity: this.maxEpisodic,
        oldestTimestamp: oldestEpisodic,
        newestTimestamp: newestEpisodic,
      },
      semantic: {
        count: this.semanticMemory.size,
        averageConfidence: Math.round(avgConfidence * 1000) / 1000,
        topKeys: [...this.semanticMemory.entries()]
          .sort((a, b) => b[1].accessCount - a[1].accessCount)
          .slice(0, 10)
          .map(([key, val]) => ({ key, accessCount: val.accessCount, confidence: val.confidence })),
      },
      working: {
        count: this.workingMemory.length,
        maxCapacity: this.maxWorking,
        expired: this.workingMemory.filter((e) => e.ttl !== null && e.ttl < Date.now()).length,
      },
      initialized: this._initialized,
      persistenceEnabled: this.enablePersistence,
      storageDir: this.storageDir,
    };
  }

  // ──────────────────────────────────────────
  // Private helpers — embeddings & similarity
  // ──────────────────────────────────────────

  /**
   * Compute a lightweight bag-of-ngrams embedding for a piece of text.
   *
   * Uses character 3-gram hashing to distribute tokens across the embedding
   * dimensions.  The result is L2-normalised so it can be used directly with
   * cosine similarity.
   *
   * @param {string} text
   * @returns {Float64Array}  Normalised embedding vector of length `embeddingDim`
   */
  _embed(text) {
    const dim = this.embeddingDim;
    const vec = new Float64Array(dim);

    if (!text || typeof text !== 'string') return vec;

    const normalized = text.toLowerCase();

    // Character 3-grams.
    for (let i = 0; i <= normalized.length - 3; i++) {
      const ngram = normalized.slice(i, i + 3);
      const hash = this._hashNgram(ngram);
      vec[Math.abs(hash) % dim] += 1;
    }

    // Character 4-grams (captures slightly longer patterns).
    for (let i = 0; i <= normalized.length - 4; i++) {
      const ngram = normalized.slice(i, i + 4);
      const hash = this._hashNgram(ngram);
      vec[Math.abs(hash) % dim] += 0.5;
    }

    // Whole-word tokens for an extra signal.
    const words = normalized.split(/\s+/).filter(Boolean);
    for (const word of words) {
      const hash = this._hashNgram(word);
      vec[Math.abs(hash) % dim] += 0.8;
    }

    // L2 normalise.
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < dim; i++) vec[i] /= norm;
    }

    return vec;
  }

  /**
   * Simple deterministic hash for a character n-gram (FNV-1a variant).
   */
  _hashNgram(str) {
    let hash = 2166136261; // FNV offset basis (32-bit)
    for (let i = 0; i < str.length; i++) {
      hash ^= str.charCodeAt(i);
      hash = (hash * 16777619) >>> 0; // FNV prime, keep 32 bits
    }
    return hash;
  }

  /**
   * Compute cosine similarity between two Float64Array vectors.
   */
  _cosineSimilarity(a, b) {
    if (a.length !== b.length) return 0;

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dot / denominator;
  }

  // ──────────────────────────────────────────
  // Private helpers — tokenisation
  // ──────────────────────────────────────────

  /**
   * Very small tokeniser — splits on non-alphanumeric and filters short tokens.
   */
  _tokenize(text) {
    return text
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length > 1);
  }

  // ──────────────────────────────────────────
  // Private helpers — fact extraction
  // ──────────────────────────────────────────

  /**
   * Extract structured facts from a directive/result pair.
   *
   * Currently detects:
   *   - Tool usage patterns   → "tool:<name>" key
   *   - Success / failure     → "result:status" key
   *   - Common error phrases  → "error:<phrase>" key
   *   - Duration patterns     → "duration:<directive_hash>" key
   *
   * @returns {{ key: string, value: string, confidence: number }[]}
   */
  _extractFacts(directive, result) {
    const facts = [];

    // ── Tool usage ────────────────────────
    const toolPattern = /(?:tool|function|command)[=:]\s*["']?(\w+)/gi;
    let match;
    while ((match = toolPattern.exec(`${directive} ${result}`)) !== null) {
      const toolName = match[1].toLowerCase();
      facts.push({
        key: `tool:${toolName}`,
        value: `Tool ${toolName} was used`,
        confidence: 0.6,
      });
    }

    // Also detect tool names mentioned as top-level keys in JSON-like results.
    const jsonToolPattern = /"tool"\s*:\s*"(\w+)"/gi;
    while ((match = jsonToolPattern.exec(result)) !== null) {
      const toolName = match[1].toLowerCase();
      facts.push({
        key: `tool:${toolName}`,
        value: `Tool ${toolName} was used`,
        confidence: 0.7,
      });
    }

    // ── Success / failure status ──────────
    const statusPattern = /\b(status|state|outcome)\s*[:=]\s*["']?(success|failure|error|failed|ok|completed)\b/gi;
    while ((match = statusPattern.exec(result)) !== null) {
      const status = match[2].toLowerCase();
      facts.push({
        key: `result:status:${status}`,
        value: `Task resulted in: ${status}`,
        confidence: status === 'error' || status === 'failure' ? 0.8 : 0.7,
      });
    }

    // ── Common error phrases ──────────────
    const errorPattern = /\b(EINVAL|ENOENT|EPERM|TypeError|ReferenceError|SyntaxError|Timeout|not found|permission denied|rate limit)\b/gi;
    while ((match = errorPattern.exec(result)) !== null) {
      const errPhrase = match[1];
      facts.push({
        key: `error:${errPhrase.toLowerCase()}`,
        value: `Encountered error: ${errPhrase}`,
        confidence: 0.75,
      });
    }

    // ── Duration / latency ────────────────
    const durationPattern = /\b(duration|elapsed|time_taken|latency)\s*[:=]\s*["']?(\d+)\s*(ms|s|sec|milliseconds?)\b/gi;
    while ((match = durationPattern.exec(result)) !== null) {
      const value = `${match[2]}${match[3]}`;
      const dirHash = this._hashNgram(directive.slice(0, 60)) >>> 0;
      facts.push({
        key: `duration:${dirHash}`,
        value,
        confidence: 0.5,
      });
    }

    return facts;
  }

  // ──────────────────────────────────────────
  // Private helpers — consolidation internals
  // ──────────────────────────────────────────

  /**
   * Merge episodic memories whose embeddings exceed `consolidationThreshold`.
   * The merged entry keeps the newer timestamp and concatenates directives.
   */
  _mergeSimilarEpisodic() {
    if (this.episodicMemory.length < 2) return;

    const merged = [];
    const consumed = new Set();

    for (let i = 0; i < this.episodicMemory.length; i++) {
      if (consumed.has(i)) continue;

      const entry = this.episodicMemory[i];
      let mergedEntry = { ...entry };
      let mergeCount = 1;

      for (let j = i + 1; j < this.episodicMemory.length; j++) {
        if (consumed.has(j)) continue;

        const other = this.episodicMemory[j];
        const sim = this._cosineSimilarity(entry.embedding, other.embedding);

        if (sim >= this.consolidationThreshold) {
          consumed.add(j);
          mergeCount++;

          // Keep the later timestamp.
          if (other.timestamp > mergedEntry.timestamp) {
            mergedEntry.timestamp = other.timestamp;
          }

          // Append the other directive for context.
          mergedEntry.directive = `${mergedEntry.directive} | ${other.directive}`;

          // Average the embeddings.
          const avgEmbedding = new Float64Array(this.embeddingDim);
          for (let k = 0; k < this.embeddingDim; k++) {
            avgEmbedding[k] = (mergedEntry.embedding[k] + other.embedding[k]) / 2;
          }
          mergedEntry.embedding = avgEmbedding;
        }
      }

      if (mergeCount > 1) {
        mergedEntry.metadata = { ...mergedEntry.metadata, mergedFrom: mergeCount };
      }

      merged.push(mergedEntry);
    }

    this.episodicMemory = merged;
  }

  /**
   * Extract recurring patterns from episodic memory as new semantic facts.
   *
   * Looks for:
   *   - Directives sharing the same leading verb pattern (e.g. "read file …")
   *   - Tools that appear frequently across episodes
   */
  _extractPatterns() {
    // Count directive verb patterns.
    const verbCounts = new Map();
    const toolCounts = new Map();

    for (const entry of this.episodicMemory) {
      // Extract leading verb-ish token (first word).
      const firstWord = (entry.directive || '').split(/\s+/)[0]?.toLowerCase();
      if (firstWord && firstWord.length > 2) {
        verbCounts.set(firstWord, (verbCounts.get(firstWord) ?? 0) + 1);
      }

      // Extract tool mentions in the result.
      const toolPattern = /"tool"\s*:\s*"(\w+)"/gi;
      let m;
      while ((m = toolPattern.exec(entry.result)) !== null) {
        const tool = m[1].toLowerCase();
        toolCounts.set(tool, (toolCounts.get(tool) ?? 0) + 1);
      }
    }

    // Store patterns that occur more than once as semantic facts.
    for (const [verb, count] of verbCounts) {
      if (count >= 2) {
        this.storeSemantic(
          `pattern:verb:${verb}`,
          `Directive verb "${verb}" appeared ${count} times across episodic memories`,
          Math.min(0.9, 0.3 + count * 0.1),
        );
      }
    }

    for (const [tool, count] of toolCounts) {
      if (count >= 2) {
        this.storeSemantic(
          `pattern:tool:${tool}`,
          `Tool "${tool}" was used ${count} times across episodic memories`,
          Math.min(0.9, 0.3 + count * 0.1),
        );
      }
    }
  }

  // ──────────────────────────────────────────
  // Private helpers — working memory TTL
  // ──────────────────────────────────────────

  /**
   * Remove expired entries from working memory.
   */
  _pruneWorkingMemory() {
    const now = Date.now();
    this.workingMemory = this.workingMemory.filter(
      (entry) => entry.ttl === null || entry.ttl > now,
    );
  }
}

export default MemorySystem;
