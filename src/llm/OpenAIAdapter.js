/**
 * OpenAIAdapter — LLMAdapter backed by the z-ai-web-dev-sdk OpenAI gateway.
 *
 * Uses z-ai for chat/completions and a deterministic hash-based fallback
 * for embeddings (the gateway does not expose an embed endpoint).
 */
import ZAI from 'z-ai-web-dev-sdk';
import { LLMAdapter } from './LLMAdapter.js';

export class OpenAIAdapter extends LLMAdapter {
  /**
   * @param {object} opts – Passed through to LLMAdapter; also:
   * @param {string} [opts.model] – OpenAI model name (default: 'gpt-4o-mini')
   */
  constructor(opts = {}) {
    super(opts);
    this.model = opts.model ?? 'gpt-4o-mini';
    /** @type {import('z-ai-web-dev-sdk').default|null} */
    this.zai = null; // lazily initialised
  }

  // ─── Internal helpers ───────────────────────────────────────────────

  /** Ensure the z-ai client is initialised exactly once. */
  async _ensureClient() {
    if (!this.zai) {
      this.zai = await ZAI.create();
    }
    return this.zai;
  }

  /**
   * Build the standardised response envelope.
   * @param {boolean} ok
   * @param {object}  extra
   * @returns {object}
   */
  _envelope(ok, extra = {}) {
    return { ok, timestamp: Date.now(), adapter: 'openai', model: this.model, ...extra };
  }

  /**
   * Simple deterministic hash-based embedding when no native embed API
   * is available.  Produces a 384-dim float vector derived from the
   * input text — suitable for cosine-similarity comparisons within the
   * same hash space but NOT compatible with OpenAI ada embeddings.
   * @param {string} text
   * @returns {number[]}
   */
  _hashEmbed(text) {
    const DIM = 384;
    const embedding = new Float64Array(DIM);
    // Walk the string in chunks and scatter values across dimensions
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      const slot = (i * 31 + code) % DIM;
      embedding[slot] += Math.sin(code * 0.01 + i * 0.1) * (1 + (code % 5));
    }
    // Normalise to unit length
    let norm = 0;
    for (let i = 0; i < DIM; i++) norm += embedding[i] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < DIM; i++) embedding[i] /= norm;
    return Array.from(embedding);
  }

  // ─── Public API ─────────────────────────────────────────────────────

  /**
   * Multi-turn chat completion via z-ai.
   * @param {string} prompt
   * @param {string} [systemPrompt]
   */
  async chat(prompt, systemPrompt) {
    try {
      const client = await this._ensureClient();

      const messages = [];
      if (systemPrompt) {
        messages.push({ role: 'system', content: systemPrompt });
      }
      messages.push({ role: 'user', content: prompt });

      const response = await client.chat.completions.create({
        model: this.model,
        messages,
        max_tokens: this.maxTokens,
        temperature: this.temperature,
      });

      const choice = response.choices?.[0];
      const content = choice?.message?.content ?? '';

      return this._envelope(true, {
        content,
        usage: response.usage ?? null,
        finishReason: choice?.finish_reason ?? null,
      });
    } catch (err) {
      return this._envelope(false, {
        error: err.message ?? String(err),
        content: '',
      });
    }
  }

  /**
   * Single-turn completion (convenience wrapper around chat).
   * @param {string} prompt
   */
  async complete(prompt) {
    try {
      const client = await this._ensureClient();

      const response = await client.chat.completions.create({
        model: this.model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: this.maxTokens,
        temperature: this.temperature,
      });

      const choice = response.choices?.[0];
      const content = choice?.message?.content ?? '';

      return this._envelope(true, {
        content,
        usage: response.usage ?? null,
        finishReason: choice?.finish_reason ?? null,
      });
    } catch (err) {
      return this._envelope(false, {
        error: err.message ?? String(err),
        content: '',
      });
    }
  }

  /**
   * Hash-based embedding (z-ai has no native embed endpoint).
   * @param {string} text
   */
  async embed(text) {
    try {
      if (typeof text !== 'string' || text.length === 0) {
        return this._envelope(false, { error: 'embed() requires a non-empty string' });
      }
      const embedding = this._hashEmbed(text);
      return this._envelope(true, { embedding });
    } catch (err) {
      return this._envelope(false, { error: err.message ?? String(err) });
    }
  }

  /**
   * Classify text into one of the candidate labels using chat completion.
   * @param {string}   text
   * @param {string[]} labels
   */
  async classify(text, labels) {
    try {
      if (!Array.isArray(labels) || labels.length === 0) {
        return this._envelope(false, { error: 'classify() requires a non-empty labels array' });
      }

      const systemPrompt = [
        'You are a classification engine. Respond with EXACTLY ONE of the following labels,',
        'and nothing else. No explanation, no punctuation, just the label.',
        '',
        'Labels: ' + labels.join(' | '),
      ].join('\n');

      const result = await this.chat(text, systemPrompt);

      if (!result.ok) return result; // propagate error envelope

      // Normalise the raw label against the candidates
      const raw = result.content.trim().toLowerCase();
      const matched = labels.find(l => l.toLowerCase() === raw)
                   ?? labels.find(l => l.toLowerCase().includes(raw))
                   ?? null;

      return this._envelope(true, {
        label: matched ?? result.content.trim(),
        confidence: matched ? 1.0 : 0.5,
        raw: result.content.trim(),
      });
    } catch (err) {
      return this._envelope(false, { error: err.message ?? String(err) });
    }
  }
}
