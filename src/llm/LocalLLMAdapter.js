/**
 * LocalLLMAdapter — LLMAdapter backed by a local model server
 * (Ollama, llama.cpp, text-generation-webui, etc.).
 *
 * Supports two API dialects:
 *   • Ollama native  — /api/chat, /api/generate, /api/embeddings
 *   • OpenAI-compatible — /v1/chat/completions
 *
 * The dialect is auto-detected from the endpoint path, or can be
 * forced via `opts.dialect` ('ollama' | 'openai').
 */
import axios from 'axios';
import { LLMAdapter } from './LLMAdapter.js';

export class LocalLLMAdapter extends LLMAdapter {
  /**
   * @param {object} opts – Passed through to LLMAdapter; also:
   * @param {string} [opts.endpoint]   – Base URL of the local server (default: 'http://localhost:11434')
   * @param {string} [opts.modelName]  – Model name known to the server (default: 'llama2')
   * @param {string} [opts.dialect]    – Force 'ollama' or 'openai'; auto-detected if omitted
   */
  constructor(opts = {}) {
    super(opts);
    this.endpoint  = opts.endpoint  ?? 'http://localhost:11434';
    this.modelName = opts.modelName ?? 'llama2';
    this.dialect   = opts.dialect   ?? null; // auto-detect
  }

  // ─── Internal helpers ───────────────────────────────────────────────

  /**
   * Determine the server dialect.  If the endpoint contains the Ollama
   * default port (11434) we assume Ollama; otherwise OpenAI-compatible.
   */
  _detectDialect() {
    if (this.dialect) return this.dialect;
    try {
      const url = new URL(this.endpoint);
      return url.port === '11434' || url.hostname.includes('ollama')
        ? 'ollama'
        : 'openai';
    } catch {
      return 'ollama'; // safe default
    }
  }

  /** Build the standardised response envelope. */
  _envelope(ok, extra = {}) {
    return {
      ok,
      timestamp: Date.now(),
      adapter: 'local',
      model: this.modelName,
      dialect: this._detectDialect(),
      ...extra,
    };
  }

  /**
   * Shared axios request wrapper with timeout and error handling.
   * @param {string} url
   * @param {object} data
   * @returns {Promise<{data:object, status:number}>}
   */
  async _post(url, data) {
    const source = axios.CancelToken.source();
    const timer = setTimeout(() => source.cancel('Request timed out'), this.timeout);

    try {
      const resp = await axios.post(url, data, {
        headers: { 'Content-Type': 'application/json' },
        cancelToken: source.token,
        timeout: this.timeout,
      });
      return { data: resp.data, status: resp.status };
    } finally {
      clearTimeout(timer);
    }
  }

  // ─── Public API ─────────────────────────────────────────────────────

  /**
   * Multi-turn chat completion.
   * @param {string} prompt
   * @param {string} [systemPrompt]
   */
  async chat(prompt, systemPrompt) {
    try {
      const dialect = this._detectDialect();

      if (dialect === 'openai') {
        // ── OpenAI-compatible endpoint (/v1/chat/completions) ──
        const messages = [];
        if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
        messages.push({ role: 'user', content: prompt });

        const url = `${this.endpoint}/v1/chat/completions`;
        const { data } = await this._post(url, {
          model: this.modelName,
          messages,
          max_tokens: this.maxTokens,
          temperature: this.temperature,
        });

        const content = data.choices?.[0]?.message?.content
                     ?? data.choices?.[0]?.text
                     ?? '';
        return this._envelope(true, {
          content,
          usage: data.usage ?? null,
          finishReason: data.choices?.[0]?.finish_reason ?? null,
        });
      }

      // ── Ollama native (/api/chat) ──
      const messages = [];
      if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
      messages.push({ role: 'user', content: prompt });

      const url = `${this.endpoint}/api/chat`;
      const { data } = await this._post(url, {
        model: this.modelName,
        messages,
        stream: false,
        options: {
          num_predict: this.maxTokens,
          temperature: this.temperature,
        },
      });

      const content = data.message?.content ?? '';
      return this._envelope(true, {
        content,
        usage: data.eval_count != null
          ? { prompt_tokens: data.prompt_eval_count ?? 0, completion_tokens: data.eval_count, total_tokens: (data.prompt_eval_count ?? 0) + data.eval_count }
          : null,
        finishReason: data.done ? 'stop' : null,
      });
    } catch (err) {
      const msg = err?.response?.data?.error ?? err.message ?? String(err);
      return this._envelope(false, { error: msg, content: '' });
    }
  }

  /**
   * Single-turn text completion.
   * @param {string} prompt
   */
  async complete(prompt) {
    try {
      const dialect = this._detectDialect();

      if (dialect === 'openai') {
        // Fall back to the chat endpoint with a single user message
        return this.chat(prompt);
      }

      // Ollama /api/generate
      const url = `${this.endpoint}/api/generate`;
      const { data } = await this._post(url, {
        model: this.modelName,
        prompt,
        stream: false,
        options: {
          num_predict: this.maxTokens,
          temperature: this.temperature,
        },
      });

      const content = data.response ?? '';
      return this._envelope(true, {
        content,
        usage: data.eval_count != null
          ? { prompt_tokens: data.prompt_eval_count ?? 0, completion_tokens: data.eval_count, total_tokens: (data.prompt_eval_count ?? 0) + data.eval_count }
          : null,
        finishReason: data.done ? 'stop' : null,
      });
    } catch (err) {
      const msg = err?.response?.data?.error ?? err.message ?? String(err);
      return this._envelope(false, { error: msg, content: '' });
    }
  }

  /**
   * Generate an embedding via the local server.
   * @param {string} text
   */
  async embed(text) {
    try {
      if (typeof text !== 'string' || text.length === 0) {
        return this._envelope(false, { error: 'embed() requires a non-empty string' });
      }

      const dialect = this._detectDialect();

      if (dialect === 'openai') {
        // Try /v1/embeddings (OpenAI-compatible)
        const url = `${this.endpoint}/v1/embeddings`;
        const { data } = await this._post(url, {
          model: this.modelName,
          input: text,
        });
        const embedding = data.data?.[0]?.embedding ?? [];
        return this._envelope(true, { embedding });
      }

      // Ollama /api/embeddings
      const url = `${this.endpoint}/api/embeddings`;
      const { data } = await this._post(url, {
        model: this.modelName,
        prompt: text,
      });

      const embedding = data.embedding ?? [];
      return this._envelope(true, { embedding });
    } catch (err) {
      const msg = err?.response?.data?.error ?? err.message ?? String(err);
      return this._envelope(false, { error: msg });
    }
  }

  /**
   * Classify text into one of the candidate labels via chat completion.
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
      const msg = err?.response?.data?.error ?? err.message ?? String(err);
      return this._envelope(false, { error: msg });
    }
  }
}
