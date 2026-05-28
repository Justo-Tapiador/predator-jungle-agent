/**
 * LLMAdapter — Abstract base class for LLM integration adapters.
 *
 * Provides a uniform interface for chat, completion, embedding, and
 * classification so higher-level PREDATOR modules can swap backends
 * without code changes.
 */
export class LLMAdapter {
  /**
   * @param {object} opts
   * @param {string}  [opts.model]       – Model identifier (default: 'default')
   * @param {number}  [opts.maxTokens]   – Max tokens per response (default: 2048)
   * @param {number}  [opts.temperature] – Sampling temperature (default: 0.7)
   * @param {number}  [opts.timeout]     – Request timeout in ms (default: 30000)
   */
  constructor(opts = {}) {
    this.model       = opts.model       ?? 'default';
    this.maxTokens   = opts.maxTokens   ?? 2048;
    this.temperature = opts.temperature ?? 0.7;
    this.timeout     = opts.timeout     ?? 30000;
  }

  /**
   * Multi-turn chat completion.
   * @param {string} prompt        – User message content
   * @param {string} [systemPrompt]– Optional system-level directive
   * @returns {Promise<{ok:boolean, content?:string, usage?:object, error?:string}>}
   */
  async chat(prompt, systemPrompt) {
    throw new Error('LLMAdapter.chat() not implemented');
  }

  /**
   * Single-turn text completion.
   * @param {string} prompt – Completion prompt
   * @returns {Promise<{ok:boolean, content?:string, usage?:object, error?:string}>}
   */
  async complete(prompt) {
    throw new Error('LLMAdapter.complete() not implemented');
  }

  /**
   * Generate an embedding vector for the given text.
   * @param {string} text – Input text
   * @returns {Promise<{ok:boolean, embedding?:number[], error?:string}>}
   */
  async embed(text) {
    throw new Error('LLMAdapter.embed() not implemented');
  }

  /**
   * Classify text into one of the provided labels.
   * @param {string}   text   – Input text
   * @param {string[]} labels – Candidate labels
   * @returns {Promise<{ok:boolean, label?:string, confidence?:number, error?:string}>}
   */
  async classify(text, labels) {
    throw new Error('LLMAdapter.classify() not implemented');
  }
}
