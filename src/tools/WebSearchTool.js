/**
 * WebSearchTool.js
 * Web search and page fetching for the PREDATOR agent system.
 * Tries z-ai-web-dev-sdk first, falls back to axios + search engine.
 * All methods are async and return structured results.
 */

import axios from 'axios';

const DEFAULT_NUM_RESULTS = 10;
const FETCH_TIMEOUT = 30000;
const SEARCH_TIMEOUT = 15000;
const MAX_CONTENT_LENGTH = 2 * 1024 * 1024; // 2 MB

class WebSearchTool {
  constructor(options = {}) {
    this.id = 'websearch';
    this.name = 'WebSearchTool';
    this.description = 'Search the web and fetch page content';
    this.timeout = options.timeout || FETCH_TIMEOUT;
    this.searchTimeout = options.searchTimeout || SEARCH_TIMEOUT;
    this.userAgent =
      options.userAgent ||
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

    // Lazy-load z-ai-web-dev-sdk on first use
    this._sdk = null;
    this._sdkLoaded = false;
  }

  async _loadSDK() {
    if (this._sdkLoaded) return;
    this._sdkLoaded = true;
    try {
      // eslint-disable-next-line import/no-unresolved
      this._sdk = await import('z-ai-web-dev-sdk');
    } catch {
      // SDK not available — will fall back to axios-based approach
    }
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

  _stripHtml(html) {
    // Remove scripts, styles, then tags
    return html
      .replace(/<script[\s\S]*?<\/script>/gi, '')
      .replace(/<style[\s\S]*?<\/style>/gi, '')
      .replace(/<[^>]+>/g, ' ')
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/\s+/g, ' ')
      .trim();
  }

  // ── Search ───────────────────────────────────────────────────────────

  async search(query, numResults = DEFAULT_NUM_RESULTS) {
    const start = Date.now();
    try {
      await this._loadSDK();
      // Strategy 1: use z-ai-web-dev-sdk if available
      if (this._sdk) {
        try {
          const results = await this._sdk.webSearch(query, { numResults });
          return this._result(
            true,
            {
              query,
              results: results.map((r) => ({
                title: r.title || '',
                url: r.url || r.link || '',
                snippet: r.snippet || r.description || '',
              })),
              count: results.length,
              source: 'z-ai-web-dev-sdk',
            },
            undefined,
            start,
          );
        } catch {
          // Fall through to axios-based search
        }
      }

      // Strategy 2: DuckDuckGo Instant Answer API (no key required)
      const ddgUrl = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
      const ddgResponse = await axios.get(ddgUrl, {
        timeout: this.searchTimeout,
        headers: { 'User-Agent': this.userAgent },
      });

      const results = [];

      // Parse DDG response
      if (ddgResponse.data) {
        const data = ddgResponse.data;

        // Abstract
        if (data.Abstract) {
          results.push({
            title: data.Heading || query,
            url: data.AbstractURL || '',
            snippet: data.Abstract,
          });
        }

        // Related topics
        if (data.RelatedTopics) {
          for (const topic of data.RelatedTopics.slice(0, numResults)) {
            if (topic.Text && topic.FirstURL) {
              results.push({
                title: topic.Text.substring(0, 100),
                url: topic.FirstURL,
                snippet: topic.Text,
              });
            }
          }
        }

        // Results
        if (data.Results) {
          for (const r of data.Results.slice(0, numResults)) {
            results.push({
              title: r.Text?.substring(0, 100) || '',
              url: r.FirstURL || '',
              snippet: r.Text || '',
            });
          }
        }
      }

      return this._result(
        true,
        {
          query,
          results: results.slice(0, numResults),
          count: Math.min(results.length, numResults),
          source: 'duckduckgo',
        },
        undefined,
        start,
      );
    } catch (err) {
      return this._result(false, undefined, `search failed: ${err.message}`, start);
    }
  }

  // ── Fetch Page ───────────────────────────────────────────────────────

  async fetchPage(url) {
    const start = Date.now();
    try {
      await this._loadSDK();
      // Strategy 1: z-ai-web-dev-sdk
      if (this._sdk) {
        try {
          const page = await this._sdk.fetchPage(url);
          return this._result(
            true,
            {
              url,
              title: page.title || '',
              content: page.content || page.html || '',
              textContent: page.textContent || this._stripHtml(page.content || page.html || ''),
              source: 'z-ai-web-dev-sdk',
            },
            undefined,
            start,
          );
        } catch {
          // Fall through to axios
        }
      }

      // Strategy 2: axios direct fetch
      const response = await axios.get(url, {
        timeout: this.timeout,
        headers: {
          'User-Agent': this.userAgent,
          Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.5',
        },
        maxContentLength: MAX_CONTENT_LENGTH,
        responseEncoding: 'utf-8',
      });

      const html = typeof response.data === 'string' ? response.data : String(response.data);
      const textContent = this._stripHtml(html);

      // Extract title
      const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
      const title = titleMatch ? titleMatch[1].trim() : '';

      return this._result(
        true,
        {
          url,
          title,
          html,
          textContent,
          contentLength: html.length,
          statusCode: response.status,
          contentType: response.headers['content-type'] || '',
          source: 'axios',
        },
        undefined,
        start,
      );
    } catch (err) {
      if (err.response) {
        return this._result(
          false,
          undefined,
          `fetchPage failed: HTTP ${err.response.status} - ${err.response.statusText}`,
          start,
        );
      }
      return this._result(false, undefined, `fetchPage failed: ${err.message}`, start);
    }
  }

  // ── Extract Links ────────────────────────────────────────────────────

  extractLinks(html) {
    const start = Date.now();
    try {
      if (typeof html !== 'string') {
        return this._result(false, undefined, 'extractLinks requires an HTML string', start);
      }

      const linkRegex = /<a\s[^>]*href\s*=\s*["']([^"']*)["'][^>]*>([\s\S]*?)<\/a>/gi;
      const links = [];
      let match;

      while ((match = linkRegex.exec(html)) !== null) {
        const href = match[1];
        const text = this._stripHtml(match[2]).trim();
        if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
          links.push({ href, text });
        }
      }

      return this._result(true, { links, count: links.length }, undefined, start);
    } catch (err) {
      return this._result(false, undefined, `extractLinks failed: ${err.message}`, start);
    }
  }
}

export default WebSearchTool;
