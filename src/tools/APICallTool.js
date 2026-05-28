/**
 * APICallTool.js
 * HTTP client tool using axios for the PREDATOR agent system.
 * Provides GET, POST, PUT, DELETE, and file download methods.
 * All methods are async and return structured results.
 */

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import { createWriteStream } from 'fs';

const DEFAULT_TIMEOUT = 30000;
const MAX_RESPONSE_SIZE = 10 * 1024 * 1024; // 10 MB
const DOWNLOAD_CHUNK_SIZE = 64 * 1024;

class APICallTool {
  constructor(options = {}) {
    this.id = 'api_call';
    this.name = 'APICallTool';
    this.description = 'Make HTTP requests (GET, POST, PUT, DELETE) and download files';
    this.defaultTimeout = options.defaultTimeout || DEFAULT_TIMEOUT;
    this.maxResponseSize = options.maxResponseSize || MAX_RESPONSE_SIZE;
    this.baseURL = options.baseURL || null;
    this.defaultHeaders = options.defaultHeaders || {};

    // Create a reusable axios instance
    this._client = axios.create({
      timeout: this.defaultTimeout,
      maxContentLength: this.maxResponseSize,
      maxBodyLength: this.maxResponseSize,
      headers: {
        'User-Agent': 'PREDATOR-Agent/1.0',
        ...this.defaultHeaders,
      },
      ...(this.baseURL && { baseURL: this.baseURL }),
    });

    // Response interceptor for common processing
    this._client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response) {
          // Server responded with error status — return structured info
          const err = new Error(`HTTP ${error.response.status}: ${error.response.statusText}`);
          err.status = error.response.status;
          err.data = error.response.data;
          err.headers = error.response.headers;
          throw err;
        }
        throw error;
      },
    );
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

  _formatResponse(response) {
    const contentType = response.headers?.['content-type'] || '';
    const isJSON = contentType.includes('application/json');
    let data = response.data;

    // Truncate large text responses
    if (typeof data === 'string' && data.length > 50000) {
      data = data.substring(0, 50000) + '\n... [response truncated]';
    }

    return {
      statusCode: response.status,
      statusText: response.statusText,
      headers: response.headers,
      data,
      contentType,
      isJSON,
    };
  }

  _mergeHeaders(customHeaders) {
    return {
      ...this.defaultHeaders,
      ...(customHeaders && typeof customHeaders === 'object' ? customHeaders : {}),
    };
  }

  // ── GET ──────────────────────────────────────────────────────────────

  async get(url, headers = {}) {
    const start = Date.now();
    try {
      const response = await this._client.get(url, {
        headers: this._mergeHeaders(headers),
        timeout: this.defaultTimeout,
      });
      return this._result(true, this._formatResponse(response), undefined, start);
    } catch (err) {
      if (err.status && err.data) {
        // Server error with response body
        return this._result(
          false,
          undefined,
          `GET ${url} failed: HTTP ${err.status} - ${err.message}`,
          start,
        );
      }
      return this._result(false, undefined, `GET ${url} failed: ${err.message}`, start);
    }
  }

  // ── POST ─────────────────────────────────────────────────────────────

  async post(url, body = null, headers = {}) {
    const start = Date.now();
    try {
      const mergedHeaders = this._mergeHeaders(headers);

      const response = await this._client.post(url, body, {
        headers: mergedHeaders,
        timeout: this.defaultTimeout,
      });

      return this._result(true, this._formatResponse(response), undefined, start);
    } catch (err) {
      if (err.status && err.data) {
        return this._result(
          false,
          undefined,
          `POST ${url} failed: HTTP ${err.status} - ${err.message}`,
          start,
        );
      }
      return this._result(false, undefined, `POST ${url} failed: ${err.message}`, start);
    }
  }

  // ── PUT ──────────────────────────────────────────────────────────────

  async put(url, body = null, headers = {}) {
    const start = Date.now();
    try {
      const mergedHeaders = this._mergeHeaders(headers);

      const response = await this._client.put(url, body, {
        headers: mergedHeaders,
        timeout: this.defaultTimeout,
      });

      return this._result(true, this._formatResponse(response), undefined, start);
    } catch (err) {
      if (err.status && err.data) {
        return this._result(
          false,
          undefined,
          `PUT ${url} failed: HTTP ${err.status} - ${err.message}`,
          start,
        );
      }
      return this._result(false, undefined, `PUT ${url} failed: ${err.message}`, start);
    }
  }

  // ── DELETE ───────────────────────────────────────────────────────────

  async delete(url, headers = {}) {
    const start = Date.now();
    try {
      const response = await this._client.delete(url, {
        headers: this._mergeHeaders(headers),
        timeout: this.defaultTimeout,
      });

      return this._result(true, this._formatResponse(response), undefined, start);
    } catch (err) {
      if (err.status && err.data) {
        return this._result(
          false,
          undefined,
          `DELETE ${url} failed: HTTP ${err.status} - ${err.message}`,
          start,
        );
      }
      return this._result(false, undefined, `DELETE ${url} failed: ${err.message}`, start);
    }
  }

  // ── Download ─────────────────────────────────────────────────────────

  async download(url, destPath) {
    const start = Date.now();
    try {
      const resolvedDest = path.resolve(destPath);

      // Ensure destination directory exists
      const dir = path.dirname(resolvedDest);
      await fs.mkdir(dir, { recursive: true });

      const response = await this._client.get(url, {
        responseType: 'stream',
        timeout: this.defaultTimeout * 2, // Downloads get double timeout
      });

      const contentLength = parseInt(response.headers?.['content-length'] || '0', 10);

      return new Promise((resolve, reject) => {
        const writer = createWriteStream(resolvedDest);
        let downloadedBytes = 0;

        response.data.on('data', (chunk) => {
          downloadedBytes += chunk.length;
          if (downloadedBytes > this.maxResponseSize) {
            writer.close();
            fs.unlink(resolvedDest).catch(() => {}); // Clean up partial file
            reject(new Error(`Download exceeded max size of ${this.maxResponseSize} bytes`));
          }
        });

        response.data.pipe(writer);

        writer.on('finish', () => {
          resolve(
            this._result(
              true,
              {
                url,
                destPath: resolvedDest,
                downloadedBytes,
                contentLength,
                statusCode: response.status,
                contentType: response.headers?.['content-type'] || '',
              },
              undefined,
              start,
            ),
          );
        });

        writer.on('error', (err) => {
          reject(err);
        });

        response.data.on('error', (err) => {
          writer.close();
          reject(err);
        });
      });
    } catch (err) {
      return this._result(false, undefined, `Download ${url} failed: ${err.message}`, start);
    }
  }
}

export default APICallTool;
