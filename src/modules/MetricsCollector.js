import { EventEmitter } from 'eventemitter3';

const BUCKET_SIZE = 1000;

/**
 * MetricsCollector — observability module for the PREDATOR agent system.
 * Provides counters, gauges, histograms, timers, and time-series storage
 * with aggregation, percentile computation, and flush-based export.
 */
export class MetricsCollector extends EventEmitter {
  constructor(opts = {}) {
    super();
    this.enableConsole = opts.enableConsole ?? false;
    this.flushInterval = opts.flushInterval ?? 60000;
    this.maxHistory = opts.maxHistory ?? 10000;

    // Metric stores
    this.counters = new Map();
    this.gauges = new Map();
    this.histograms = new Map();
    this.timers = new Map();
    this.timeSeries = new Map();

    this._flushTimer = null;
    this._timerIdCounter = 0;
  }

  // ── Counters ────────────────────────────────────────────────────────

  /**
   * Increment a counter by the given value (default 1).
   * @param {string} name
   * @param {number} [value=1]
   */
  incrementCounter(name, value = 1) {
    const current = this.counters.get(name) ?? 0;
    const next = current + value;
    this.counters.set(name, next);

    if (this.enableConsole) {
      console.log(`[metric] counter ${name} = ${next} (+${value})`);
    }

    this.emit('counter', { name, value: next, delta: value });
    return next;
  }

  /**
   * Get the current value of a counter.
   * @param {string} name
   * @returns {number}
   */
  getCounter(name) {
    return this.counters.get(name) ?? 0;
  }

  // ── Gauges ──────────────────────────────────────────────────────────

  /**
   * Set a gauge to an absolute value.
   * @param {string} name
   * @param {number} value
   */
  setGauge(name, value) {
    const prev = this.gauges.get(name);
    this.gauges.set(name, value);

    if (this.enableConsole) {
      console.log(`[metric] gauge ${name} = ${value}`);
    }

    this.emit('gauge', { name, value, prev });
    return value;
  }

  /**
   * Get the current value of a gauge.
   * @param {string} name
   * @returns {number|undefined}
   */
  getGauge(name) {
    return this.gauges.get(name);
  }

  // ── Histograms ──────────────────────────────────────────────────────

  /**
   * Record an observation in a histogram.
   * Internally stores raw values in buckets of BUCKET_SIZE for on-demand
   * percentile computation.
   * @param {string} name
   * @param {number} value
   */
  observeHistogram(name, value) {
    let hist = this.histograms.get(name);
    if (!hist) {
      hist = { sum: 0, count: 0, min: Infinity, max: -Infinity, buckets: [[]] };
      this.histograms.set(name, hist);
    }

    hist.sum += value;
    hist.count += 1;
    if (value < hist.min) hist.min = value;
    if (value > hist.max) hist.max = value;

    // Append to the latest bucket; spill into a new bucket when full
    const currentBucket = hist.buckets[hist.buckets.length - 1];
    if (currentBucket.length >= BUCKET_SIZE) {
      hist.buckets.push([value]);
    } else {
      currentBucket.push(value);
    }

    if (this.enableConsole) {
      console.log(`[metric] histogram ${name} observed ${value} (count=${hist.count})`);
    }

    this.emit('histogram', { name, value, count: hist.count });
    return hist.count;
  }

  /**
   * Compute and return histogram statistics including percentiles.
   * @param {string} name
   * @returns {{ count, sum, min, max, mean, p50, p95, p99 }|null}
   */
  getHistogram(name) {
    const hist = this.histograms.get(name);
    if (!hist || hist.count === 0) return null;

    // Flatten all buckets into a single sorted array for percentile computation
    const allValues = hist.buckets.flat();
    allValues.sort((a, b) => a - b);

    const mean = hist.sum / hist.count;
    const p50 = this._percentile(allValues, 50);
    const p95 = this._percentile(allValues, 95);
    const p99 = this._percentile(allValues, 99);

    return {
      count: hist.count,
      sum: hist.sum,
      min: hist.min,
      max: hist.max,
      mean,
      p50,
      p95,
      p99,
    };
  }

  /**
   * Compute a percentile from a sorted array of values.
   * Uses the "nearest-rank" method.
   * @param {number[]} sorted
   * @param {number} p — percentile (0–100)
   * @returns {number}
   * @private
   */
  _percentile(sorted, p) {
    if (sorted.length === 0) return 0;
    if (sorted.length === 1) return sorted[0];

    const rank = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(rank);
    const upper = Math.ceil(rank);
    if (lower === upper) return sorted[lower];

    // Linear interpolation
    const fraction = rank - lower;
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
  }

  // ── Timers ──────────────────────────────────────────────────────────

  /**
   * Start a timer and return a unique timer ID.
   * @param {string} name
   * @returns {number} timerId
   */
  startTimer(name) {
    let timer = this.timers.get(name);
    if (!timer) {
      timer = { sum: 0, count: 0, active: new Map() };
      this.timers.set(name, timer);
    }

    const id = ++this._timerIdCounter;
    timer.active.set(id, Date.now());

    if (this.enableConsole) {
      console.log(`[metric] timer ${name} started (id=${id})`);
    }

    return id;
  }

  /**
   * Stop a previously started timer and record the elapsed duration in ms.
   * @param {string} name
   * @param {number} timerId — ID returned by startTimer
   * @returns {number} elapsed duration in ms, or 0 if the timer ID was not found
   */
  stopTimer(name, timerId) {
    const timer = this.timers.get(name);
    if (!timer) return 0;

    const startTime = timer.active.get(timerId);
    if (startTime === undefined) return 0;

    timer.active.delete(timerId);
    const elapsed = Date.now() - startTime;
    timer.sum += elapsed;
    timer.count += 1;

    if (this.enableConsole) {
      console.log(`[metric] timer ${name} stopped (id=${timerId}, elapsed=${elapsed}ms)`);
    }

    this.emit('timer', { name, elapsed, timerId, count: timer.count });
    return elapsed;
  }

  // ── Time Series ─────────────────────────────────────────────────────

  /**
   * Record a time-stamped value.
   * Trims oldest entries when exceeding maxHistory.
   * @param {string} name
   * @param {number} value
   * @param {number} [timestamp=Date.now()]
   */
  recordTimeSeries(name, value, timestamp = Date.now()) {
    let series = this.timeSeries.get(name);
    if (!series) {
      series = [];
      this.timeSeries.set(name, series);
    }

    series.push({ timestamp, value });

    // Trim from the front when exceeding maxHistory
    if (series.length > this.maxHistory) {
      series.splice(0, series.length - this.maxHistory);
    }

    if (this.enableConsole) {
      console.log(`[metric] timeseries ${name} = ${value} at ${timestamp}`);
    }

    this.emit('timeseries', { name, value, timestamp });
  }

  /**
   * Retrieve time-series data with optional filtering.
   * @param {string} name
   * @param {{ since?: number, until?: number, limit?: number }} [opts]
   * @returns {{ timestamp: number, value: number }[]}
   */
  getTimeSeries(name, opts = {}) {
    const series = this.timeSeries.get(name);
    if (!series) return [];

    let result = series;

    if (opts.since != null) {
      result = result.filter((p) => p.timestamp >= opts.since);
    }
    if (opts.until != null) {
      result = result.filter((p) => p.timestamp <= opts.until);
    }
    if (opts.limit != null && opts.limit > 0) {
      // Return the most recent `limit` points
      result = result.slice(-opts.limit);
    }

    return result;
  }

  // ── Convenience: recordStep ─────────────────────────────────────────

  /**
   * Record a complete TPS step with all relevant metrics extracted from
   * the provided step data object.
   *
   * Expected stepData shape:
   *   { step, craving?, cascadeRisk?, praxisNorm?, tokenUsage?, energyUsage?, phase? }
   *
   * @param {{ step: number, craving?: number, cascadeRisk?: number, praxisNorm?: number, tokenUsage?: number, energyUsage?: number, phase?: string }} stepData
   */
  recordStep(stepData) {
    // Step counter — total number of steps observed
    this.incrementCounter('steps.total');

    // Craving gauge — current craving level
    if (stepData.craving != null) {
      this.setGauge('step.craving', stepData.craving);
    }

    // Cascade risk gauge — current cascade risk
    if (stepData.cascadeRisk != null) {
      this.setGauge('step.cascade_risk', stepData.cascadeRisk);
    }

    // Praxis norm histogram — distribution of praxis norm values
    if (stepData.praxisNorm != null) {
      this.observeHistogram('step.praxis_norm', stepData.praxisNorm);
    }

    // Token usage gauge — current token usage
    if (stepData.tokenUsage != null) {
      this.setGauge('step.token_usage', stepData.tokenUsage);
    }

    // Energy usage gauge — current energy usage
    if (stepData.energyUsage != null) {
      this.setGauge('step.energy_usage', stepData.energyUsage);
    }

    // Phase distribution counter — track which phases occur
    if (stepData.phase != null) {
      this.incrementCounter(`step.phase.${stepData.phase}`);
    }

    this.emit('step', stepData);
  }

  // ── Convenience: recordTask ─────────────────────────────────────────

  /**
   * Record a complete task result with all relevant metrics extracted from
   * the provided task result object.
   *
   * Expected taskResult shape:
   *   { success, quality?, stepCount?, wallClockTime?, tokenEfficiency?, extinctionCount? }
   *
   * @param {{ success: boolean, quality?: number, stepCount?: number, wallClockTime?: number, tokenEfficiency?: number, extinctionCount?: number }} taskResult
   */
  recordTask(taskResult) {
    // Task completion counter — success / failure
    const status = taskResult.success ? 'success' : 'failure';
    this.incrementCounter(`tasks.${status}`);
    this.incrementCounter('tasks.total');

    // Quality histogram — distribution of quality scores
    if (taskResult.quality != null) {
      this.observeHistogram('task.quality', taskResult.quality);
    }

    // Step count histogram — distribution of steps per task
    if (taskResult.stepCount != null) {
      this.observeHistogram('task.step_count', taskResult.stepCount);
    }

    // Wall clock time — record as timer observation
    if (taskResult.wallClockTime != null) {
      // Use the timer mechanism but directly record the duration
      let timer = this.timers.get('task.wall_clock_time');
      if (!timer) {
        timer = { sum: 0, count: 0, active: new Map() };
        this.timers.set('task.wall_clock_time', timer);
      }
      timer.sum += taskResult.wallClockTime;
      timer.count += 1;
    }

    // Token efficiency gauge — latest token efficiency
    if (taskResult.tokenEfficiency != null) {
      this.setGauge('task.token_efficiency', taskResult.tokenEfficiency);
    }

    // Extinction count counter — cumulative extinctions
    if (taskResult.extinctionCount != null) {
      this.incrementCounter('task.extinction_count', taskResult.extinctionCount);
    }

    this.emit('task', taskResult);
  }

  // ── Aggregation ─────────────────────────────────────────────────────

  /**
   * Return a summary snapshot of all collected metrics.
   * @returns {{
   *   counters: Object,
   *   gauges: Object,
   *   histograms: Object,
   *   timers: Object,
   *   timeSeries: { count: number, names: string[] }
   * }}
   */
  getSummary() {
    const counters = {};
    for (const [name, value] of this.counters) {
      counters[name] = value;
    }

    const gauges = {};
    for (const [name, value] of this.gauges) {
      gauges[name] = value;
    }

    const histograms = {};
    for (const [name] of this.histograms) {
      histograms[name] = this.getHistogram(name);
    }

    const timers = {};
    for (const [name, timer] of this.timers) {
      timers[name] = {
        sum: timer.sum,
        count: timer.count,
        active: timer.active.size,
        mean: timer.count > 0 ? timer.sum / timer.count : 0,
      };
    }

    const timeSeriesInfo = {};
    for (const [name, series] of this.timeSeries) {
      timeSeriesInfo[name] = {
        count: series.length,
        first: series.length > 0 ? series[0] : null,
        last: series.length > 0 ? series[series.length - 1] : null,
      };
    }

    return { counters, gauges, histograms, timers, timeSeries: timeSeriesInfo };
  }

  // ── Reset ───────────────────────────────────────────────────────────

  /**
   * Reset a specific metric by name across all stores.
   * @param {string} name
   */
  reset(name) {
    this.counters.delete(name);
    this.gauges.delete(name);
    this.histograms.delete(name);
    this.timers.delete(name);
    this.timeSeries.delete(name);

    if (this.enableConsole) {
      console.log(`[metric] reset ${name}`);
    }

    this.emit('reset', { name });
  }

  /**
   * Reset all metric stores.
   */
  resetAll() {
    this.counters.clear();
    this.gauges.clear();
    this.histograms.clear();
    this.timers.clear();
    this.timeSeries.clear();

    if (this.enableConsole) {
      console.log('[metric] reset all');
    }

    this.emit('resetAll');
  }

  // ── Lifecycle ───────────────────────────────────────────────────────

  /**
   * Start the auto-flush timer.
   */
  start() {
    if (this._flushTimer) return; // already running

    this._flushTimer = setInterval(() => {
      this.flush().catch((err) => {
        this.emit('error', err);
      });
    }, this.flushInterval);

    // Ensure the timer does not prevent the process from exiting
    if (this._flushTimer.unref) {
      this._flushTimer.unref();
    }

    if (this.enableConsole) {
      console.log(`[metric] auto-flush started (interval=${this.flushInterval}ms)`);
    }

    this.emit('start');
  }

  /**
   * Stop the auto-flush timer.
   */
  stop() {
    if (this._flushTimer) {
      clearInterval(this._flushTimer);
      this._flushTimer = null;
    }

    if (this.enableConsole) {
      console.log('[metric] auto-flush stopped');
    }

    this.emit('stop');
  }

  /**
   * Flush all metrics and emit a 'flush' event with the complete snapshot.
   * External exporters can listen for this event to push metrics to
   * Prometheus, StatsD, Datadog, etc.
   * @returns {Promise<void>}
   */
  async flush() {
    const snapshot = this.getSummary();
    snapshot.flushTimestamp = Date.now();

    if (this.enableConsole) {
      console.log('[metric] flush:', JSON.stringify(snapshot, null, 2));
    }

    this.emit('flush', snapshot);
  }
}

export default MetricsCollector;
