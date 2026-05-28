/**
 * PREDATOR Web Dashboard — Client-Side Logic
 *
 * Connects to the PREDATOR Web Server via WebSocket for real-time
 * event streaming and provides interactive chart visualizations
 * using Chart.js.
 *
 * Architecture:
 *   ┌──────────────────┐     WS      ┌──────────────┐
 *   │  Dashboard.js     │◄──────────►│  Web Server   │
 *   │  (This file)      │  Commands  │  (server.js)  │
 *   └──────────────────┘             └──────┬───────┘
 *         │                                  │
 *         ▼                                  ▼
 *   ┌──────────┐                      ┌──────────┐
 *   │ Chart.js │                      │ Predator  │
 *   │ Canvases │                      │  Agent    │
 *   └──────────┘                      └──────────┘
 */

// ═══════════════════════ GLOBALS ═══════════════════════
let ws = null;
let reconnectTimer = null;
let requestCounter = 0;
const WS_URL = `ws://${window.location.host}/ws`;

// Chart instances
let qualityChart = null;
let progressChart = null;
let cascadeRiskChart = null;
let resourceUsageChart = null;
let metricsTimeSeriesChart = null;
let cascadeDetailChart = null;

// Data buffers for charts
const MAX_DATA_POINTS = 60;
const qualityData = [];
const progressData = [];
const cascadeRiskData = [];
const tokenUsageData = [];
const energyUsageData = [];

// Event log buffer
const MAX_EVENTS = 500;
const events = [];

// State
let agentCreatedAt = null;
let cascadeEvents = 0;
let safetyBlocks = 0;

// ═══════════════════════ INITIALIZATION ═══════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initCharts();
  bindActions();
  connectWebSocket();
  startClockUpdate();
});

// ═══════════════════════ NAVIGATION ═══════════════════════
function initNavigation() {
  const navBtns = document.querySelectorAll('.nav-btn');
  const panels  = document.querySelectorAll('.panel');

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.panel;

      navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      panels.forEach(p => p.classList.remove('active'));
      document.getElementById(`panel-${target}`).classList.add('active');

      // Resize charts when panel becomes visible
      setTimeout(() => {
        Chart.getCharts?.().forEach(c => c.resize());
      }, 50);
    });
  });
}

// ═══════════════════════ WEBSOCKET ═══════════════════════
function connectWebSocket() {
  updateConnectionStatus('connecting');

  try {
    ws = new WebSocket(WS_URL);
  } catch (err) {
    updateConnectionStatus('disconnected');
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    updateConnectionStatus('connected');
    clearReconnect();
    addSystemEvent('WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleServerMessage(msg);
    } catch (err) {
      console.error('[WS] Parse error:', err);
    }
  };

  ws.onclose = () => {
    updateConnectionStatus('disconnected');
    addSystemEvent('WebSocket disconnected');
    scheduleReconnect();
  };

  ws.onerror = (err) => {
    console.error('[WS] Error:', err);
    updateConnectionStatus('disconnected');
  };
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWebSocket();
  }, 3000);
}

function clearReconnect() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function sendCommand(type, payload = {}) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.warn('[WS] Not connected. Cannot send:', type);
    return;
  }
  const id = ++requestCounter;
  ws.send(JSON.stringify({ type, id, payload }));
  return id;
}

function updateConnectionStatus(status) {
  const indicator = document.getElementById('connectionStatus');
  const dot = indicator.querySelector('.dot');
  const text = indicator.querySelector('.status-text');

  dot.className = `dot ${status}`;
  const labels = { connected: 'Connected', disconnected: 'Disconnected', connecting: 'Connecting...' };
  text.textContent = labels[status] ?? status;
}

// ═══════════════════════ MESSAGE HANDLER ═══════════════════════
function handleServerMessage(msg) {
  const { event, data, timestamp } = msg;

  // Log all events
  addEvent(event, data, timestamp);

  switch (event) {
    // ── Connection & Status ──────────────────────────────────────────
    case 'connected':
      document.getElementById('agentId').textContent = `Agent: ${data.agentStatus?.id?.slice(0, 8) ?? '—'}`;
      agentCreatedAt = data.agentStatus?.createdAt;
      updateModuleTable(data.agentStatus);
      break;

    case 'status:update':
      updateKPIs(data);
      updateModuleTable(data);
      if (data.createdAt) agentCreatedAt = data.createdAt;
      break;

    case 'metrics:snapshot':
    case 'metrics:update':
      updateMetricsTables(data);
      break;

    case 'cascade:history':
      updateCascadeChart(data);
      break;

    // ── Task Events ──────────────────────────────────────────────────
    case 'task:accepted':
      updateCurrentTask({ directive: data.directive, status: 'Running' });
      break;

    case 'task:result':
      updateCurrentTask(null);
      qualityData.push(data.quality ?? 0);
      trimBuffer(qualityData);
      updateQualityChart();
      break;

    case 'task:error':
      updateCurrentTask(null);
      break;

    case 'task:complete':
      updateCurrentTask(null);
      qualityData.push(data.quality ?? 0);
      trimBuffer(qualityData);
      updateQualityChart();
      refreshHistory();
      break;

    // ── Step Events ──────────────────────────────────────────────────
    case 'step:complete':
      progressData.push(data.progress ?? 0);
      trimBuffer(progressData);
      updateProgressChart();
      if (data.rho != null) {
        cascadeRiskData.push(data.rho);
        trimBuffer(cascadeRiskData);
        updateCascadeRiskChart();
      }
      break;

    // ── Cascade Events ──────────────────────────────────────────────
    case 'cascade:warning':
      cascadeEvents++;
      document.getElementById('kpi-cascade').textContent = cascadeEvents;
      break;

    case 'cascade:critical':
      cascadeEvents++;
      document.getElementById('kpi-cascade').textContent = cascadeEvents;
      break;

    // ── Safety Events ───────────────────────────────────────────────
    case 'safety:blocked':
    case 'safety:stepBlocked':
      safetyBlocks++;
      document.getElementById('kpi-safety').textContent = safetyBlocks;
      break;

    // ── Memory Events ───────────────────────────────────────────────
    case 'memory:recall':
    case 'memory:store':
      // Handled by event log only
      break;

    // ── Owner Events ────────────────────────────────────────────────
    case 'owner:escalation':
      updateCurrentTask({
        status: 'ESCALATED — Awaiting Owner Feedback',
        escalation: data,
      });
      break;

    case 'owner:feedback_received':
      updateCurrentTask({ status: 'Resuming after feedback...' });
      break;

    // ── Chain Events ────────────────────────────────────────────────
    case 'chain:accepted':
      updateCurrentTask({ directive: `Chain (${data.directiveCount} steps)`, status: 'Running' });
      break;

    case 'chain:result':
    case 'chain:complete':
      updateCurrentTask(null);
      refreshHistory();
      break;

    // ── Train Events ────────────────────────────────────────────────
    case 'train:start':
    case 'train:accepted':
      updateCurrentTask({ directive: 'Training Pipeline', status: 'Training' });
      break;

    case 'train:complete':
    case 'train:result':
      updateCurrentTask(null);
      break;

    // ── State Events ────────────────────────────────────────────────
    case 'state:serialized':
      // Offer download
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `predator-state-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      break;

    // ── Shutdown ────────────────────────────────────────────────────
    case 'shutdown:initiated':
      updateCurrentTask({ directive: 'Shutting down...', status: 'Shutting Down' });
      break;

    case 'shutdown:complete':
      updateConnectionStatus('disconnected');
      break;
  }
}

// ═══════════════════════ CHARTS INITIALIZATION ═══════════════════════
function initCharts() {
  // Global Chart.js defaults
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = '#1e3a5f';
  Chart.defaults.font.family = "'JetBrains Mono', 'Fira Code', monospace";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.animation.duration = 300;

  // ── Quality Chart ─────────────────────────────────────────────────
  qualityChart = new Chart(document.getElementById('chart-quality'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Quality Score',
        data: [],
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        pointHoverRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, grid: { color: '#1a2e4a' } },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { display: true, position: 'top' },
      },
    },
  });

  // ── Progress Chart ────────────────────────────────────────────────
  progressChart = new Chart(document.getElementById('chart-progress'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Step Progress',
        data: [],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, grid: { color: '#1a2e4a' } },
        x: { grid: { display: false } },
      },
    },
  });

  // ── Cascade Risk Chart ────────────────────────────────────────────
  cascadeRiskChart = new Chart(document.getElementById('chart-cascade-risk'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Cascade Risk (rho)',
        data: [],
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, grid: { color: '#1a2e4a' } },
        x: { grid: { display: false } },
      },
      plugins: {
        annotation: {
          annotations: {
            warnLine: {
              type: 'line', yMin: 0.35, yMax: 0.35,
              borderColor: 'rgba(245, 158, 11, 0.5)', borderDash: [6, 3],
              label: { content: 'Warn', display: true, position: 'start' },
            },
            criticalLine: {
              type: 'line', yMin: 0.65, yMax: 0.65,
              borderColor: 'rgba(239, 68, 68, 0.5)', borderDash: [6, 3],
              label: { content: 'Critical', display: true, position: 'start' },
            },
          },
        },
      },
    },
  });

  // ── Resource Usage Chart ──────────────────────────────────────────
  resourceUsageChart = new Chart(document.getElementById('chart-resource-usage'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Token Usage',
          data: [],
          backgroundColor: 'rgba(6, 182, 212, 0.6)',
          borderColor: '#06b6d4',
          borderWidth: 1,
        },
        {
          label: 'Energy Usage',
          data: [],
          backgroundColor: 'rgba(139, 92, 246, 0.6)',
          borderColor: '#8b5cf6',
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, grid: { color: '#1a2e4a' } },
        x: { grid: { display: false } },
      },
    },
  });

  // ── Metrics Time Series ───────────────────────────────────────────
  metricsTimeSeriesChart = new Chart(document.getElementById('chart-metrics-timeseries'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Value',
        data: [],
        borderColor: '#06b6d4',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { grid: { color: '#1a2e4a' } },
        x: { grid: { display: false } },
      },
    },
  });

  // ── Cascade Detail Chart ──────────────────────────────────────────
  cascadeDetailChart = new Chart(document.getElementById('chart-cascade-detail'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Overall Cascade Risk',
        data: [],
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, grid: { color: '#1a2e4a' } },
        x: { grid: { display: false } },
      },
    },
  });
}

// ═══════════════════════ CHART UPDATES ═══════════════════════
function trimBuffer(arr) {
  while (arr.length > MAX_DATA_POINTS) arr.shift();
}

function updateQualityChart() {
  qualityChart.data.labels = qualityData.map((_, i) => `T${i + 1}`);
  qualityChart.data.datasets[0].data = [...qualityData];
  qualityChart.update('none');
}

function updateProgressChart() {
  progressChart.data.labels = progressData.map((_, i) => `S${i + 1}`);
  progressChart.data.datasets[0].data = [...progressData];
  progressChart.update('none');
}

function updateCascadeRiskChart() {
  cascadeRiskChart.data.labels = cascadeRiskData.map((_, i) => `S${i + 1}`);
  cascadeRiskChart.data.datasets[0].data = [...cascadeRiskData];
  cascadeRiskChart.update('none');
}

function updateCascadeChart(historyData) {
  if (!Array.isArray(historyData) || historyData.length === 0) return;

  const labels = historyData.map(h => new Date(h.timestamp).toLocaleTimeString());
  const risks = historyData.map(h => h.overallRisk ?? 0);

  cascadeDetailChart.data.labels = labels;
  cascadeDetailChart.data.datasets[0].data = risks;
  cascadeDetailChart.update('none');
}

// ═══════════════════════ KPI UPDATES ═══════════════════════
function updateKPIs(status) {
  if (!status) return;

  const statusText = status.shuttingDown ? 'Shutting Down' : (status.currentTask ? 'Running' : 'Idle');
  document.getElementById('kpi-status').textContent = statusText;
  document.getElementById('kpi-status').className = `kpi-value ${status.currentTask ? 'pulse' : ''}`;

  document.getElementById('kpi-tasks').textContent = status.historyCount ?? 0;
  document.getElementById('kpi-cascade').textContent = cascadeEvents;
  document.getElementById('kpi-safety').textContent = safetyBlocks;

  if (agentCreatedAt) {
    const uptimeSec = Math.floor((Date.now() - agentCreatedAt) / 1000);
    const hours = Math.floor(uptimeSec / 3600);
    const mins = Math.floor((uptimeSec % 3600) / 60);
    const secs = uptimeSec % 60;
    document.getElementById('kpi-uptime').textContent = `${hours}h ${mins}m ${secs}s`;
  }
}

function updateModuleTable(status) {
  if (!status) return;
  const tbody = document.querySelector('#module-table tbody');
  const modules = [
    { name: 'ANNPsi Backbone',  enabled: status.backbone === 'ok' || status.backbone !== undefined },
    { name: 'Memory System',    enabled: status.memoryEnabled },
    { name: 'Safety Guardrails',enabled: status.safetyEnabled },
    { name: 'Metrics Collector',enabled: status.metricsEnabled },
    { name: 'Plugin Manager',   enabled: status.pluginsEnabled },
    { name: 'Cascade Monitor',  enabled: status.cascadeMonitor === 'ok' || status.cascadeMonitor !== undefined },
  ];

  tbody.innerHTML = modules.map(m => `
    <tr>
      <td>${m.name}</td>
      <td class="${m.enabled ? 'status-enabled' : 'status-disabled'}">${m.enabled ? 'Enabled' : 'Disabled'}</td>
    </tr>
  `).join('');
}

// ═══════════════════════ METRICS TABLES ═══════════════════════
function updateMetricsTables(data) {
  if (!data) return;

  // Counters
  const countersTbody = document.querySelector('#metrics-counters tbody');
  if (data.counters) {
    countersTbody.innerHTML = Object.entries(data.counters).map(([k, v]) =>
      `<tr><td>${k}</td><td>${v}</td></tr>`
    ).join('');
  }

  // Gauges
  const gaugesTbody = document.querySelector('#metrics-gauges tbody');
  if (data.gauges) {
    gaugesTbody.innerHTML = Object.entries(data.gauges).map(([k, v]) =>
      `<tr><td>${k}</td><td>${typeof v === 'number' ? v.toFixed(4) : v}</td></tr>`
    ).join('');
  }

  // Histograms
  const histTbody = document.querySelector('#metrics-histograms tbody');
  if (data.histograms) {
    histTbody.innerHTML = Object.entries(data.histograms).map(([k, v]) => {
      if (!v) return '';
      return `<tr>
        <td>${k}</td>
        <td>${v.count}</td>
        <td>${v.mean?.toFixed(4) ?? '—'}</td>
        <td>${v.p50?.toFixed(4) ?? '—'}</td>
        <td>${v.p95?.toFixed(4) ?? '—'}</td>
        <td>${v.p99?.toFixed(4) ?? '—'}</td>
      </tr>`;
    }).join('');
  }

  // Timers
  const timersTbody = document.querySelector('#metrics-timers tbody');
  if (data.timers) {
    timersTbody.innerHTML = Object.entries(data.timers).map(([k, v]) =>
      `<tr>
        <td>${k}</td>
        <td>${v.count}</td>
        <td>${v.mean?.toFixed(1) ?? '—'}</td>
        <td>${v.sum?.toFixed(1) ?? '—'}</td>
      </tr>`
    ).join('');
  }

  // Update average quality KPI from histogram
  if (data.histograms?.['task.quality']) {
    const avg = data.histograms['task.quality'].mean;
    if (avg != null) {
      document.getElementById('kpi-quality').textContent = avg.toFixed(3);
    }
  }
}

// ═══════════════════════ CURRENT TASK ═══════════════════════
function updateCurrentTask(info) {
  const display = document.getElementById('current-task-display');

  if (!info) {
    display.innerHTML = '<p class="placeholder">No task currently running.</p>';
    return;
  }

  let html = '';

  if (info.directive) {
    html += `<div class="task-info-row">
      <span class="task-info-label">Directive</span>
      <span class="task-info-value">${escapeHtml(info.directive)}</span>
    </div>`;
  }

  html += `<div class="task-info-row">
    <span class="task-info-label">Status</span>
    <span class="task-info-value ${info.status === 'Running' ? 'pulse' : ''}">${escapeHtml(info.status ?? 'Unknown')}</span>
  </div>`;

  if (info.escalation) {
    html += `<div class="task-info-row">
      <span class="task-info-label">Escalation</span>
      <span class="task-info-value" style="color: var(--accent-pink);">
        rho = ${info.escalation.rho?.toFixed(4) ?? '—'} — Owner input required
      </span>
    </div>`;
  }

  display.innerHTML = html;
}

// ═══════════════════════ EVENT LOG ═══════════════════════
function addEvent(eventName, data, timestamp) {
  const ts = timestamp ?? Date.now();
  const timeStr = new Date(ts).toLocaleTimeString();

  events.push({ time: timeStr, name: eventName, data });
  if (events.length > MAX_EVENTS) events.shift();

  renderEventLog();
}

function addSystemEvent(message) {
  events.push({ time: new Date().toLocaleTimeString(), name: 'system', data: { message } });
  renderEventLog();
}

function renderEventLog() {
  const container = document.getElementById('event-log');
  const autoScroll = document.getElementById('auto-scroll')?.checked;

  const eventClass = (name) => {
    const prefix = name.split(':')[0];
    return `e-${name}`;
  };

  container.innerHTML = events.slice(-100).map(e => `
    <div class="event-entry">
      <span class="event-time">${e.time}</span>
      <span class="event-name ${eventClass(e.name)}">${e.name}</span>
      <span class="event-data">${truncate(JSON.stringify(e.data), 120)}</span>
    </div>
  `).join('');

  if (autoScroll) {
    container.scrollTop = container.scrollHeight;
  }
}

// ═══════════════════════ HISTORY ═══════════════════════
async function refreshHistory() {
  try {
    const res = await fetch('/api/history');
    const json = await res.json();
    if (!json.ok) return;

    const container = document.getElementById('history-list');
    const items = json.data;

    if (items.length === 0) {
      container.innerHTML = '<p class="placeholder">No tasks executed yet.</p>';
      return;
    }

    container.innerHTML = items.slice(-20).reverse().map(item => `
      <div class="history-entry">
        <span class="history-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
        <span>${escapeHtml(item.directive?.slice(0, 60) ?? '—')}</span>
        <span style="float:right; color: ${item.status === 'success' ? 'var(--status-ok)' : 'var(--text-muted)'}">
          q=${item.quality?.toFixed(3) ?? '—'}
        </span>
      </div>
    `).join('');
  } catch (err) {
    console.error('[History] Fetch error:', err);
  }
}

// ═══════════════════════ CASCADE PANEL ═══════════════════════
async function loadCascadeData() {
  try {
    const [cascadeRes, historyRes, extRes, intRes] = await Promise.all([
      fetch('/api/cascade'),
      fetch('/api/cascade/history?limit=50'),
      fetch('/api/cascade/extinctions'),
      fetch('/api/cascade/interventions'),
    ]);

    const cascade = await cascadeRes.json();
    const history = await historyRes.json();
    const extinctions = await extRes.json();
    const interventions = await intRes.json();

    // Thresholds
    if (cascade.ok && cascade.data) {
      const d = cascade.data;
      document.getElementById('cascade-thresholds').innerHTML = `
        <div class="threshold-row warn">
          <span class="threshold-label">Warning</span>
          <span class="threshold-value">${d.rhoWarn}</span>
        </div>
        <div class="threshold-row moderate">
          <span class="threshold-label">Moderate</span>
          <span class="threshold-value">${d.rhoModerate}</span>
        </div>
        <div class="threshold-row critical">
          <span class="threshold-label">Critical</span>
          <span class="threshold-value">${d.rhoCritical}</span>
        </div>
        <div class="threshold-row">
          <span class="threshold-label">Self-Healing</span>
          <span class="threshold-value">${d.selfHealing ? 'Enabled' : 'Disabled'}</span>
        </div>
        <div class="threshold-row">
          <span class="threshold-label">Extinctions</span>
          <span class="threshold-value">${d.extinctionCount}</span>
        </div>
        <div class="threshold-row">
          <span class="threshold-label">Interventions</span>
          <span class="threshold-value">${d.interventionCount}</span>
        </div>
      `;

      // Layer risk trends
      if (d.layerRiskTrends && Object.keys(d.layerRiskTrends).length > 0) {
        document.getElementById('layer-risk-trends').innerHTML = Object.entries(d.layerRiskTrends).map(([layer, info]) => {
          const riskClass = info.current >= 0.65 ? 'risk-high' : info.current >= 0.35 ? 'risk-moderate' : 'risk-low';
          return `<div class="layer-risk-item">
            <span class="layer-name">${layer}</span>
            <span class="risk-value ${riskClass}">${info.current.toFixed(4)} (trend: ${info.trend >= 0 ? '+' : ''}${info.trend.toFixed(4)})</span>
          </div>`;
        }).join('');
      }
    }

    // Cascade risk history chart
    if (history.ok && history.data) {
      updateCascadeChart(history.data);
    }

    // Extinction log
    if (extinctions.ok && extinctions.data) {
      const extContainer = document.getElementById('extinction-log');
      if (extinctions.data.length === 0) {
        extContainer.innerHTML = '<p class="placeholder">No extinctions recorded.</p>';
      } else {
        extContainer.innerHTML = extinctions.data.map(e => `
          <div class="log-entry">
            <span class="log-time">${new Date(e.t).toLocaleTimeString()}</span>
            Unit ${e.unitId} — ${e.cause} (extinctions: ${e.extinctions})
          </div>
        `).join('');
      }
    }

    // Intervention log
    if (interventions.ok && interventions.data) {
      const intContainer = document.getElementById('intervention-log');
      if (interventions.data.length === 0) {
        intContainer.innerHTML = '<p class="placeholder">No interventions recorded.</p>';
      } else {
        intContainer.innerHTML = interventions.data.map(i => `
          <div class="log-entry">
            <span class="log-time">${new Date(i.t).toLocaleTimeString()}</span>
            <span style="color: var(--accent-orange)">${i.type}</span> — risk: ${i.risk?.toFixed(4) ?? '—'}
          </div>
        `).join('');
      }
    }
  } catch (err) {
    console.error('[Cascade] Load error:', err);
  }
}

// ═══════════════════════ MEMORY PANEL ═══════════════════════
async function searchMemory(query) {
  const resultsDiv = document.getElementById('memory-results');
  if (!query.trim()) {
    resultsDiv.innerHTML = '<p class="placeholder">Enter a query to search agent memory.</p>';
    return;
  }

  try {
    const res = await fetch(`/api/memory/${encodeURIComponent(query)}?limit=20`);
    const json = await res.json();

    if (!json.ok) {
      resultsDiv.innerHTML = `<p class="placeholder">Error: ${json.error}</p>`;
      return;
    }

    if (!json.data || json.data.length === 0) {
      resultsDiv.innerHTML = '<p class="placeholder">No memories found.</p>';
      return;
    }

    resultsDiv.innerHTML = json.data.map(m => `
      <div class="memory-item">
        <span class="memory-score">${m.score?.toFixed(3) ?? '—'}</span>
        <span>${escapeHtml(JSON.stringify(m).slice(0, 200))}</span>
      </div>
    `).join('');
  } catch (err) {
    resultsDiv.innerHTML = `<p class="placeholder">Error: ${err.message}</p>`;
  }
}

// ═══════════════════════ BUTTON BINDINGS ═══════════════════════
function bindActions() {
  // Task execution
  document.getElementById('btn-execute').addEventListener('click', () => {
    const directive = document.getElementById('directive-input').value.trim();
    if (!directive) return;

    let budget;
    const budgetStr = document.getElementById('budget-input').value.trim();
    if (budgetStr) {
      try { budget = JSON.parse(budgetStr); } catch { budget = undefined; }
    }

    sendCommand('task:execute', { directive, budget });
  });

  // Chain execution
  document.getElementById('btn-chain').addEventListener('click', () => {
    const text = document.getElementById('chain-input').value.trim();
    if (!text) return;
    const directives = text.split('\n').map(s => s.trim()).filter(Boolean);
    sendCommand('chain:execute', { directives });
  });

  // Owner feedback
  document.getElementById('btn-feedback').addEventListener('click', () => {
    const text = document.getElementById('feedback-input').value.trim();
    if (!text) return;
    let feedback;
    try { feedback = JSON.parse(text); } catch { feedback = { instruction: text }; }
    sendCommand('owner:feedback', { feedback });
  });

  // Serialize state
  document.getElementById('btn-serialize').addEventListener('click', () => {
    sendCommand('state:serialize');
  });

  // Start training
  document.getElementById('btn-train').addEventListener('click', () => {
    sendCommand('train:start', { config: { epochs: 10, learningRate: 0.001 } });
  });

  // Shutdown
  document.getElementById('btn-shutdown').addEventListener('click', () => {
    if (confirm('Are you sure you want to shutdown the PREDATOR agent?')) {
      sendCommand('shutdown');
    }
  });

  // Clear events
  document.getElementById('btn-clear-events').addEventListener('click', () => {
    events.length = 0;
    renderEventLog();
  });

  // Metrics reset
  document.getElementById('btn-metrics-reset').addEventListener('click', async () => {
    try {
      await fetch('/api/metrics/reset', { method: 'POST' });
    } catch (err) {
      console.error('[Metrics] Reset error:', err);
    }
  });

  // Memory search
  document.getElementById('btn-memory-search').addEventListener('click', () => {
    searchMemory(document.getElementById('memory-query').value);
  });

  document.getElementById('memory-query').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      searchMemory(e.target.value);
    }
  });

  // Load cascade data when switching to that panel
  document.querySelector('[data-panel="cascade"]').addEventListener('click', () => {
    loadCascadeData();
  });

  // Load history when switching to overview
  document.querySelector('[data-panel="overview"]').addEventListener('click', () => {
    refreshHistory();
  });

  // Load metrics when switching to metrics panel
  document.querySelector('[data-panel="metrics"]').addEventListener('click', () => {
    fetchMetricsSnapshot();
  });
}

async function fetchMetricsSnapshot() {
  try {
    const res = await fetch('/api/metrics');
    const json = await res.json();
    if (json.ok) updateMetricsTables(json.data);
  } catch (_) {}
}

// ═══════════════════════ CLOCK ═══════════════════════
function startClockUpdate() {
  const update = () => {
    document.getElementById('footer-time').textContent = new Date().toLocaleString();
  };
  update();
  setInterval(update, 1000);
}

// ═══════════════════════ HELPERS ═══════════════════════
function escapeHtml(str) {
  if (typeof str !== 'string') str = String(str);
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function truncate(str, maxLen) {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 3) + '...';
}
