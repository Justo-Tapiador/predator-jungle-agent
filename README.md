# PREDATOR JUNGLE v2.0

### Praxic Reinforcement and Extinction-Driven Agentic Task Orchestrator and Realizer

> A Hierarchically Governed Deep Agentic AI System built on the Artificial Junky Neuron (AJN) Framework — **Enhanced v2.0** with real transformers, persistent memory, LLM integration, safety guardrails, and plugin architecture.

[![Node.js Version](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com/Justo-Tapiador/predator-jungle-agent)

---

<div align="center">
<div style="font-size:18pt;text-align:center;">🚨<b>WARNING: This Predator Agent needs  to be <br> trained in order to function correctly! </b>🚨</div>
<img src="https://raw.githubusercontent.com/justo-tapiador/predator-jungle-agent/main/web/public/predator-5.jpg" alt="Train me Please!"  width="512"/>
</div>

## What's New in v2.0

This is a major enhancement of the original PREDATOR agent system. Every component has been improved, and several entirely new subsystems have been added.

### Core Neural Architecture Improvements

| Feature | v0.1 | v2.0 |
|---------|------|------|
| **Transformer layers** | Simulated (`tanh` + random noise) | Real multi-head self-attention with Q/K/V projections, layer norm, GELU activation |
| **AJN gradient updates** | Simple gradient approximation | Momentum-based optimization with cosine annealing LR |
| **AJN stability** | No phase hysteresis | Hysteresis bands prevent phase oscillation |
| **Experience learning** | None | Experience replay buffer for stable learning |
| **Inter-unit learning** | None | Hebbian co-activation traces |
| **Entropy control** | None | Entropy-regularized praxis sampling prevents premature collapse |

### New Subsystems

| Subsystem | Description |
|-----------|-------------|
| **Persistent Memory** | Episodic, semantic, and working memory with vector similarity retrieval |
| **Safety Guardrails** | Pre-execution safety checks with 3 severity levels (permissive/standard/strict) |
| **Metrics Collector** | Counters, gauges, histograms, timers, and time-series for full observability |
| **Plugin Architecture** | Hook-based plugin system with priority ordering and cancellation support |
| **LLM Integration** | OpenAI (z-ai-sdk) and local (Ollama) LLM adapters for semantic directive understanding |
| **Real Tool Implementations** | FileSystem, WebSearch, CodeExecution, API, Database tools replacing stubs |
| **Tool Registry** | Centralized tool management with schema validation and PSE integration |
| **State Serialization** | Full agent checkpointing with versioned JSON files and base64-encoded Float64Arrays |
| **Dataset Loader** | JSON/CSV loading, synthetic data generation, data augmentation, and batched iteration |
| **Checkpoint Manager** | Training state persistence with auto-pruning and export capabilities |
| **Configuration System** | Zod-validated config with environment overrides and multi-environment support |
| **Docker Support** | Multi-stage Dockerfile and docker-compose with optional Ollama sidecar |

---

## Architecture

```
Owner Directive (natural language)
        |
        v
+-----------------------------------------+
|  Hierarchical Command Interpreter (HCI) |  <-- LLM-enhanced parsing + task decomposition
+-----------------------------------------+
        |  addiction target injection
        v
+-----------------------------------------+
|  ANN-Psi Backbone (12 layers)           |
|                                         |
|  L1-L2  Hybrid AJN  (sensory encoding)  |
|  L3     Hetero AJN  K=8  (features)     |
|  L4-L5  REAL Transformer (context attn) |  <-- NEW: Real self-attention
|  L6     Hetero AJN  K=16 (concepts)     |
|  L7     Hybrid AJN  (modulation)        |
|  L8-L9  REAL Transformer (reasoning)    |  <-- NEW: Real self-attention
|  L10    Hetero AJN  K=32 (high-order)   |
|  L11    Hybrid AJN  (praxic assembly)   |
|  L12    Output AJN  (TPS emission)      |
+-----------------------------------------+
        |  praxis tensors
        v
+-----------------------------------------+
|  Safety Guardrails                      |  <-- NEW: Pre-execution safety checks
+-----------------------------------------+
        |
        v
+-----------------------------------------+
|  Token-Energy Arbitrator (TEA)          |  <-- PID-controlled emission rate
+-----------------------------------------+
        |  filtered praxis stream
        v
+-----------------------------------------+
|  Praxic Stream Executor (PSE)           |  <-- Retry logic, timeout, parallel exec
+-----------------------------------------+
        |
        v
   Environment (tools, APIs, file system)
        |  feedback
        +-------------------------------------> Memory System --> ANN-Psi (next step)
                                             Metrics Collector --> Observability
                                             Plugin Hooks --> Extensibility
```

---

## Installation

```bash
git clone https://github.com/Justo-Tapiador/predator-jungle-agent.git
cd predator-jungle-agent
npm install
```

Requires **Node.js >= 18.0.0**.

### Docker

```bash
docker build -t predator-jungle-agent -f docker/Dockerfile .
docker run -p 3000:3000 -v $(pwd)/data:/app/data predator-jungle-agent
```

### With Local LLM (Ollama)

```bash
docker-compose --profile llm up
```

---

## Quick Start

```bash
# Run a task directly
node scripts/cli.js task "Debug all failing unit tests in the auth module" --priority critical

# Interactive demo (3 tasks, full training)
node scripts/cli.js demo

# Web dashboard (http://localhost:3000)
node scripts/cli.js web

# Full benchmark suite
npm run benchmark

# Manage checkpoints
node scripts/cli.js checkpoint save --label "phase1_done"
node scripts/cli.js checkpoint list
node scripts/cli.js checkpoint load --id <checkpoint-id>
```

---

## Programmatic API

### Basic Usage

```javascript
import { Predator } from './src/index.js';

const agent = new Predator({
  enableMemory: true,
  enableSafety: true,
  enableMetrics: true,
  safetyLevel: 'standard',
});

// Train all 4 phases
await agent.train();

// Execute an Owner directive
const result = await agent.execute(
  'Implement and test a rate-limiting middleware',
  { priority: 'expedited', budget: { tokens: 30000, energy: 1.0 } }
);

console.log(`Quality: ${(result.quality * 100).toFixed(1)}%`);
console.log(`Tokens used: ${result.tokenUsage.tokensUsed}`);
```

### With LLM Integration

```javascript
import { Predator, OpenAIAdapter } from './src/index.js';

const llm = new OpenAIAdapter({ model: 'gpt-4', temperature: 0.7 });
const agent = new Predator({ llmAdapter: llm });

// HCI will now use LLM for semantic directive understanding
const result = await agent.execute('Analyze and improve performance of the search algorithm');
```

### With Custom Tools

```javascript
import { Predator, ToolRegistry, FileSystemTool, DatabaseTool } from './src/index.js';

const agent = new Predator();
const registry = new ToolRegistry();
registry.registerMany([new FileSystemTool(), new DatabaseTool()]);

// Register custom tool
agent.registerTool('my_api', async (args) => {
  const response = await fetch(args.url);
  return { status: response.status, data: await response.json() };
});
```

### With Plugins

```javascript
import { Predator } from './src/index.js';

const agent = new Predator();

agent.use({
  name: 'my-plugin',
  version: '1.0.0',
  hooks: {
    taskComplete: (result) => {
      console.log(`Task completed with quality: ${result.quality}`);
    },
    beforeStep: (context) => {
      // Modify context before each step
      return context;
    },
    extinction: (event) => {
      console.warn(`Extinction on unit ${event.id}`);
    },
  },
});
```

### State Persistence

```javascript
import { Predator, StateSerializer } from './src/index.js';

const agent = new Predator();
await agent.train();

// Save checkpoint
const serializer = new StateSerializer({ checkpointDir: './checkpoints' });
await serializer.save(agent, 'after_training');

// Later, restore from checkpoint
const agent2 = new Predator();
await serializer.load(agent2, 'after_training');
```

---

## Training Pipeline

The 4-phase training pipeline has been enhanced with:

- **Configurable learning rate schedules**: cosine, linear, step, constant
- **Early stopping**: with configurable patience per phase
- **Checkpointing**: automatic state saves after each phase
- **Data augmentation**: noise injection, permutation, adversarial perturbation
- **Dataset loading**: JSON, CSV, and synthetic data generation
- **Metrics recording**: loss curves, saturation rates, resilience scores

```javascript
await agent.train({
  epochsI:     10,   // Phase I:  Large-scale pre-training
  epochsII_T1:  5,   // Phase II: Addiction seeding
  epochsII_T2:  5,   //           Tolerance building
  epochsII_T3:  5,   //           Frustration hardening
  epochsIII:    8,   // Phase III: Hierarchical fine-tuning (HIFT)
  epochsIV:     6,   // Phase IV:  Adversarial frustration hardening
  enableCheckpoints: true,
  earlyStoppingPatience: 5,
});
```

---

## New Modules Deep Dive

### Memory System

Three-tier memory architecture for persistent knowledge across tasks:

- **Episodic Memory**: Stores task execution records with vector embeddings for similarity-based retrieval
- **Semantic Memory**: Key-value store for extracted facts and patterns with confidence scores
- **Working Memory**: Short-term buffer with TTL support for current task context

```javascript
const mem = new MemorySystem({ storageDir: './data/memory' });
await mem.init();

// Store task experience
await mem.store('debug auth module', { quality: 0.85, steps: 12 }, {});

// Recall relevant memories
const memories = await mem.recall('authentication bug', { limit: 5, minSimilarity: 0.6 });

// Store semantic facts
await mem.storeSemantic('auth_module_path', '/src/auth/', 0.9);
```

### Safety Guardrails

Pre-execution safety filter with three severity levels:

| Level | Behavior |
|-------|----------|
| **Permissive** | Minimal checks, no budget caps |
| **Standard** | Full checks, protected paths, rate limits |
| **Strict** | All checks + rollback plans required for destructive actions |

```javascript
const safety = new SafetyGuardrails({
  safetyLevel: 'strict',
  protectedPaths: ['/etc', '/root', '/production'],
  rateLimits: { perMinute: 30, perHour: 200 },
});
```

### Metrics Collector

Full observability stack with five metric types:

```javascript
const metrics = new MetricsCollector({ enableConsole: true });
metrics.start();

metrics.incrementCounter('tasks_completed');
metrics.setGauge('current_craving', 0.65);
metrics.observeHistogram('quality', 0.85);
metrics.startTimer('task_execution');
// ... later
metrics.stopTimer('task_execution', timerId);

const summary = metrics.getSummary();
```

### Plugin System

Hook-based plugin architecture with priority ordering and cancellation:

| Hook | Cancellable | Description |
|------|-------------|-------------|
| `beforeStep` | Yes | Before each TPS step |
| `afterStep` | No | After each TPS step |
| `beforeEmit` | Yes | Before emitting a praxis |
| `afterEmit` | No | After emitting a praxis |
| `taskComplete` | No | When a task completes |
| `directiveReceived` | Yes | When a directive is received |
| `extinction` | No | On extinction events |
| `trainingEpoch` | No | After each training epoch |

---

## Real Tool Implementations

All v0.1 stub tools have been replaced with functional implementations:

| Tool | Capabilities |
|------|-------------|
| **FileSystemTool** | Read, write, append, delete, list, stat, mkdir, copy, move, search |
| **WebSearchTool** | Web search (z-ai-sdk + DuckDuckGo fallback), page fetch, link extraction |
| **CodeExecutionTool** | Sandboxed JS execution via `vm` module, shell commands with safety checks |
| **APICallTool** | GET, POST, PUT, DELETE with timeout and retry |
| **DatabaseTool** | In-memory SQL with full CRUD, optional better-sqlite3 backend |
| **ToolRegistry** | Central registration, schema validation, PSE handler generation |

---

## LLM Integration

### OpenAI Adapter (via z-ai-web-dev-sdk)

```javascript
import { OpenAIAdapter } from './src/llm/OpenAIAdapter.js';

const llm = new OpenAIAdapter({ model: 'gpt-4', temperature: 0.7 });
const response = await llm.chat('Analyze this task directive...', 'You are a helpful assistant.');
const classification = await llm.classify('Debug the auth module', ['coding', 'research', 'ops']);
```

### Local LLM Adapter (Ollama / llama.cpp)

```javascript
import { LocalLLMAdapter } from './src/llm/LocalLLMAdapter.js';

const llm = new LocalLLMAdapter({ endpoint: 'http://localhost:11434', modelName: 'llama2' });
const response = await llm.chat('Analyze this directive...');
```

---

## Configuration

Configuration is managed through a layered system:

1. **`config/default.json`** — Default values
2. **`config/production.json`** — Environment-specific overrides
3. **`.env`** — Environment variable overrides
4. **Zod schema validation** — Runtime type checking

```bash
# Copy and customize environment
cp .env.example .env
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `none` | LLM backend: none, openai, local |
| `LLM_ENDPOINT` | `http://localhost:11434` | Local LLM endpoint |
| `SAFETY_LEVEL` | `standard` | Safety guardrail level |
| `MEMORY_DIR` | `./data/memory` | Memory persistence directory |
| `PORT` | `3000` | Web dashboard port |
| `LOG_LEVEL` | `info` | Logging verbosity |

---

## Project Structure

```
predator-jungle-agent/
+-- src/
|   +-- core/
|   |   +-- ArtificialJunkyNeuron.js   # Enhanced AJN (momentum, replay, Hebbian)
|   |   +-- ANNPsi.js                  # 12-layer backbone with real transformers
|   |   +-- Predator.js                # Main orchestrator (memory, safety, metrics, plugins)
|   |   +-- StateSerializer.js         # NEW: Checkpoint save/load
|   +-- layers/
|   |   +-- AJNLayer.js                # Enhanced with EMA cascade risk, serialization
|   |   +-- TransformerBlock.js        # NEW: Real multi-head self-attention
|   +-- modules/
|   |   +-- HierarchicalCommandInterpreter.js  # LLM-enhanced, task decomposition
|   |   +-- TokenEnergyArbitrator.js           # PID-controlled, energy recycling
|   |   +-- PraxicStreamExecutor.js            # Retry logic, parallel execution
|   |   +-- CascadeMonitor.js                  # Predictive, self-healing
|   |   +-- MemorySystem.js                    # NEW: 3-tier persistent memory
|   |   +-- SafetyGuardrails.js                # NEW: Pre-execution safety
|   |   +-- MetricsCollector.js                # NEW: Full observability
|   |   +-- PluginManager.js                   # NEW: Hook-based plugins
|   +-- tools/
|   |   +-- FileSystemTool.js          # NEW: Real file operations
|   |   +-- WebSearchTool.js           # NEW: Real web search
|   |   +-- CodeExecutionTool.js       # NEW: Sandboxed code execution
|   |   +-- APICallTool.js             # NEW: HTTP client
|   |   +-- DatabaseTool.js            # NEW: SQL database operations
|   |   +-- ToolRegistry.js            # NEW: Central tool management
|   +-- training/
|   |   +-- TrainingPipeline.js        # Enhanced: LR schedules, early stopping, checkpoints
|   |   +-- DatasetLoader.js           # NEW: Data loading and augmentation
|   |   +-- CheckpointManager.js       # NEW: Training state persistence
|   +-- llm/
|   |   +-- LLMAdapter.js             # NEW: Abstract LLM interface
|   |   +-- OpenAIAdapter.js           # NEW: z-ai-web-dev-sdk integration
|   |   +-- LocalLLMAdapter.js         # NEW: Ollama/llama.cpp support
|   +-- index.js                       # Public exports
+-- config/
|   +-- default.json                   # Default configuration
|   +-- production.json                # Production overrides
|   +-- schema.js                      # Zod validation schema
+-- plugins/
|   +-- example-plugin.js              # Example plugins (task logger, budget alert)
+-- scripts/
|   +-- cli.js                         # Enhanced CLI
|   +-- benchmark.js                   # Enhanced benchmarks
|   +-- lib/
|       +-- demo.js
|       +-- format.js
+-- web/
|   +-- server.js                      # Express + WebSocket server
|   +-- public/
|       +-- index.html
+-- tests/
|   +-- predator.test.js               # Comprehensive test suite
+-- docker/
|   +-- Dockerfile                     # Multi-stage production build
|   +-- docker-compose.yml             # With optional Ollama sidecar
+-- .env.example                       # Environment variable template
+-- package.json
```

---

## Complete Improvement Summary

### Neural Architecture (6 improvements)
1. **Real Transformer Blocks** — Multi-head self-attention with Q/K/V projections, layer normalization, GELU activation, Xavier initialization, positional encoding, residual connections
2. **Momentum-based AJN Optimization** — Velocity buffers for mu and sigma updates with momentum coefficient
3. **Cosine Annealing Learning Rate** — Adaptive LR that oscillates between eta and etaMin for better convergence
4. **Experience Replay Buffer** — Stores past (stimulus, praxis, reward) tuples and replays them for stable learning
5. **Hebbian Co-activation Traces** — Inter-unit learning that strengthens connections between co-activated neurons
6. **Phase Transition Hysteresis** — Prevents oscillation between phases near transition boundaries

### New Subsystems (8 new modules)
7. **Persistent Memory System** — Episodic, semantic, and working memory with vector similarity retrieval and automatic consolidation
8. **Safety Guardrails** — Pre-execution safety checks with protected paths, blocked commands, rate limits, and 3 severity levels
9. **Metrics Collector** — Full observability with counters, gauges, histograms, timers, and time-series data
10. **Plugin Architecture** — Hook-based plugin system with priority ordering, cancellation, and lifecycle management
11. **LLM Integration** — Abstract adapter with OpenAI (z-ai-sdk) and local (Ollama) implementations
12. **State Serialization** — Full agent checkpointing with versioned JSON files and Float64Array compression
13. **Dataset Loader** — JSON/CSV loading, synthetic generation, validation, augmentation, and batched iteration
14. **Checkpoint Manager** — Training state persistence with auto-pruning and export

### Tool Implementations (6 real tools)
15. **FileSystemTool** — Complete file I/O with sandbox directory, protected paths, size limits
16. **WebSearchTool** — z-ai-sdk integration with DuckDuckGo fallback
17. **CodeExecutionTool** — Sandboxed JS via `vm` module with safety restrictions
18. **APICallTool** — Full HTTP client with timeout, retry, and streaming downloads
19. **DatabaseTool** — In-memory SQL with optional better-sqlite3 backend
20. **ToolRegistry** — Central management with schema validation and PSE integration

### Module Enhancements (7 improvements)
21. **LLM-Enhanced HCI** — Semantic directive parsing, task decomposition, extended constraint patterns with severity levels
22. **PID-Controlled TEA** — Proportional-integral-derivative controller for smoother budget consumption, energy recycling
23. **Enhanced PSE** — Retry logic with exponential backoff, timeout enforcement, parallel execution, structured audit log
24. **Predictive Cascade Monitor** — Trend analysis, graduated interventions, self-healing with learned stimulus patterns
25. **Enhanced Training Pipeline** — LR schedules, early stopping, data augmentation, per-phase checkpointing
26. **EMA Cascade Risk** — Exponential moving average for smoother cascade risk estimation in AJN layers
27. **Softmax Temperature Competition** — Temperature-scaled class selection in heterogeneous layers

### Infrastructure (6 improvements)
28. **Configuration System** — Zod-validated config with multi-environment support and env variable overrides
29. **Docker Support** — Multi-stage Dockerfile with non-root user, health checks, and docker-compose with Ollama sidecar
30. **Enhanced CLI** — New commands for checkpoint management, tool listing, safety configuration
31. **Comprehensive Test Suite** — Tests for all new modules including Memory, Safety, Metrics, Plugins, Tools
32. **Example Plugins** — Task logger and token budget alert plugins demonstrating the plugin architecture
33. **Environment Configuration** — `.env.example` with all configurable parameters documented

---

## The AJN Six-Phase Lifecycle

Each neuron autonomously cycles through:

| Phase | Name | Behaviour |
|-------|------|-----------|
| 1 | **Random** | High-entropy exploration |
| 2 | **Reinforce** | Bias developing toward stimulus (with momentum) |
| 3 | **Saturation** | Craving satisfied; praxes suppressed |
| 4 | **Withdrawal** | Threshold decaying; craving returns |
| 5 | **Frustration** | Failure; covariance expanding chaotically |
| 6 | **Extinction** | Addiction dissolved; full reset |

---

## Theoretical Basis

PREDATOR's efficiency advantages derive directly from AJN saturation dynamics:

- **Token suppression in saturation** — During Phase 3, `||P_t||_F -> 0`, so near-zero tokens are emitted when the task is progressing well.
- **Chaotic intensification in frustration** — During Phase 5, covariance expands, generating high-diversity praxes that break stuck states.
- **Cascade prevention** — Lateral inhibition coupling reduces group extinction probability below the independent bound.
- **Adaptive compute** — The TEA combines PID-controlled energy-aware suppression with craving-driven override.
- **Self-healing** — The Cascade Monitor learns successful stimulus patterns and injects them proactively when risk trends upward.

**Primary References:**
- Tapiador Garcia, J. (2024). *Agentic Theory: Definition of the Artificial Junky Neuron (AJN).* WALLERMAX-AI 2604.00012.
- Tapiador Garcia, J. (2024). *Agentic Theory II: The AJN and ANN-Psi.* WALLERMAX-AI 2604.00013.
- Tapiador Garcia, J. (2024). *Agentic Theory III: Stimulus Tensor Propagation.* WALLERMAX-AI 2604.00014.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

**v2.0 Enhancements (c) 2024-2026. Based on Agentic Theory by Justo Tapiador Garcia (UA).**
