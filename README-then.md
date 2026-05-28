# PREDATOR-CG

# Creative & Generative Specialization

> **P**raxic **R**einforcement and **E**xtinction-**D**riven **A**gentic **T**ask **O**rchestrator and **R**ealizer
> — **Creative & Generative Edition**

A deep agentic AI system built on the **Artificial Junky Neuron (AJN)** framework, specialized for **creative writing, generative art, music composition, storytelling, brainstorming, and open-ended ideation**. Fully implemented in Node.js.

---

**Based on the Agentic Theory by Justo Tapiador García**
Universidad de Alicante (UA)
Preprints: WALLERMAX-AI 2604.00012 · 2604.00013 · 2604.00014

---

## Table of Contents

- [What Makes PREDATOR-CG Different?](#what-makes-predator-cg-different)
- [Creative Task Domains](#creative-task-domains)
- [Architecture](#architecture)
- [New Features for Creative Tasks](#new-features-for-creative-tasks)
- [The AJN Six-Phase Lifecycle](#the-ajn-six-phase-lifecycle)
- [Creative-Optimized Training Pipeline](#creative-optimized-training-pipeline)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Programmatic API](#programmatic-api)
- [Creative Domain Examples](#creative-domain-examples)
- [LLM Integration](#llm-integration)
- [Web Dashboard](#web-dashboard)
- [Safety Guardrails for Creative Agents](#safety-guardrails-for-creative-agents)
- [Memory System for Creative Context](#memory-system-for-creative-context)
- [Plugin Architecture](#plugin-architecture)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Theoretical Basis](#theoretical-basis)
- [License](#license)

---

## What Makes PREDATOR-CG Different?

PREDATOR-CG is a creative-specialized variant of the base PREDATOR agent. While the original PREDATOR excels at structured task orchestration (debugging, refactoring, testing), PREDATOR-CG is tuned, trained, and equipped for **open-ended creative generation** where there is no single correct answer.

The key differences are:

| Aspect | Base PREDATOR | PREDATOR-CG |
|--------|--------------|-------------|
| **Primary domain** | Structured task execution | Creative & generative tasks |
| **Quality metric** | Task completion, success rate | Novelty, coherence, aesthetic value |
| **Cascade tolerance** | Low — early intervention | Higher — creative chaos is productive |
| **Extinction behaviour** | Reset and retry | Divergent exploration before reset |
| **Training emphasis** | Convergence, precision | Divergence, diversity, style consistency |
| **Memory priority** | Error patterns, tool results | Style signatures, narrative arcs, aesthetic preferences |
| **Safety focus** | Prevent destructive actions | Prevent plagiarism, content policy violations |
| **TEA budget** | Token-efficient (suppress in saturation) | Token-generous (explore in saturation) |
| **LLM role** | Directive parsing | Co-creative partner, style transfer |

PREDATOR-CG leverages the same AJN addiction dynamics but flips the interpretation: **frustration phases become creative exploration**, **saturation becomes stylistic coherence**, and **extinction becomes a pivot to a new creative direction**.

---

## Creative Task Domains

PREDATOR-CG is designed to excel in the following creative domains:

### Creative Writing & Storytelling
Generate short stories, novels, screenplays, poetry, and interactive fiction. The agent maintains narrative coherence across long-form content through its persistent memory system, tracking character arcs, plot threads, and thematic consistency. The AJN's frustration phase drives unexpected plot twists and creative divergence when narratives stall, while saturation ensures tonal consistency within scenes.

### Generative Art & Visual Design
Produce prompts and parameter sets for image generation models (Stable Diffusion, DALL-E, Midjourney), design color palettes, compose layout specifications, and iterate on visual concepts through multi-step refinement. The agent's praxic stream executor dispatches to image-generation APIs and evaluates aesthetic feedback from CLIP-based scoring or human ratings.

### Music Composition & Sound Design
Compose melodies, harmonies, rhythmic patterns, and full arrangements encoded as MIDI, ABC notation, or MusicXML. The Transformer layers (L4–L5 and L8–L9) capture long-range harmonic structure, while AJN layers inject stylistic variation and avoid repetitive patterns through controlled frustration dynamics.

### Brainstorming & Ideation
Facilitate divergent thinking sessions, generate creative concepts, produce mind maps, and synthesize novel combinations of existing ideas. The agent deliberately enters frustration phases to break out of conventional thinking patterns, then uses the cascade monitor to ensure the exploration remains productive rather than chaotic.

### Content Creation & Copywriting
Draft marketing copy, social media posts, blog articles, product descriptions, and email sequences. Style consistency is maintained through the memory system's semantic store, which records brand voice parameters, tone preferences, and successful content patterns for recall during future generation tasks.

### Game Design & Worldbuilding
Create game mechanics, lore, quest lines, NPC dialogue, procedural generation rules, and interactive narrative systems. The hierarchical command interpreter decomposes high-level worldbuilding directives into specific creative subtasks, each with its own budget and style constraints.

---

## Architecture

```
Owner Directive (creative prompt / style specification)
        │
        ▼
┌─────────────────────────────────────────┐
│  Hierarchical Command Interpreter (HCI) │  ← Parses creative directives into
│                                         │    addiction targets + style constraints
└─────────────────────────────────────────┘
        │  addiction target injection + style vector
        ▼
┌─────────────────────────────────────────┐
│  ANN-Ψ Backbone (12 layers)             │
│                                         │
│  L1–L2  Hybrid AJN  (sensory encoding)  │  ← Style features, mood, genre signals
│  L3     Hetero AJN  K=8  (features)     │  ← Rhythm, tone, vocabulary patterns
│  L4–L5  Transformer (contextual attn)   │  ← Long-range narrative/thematic coherence
│  L6     Hetero AJN  K=16 (concepts)     │  ← Abstract concepts, metaphors, motifs
│  L7     Hybrid AJN  (modulation)        │  ← Style blending, genre mixing
│  L8–L9  Transformer (high-level reason) │  ← Plot structure, harmonic progression
│  L10    Hetero AJN  K=32 (high-order)   │  ← Creative strategy, aesthetic judgment
│  L11    Hybrid AJN  (praxic assembly)   │  ← Output assembly with style constraints
│  L12    Output AJN  (TPS emission)      │  ← Generated content tokens + metadata
└─────────────────────────────────────────┘
        │  creative praxis tensors
        ▼
┌─────────────────────────────────────────┐
│  Token-Energy Arbitrator (TEA)          │  ← Generous budget for exploration;
│                                         │    craving-driven override for creative
│                                         │    bursts; suppress only on convergence
└─────────────────────────────────────────┘
        │  filtered creative praxis stream
        ▼
┌─────────────────────────────────────────┐
│  Praxic Stream Executor (PSE)           │  ← Dispatch to creative tools:
│                                         │    LLM generation, image APIs, MIDI,
│                                         │    scoring functions, style validators
└─────────────────────────────────────────┘
        │
        ▼
   Creative Output (text, images, music, structured data)
        │  feedback (aesthetic score, coherence, novelty)
        └──────────────────────────────────► ANN-Ψ (next step)
```

---

## New Features for Creative Tasks

### 1. Style Vector Injection

Every directive can carry a **style vector** — a set of named parameters that influence generation at every layer:

```js
const result = await agent.execute(
  'Write a noir detective monologue about a rainy night',
  {
    style: {
      genre: 'noir',
      tone: 'melancholic',
      vocabulary: 'hardboiled',
      pacing: 'slow-burn',
      influences: ['Chandler', 'Hammett'],
    },
  }
);
```

Style vectors are injected as additional stimulus at L1–L2 and propagated through the backbone, biasing AJN addiction targets and Transformer attention patterns toward the specified aesthetic.

### 2. Divergence-Aware Cascade Monitor

In the base PREDATOR, rising cascade risk triggers intervention. In PREDATOR-CG, the cascade monitor distinguishes between **productive divergence** (creative exploration) and **destructive cascade** (loss of coherence):

- **Productive divergence** (rho 0.35–0.55): No intervention — the agent is exploring. Extended frustration is allowed.
- **Creative instability** (rho 0.55–0.75): Gentle re-anchoring via style vector reminders, no forced owner escalation.
- **Coherence collapse** (rho > 0.75): Full intervention with owner escalation — the output has lost narrative/thematic coherence.

This graduated approach prevents premature shutdown of creative exploration while still guarding against degenerate output.

### 3. Novelty-Enhanced Quality Computation

The quality metric for creative tasks incorporates **novelty** as a first-class signal alongside success rate and progress:

```
quality = 0.30 × successRate
        + 0.15 × avgProgress
        + 0.25 × noveltyScore        // NEW: distance from training distribution
        + 0.20 × coherenceScore       // NEW: self-consistency of output
        + 0.10 × ownerAlignment
        − penalties (cascade, extinction)
```

The novelty score is computed as the cosine distance between the generated output's embedding and the centroid of similar outputs in the training set. Higher distance means more novel output. The coherence score measures internal consistency (e.g., character names remain stable, chord progressions resolve properly).

### 4. Creative Memory Tiers

The memory system is extended with two creative-specific tiers:

- **Style Memory**: Stores style signatures (genre, tone, vocabulary distributions) from successful outputs. When a new directive arrives, the most relevant style signatures are recalled and injected as style vectors.
- **Inspiration Memory**: Stores cross-domain creative fragments (a striking metaphor, an unusual chord progression, a vivid color combination) that can be injected as creative prompts during frustration phases to break out of repetitive patterns.

### 5. Multi-Modal Output Tools

PREDATOR-CG includes specialized creative tools:

| Tool | Purpose | Output Format |
|------|---------|---------------|
| `TextGenerator` | Prose, poetry, dialogue, scripts | Plain text, Markdown, Fountain |
| `ImagePromptGenerator` | Prompts for image models | Structured prompt + parameters |
| `MusicComposer` | Melodies, harmonies, arrangements | MIDI, ABC notation, MusicXML |
| `StructurePlanner` | Outlines, story arcs, chapter plans | JSON hierarchical structure |
| `StyleAnalyzer` | Analyze and extract style from reference text | Style vector JSON |
| `CoherenceChecker` | Validate narrative/thematic consistency | Score + violation list |
| `NoveltyScorer` | Measure output originality vs. training data | 0–1 novelty score |

### 6. LLM Co-Creative Partnership

In PREDATOR-CG, the LLM adapter is not just a directive parser — it is a **co-creative partner**:

- **Style Transfer**: Send generated output to the LLM with a style prompt ("Rewrite this in the style of Borges") and feed the result back through the backbone for refinement.
- **Creative Feedback Loop**: After each generation step, the LLM evaluates the output for coherence, style adherence, and aesthetic quality, providing structured feedback that updates craving targets.
- **Prompt Engineering**: The LLM dynamically crafts and refines prompts for external generation APIs based on the backbone's creative state (craving level, cascade risk, current style vector).

### 7. Generation Modes

PREDATOR-CG supports three distinct generation modes selectable per directive:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Explore** | High frustration tolerance, generous token budget, minimal cascade intervention | Brainstorming, ideation, first drafts |
| **Refine** | Moderate parameters, balanced exploration/coherence | Second drafts, style polishing, iteration |
| **Converge** | Strict cascade monitoring, tight token budget, high coherence weight | Final output, publication-ready content |

```js
const draft = await agent.execute('Brainstorm 10 concepts for a sci-fi short story', { mode: 'explore' });
const refined = await agent.execute('Refine concept #3 into a 2000-word story', { mode: 'refine' });
const final = await agent.execute('Polish the story for publication', { mode: 'converge' });
```

---

## The AJN Six-Phase Lifecycle

Each neuron autonomously cycles through six phases. In PREDATOR-CG, each phase has a creative interpretation:

| Phase | Name | Standard Behaviour | Creative Interpretation |
|-------|------|--------------------|------------------------|
| 1 | **Random** | High-entropy exploration | Free association, stream of consciousness |
| 2 | **Reinforce** | Bias developing toward stimulus | Style acquisition, genre internalization |
| 3 | **Saturation** | Craving satisfied; praxes suppressed | Stylistic coherence, voice consolidation |
| 4 | **Withdrawal** | Threshold decaying; craving returns | Creative restlessness, desire for novelty |
| 5 | **Frustration** | Failure; covariance expanding chaotically | Divergent thinking, creative breakthrough |
| 6 | **Extinction** | Addiction dissolved; full reset | Creative pivot, new direction |

The creative reinterpretation of the frustration phase is the key innovation: instead of treating high covariance as a failure mode, PREDATOR-CG harnesses it as a source of creative diversity. When AJN units enter frustration, their expanded covariance generates novel praxis combinations that break repetitive patterns and produce unexpected creative output.

---

## Creative-Optimized Training Pipeline

PREDATOR-CG uses the same 4-phase training pipeline as the base system, but with creative-specific adaptations at each phase:

### Phase I: Large-Scale Pre-Training (Creative Foundation)

**Objective**: Establish broad creative competence across multiple domains.

- **Dataset**: Curated corpora of creative works — literature, poetry, song lyrics, visual art descriptions, game narratives, screenplays. Minimum 100K samples across all creative domains.
- **Learning rate**: Cosine schedule from 0.05 to 1e-6 over 10 epochs.
- **Batch size**: 32 samples.
- **Duration**: ~2–4 hours on a single GPU (NVIDIA RTX 4090) or ~8–12 hours CPU-only.
- **Checkpoint**: Saved after completion for resumability.

```js
await agent.train({
  epochsI: 10,
  creativeDataset: './data/creative-foundation/',
  lrSchedule: 'cosine',
  lrMax: 0.05,
});
```

### Phase II: Addiction Shaping (Style & Genre Specialization)

**Objective**: Shape AJN addiction targets toward creative domain patterns.

This phase uses a **3-tier curriculum** with progressive difficulty:

| Tier | Focus | Difficulty | Epochs |
|------|-------|-----------|--------|
| T1 | Single-genre style acquisition | 0.3 | 5 |
| T2 | Multi-genre style blending | 0.6 | 5 |
| T3 | Constraint-aware generation (e.g., sonnet form, blues progression) | 0.9 | 5 |

- **Dataset**: Genre-labeled creative samples with style annotations.
- **Augmentation**: Noise injection (stylistic variation), permutation (reordering creative elements), and cross-genre mixing.
- **Early stopping**: Patience of 5 epochs per tier — stops when the agent has internalized the style patterns.
- **Duration**: ~3–6 hours GPU, ~10–18 hours CPU.

### Phase III: Hierarchical Fine-Tuning (HIFT — Creative Directives)

**Objective**: Specialize the agent for specific creative directive types.

Each creative directive type receives dedicated fine-tuning with half the base learning rate:

```js
await agent.train({
  epochsIII: 8,
  directives: [
    'write a short story',
    'compose a melody',
    'generate a color palette',
    'create a character profile',
    'design a game level',
  ],
  lrMax: 0.025, // Half of base
});
```

- **Duration**: ~2–4 hours GPU, ~6–12 hours CPU (depends on directive count).
- **Key difference from base**: Directives are creative task types, not programming tasks.

### Phase IV: Adversarial Frustration Hardening (Creative Resilience)

**Objective**: Train the agent to maintain creative quality under adversarial conditions — contradictory style constraints, extreme genre mixing, and coherence challenges.

Progressive difficulty escalation from 0.1 to 1.0 across 6 epochs:

- **Low difficulty (0.1–0.3)**: Mild style contradictions ("write a cheerful gothic poem").
- **Medium difficulty (0.3–0.6)**: Significant constraint conflicts ("compose a minimalist maximalist symphony").
- **High difficulty (0.6–1.0)**: Extreme creative challenges ("write a silent opera in the style of Metallica performed by a string quartet").

The agent learns to find creative resolutions to contradictory constraints rather than simply failing — a crucial skill for real-world creative tasks where briefs are often ambiguous or paradoxical.

- **Early stopping**: Triggers when adversarial robustness exceeds 0.95.
- **Duration**: ~2–3 hours GPU, ~6–10 hours CPU.

### Full Training Summary

| Phase | GPU Time | CPU Time | Checkpoint Size |
|-------|----------|----------|-----------------|
| I: Pre-training | 2–4h | 8–12h | ~200 MB |
| II: Addiction Shaping | 3–6h | 10–18h | ~250 MB |
| III: HIFT | 2–4h | 6–12h | ~150 MB |
| IV: Adversarial Hardening | 2–3h | 6–10h | ~100 MB |
| **Total** | **9–17h** | **30–52h** | **~700 MB** |

Training can be resumed from any checkpoint:
```js
await agent.trainingPipeline.resumeFromCheckpoint('phase-II-epoch-12', config);
```

---

## Hardware Requirements

### Minimum Requirements (CPU-Only Inference)

| Component | Specification |
|-----------|--------------|
| CPU | 4+ cores, x86-64 (Intel i5 8th gen / AMD Ryzen 5 3600 or better) |
| RAM | 8 GB |
| Storage | 2 GB free (agent state + checkpoints + memory) |
| Node.js | ≥ 18.0.0 |

CPU-only inference runs the ANN-Ψ backbone with `dModel=128, nHeads=4, dFF=512` (reduced configuration). Expect response times of 2–10 seconds per creative step depending on directive complexity.

### Recommended Requirements (GPU-Accelerated Inference)

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 3060 or better (12 GB VRAM) with CUDA 11.8+ |
| CPU | 8+ cores |
| RAM | 16 GB |
| Storage | 10 GB free (datasets + checkpoints + creative outputs) |
| Node.js | ≥ 18.0.0 |

GPU-accelerated inference uses the full `dModel=256, nHeads=8, dFF=1024` configuration. Expect response times of 0.2–1.5 seconds per creative step.

### Training Requirements

| Training Phase | Minimum GPU | Minimum VRAM | Minimum RAM | Dataset Size |
|----------------|-------------|-------------|-------------|-------------|
| Phase I | NVIDIA RTX 3060 | 12 GB | 16 GB | 100K+ samples |
| Phase II | NVIDIA RTX 3060 | 12 GB | 16 GB | 50K+ labeled samples |
| Phase III | NVIDIA RTX 2080 Ti | 8 GB | 16 GB | Per-directive subsets |
| Phase IV | NVIDIA RTX 3060 | 12 GB | 16 GB | Adversarial pairs |

**CPU-only training** is possible but not recommended — each phase will take 3–5x longer. For CPU-only training, use the reduced backbone configuration (`dModel=128`).

### Cloud Training Recommendations

| Provider | GPU Instance | Cost/Hour (approx.) | Full Training Time |
|----------|-------------|---------------------|-------------------|
| Google Colab | T4 (free tier) | Free | ~20–30 hours |
| AWS | p3.2xlarge (V100) | $3.04 | ~8–12 hours |
| Lambda Labs | A100 40GB | $1.10 | ~4–6 hours |
| RunPod | RTX 4090 | $0.74 | ~5–8 hours |

### Memory & Storage Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| ANN-Ψ Backbone (full) | ~80 MB | dModel=256, 12 layers |
| ANN-Ψ Backbone (reduced) | ~25 MB | dModel=128, 12 layers |
| Training Checkpoint | 100–250 MB | Depends on phase and epoch |
| Episodic Memory (1000 entries) | ~5 MB | With 64-dim embeddings |
| Semantic Memory | ~1 MB | Key-value facts |
| Style Memory | ~2 MB | Style signatures |
| Inspiration Memory | ~3 MB | Cross-domain fragments |
| Creative Dataset | 500 MB – 5 GB | Depends on domain coverage |
| Generated Outputs | Variable | Text: ~1 KB/page; Images: prompt only |

---

## Installation

```bash
git clone https://github.com/justo-tapiador/predator-jungle-agent.git
cd predator-jungle-agent
npm install
```

Requires **Node.js ≥ 18**.

### Optional: GPU Acceleration

For GPU-accelerated training and inference, install the CUDA backend:

```bash
# Requires NVIDIA CUDA Toolkit 11.8+ and cuDNN 8+
npm install @nvidia/cuda-bindings  # Hypothetical — use actual bindings for your setup
```

### Environment Variables

```bash
# .env file
PREDATOR_PORT=3000
PREDATOR_HOST=0.0.0.0

# LLM Integration (for co-creative partnership)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Optional: Local LLM
LOCAL_LLM_ENDPOINT=http://localhost:11434/api/generate
LOCAL_LLM_MODEL=llama3

# Image Generation
IMAGE_GEN_API_KEY=...
IMAGE_GEN_ENDPOINT=https://api.stability.ai/v1/generation

# Memory persistence
MEMORY_STORAGE_DIR=./data/memory
```

---

## Quick Start

```bash
# Creative writing task
npm run cli -- task "Write a haiku about quantum entanglement" --mode explore

# Music composition
npm run cli -- task "Compose a 16-bar jazz blues melody in Bb" --mode refine

# Brainstorming session
npm run cli -- task "Generate 20 concepts for a time-travel card game" --mode explore

# Interactive demo (3 creative tasks with full training)
npm run cli -- demo

# Web dashboard (http://localhost:3000)
npm run web

# Full benchmark suite
npm run benchmark
```

---

## Programmatic API

### Basic Creative Generation

```js
import { Predator } from './src/index.js';

const agent = new Predator({
  dModel: 256,         // Full creative capacity
  nHeads: 8,
  dFF: 1024,
  maxSteps: 200,
  enableMemory: true,   // Persistent style & narrative memory
  enableSafety: true,   // Content policy guardrails
  enableMetrics: true,  // Novelty & coherence tracking
  enablePlugins: true,  // Creative tool plugins
  rhoWarn: 0.55,        // Higher tolerance for creative divergence
  rhoCritical: 0.75,    // Only intervene on coherence collapse
});

// Execute a creative directive
const result = await agent.execute(
  'Write a noir detective monologue about AI sentience',
  {
    style: {
      genre: 'noir',
      tone: 'philosophical',
      perspective: 'first-person',
      pacing: 'contemplative',
    },
    mode: 'refine',
    budget: { tokens: 50000, energy: 1.5 },  // Generous creative budget
  }
);

console.log(`Quality: ${(result.quality * 100).toFixed(1)}%`);
console.log(`Novelty: ${(result.noveltyScore * 100).toFixed(1)}%`);
console.log(`Coherence: ${(result.coherenceScore * 100).toFixed(1)}%`);
console.log(`Output: ${result.generatedContent}`);
```

### Multi-Step Creative Workflow

```js
// Step 1: Explore — brainstorm story concepts
const concepts = await agent.execute(
  'Brainstorm 5 concepts for a cyberpunk short story about memory trading',
  { mode: 'explore' }
);

// Step 2: Refine — develop the best concept
const story = await agent.execute(
  `Expand concept #3 into a 3000-word short story with vivid descriptions`,
  { mode: 'refine', style: { genre: 'cyberpunk', tone: 'bittersweet' } }
);

// Step 3: Converge — polish for publication
const polished = await agent.execute(
  'Polish the story: improve dialogue, tighten pacing, enhance the ending',
  { mode: 'converge' }
);
```

### Chain Execution for Complex Projects

```js
// Generate a complete creative project in one chain
const results = await agent.executeChain([
  'Create a worldbuilding document for a underwater civilization',
  'Design 5 character profiles for this world',
  'Write a 2000-word story set in this world featuring these characters',
  'Generate an image prompt for the story cover art',
  'Compose a 30-second theme melody in the world\'s musical style',
]);
```

### Creative Tools

```js
// Register creative tools
agent.registerTool('image-gen', async (args) => {
  // Call your image generation API
  const response = await fetch('https://api.stability.ai/v1/generation/stable-diffusion-xl/text-to-image', {
    method: 'POST',
    headers: { Authorization: `Bearer ${process.env.STABILITY_API_KEY}` },
    body: JSON.stringify({ text_prompts: [{ text: args.prompt }], cfg_scale: 7, steps: 30 }),
  });
  return await response.json();
}, { description: 'Generate images from text prompts' });

agent.registerTool('midi-compose', async (args) => {
  // Convert ABC notation to MIDI
  return { format: 'midi', data: convertAbcToMidi(args.notation) };
}, { description: 'Compose music from notation' });

agent.registerTool('style-analyzer', async (args) => {
  // Analyze text style and return a style vector
  return analyzeStyle(args.referenceText);
}, { description: 'Extract style vector from reference text' });
```

### Event-Driven Creative Monitoring

```js
// Track creative quality in real-time
agent.on('step:complete', (s) => {
  console.log(`Step ${s.step}: progress=${(s.progress*100).toFixed(0)}% rho=${s.rho?.toFixed(3)}`);
});

agent.on('cascade:warning', (e) => {
  console.log(`Creative divergence detected (rho=${e.rho.toFixed(3)}) — allowing exploration`);
});

agent.on('cascade:critical', (e) => {
  console.warn(`Coherence collapse (rho=${e.rho.toFixed(3)}) — re-anchoring style`);
});

agent.on('owner:escalation', (e) => {
  console.warn(`Creative deadlock — Owner input requested`);
  agent.resume({ alignment: 0.7, instruction: 'Try a different perspective' });
});

// Track novelty and coherence scores
agent.on('task:complete', (result) => {
  console.log(`Novelty: ${result.noveltyScore?.toFixed(3)}`);
  console.log(`Coherence: ${result.coherenceScore?.toFixed(3)}`);
  console.log(`Quality: ${result.quality?.toFixed(3)}`);
});
```

---

## Creative Domain Examples

### Poetry Generation

```js
const poem = await agent.execute(
  'Write a villanelle about the passage of time',
  {
    style: { form: 'villanelle', tone: 'elegiac', imagery: 'natural' },
    mode: 'converge',  // Strict form requires convergence mode
    budget: { tokens: 20000 },
  }
);
```

### Collaborative Storytelling

```js
// Human provides the seed, agent expands, human guides
const chapter1 = await agent.execute(
  'Write the opening chapter of a mystery set in 1920s Shanghai',
  { mode: 'refine', style: { genre: 'mystery', era: '1920s', setting: 'Shanghai' } }
);

// Continue with memory of previous chapter
const chapter2 = await agent.execute(
  'Continue the story: the detective discovers a hidden message in the jade pendant',
  { mode: 'refine' }
);
```

### Music Composition

```js
const melody = await agent.execute(
  'Compose a 32-bar jazz standard in the style of Thelonious Monk',
  {
    style: { genre: 'jazz', subgenre: 'bebop', influence: 'monk', key: 'Eb', timeSignature: '4/4' },
    mode: 'explore',
    budget: { tokens: 40000 },
  }
);
```

### Visual Art Direction

```js
const artwork = await agent.execute(
  'Create a detailed prompt for a surreal painting blending Dalí and Miyazaki',
  {
    style: { movement: 'surrealism', influences: ['Dalí', 'Miyazaki'], medium: 'oil painting' },
    mode: 'refine',
  }
);
```

---

## LLM Integration

PREDATOR-CG supports two LLM integration modes:

### OpenAI Adapter

```js
import { OpenAIAdapter } from './src/llm/OpenAIAdapter.js';

const llm = new OpenAIAdapter({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4',
  temperature: 0.9,   // Higher temperature for creative tasks
  maxTokens: 4096,
});

// Use LLM for style transfer
const rewritten = await llm.chat([
  { role: 'system', content: 'You are a creative writing assistant specializing in style transfer.' },
  { role: 'user', content: `Rewrite the following in the style of Borges:\n\n${generatedText}` },
]);
```

### Local LLM Adapter

```js
import { LocalLLMAdapter } from './src/llm/LocalLLMAdapter.js';

const llm = new LocalLLMAdapter({
  endpoint: 'http://localhost:11434/api/generate',
  model: 'llama3',
  temperature: 0.95,
});

// Use for offline creative generation
const idea = await llm.generate('Suggest an unexpected plot twist for a sci-fi mystery');
```

The LLM adapter is used within the PSE for:
1. **Co-creative refinement** — iteratively improve generated content
2. **Style transfer** — apply new styles to existing output
3. **Creative evaluation** — assess novelty, coherence, and aesthetic quality
4. **Prompt engineering** — craft prompts for external generation APIs

---

## Web Dashboard

```bash
npm run web
# → http://localhost:3000
```

The real-time dashboard provides monitoring for:

### Overview Panel
- Creative quality score (novelty + coherence + success)
- Cascade risk timeline with creative divergence zones
- Step progress chart with frustration phase markers
- Token/energy consumption for creative exploration
- Module status (memory, safety, metrics, plugins)

### Task Control Panel
- Directive submission with style vector editor
- Generation mode selector (Explore / Refine / Converge)
- Owner feedback for creative escalation events
- Quick actions: serialize state, start training, shutdown

### Metrics Panel
- Novelty score histogram (p50, p95, p99)
- Coherence score distribution
- Style consistency gauges
- Creative step counters by phase
- Token efficiency per generation mode

### Cascade Monitor Panel
- Graduated threshold display (divergence → instability → collapse)
- Per-layer risk trends with creative interpretation
- Extinction log with creative pivot events
- Self-healing stimulus injection history

### Memory Panel
- Style memory search and recall
- Inspiration memory browsing
- Episodic memory query for past creative outputs
- Semantic memory exploration

### Event Log
- Real-time event stream with color-coded creative events
- Auto-scrolling with filter support
- Event categories: generation, cascade, safety, memory, training

---

## Safety Guardrails for Creative Agents

Creative agents need safety guardrails adapted to their domain:

### Content Policy Enforcement
- **Plagiarism detection**: Compares generated text against training data to flag near-duplicate outputs (configurable similarity threshold).
- **Content policy**: Blocks generation of content that violates configurable policy rules (violence, hate speech, etc.).
- **Attribution tracking**: When style transfer is used, the system tracks and reports the source influences.

### Creative Safety Levels

```js
const agent = new Predator({
  enableSafety: true,
  safetyConfig: {
    safetyLevel: 'standard',  // 'permissive' | 'standard' | 'strict'
    plagiarismThreshold: 0.85,  // Flag outputs > 85% similar to training data
    contentPolicyRules: ['no-hate-speech', 'no-graphic-violence'],
    maxGenerationLength: 10000,  // Characters per output
  },
});
```

| Level | Plagiarism Check | Content Policy | Rate Limit |
|-------|-----------------|----------------|------------|
| Permissive | Off | Minimal | 120/min |
| Standard | On (0.85 threshold) | Standard rules | 60/min |
| Strict | On (0.70 threshold) | Comprehensive | 30/min |

### Audit Trail

All safety decisions are logged with full context:

```js
const audit = agent.safety.getAuditTrail(50);
// Each entry: { timestamp, checkType, allowed, reason, severity }
```

---

## Memory System for Creative Context

The three-tier memory system is optimized for creative work:

### Episodic Memory
Stores complete creative task records with style vectors, generated outputs, quality scores, and feedback. Enables the agent to maintain narrative consistency across multi-session projects and recall successful creative strategies.

```js
// Store a creative task result
await agent.memory.store(
  'Write a noir monologue about AI sentience',
  { content: '...', quality: 0.85, novelty: 0.72, coherence: 0.91 },
  { style: { genre: 'noir' }, mode: 'refine' }
);

// Recall similar creative tasks for inspiration
const memories = await agent.memory.recall('detective story', { limit: 5 });
```

### Semantic Memory
Stores extracted creative facts — genre conventions, style rules, vocabulary distributions, structural patterns. These facts guide future generation without requiring exact episode recall.

```js
// Query semantic memory for genre conventions
const conventions = await agent.memory.recallSemantic('genre:noir');
// → { value: 'Common noir elements: femme fatale, moral ambiguity, urban decay...', confidence: 0.85 }
```

### Working Memory
Short-term buffer for the current creative session — active style vector, in-progress narrative state, current character list, thematic threads.

```js
// Push current creative context
await agent.memory.pushWorking('active_characters', ['Detective Chen', 'Dr. Okafor'], 3600000);
await agent.memory.pushWorking('current_theme', 'identity vs. memory', 3600000);
```

### Memory Consolidation

Periodic consolidation merges similar creative episodes, extracts recurring style patterns, and prunes low-confidence semantic facts:

```js
await agent.memory.consolidate();
// Merges similar episodes, extracts style patterns, prunes low-confidence facts
```

---

## Plugin Architecture

Extend PREDATOR-CG with creative plugins:

```js
const rhymePlugin = {
  name: 'rhyme-checker',
  version: '1.0.0',
  description: 'Validates rhyme schemes in generated poetry',

  hooks: {
    afterStep: {
      handler: (context, result) => {
        if (context.taskType === 'poetry') {
          const rhymeScore = checkRhymeScheme(result.generatedText, context.expectedScheme);
          result.rhymeScore = rhymeScore;
        }
      },
      priority: 30,
    },
  },

  init(pluginManager) {
    console.log('Rhyme checker plugin initialized');
  },

  destroy() {
    console.log('Rhyme checker plugin destroyed');
  },
};

agent.use(rhymePlugin);
```

### Available Hook Points for Creative Plugins

| Hook | Trigger | Use Case |
|------|---------|----------|
| `beforeStep` | Before each TPS step | Inject style modifications, set creative parameters |
| `afterStep` | After each TPS step | Score output, validate form constraints |
| `beforeEmit` | Before emitting praxis | Filter/modify generated content |
| `afterEmit` | After emitting praxis | Log output, trigger external processes |
| `taskComplete` | Task finished | Archive output, update style memory |
| `directiveReceived` | New directive arrives | Pre-process, augment with style data |
| `extinction` | AJN extinction event | Log creative pivots, inject inspiration |
| `trainingEpoch` | Training epoch ends | Track creative quality metrics |

---

## Project Structure

```
predator-cg/
├── src/
│   ├── core/
│   │   ├── ArtificialJunkyNeuron.js        # AJN unit (5-tuple definition)
│   │   ├── ANNPsi.js                       # 12-layer ANN-Ψ backbone
│   │   ├── Predator.js                     # Main agent orchestrator (creative mode)
│   │   └── StateSerializer.js              # Checkpoint serialization
│   ├── layers/
│   │   ├── AJNLayer.js                     # 3 integration paradigms
│   │   └── TransformerBlock.js             # Self-attention blocks
│   ├── modules/
│   │   ├── HierarchicalCommandInterpreter.js  # HCI (creative directive parsing)
│   │   ├── TokenEnergyArbitrator.js           # TEA + PSE (creative budget)
│   │   ├── CascadeMonitor.js                  # Divergence-aware cascade prevention
│   │   ├── MemorySystem.js                    # 3-tier creative memory
│   │   ├── SafetyGuardrails.js                # Content policy + plagiarism detection
│   │   ├── MetricsCollector.js                # Novelty + coherence tracking
│   │   └── PluginManager.js                   # Creative plugin architecture
│   ├── tools/
│   │   ├── ToolRegistry.js                    # Creative tool registry
│   │   ├── WebSearchTool.js                   # Research & inspiration
│   │   ├── CodeExecutionTool.js               # Script-based generation
│   │   ├── APICallTool.js                     # External API integration
│   │   ├── DatabaseTool.js                    # Creative asset storage
│   │   └── FileSystemTool.js                  # Output file management
│   ├── training/
│   │   ├── TrainingPipeline.js                # 4-phase creative training
│   │   ├── DatasetLoader.js                   # Creative dataset management
│   │   └── CheckpointManager.js               # Training state persistence
│   ├── llm/
│   │   ├── LLMAdapter.js                      # Base LLM interface
│   │   ├── OpenAIAdapter.js                   # OpenAI integration
│   │   └── LocalLLMAdapter.js                 # Local LLM (Ollama, etc.)
│   └── index.js                               # Public exports
├── web/
│   ├── server.js                              # Express + WebSocket server
│   ├── routes/
│   │   └── api.js                             # REST API endpoints
│   └── public/
│       ├── index.html                         # Dashboard UI
│       ├── css/dashboard.css                  # Scientific dark theme
│       └── js/dashboard.js                    # Client-side logic + Chart.js
├── config/
│   ├── default.json                           # Default configuration
│   ├── production.json                        # Production overrides
│   └── schema.js                              # Configuration validation
├── docker/
│   ├── Dockerfile                             # Container build
│   └── docker-compose.yml                     # Orchestrated deployment
├── plugins/
│   └── example-plugin.js                      # Example creative plugin
├── scripts/
│   ├── cli.js                                 # Commander CLI
│   ├── benchmark.js                           # Creative quality benchmarks
│   └── lib/
│       ├── demo.js                            # Interactive creative demo
│       └── format.js                          # Output formatting
├── tests/
│   ├── predator.test.js                       # Main test suite
│   ├── unit/                                  # Unit tests
│   └── integration/                           # Integration tests
├── data/
│   ├── memory/                                # Persistent memory storage
│   └── checkpoints/                           # Training checkpoints
├── .env.example                               # Environment template
├── package.json                               # Project manifest
└── README.md                                  # This file
```

---

## Configuration Reference

### Agent Configuration

```js
const agent = new Predator({
  // Backbone architecture
  dModel: 256,           // Transformer model dimension (128 for CPU, 256 for GPU)
  nHeads: 8,             // Attention heads (4 for CPU, 8 for GPU)
  dFF: 1024,             // Feed-forward dimension (512 for CPU, 1024 for GPU)

  // Execution limits
  maxSteps: 200,         // Max TPS steps per task

  // Creative thresholds
  rhoWarn: 0.55,         // Creative divergence warning (higher than base)
  rhoCritical: 0.75,     // Coherence collapse threshold (higher than base)

  // Feature flags
  enableMemory: true,    // Persistent creative memory
  enableSafety: true,    // Content policy guardrails
  enableMetrics: true,   // Novelty & coherence tracking
  enablePlugins: true,   // Creative tool plugins

  // Creative-specific defaults
  defaultMode: 'refine', // Default generation mode
  noveltyWeight: 0.25,   // Novelty score weight in quality computation
  coherenceWeight: 0.20, // Coherence score weight
});
```

### Training Configuration

```js
await agent.train({
  // Phase I: Creative Foundation
  epochsI: 10,
  creativeDataset: './data/creative-foundation/',

  // Phase II: Style & Genre Specialization
  epochsII_T1: 5,   // Single-genre acquisition
  epochsII_T2: 5,   // Multi-genre blending
  epochsII_T3: 5,   // Constraint-aware generation

  // Phase III: Creative HIFT
  epochsIII: 8,
  directives: [
    'write a short story',
    'compose a melody',
    'generate a color palette',
    'create a character profile',
    'design a game level',
  ],

  // Phase IV: Adversarial Creative Hardening
  epochsIV: 6,

  // Shared parameters
  batchSize: 32,
  lrSchedule: 'cosine',  // 'cosine' | 'linear' | 'step' | 'constant'
  lrMax: 0.05,
  lrMin: 1e-6,
  enableCheckpoints: true,
  earlyStoppingPatience: 5,
  onProgress: (progress) => console.log(progress),
});
```

### Safety Configuration

```js
safetyConfig: {
  safetyLevel: 'standard',
  plagiarismThreshold: 0.85,
  contentPolicyRules: ['no-hate-speech', 'no-graphic-violence'],
  maxGenerationLength: 10000,
  protectedPaths: ['/etc', '/root', '/sys'],
  rateLimits: { perMinute: 60, perHour: 500 },
  allowedTools: null,  // null = all creative tools allowed
}
```

---

## Testing

```bash
# Full test suite
node --experimental-vm-modules tests/predator.test.js

# Unit tests only
node --experimental-vm-modules tests/unit/run.js

# Integration tests only
node --experimental-vm-modules tests/integration/run.js
```

### Creative Quality Benchmarks

```bash
npm run benchmark
```

Benchmarks measure:
- **Novelty**: Average cosine distance from training distribution
- **Coherence**: Self-consistency score across output
- **Style adherence**: Match between requested and delivered style
- **Diversity**: Variance across multiple outputs for the same prompt
- **Latency**: Time per creative step (CPU vs GPU)

---

## Theoretical Basis

PREDATOR-CG's creative advantages derive from the AJN saturation dynamics, reinterpreted for generative tasks:

### Frustration as Creative Engine
During Phase 5 (Frustration), AJN covariance expands chaotically, generating high-diversity praxes. In structured tasks this is a failure mode, but in creative generation it is the primary source of novelty. The expanded covariance space produces unexpected combinations — a metaphor from biology applied to music, a narrative structure borrowed from architecture, a color palette inspired by astronomical data. PREDATOR-CG deliberately sustains frustration phases longer than the base system, with the cascade monitor intervening only when coherence collapses rather than when mere divergence is detected.

### Saturation as Stylistic Coherence
During Phase 3 (Saturation), praxes are suppressed as the neuron's craving is satisfied. In creative mode, this suppression manifests as stylistic consistency — the generated output conforms to an established voice, tone, and structural pattern. The token suppression mechanism (`‖P_t‖_F → 0`) means that when the agent has found a satisfying creative direction, it emits fewer but more refined tokens, concentrating quality rather than producing quantity.

### Cascade Prevention as Coherence Maintenance
Lateral inhibition coupling between AJN units prevents group extinction. In creative terms, this means that when one creative thread begins to fail (a plot thread dead-ends, a melody becomes repetitive), neighboring units provide stabilizing signals that re-anchor the generation. The cascade monitor's graduated intervention levels — productive divergence, creative instability, coherence collapse — map directly to the creative process of exploration, floundering, and recovery.

### Adaptive Compute as Creative Budget
The TEA combines energy-aware suppression with craving-driven override. In creative mode, the craving-driven override is more generous: when the agent detects that it is in a productive creative flow (high progress, moderate cascade risk), it allocates additional compute to sustain the flow. When the agent is stuck (low progress, high cascade risk), it redirects budget toward exploration and frustration-phase activation.

### Memory as Creative Context
The three-tier memory system provides the temporal depth that creative work requires. Episodic memory maintains narrative continuity across multi-session projects. Semantic memory stores genre conventions and style rules that guide generation without requiring exact recall. Working memory holds the current creative state — active characters, themes, and style parameters — ensuring that each step builds coherently on the last.

---

## License

MIT

---

*Agentic Theory original concepts © Justo Tapiador García, Universidad de Alicante (UA), 2024–2026.*
*Creative & Generative specialization — PREDATOR-CG Edition.*
