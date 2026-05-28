/**
 * PREDATOR – Main entry point (v2.0)
 * ─────────────────────────────────────────────────────────────────────────────
 * Praxic Reinforcement and Extinction-Driven Agentic Task Orchestrator
 * and Realizer (PREDATOR)
 *
 * Based on the Agentic Theory by Justo Tapiador García (UA)
 * Preprints: WALLERMAX-AI 2604.00012, 2604.00013, 2604.00014
 * ─────────────────────────────────────────────────────────────────────────────
 */

// Core
export { Predator }           from './core/Predator.js';
export { ANNPsi }             from './core/ANNPsi.js';
export { ArtificialJunkyNeuron, AJNPhase } from './core/ArtificialJunkyNeuron.js';
export { StateSerializer }    from './core/StateSerializer.js';

// Layers
export { HomogeneousAJNLayer, HeterogeneousAJNLayer, HybridAJNLayer }
                              from './layers/AJNLayer.js';
export { TransformerBlock }   from './layers/TransformerBlock.js';

// Modules
export { HierarchicalCommandInterpreter }
                              from './modules/HierarchicalCommandInterpreter.js';
export { TokenEnergyArbitrator, PraxicStreamExecutor }
                              from './modules/TokenEnergyArbitrator.js';
export { CascadeMonitor }     from './modules/CascadeMonitor.js';
export { MemorySystem }       from './modules/MemorySystem.js';
export { SafetyGuardrails }   from './modules/SafetyGuardrails.js';
export { MetricsCollector }   from './modules/MetricsCollector.js';
export { PluginManager }      from './modules/PluginManager.js';

// Tools
export { FileSystemTool }     from './tools/FileSystemTool.js';
export { WebSearchTool }      from './tools/WebSearchTool.js';
export { CodeExecutionTool }  from './tools/CodeExecutionTool.js';
export { APICallTool }        from './tools/APICallTool.js';
export { DatabaseTool }       from './tools/DatabaseTool.js';
export { ToolRegistry }       from './tools/ToolRegistry.js';

// Training
export { TrainingPipeline }   from './training/TrainingPipeline.js';
export { DatasetLoader }      from './training/DatasetLoader.js';
export { CheckpointManager }  from './training/CheckpointManager.js';

// LLM
export { LLMAdapter }         from './llm/LLMAdapter.js';
export { OpenAIAdapter }      from './llm/OpenAIAdapter.js';
export { LocalLLMAdapter }    from './llm/LocalLLMAdapter.js';

// Web Dashboard
// Note: The web server is a standalone entry point. Start it with: npm run web
// Import dynamically: const { predator } = await import('../web/server.js');
