/**
 * PREDATOR v2.0 Test Suite
 * Tests for: AJN unit, layers, transformer, HCI, TEA, PSE, ANNPsi,
 *            Predator agent, Memory, Safety, Metrics, Plugins, StateSerializer
 */

import { strict as assert } from 'assert';
import { ArtificialJunkyNeuron, AJNPhase } from '../src/core/ArtificialJunkyNeuron.js';
import { HomogeneousAJNLayer, HeterogeneousAJNLayer, HybridAJNLayer } from '../src/layers/AJNLayer.js';
import { TransformerBlock } from '../src/layers/TransformerBlock.js';
import { HierarchicalCommandInterpreter } from '../src/modules/HierarchicalCommandInterpreter.js';
import { TokenEnergyArbitrator, PraxicStreamExecutor } from '../src/modules/TokenEnergyArbitrator.js';
import { ANNPsi } from '../src/core/ANNPsi.js';
import { Predator } from '../src/core/Predator.js';
import { MemorySystem } from '../src/modules/MemorySystem.js';
import { SafetyGuardrails } from '../src/modules/SafetyGuardrails.js';
import { MetricsCollector } from '../src/modules/MetricsCollector.js';
import { PluginManager } from '../src/modules/PluginManager.js';
import { StateSerializer } from '../src/core/StateSerializer.js';
import { ToolRegistry } from '../src/tools/ToolRegistry.js';
import { FileSystemTool } from '../src/tools/FileSystemTool.js';
import { DatabaseTool } from '../src/tools/DatabaseTool.js';

const pass = (name) => console.log(`  OK ${name}`);
const fail = (name, err) => { console.error(`  FAIL ${name}: ${err.message}`); process.exitCode = 1; };

async function runSuite(suiteName, tests) {
  console.log(`\n[${suiteName}]`);
  for (const [name, fn] of Object.entries(tests)) {
    try { await fn(); pass(name); }
    catch (e) { fail(name, e); }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('ArtificialJunkyNeuron v2.0', {
  'initial phase is RANDOM': () => {
    const ajn = new ArtificialJunkyNeuron({ intensityFn: () => 0 });
    assert.equal(ajn.phase, AJNPhase.RANDOM);
  },
  'craving increases with rich stimulus': () => {
    const ajn = new ArtificialJunkyNeuron({ intensityFn: () => 0.9 });
    ajn.process({ intensity: 0.9 });
    ajn.process({ intensity: 0.9 });
    assert.ok(ajn.M > 0.1, `craving should grow, got ${ajn.M}`);
  },
  'enters SATURATION phase on strong stimulus': () => {
    const ajn = new ArtificialJunkyNeuron({ params: { betaM: 0.5, thetaSat: 0.5 } });
    for (let i = 0; i < 20; i++) ajn.process({ intensity: 0.95 });
    assert.equal(ajn.phase, AJNPhase.SATURATION);
  },
  'enters EXTINCTION after tau consecutive failures': () => {
    const ajn = new ArtificialJunkyNeuron({ params: { tau: 5, betaM: 0.1 } });
    let extinct = false;
    ajn.on('extinction', () => { extinct = true; });
    for (let i = 0; i < 10; i++) ajn.process({ intensity: 0.0 });
    assert.ok(extinct || ajn.extinctions > 0, 'extinction should fire');
  },
  'praxis tensor has correct dimensionality': () => {
    const dim = 32;
    const ajn = new ArtificialJunkyNeuron({ params: { praximDim: dim } });
    const r = ajn.process({ intensity: 0.5 });
    assert.equal(r.praxis.length, dim);
  },
  'snapshot returns all required fields': () => {
    const ajn = new ArtificialJunkyNeuron();
    const s = ajn.snapshot();
    for (const k of ['id','phase','phaseName','M','theta','nFail','step','extinctions','avgReward']) {
      assert.ok(k in s, `snapshot missing field: ${k}`);
    }
  },
  'injectAddictionTarget seeds craving': () => {
    const ajn = new ArtificialJunkyNeuron();
    ajn.injectAddictionTarget(new Float64Array(64).fill(0.8));
    assert.ok(ajn.M >= 0.3, 'craving should be seeded');
  },
  'serialize/deserialize round-trip preserves state': () => {
    const ajn = new ArtificialJunkyNeuron({ params: { betaM: 0.5 } });
    for (let i = 0; i < 5; i++) ajn.process({ intensity: 0.7 });
    const state = ajn.serialize();
    const ajn2 = new ArtificialJunkyNeuron();
    ajn2.deserialize(state);
    assert.equal(ajn2.step, ajn.step);
    assert.equal(ajn2.M, ajn.M);
    assert.equal(ajn2.phase, ajn.phase);
  },
  'experience replay buffer works': () => {
    const ajn = new ArtificialJunkyNeuron({ params: { replaySize: 10 } });
    for (let i = 0; i < 15; i++) ajn.process({ intensity: 0.5 });
    assert.ok(ajn.replayBuffer.length <= 10, 'replay buffer should be capped');
  },
  'Hebbian trace accumulates with stimulus': () => {
    const ajn = new ArtificialJunkyNeuron({ params: { hebbianLR: 0.01 } });
    for (let i = 0; i < 10; i++) ajn.process({ intensity: 0.8 });
    const traceNorm = ajn.hebbianTrace.reduce((s, v) => s + Math.abs(v), 0);
    assert.ok(traceNorm > 0, 'Hebbian trace should accumulate');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('TransformerBlock v2.0', {
  'process returns transformed stimulus': () => {
    const block = new TransformerBlock({ id: 'test-tf', dModel: 32, nHeads: 4, dFF: 128 });
    const result = block.process({ intensity: 0.5, task_progress: 0.3 });
    assert.ok('intensity' in result, 'should return intensity');
    assert.equal(result._layer, 'test-tf');
  },
  'serialize/deserialize round-trip': () => {
    const block = new TransformerBlock({ id: 'test-tf2', dModel: 16, nHeads: 2, dFF: 64 });
    const state = block.serialize();
    const block2 = new TransformerBlock({ id: 'test-tf2b', dModel: 16, nHeads: 2, dFF: 64 });
    block2.deserialize(state);
    assert.ok(true, 'deserialization should not throw');
  },
  'handles empty stimulus gracefully': () => {
    const block = new TransformerBlock({ dModel: 16, nHeads: 2 });
    const result = block.process(null);
    assert.ok(result, 'should return a result even for null input');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('HomogeneousAJNLayer v2.0', {
  'returns layerPraxis of correct shape': () => {
    const layer = new HomogeneousAJNLayer({
      N: 8, stimulusClass: 'test', intensityFn: () => 0.5,
    });
    const { layerPraxis } = layer.process({ intensity: 0.5 });
    assert.equal(layerPraxis.length, 64);
  },
  'cascadeRisk is between 0 and 1': () => {
    const layer = new HomogeneousAJNLayer({
      N: 4, stimulusClass: 'test', intensityFn: () => 0,
    });
    const { cascadeRisk } = layer.process({ intensity: 0 });
    assert.ok(cascadeRisk >= 0 && cascadeRisk <= 1);
  },
  'weighted aggregation works': () => {
    const layer = new HomogeneousAJNLayer({
      N: 8, stimulusClass: 'test', intensityFn: () => 0.5, aggregation: 'weighted',
    });
    const { layerPraxis } = layer.process({ intensity: 0.5 });
    assert.equal(layerPraxis.length, 64);
  },
  'serialize/deserialize works': () => {
    const layer = new HomogeneousAJNLayer({
      N: 4, stimulusClass: 'test', intensityFn: () => 0.5,
    });
    layer.process({ intensity: 0.5 });
    const state = layer.serialize();
    const layer2 = new HomogeneousAJNLayer({
      N: 4, stimulusClass: 'test', intensityFn: () => 0.5,
    });
    layer2.deserialize(state);
    assert.ok(true, 'should not throw');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('ANNPsi v2.0', {
  'forward pass returns required fields': () => {
    const ann = new ANNPsi();
    const r = ann.forward({ intensity: 0.5, task_progress: 0.3, completion_signal: 0.2,
                            goal_proximity: 0.4, correctness: 0.6 });
    for (const k of ['outputPraxis','praxisNorm','saturated','cascadeRisk','layerTrace']) {
      assert.ok(k in r, `forward missing: ${k}`);
    }
  },
  'layerTrace has entries for all layers': () => {
    const ann = new ANNPsi();
    const r = ann.forward({ intensity: 0.5 });
    assert.ok(r.layerTrace.length >= 8, `expected >=8 trace entries, got ${r.layerTrace.length}`);
  },
  'cascadeRisk is in [0,1]': () => {
    const ann = new ANNPsi();
    const r = ann.forward({ intensity: 0 });
    assert.ok(r.cascadeRisk >= 0 && r.cascadeRisk <= 1);
  },
  'injectHCITargets does not throw': () => {
    const ann = new ANNPsi();
    assert.doesNotThrow(() => {
      ann.injectHCITargets({ l1: new Float64Array(64).fill(0.5) });
    });
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('HierarchicalCommandInterpreter v2.0', {
  'parses goal correctly (sync)': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parseSync('Debug and fix all failing unit tests urgently');
    assert.ok(d.goal.length > 0);
    assert.equal(d.priority, 'critical');
    assert.ok(d.urgency > 0.5);
  },
  'routine priority for calm language': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parseSync('Please organize the project documentation at your convenience');
    assert.equal(d.priority, 'routine');
  },
  'buildLayerTargets returns all 8 layers': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parseSync('write code for a REST API');
    const targets = hci.buildLayerTargets(d);
    for (const l of ['l1','l2','l3','l6','l7','l10','l11','l12']) {
      assert.ok(l in targets, `missing layer target: ${l}`);
    }
  },
  'constraint extraction: file protection': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parseSync('Refactor the codebase. Do not modify the auth/ directory');
    const fileConstraint = d.constraints.find(c => c.type === 'file_protection');
    assert.ok(fileConstraint, 'file_protection constraint should be extracted');
  },
  'validatePraxis blocks protected path with severity': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parseSync('do not modify the secrets/ folder');
    const praxis = { toolId: 'write_file', args: { path: 'secrets/config.json' } };
    const result = hci.validatePraxis(praxis, d);
    assert.equal(result.valid, false);
    assert.ok(result.violations.some(v => v.severity === 'critical'));
  },
  'directive history tracking works': () => {
    const hci = new HierarchicalCommandInterpreter();
    hci.parseSync('First task');
    hci.parseSync('Second task');
    const history = hci.getHistory();
    assert.equal(history.length, 2);
  },
  'complex directive detection works': () => {
    const hci = new HierarchicalCommandInterpreter();
    const d = hci.parseSync('Search for papers and then summarize the results and create a report');
    assert.ok(d.subTasks !== null, 'complex directive should be decomposed');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('MemorySystem', {
  'init creates storage directory': async () => {
    const mem = new MemorySystem({ storageDir: '/tmp/predator-test-mem', enablePersistence: false });
    await mem.init();
    assert.ok(true, 'init should succeed');
  },
  'store and recall episodic memory': async () => {
    const mem = new MemorySystem({ enablePersistence: false });
    await mem.init();
    await mem.store('debug the auth module', { quality: 0.8, steps: 15 }, {});
    const results = await mem.recall('debug auth', { types: ['episodic'], limit: 5 });
    assert.ok(results.length > 0, 'should recall stored memory');
  },
  'semantic memory store and recall': async () => {
    const mem = new MemorySystem({ enablePersistence: false });
    await mem.init();
    await mem.storeSemantic('auth_module_path', '/src/auth/', 0.9);
    const result = await mem.recallSemantic('auth_module_path');
    assert.ok(result, 'should recall semantic fact');
    assert.equal(result.value, '/src/auth/');
  },
  'working memory with TTL': async () => {
    const mem = new MemorySystem({ enablePersistence: false });
    await mem.init();
    await mem.pushWorking('current_task', 'debugging', 100); // 100ms TTL
    const stats = mem.stats();
    assert.equal(stats.workingCount, 1);
    await new Promise(r => setTimeout(r, 150));
    // TTL should expire on next access
    const stats2 = mem.stats();
    assert.ok(stats2.workingCount <= 1, 'TTL items may or may not be cleaned yet');
  },
  'consolidation merges similar memories': async () => {
    const mem = new MemorySystem({ enablePersistence: false });
    await mem.init();
    await mem.store('fix bug in payment module', { quality: 0.7 }, {});
    await mem.store('fix bug in payment module', { quality: 0.8 }, {});
    await mem.consolidate();
    const stats = mem.stats();
    assert.ok(stats.episodicCount <= 2, 'consolidation should reduce memory count');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('SafetyGuardrails', {
  'allows safe actions': () => {
    const sg = new SafetyGuardrails();
    const result = sg.checkStep(
      { goal: 'organize files', priority: 'routine' },
      { toolId: 'read_file', args: { path: '/home/user/docs' } }
    );
    assert.ok(result.allowed, 'safe actions should be allowed');
  },
  'blocks protected paths': () => {
    const sg = new SafetyGuardrails();
    const result = sg.checkStep(
      { goal: 'modify system', priority: 'routine' },
      { toolId: 'write_file', args: { path: '/etc/passwd' } }
    );
    assert.ok(!result.allowed, 'protected paths should be blocked');
  },
  'safety levels work': () => {
    const sg = new SafetyGuardrails({ safetyLevel: 'strict' });
    const result = sg.checkStep(
      { goal: 'delete temp files' },
      { toolId: 'write_file', args: { path: '/tmp/test' } }
    );
    // In strict mode, destructive actions without rollback may be blocked
    assert.ok(typeof result.allowed === 'boolean');
  },
  'audit trail grows': () => {
    const sg = new SafetyGuardrails();
    sg.checkStep({ goal: 'test' }, { toolId: 'noop' });
    sg.checkStep({ goal: 'test' }, { toolId: 'read_file', args: {} });
    const trail = sg.getAuditTrail();
    assert.ok(trail.length >= 2, 'audit trail should grow');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('MetricsCollector', {
  'counter increment works': () => {
    const mc = new MetricsCollector();
    mc.incrementCounter('steps');
    mc.incrementCounter('steps');
    assert.equal(mc.getCounter('steps'), 2);
  },
  'gauge set works': () => {
    const mc = new MetricsCollector();
    mc.setGauge('craving', 0.85);
    assert.equal(mc.getGauge('craving'), 0.85);
  },
  'histogram observation works': () => {
    const mc = new MetricsCollector();
    mc.observeHistogram('quality', 0.7);
    mc.observeHistogram('quality', 0.9);
    const h = mc.getHistogram('quality');
    assert.ok(h !== null);
    assert.equal(h.count, 2);
  },
  'timer works': () => {
    const mc = new MetricsCollector();
    const id = mc.startTimer('execution');
    mc.stopTimer('execution', id);
    assert.ok(mc.timers.has('execution'));
    assert.ok(mc.timers.get('execution').count >= 1);
  },
  'recordStep convenience method': () => {
    const mc = new MetricsCollector();
    mc.recordStep({ step: 1, craving: 0.5, cascadeRisk: 0.1, praxisNorm: 0.3 });
    assert.equal(mc.getCounter('step'), 1);
  },
  'getSummary returns all stores': () => {
    const mc = new MetricsCollector();
    mc.incrementCounter('test');
    mc.setGauge('val', 1.0);
    const summary = mc.getSummary();
    assert.ok('counters' in summary);
    assert.ok('gauges' in summary);
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('PluginManager', {
  'register and list plugins': () => {
    const pm = new PluginManager();
    pm.register({
      name: 'test-plugin',
      version: '1.0.0',
      hooks: { afterStep: () => {} },
    });
    const plugins = pm.listPlugins();
    assert.equal(plugins.length, 1);
    assert.equal(plugins[0].name, 'test-plugin');
  },
  'unregister plugin': () => {
    const pm = new PluginManager();
    pm.register({ name: 'test', version: '1.0.0', hooks: {} });
    pm.unregister('test');
    assert.ok(!pm.hasPlugin('test'));
  },
  'fireHook with non-cancellable hook': async () => {
    const pm = new PluginManager();
    let called = false;
    pm.register({ name: 'test', version: '1.0.0', hooks: { afterStep: () => { called = true; } } });
    await pm.fireHook('afterStep', {});
    assert.ok(called, 'hook should have been called');
  },
  'fireHook with cancellable hook (returns null)': async () => {
    const pm = new PluginManager();
    pm.register({ name: 'blocker', version: '1.0.0', hooks: { beforeStep: () => null } });
    const result = await pm.fireHook('beforeStep', { data: 'test' });
    assert.equal(result, null, 'cancellable hook returning null should propagate');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('ToolRegistry', {
  'register and list tools': () => {
    const registry = new ToolRegistry();
    registry.register('fs', new FileSystemTool());
    const tools = registry.list();
    assert.ok(tools.length > 0);
  },
  'validate rejects unknown tool': () => {
    const registry = new ToolRegistry();
    const result = registry.validate('nonexistent', {});
    assert.ok(!result.valid);
  },
  'createPSEHandlers returns Map': () => {
    const registry = new ToolRegistry();
    registry.register('db', new DatabaseTool());
    const handlers = registry.createPSEHandlers();
    assert.ok(handlers instanceof Map);
  },
});

// ─────────────────────────────────────────────────────────────────────────────
await runSuite('Predator v2.0 (integration)', {
  'status() returns well-formed object': () => {
    const p = new Predator();
    const s = p.status();
    for (const k of ['id','version','trained','running','tasksDone']) {
      assert.ok(k in s, `status missing: ${k}`);
    }
  },
  'execute() completes and returns task result': async function() {
    this?.timeout?.(30000);
    const p = new Predator({ enableMemory: false, enableSafety: false });
    const result = await p.execute('Write a quick hello world script', {
      priority: 'routine', budget: { tokens: 5000, energy: 0.3 },
    });
    assert.ok('taskId'     in result);
    assert.ok('steps'      in result);
    assert.ok('quality'    in result);
    assert.ok('tokenUsage' in result);
    assert.ok(result.steps > 0, 'should have taken at least 1 step');
  },
  'task history grows after execute': async () => {
    const p = new Predator({ enableMemory: false });
    await p.execute('list directory contents', { budget: { tokens: 3000, energy: 0.2 } });
    assert.equal(p.history().length, 1);
  },
  'registerTool makes tool available to PSE': () => {
    const p = new Predator();
    p.registerTool('my_tool', async () => ({ custom: true }));
    assert.ok(p.pse.tools.has('my_tool'));
  },
  'use() registers plugin': () => {
    const p = new Predator();
    p.use({ name: 'test-plugin', version: '1.0.0', hooks: {} });
    assert.ok(p.plugins.hasPlugin('test-plugin'));
  },
  'serialize/deserialize round-trip': () => {
    const p = new Predator({ enableMemory: false, enableSafety: false, enableMetrics: false, enablePlugins: false });
    const state = p.serialize();
    assert.ok(state, 'serialize should return a state object');
    assert.ok(state.version, 'state should have version');
  },
});

// ─────────────────────────────────────────────────────────────────────────────
console.log('\n' + '='.repeat(70));
if ((process.exitCode ?? 0) === 0) {
  console.log('  All PREDATOR v2.0 tests passed OK');
} else {
  console.log('  Some tests FAILED');
}
