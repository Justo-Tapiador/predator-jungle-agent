#!/usr/bin/env node
/**
 * PREDATOR Benchmark (v2.0) – Inference efficiency analysis
 * Measures tokens, energy, latency and quality across task types
 * with the enhanced v2.0 feature set
 */

import chalk from 'chalk';
import { Predator } from '../src/index.js';

const BENCHMARK_TASKS = [
  { label: 'Autonomous Debugging',      directive: 'Debug all failing tests in the authentication module',         priority: 'critical',  tokens: 20000 },
  { label: 'Research Assistance',       directive: 'Search and synthesize latest papers on neural efficiency',     priority: 'expedited', tokens: 15000 },
  { label: 'Multi-tool Orchestration',  directive: 'Search, analyze, and refactor the payment processing code',    priority: 'routine',   tokens: 25000 },
  { label: 'Code Generation',           directive: 'Implement a REST API for user management with tests',          priority: 'routine',   tokens: 18000 },
  { label: 'Real-time Monitoring',      directive: 'Monitor all API endpoints and report anomalies urgently',      priority: 'critical',  tokens: 10000 },
  { label: 'Security Audit',            directive: 'Audit all authentication endpoints for security vulnerabilities', priority: 'expedited', tokens: 15000 },
  { label: 'Documentation Gen',         directive: 'Generate comprehensive API documentation from source code',    priority: 'routine',   tokens: 12000 },
];

const RUNS = 3;

async function benchmark() {
  console.log(chalk.red.bold('\n  PREDATOR v2.0 Benchmark Suite\n'));
  console.log(chalk.gray('  Features: Real Transformers | Memory | Safety | Metrics | Plugins\n'));

  const agent = new Predator({ enableMemory: true, enableMetrics: true });

  process.stdout.write(chalk.yellow('  Training agent...'));
  await agent.train({ epochsI: 3, epochsII_T1: 2, epochsII_T2: 2, epochsII_T3: 2, epochsIII: 3, epochsIV: 2 });
  console.log(chalk.green(' done\n'));

  const results = [];

  for (const task of BENCHMARK_TASKS) {
    process.stdout.write(chalk.cyan(`  > ${task.label.padEnd(30)}`));

    const runs = [];
    for (let r = 0; r < RUNS; r++) {
      const start = performance.now();
      const res = await agent.execute(task.directive, {
        priority: task.priority,
        budget: { tokens: task.tokens, energy: 1.0 },
      });
      const elapsed = performance.now() - start;
      runs.push({
        steps:   res.steps,
        tokens:  res.tokenUsage.tokensUsed,
        energy:  res.tokenUsage.energyUsed,
        quality: res.quality,
        ms:      elapsed,
        cascades: res.cascadeEvents,
        extinctions: res.extinctions,
        satSteps: res.stepRecords ? res.stepRecords.filter(s => s.phase === 'SATURATED').length : 0,
      });
    }

    const avg = (key) => runs.reduce((s, r) => s + r[key], 0) / runs.length;
    const summary = {
      label:       task.label,
      priority:    task.priority,
      avgSteps:    avg('steps').toFixed(1),
      avgTokens:   avg('tokens').toFixed(0),
      avgEnergy:   avg('energy').toFixed(4),
      avgQuality:  avg('quality').toFixed(3),
      avgMs:       avg('ms').toFixed(0),
      tokenEff:    ((1 - avg('tokens') / task.tokens) * 100).toFixed(1),
      satRate:     ((avg('satSteps') / avg('steps')) * 100).toFixed(1),
      tokPerStep:  (avg('tokens') / avg('steps')).toFixed(1),
    };
    results.push(summary);

    console.log(
      chalk.white(`steps=${summary.avgSteps.padStart(5)} `) +
      chalk.blue(`tok=${summary.avgTokens.padStart(6)} `) +
      chalk.green(`Q=${(summary.avgQuality*100).toFixed(0).padStart(3)}% `) +
      chalk.yellow(`sat=${summary.satRate}% `) +
      chalk.gray(`${summary.avgMs}ms`)
    );
  }

  // ── Summary table ──────────────────────────────────────────────────────────
  console.log(chalk.cyan('\n  +-- Inference Efficiency Summary ' + '-'.repeat(40) + '+'));
  console.log(chalk.cyan('  |') +
    chalk.gray(' Task'.padEnd(33)) +
    chalk.blue('Tok/Step'.padEnd(10)) +
    chalk.green('Quality'.padEnd(10)) +
    chalk.yellow('SatRate'.padEnd(10)) +
    chalk.white('BudgetRem') +
    chalk.cyan(' |'));
  console.log(chalk.cyan('  +' + '-'.repeat(69) + '+'));

  for (const r of results) {
    console.log(chalk.cyan('  |') +
      chalk.white(r.label.slice(0,30).padEnd(33)) +
      chalk.blue(r.tokPerStep.padEnd(10)) +
      chalk.green(((r.avgQuality)*100).toFixed(0)+'%'.padEnd(10)) +
      chalk.yellow(r.satRate+'%'.padEnd(10)) +
      chalk.green(r.tokenEff+'%') +
      chalk.cyan(' |'));
  }
  console.log(chalk.cyan('  +' + '-'.repeat(69) + '+\n'));

  // ── Key insights ───────────────────────────────────────────────────────────
  const avgSat = results.reduce((s, r) => s + parseFloat(r.satRate), 0) / results.length;
  const avgBudget = results.reduce((s, r) => s + parseFloat(r.tokenEff), 0) / results.length;
  console.log(chalk.gray(`  Average saturation rate: ${avgSat.toFixed(1)}% of steps suppressed`));
  console.log(chalk.gray(`  Average token budget remaining: ${avgBudget.toFixed(1)}%`));
  console.log(chalk.gray(`  (Higher saturation = better efficiency per AJN theory prediction)\n`));

  // ── v2.0 features impact ───────────────────────────────────────────────────
  if (agent.metrics) {
    const metricsSummary = agent.metrics.getSummary();
    console.log(chalk.cyan('  v2.0 Feature Metrics:'));
    console.log(chalk.gray(`  Safety checks: ${metricsSummary.counters?.get?.('safety_checks') ?? 'N/A'}`));
    console.log(chalk.gray(`  Memory lookups: ${metricsSummary.counters?.get?.('memory_lookups') ?? 'N/A'}`));
    console.log(chalk.gray(`  Plugin hooks: ${metricsSummary.counters?.get?.('plugin_hooks') ?? 'N/A'}`));
  }
}

benchmark().catch(console.error);
