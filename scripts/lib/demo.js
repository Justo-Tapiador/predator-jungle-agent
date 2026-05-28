/**
 * PREDATOR built-in demonstration (v2.0 – Enhanced)
 * Runs 3 tasks showcasing different priority classes and domains
 * with the enhanced v2.0 feature set
 */
import chalk  from 'chalk';
import ora    from 'ora';
import { Predator } from '../../src/index.js';

const DEMO_TASKS = [
  {
    directive: 'Debug and fix all failing unit tests in the authentication module',
    opts:      { priority: 'critical', budget: { tokens: 30_000, energy: 0.8 } },
    label:     'Autonomous Debugging (critical)',
  },
  {
    directive: 'Search and summarize the latest research on transformer efficiency improvements',
    opts:      { priority: 'expedited', budget: { tokens: 20_000, energy: 0.6 } },
    label:     'Scientific Research Assistance (expedited)',
  },
  {
    directive: 'Organize and refactor the database access layer with proper connection pooling',
    opts:      { priority: 'routine', budget: { tokens: 15_000, energy: 0.5 } },
    label:     'Code Refactoring (routine)',
  },
];

export async function runDemo() {
  console.log(chalk.red.bold(`
+======================================================================+
|            PREDATOR  v2.0  -  Built-in Demonstration                 |
|   Agentic Theory · Justo Tapiador García (UA) · 2024-2026            |
|   Enhanced: Real Transformers · Memory · Safety · Plugins · LLM      |
+======================================================================+
`));

  // Build and train once
  const agent = new Predator({
    enableMemory: true,
    enableSafety: true,
    enableMetrics: true,
  });

  const spinner = ora({ text: chalk.yellow('Initialising PREDATOR (4-phase training)...'), color: 'yellow' }).start();

  let lastPhase = null;
  agent.on('phaseStart', e => {
    if (lastPhase !== e.phase) {
      lastPhase = e.phase;
      spinner.text = chalk.yellow(`Phase ${e.phase}: ${e.name}...`);
    }
  });

  await agent.train({
    epochsI: 4, epochsII_T1: 3, epochsII_T2: 3, epochsII_T3: 3,
    epochsIII: 4, epochsIV: 3,
  });
  spinner.succeed(chalk.green('Training complete'));

  // Show enabled features
  console.log(chalk.cyan('\n  Enabled Features:'));
  console.log(chalk.white('    Memory System:    ') + (agent.memory ? chalk.green('ON') : chalk.gray('OFF')));
  console.log(chalk.white('    Safety Guardrails:') + (agent.safety ? chalk.green('ON') : chalk.gray('OFF')));
  console.log(chalk.white('    Metrics:          ') + (agent.metrics ? chalk.green('ON') : chalk.gray('OFF')));
  console.log(chalk.white('    Plugins:          ') + (agent.plugins ? chalk.green('ON') : chalk.gray('OFF')));

  // Run demo tasks
  for (let i = 0; i < DEMO_TASKS.length; i++) {
    const t = DEMO_TASKS[i];
    console.log(chalk.cyan.bold(`\n  [${i+1}/${DEMO_TASKS.length}] ${t.label}`));
    console.log(chalk.gray(`  "${t.directive}"`));

    const spin = ora({ text: chalk.magenta('Running...'), color: 'magenta' }).start();
    const result = await agent.execute(t.directive, t.opts);
    spin.succeed(result.success ? chalk.green('Success') : chalk.yellow('Partial'));

    // Efficiency mini-chart
    const steps   = result.stepRecords;
    const satSteps = steps ? steps.filter(s => s.phase === 'SATURATED').length : 0;
    const tokenEff = ((1 - result.tokenUsage.tokensUsed / result.tokenUsage.tokenBudget) * 100).toFixed(1);

    console.log(chalk.gray(
      `  Steps: ${result.steps} | Saturated: ${satSteps}/${result.steps} | ` +
      `Token budget remaining: ${tokenEff}% | Quality: ${(result.quality*100).toFixed(1)}%`
    ));

    // Show memory stats if available
    if (agent.memory) {
      const memStats = agent.memory.stats();
      console.log(chalk.gray(
        `  Memory: ${memStats.episodicCount} episodic | ${memStats.semanticCount} semantic | ${memStats.workingCount} working`
      ));
    }
  }

  // Show metrics summary if available
  if (agent.metrics) {
    const summary = agent.metrics.getSummary();
    console.log(chalk.cyan('\n  Metrics Summary:'));
    if (summary.counters) {
      const taskSuccess = summary.counters.get?.('task_success') ?? 0;
      const taskFailed = summary.counters.get?.('task_failed') ?? 0;
      console.log(chalk.gray(`  Tasks: ${taskSuccess} success, ${taskFailed} failed`));
    }
  }

  console.log(chalk.green.bold('\n  Demo complete. PREDATOR v2.0 is operational.\n'));
}
