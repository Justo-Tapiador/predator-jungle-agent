#!/usr/bin/env node
/**
 * PREDATOR CLI (v2.0 – Enhanced)
 * Usage:
 *   predator task "Implement a REST API for user management"
 *   predator train [--epochs 10]
 *   predator status
 *   predator demo
 *   predator checkpoint save [--label my-checkpoint]
 *   predator checkpoint load [--id <checkpoint-id>]
 *   predator tools list
 */

import { Command }  from 'commander';
import chalk        from 'chalk';
import ora          from 'ora';
import { Predator } from '../src/index.js';
import { StateSerializer } from '../src/core/StateSerializer.js';
import { ToolRegistry } from '../src/tools/ToolRegistry.js';
import { FileSystemTool } from '../src/tools/FileSystemTool.js';
import { WebSearchTool } from '../src/WebSearchTool.js';
import { CodeExecutionTool } from '../src/tools/CodeExecutionTool.js';
import { APICallTool } from '../src/tools/APICallTool.js';
import { DatabaseTool } from '../src/tools/DatabaseTool.js';
import { formatTaskResult, formatStatus, formatTrainingHistory } from './lib/format.js';

const program = new Command();

program
  .name('predator')
  .description(chalk.red.bold('PREDATOR') + ' – Deep Agentic AI System (AJN Framework v2.0)\n' +
               chalk.gray('  Based on Agentic Theory by Justo Tapiador García (UA)'))
  .version('2.0.0');

// ── task command ──────────────────────────────────────────────────────────────
program
  .command('task <directive>')
  .description('Execute a task from a natural language Owner directive')
  .option('-p, --priority <level>', 'Priority: routine | expedited | critical', 'routine')
  .option('-t, --tokens <n>',       'Token budget', '50000')
  .option('-e, --energy <n>',       'Energy budget (normalized)', '1.0')
  .option('--no-train',             'Skip quick pre-training')
  .option('--safety <level>',       'Safety level: permissive | standard | strict', 'standard')
  .option('--memory',               'Enable persistent memory', true)
  .option('--no-memory',            'Disable persistent memory')
  .action(async (directive, opts) => {
    console.log(banner());

    const agent = new Predator({
      safetyLevel: opts.safety,
      enableMemory: opts.memory !== false,
    });
    attachProgressListeners(agent);

    if (opts.train !== false) {
      const spinner = ora({ text: chalk.yellow('Running quick pre-training...'), color: 'yellow' }).start();
      await agent.train({ epochsI: 3, epochsII_T1: 2, epochsII_T2: 2, epochsII_T3: 2,
                          epochsIII: 3, epochsIV: 2 });
      spinner.succeed(chalk.green('Pre-training complete'));
    }

    console.log(chalk.cyan('\n+-- Owner Directive ' + '-'.repeat(60)));
    console.log(chalk.white(`|  ${directive}`));
    console.log(chalk.cyan(`|  Priority: ${opts.priority}  |  Budget: ${opts.tokens} tokens`));
    console.log(chalk.cyan('+' + '-'.repeat(70) + '\n'));

    const spinner = ora({ text: chalk.magenta('PREDATOR executing task...'), color: 'magenta' }).start();
    const result  = await agent.execute(directive, {
      priority: opts.priority,
      budget: { tokens: parseInt(opts.tokens), energy: parseFloat(opts.energy) },
    });
    spinner.succeed(chalk.green('Task complete'));

    console.log(formatTaskResult(result));

    // Save checkpoint
    if (opts.memory) {
      const serializer = new StateSerializer();
      await serializer.save(agent, 'task_complete');
    }
  });

// ── train command ─────────────────────────────────────────────────────────────
program
  .command('train')
  .description('Run the full 4-phase PREDATOR training pipeline')
  .option('--epochs-i  <n>',   'Phase I epochs',  '10')
  .option('--epochs-ii <n>',   'Phase II epochs per stage', '5')
  .option('--epochs-iii <n>',  'Phase III epochs', '8')
  .option('--epochs-iv <n>',   'Phase IV epochs',  '6')
  .option('--checkpoint',      'Save checkpoints after each phase', true)
  .action(async (opts) => {
    console.log(banner());
    const agent = new Predator();
    attachTrainingListeners(agent);

    const history = await agent.train({
      epochsI:     parseInt(opts.epochsI  ?? 10),
      epochsII_T1: parseInt(opts.epochsIi ??  5),
      epochsII_T2: parseInt(opts.epochsIi ??  5),
      epochsII_T3: parseInt(opts.epochsIi ??  5),
      epochsIII:   parseInt(opts.epochsIii ?? 8),
      epochsIV:    parseInt(opts.epochsIv  ?? 6),
    });

    console.log(formatTrainingHistory(history));
    console.log(chalk.green.bold('\n  Training pipeline complete.'));

    if (opts.checkpoint) {
      const serializer = new StateSerializer();
      await serializer.save(agent, 'training_complete');
      console.log(chalk.gray('  Checkpoint saved.'));
    }
  });

// ── status command ────────────────────────────────────────────────────────────
program
  .command('status')
  .description('Print current PREDATOR system status')
  .action(() => {
    console.log(banner());
    const agent = new Predator();
    console.log(formatStatus(agent.status()));
  });

// ── demo command ──────────────────────────────────────────────────────────────
program
  .command('demo')
  .description('Run the built-in demonstration with 3 representative tasks')
  .action(async () => {
    const { runDemo } = await import('./lib/demo.js');
    await runDemo();
  });

// ── checkpoint command ────────────────────────────────────────────────────────
program
  .command('checkpoint <action>')
  .description('Manage agent checkpoints (save | load | list)')
  .option('--label <label>',    'Checkpoint label (for save)', '')
  .option('--id <id>',          'Checkpoint ID (for load)')
  .action(async (action, opts) => {
    const serializer = new StateSerializer();

    if (action === 'save') {
      const agent = new Predator();
      const spinner = ora({ text: chalk.yellow('Quick training...'), color: 'yellow' }).start();
      await agent.train({ epochsI: 2, epochsII_T1: 1, epochsII_T2: 1, epochsII_T3: 1,
                          epochsIII: 2, epochsIV: 1 });
      spinner.succeed(chalk.green('Training done'));

      const path = await serializer.save(agent, opts.label || 'manual');
      console.log(chalk.green(`  Checkpoint saved: ${path}`));
    } else if (action === 'load') {
      if (!opts.id) {
        console.log(chalk.yellow('  Please specify --id <checkpoint-id>'));
        return;
      }
      const agent = new Predator();
      await serializer.load(agent, opts.id);
      console.log(chalk.green(`  Checkpoint loaded: ${opts.id}`));
      console.log(formatStatus(agent.status()));
    } else if (action === 'list') {
      const checkpoints = await serializer.listCheckpoints();
      if (checkpoints.length === 0) {
        console.log(chalk.yellow('  No checkpoints found.'));
        return;
      }
      console.log(chalk.cyan('\n  Available Checkpoints:'));
      for (const cp of checkpoints) {
        console.log(chalk.white(`    ${cp.id.padEnd(30)} ${cp.label?.padEnd(20) ?? ''} ${cp.timestamp}`));
      }
    }
  });

// ── tools command ─────────────────────────────────────────────────────────────
program
  .command('tools <action>')
  .description('Manage tools (list | register)')
  .action(async (action) => {
    if (action === 'list') {
      const registry = new ToolRegistry();
      registry.registerMany([
        new FileSystemTool(),
        new WebSearchTool(),
        new CodeExecutionTool(),
        new APICallTool(),
        new DatabaseTool(),
      ]);
      const tools = registry.list();
      console.log(chalk.cyan('\n  Registered Tools:'));
      for (const tool of tools) {
        console.log(chalk.white(`    ${tool.id.padEnd(20)} ${tool.description ?? ''}`));
      }
    }
  });

program.parse();

// ── Helpers ───────────────────────────────────────────────────────────────────

function banner() {
  return chalk.red(`
  ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗
  ██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
  ██████╔╝██████╔╝█████╗  ██║  ██║███████║   ██║   ██║   ██║██████╔╝
  ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██╔══██║   ██║   ██║   ██║██╔══██╗
  ██║     ██║  ██║███████╗██████╔╝██║  ██║   ██║   ╚██████╔╝██║  ██║
  ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
`) + chalk.gray('  v2.0 · Agentic Theory · Justo Tapiador García (UA) · 2024-2026\n');
}

function attachProgressListeners(agent) {
  agent.on('tpsStep', r => {
    if (r.step % 10 === 0) {
      process.stdout.write(
        chalk.gray(`  [step ${String(r.step).padStart(3)}] `) +
        chalk.yellow(`craving=${r.craving?.toFixed(3) ?? 'n/a'} `) +
        chalk.blue(`cascadeRisk=${r.cascadeRisk?.toFixed(3) ?? 'n/a'} `) +
        chalk.green(`tokens+=${r.tokensOut ?? 0}`) + '\n'
      );
    }
  });
  agent.on('ownerEscalation', e => {
    console.log(chalk.red.bold(`\n  OWNER ESCALATION: ${e.message}`));
  });
  agent.on('extinction', e => {
    console.log(chalk.red(`  Extinction event on unit ${e.id} (#${e.extinctions})`));
  });
  agent.on('safetyBlock', e => {
    console.log(chalk.red(`  SAFETY BLOCK: ${e.reason}`));
  });
}

function attachTrainingListeners(agent) {
  let lastPhase = null;
  agent.on('phaseStart', e => {
    if (lastPhase !== e.phase) {
      lastPhase = e.phase;
      console.log(chalk.cyan.bold(`\n  Phase ${e.phase}: ${e.name}`));
    }
  });
  agent.on('trainingProgress', p => {
    const bar = String.fromCharCode(9608).repeat(Math.floor((p.epoch / p.epochs) * 20)).padEnd(20, String.fromCharCode(9617));
    process.stdout.write(
      `\r  [${bar}] ${p.epoch}/${p.epochs}` +
      (p.loss     !== undefined ? chalk.gray(` loss=${p.loss.toFixed(4)}`) : '') +
      (p.satRate  !== undefined ? chalk.green(` sat=${p.satRate.toFixed(2)}`) : '') +
      (p.recoveryRate !== undefined ? chalk.yellow(` rec=${p.recoveryRate.toFixed(2)}`) : '')
    );
  });
}
