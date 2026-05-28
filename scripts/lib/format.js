/**
 * CLI output formatters (v2.0 – Enhanced)
 */
import chalk from 'chalk';

export function formatTaskResult(r) {
  const successIcon = r.success ? chalk.green('OK') : chalk.red('FAIL');
  const qualityBar  = String.fromCharCode(9608).repeat(Math.floor(r.quality * 20)).padEnd(20, String.fromCharCode(9617));
  const tokenPct    = ((r.tokenUsage.tokensUsed / r.tokenUsage.tokenBudget) * 100).toFixed(1);
  const energyPct   = ((r.tokenUsage.energyUsed / r.tokenUsage.energyBudget) * 100).toFixed(1);
  const budgetWarn  = r.tokenUsage.budgetWarning ?? 'ok';
  const warnIcon    = budgetWarn === 'critical' ? chalk.red(' !') : budgetWarn === 'warning' ? chalk.yellow(' ~') : '';

  return `
${chalk.cyan('+-- Task Result ' + '-'.repeat(60))}
${chalk.cyan('|')} ${successIcon} ${chalk.white.bold((r.goal ?? '').slice(0, 60))}
${chalk.cyan('|')}
${chalk.cyan('|')} ${chalk.gray('Quality   ')} [${chalk.green(qualityBar)}] ${(r.quality * 100).toFixed(1)}%
${chalk.cyan('|')} ${chalk.gray('Steps     ')} ${r.steps}  |  ${chalk.gray('Emitted')} ${r.feedbackCount} praxes
${chalk.cyan('|')} ${chalk.gray('Tokens    ')} ${r.tokenUsage.tokensUsed.toLocaleString()} / ${r.tokenUsage.tokenBudget.toLocaleString()} (${tokenPct}%)${warnIcon}
${chalk.cyan('|')} ${chalk.gray('Energy    ')} ${r.tokenUsage.energyUsed.toFixed(4)} / ${r.tokenUsage.energyBudget.toFixed(2)} (${energyPct}%)
${chalk.cyan('|')} ${chalk.gray('Wall time ')} ${r.wallClockMs}ms
${chalk.cyan('|')} ${chalk.gray('Cascades  ')} ${r.cascadeEvents}  |  ${chalk.gray('Extinctions')} ${r.extinctions}
${chalk.cyan('+' + '-'.repeat(70))}`;
}

export function formatStatus(s) {
  const trained = s.trained ? chalk.green('yes') : chalk.yellow('no');
  const running = s.running ? chalk.green('yes') : chalk.gray('no');

  return `
${chalk.cyan('+-- PREDATOR Status ' + '-'.repeat(58))}
${chalk.cyan('|')} ${chalk.gray('ID       ')} ${s.id}
${chalk.cyan('|')} ${chalk.gray('Version  ')} ${s.version ?? '2.0.0'}
${chalk.cyan('|')} ${chalk.gray('Trained  ')} ${trained}
${chalk.cyan('|')} ${chalk.gray('Running  ')} ${running}
${chalk.cyan('|')} ${chalk.gray('Tasks    ')} ${s.tasksDone} completed
${chalk.cyan('|')} ${chalk.gray('Extinct. ')} ${s.extinctions} events logged
${chalk.cyan('|')} ${chalk.gray('Memory   ')} ${s.memoryStats ? `${s.memoryStats.episodicCount} ep | ${s.memoryStats.semanticCount} sem` : 'N/A'}
${chalk.cyan('|')} ${chalk.gray('Safety   ')} ${s.safetyLevel ?? 'standard'}
${chalk.cyan('+' + '-'.repeat(70))}`;
}

export function formatTrainingHistory(h) {
  const avg = (arr, key) => arr.reduce((s, r) => s + (r[key] ?? 0), 0) / Math.max(arr.length, 1);

  return `
${chalk.cyan('+-- Training Summary ' + '-'.repeat(58))}
${chalk.cyan('|')} ${chalk.yellow('Phase I  ')}  Epochs: ${h.phaseI.length}   Final loss: ${h.phaseI.at(-1)?.loss?.toFixed(4) ?? 'n/a'}
${chalk.cyan('|')} ${chalk.yellow('Phase II ')}  Epochs: ${h.phaseII.length}  Avg sat rate: ${avg(h.phaseII, 'saturationRate').toFixed(3)}
${chalk.cyan('|')} ${chalk.yellow('Phase III')}  Epochs: ${h.phaseIII.length}  Final loss: ${h.phaseIII.at(-1)?.totalLoss?.toFixed(4) ?? 'n/a'}
${chalk.cyan('|')} ${chalk.yellow('Phase IV ')}  Epochs: ${h.phaseIV.length}   Resilience: ${avg(h.phaseIV, 'resilience').toFixed(3)}
${chalk.cyan('+' + '-'.repeat(70))}`;
}
