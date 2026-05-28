/**
 * Configuration schema validation for PREDATOR Agent v2.0
 * Uses Zod for runtime type checking and validation
 */

import { z } from 'zod';

export const AgentConfigSchema = z.object({
  agent: z.object({
    version: z.string().default('2.0.0'),
    dModel: z.number().int().min(16).max(512).default(64),
    nHeads: z.number().int().min(1).max(32).default(4),
    dFF: z.number().int().min(64).max(2048).default(256),
    maxSteps: z.number().int().min(10).max(10000).default(200),
  }).default({}),

  ajn: z.object({
    betaM: z.number().min(0).max(1).default(0.85),
    lambdaUp: z.number().min(0).max(1).default(0.30),
    delta: z.number().min(0).max(1).default(0.02),
    thetaSat: z.number().min(0).max(1).default(0.75),
    tau: z.number().int().min(1).max(200).default(20),
    eta: z.number().min(0).max(1).default(0.05),
    praximDim: z.number().int().min(8).max(512).default(64),
  }).default({}),

  budget: z.object({
    defaultTokens: z.number().int().min(1000).default(50000),
    defaultEnergy: z.number().min(0.1).max(100).default(1.0),
    defaultWallClockMs: z.number().int().min(1000).default(300000),
  }).default({}),

  cascade: z.object({
    rhoWarn: z.number().min(0).max(1).default(0.35),
    rhoModerate: z.number().min(0).max(1).default(0.50),
    rhoCritical: z.number().min(0).max(1).default(0.65),
    pollMs: z.number().int().min(100).max(10000).default(500),
    selfHealing: z.boolean().default(true),
  }).default({}),

  memory: z.object({
    storageDir: z.string().default('./data/memory'),
    maxEpisodic: z.number().int().min(10).max(100000).default(1000),
    maxWorking: z.number().int().min(1).max(100).default(10),
    enablePersistence: z.boolean().default(true),
  }).default({}),

  safety: z.object({
    safetyLevel: z.enum(['permissive', 'standard', 'strict']).default('standard'),
    protectedPaths: z.array(z.string()).default(['/etc', '/root', '/sys', '/proc']),
    maxFileSize: z.number().int().min(1024).default(10485760),
    rateLimits: z.object({
      perMinute: z.number().int().min(1).default(60),
      perHour: z.number().int().min(1).default(500),
    }).default({}),
  }).default({}),

  training: z.object({
    epochsI: z.number().int().min(1).default(10),
    epochsII_T1: z.number().int().min(1).default(5),
    epochsII_T2: z.number().int().min(1).default(5),
    epochsII_T3: z.number().int().min(1).default(5),
    epochsIII: z.number().int().min(1).default(8),
    epochsIV: z.number().int().min(1).default(6),
    batchSize: z.number().int().min(1).max(1024).default(32),
    enableCheckpoints: z.boolean().default(true),
    earlyStoppingPatience: z.number().int().min(1).default(5),
  }).default({}),

  tools: z.object({
    timeout: z.number().int().min(1000).max(300000).default(30000),
    maxRetries: z.number().int().min(0).max(10).default(2),
    maxConcurrent: z.number().int().min(1).max(20).default(3),
  }).default({}),

  llm: z.object({
    provider: z.enum(['none', 'openai', 'local']).default('none'),
    model: z.string().default('default'),
    temperature: z.number().min(0).max(2).default(0.7),
    maxTokens: z.number().int().min(1).max(32768).default(2048),
  }).default({}),

  logging: z.object({
    level: z.enum(['trace', 'debug', 'info', 'warn', 'error', 'fatal']).default('info'),
    enableConsole: z.boolean().default(true),
    enableFile: z.boolean().default(false),
  }).default({}),
});

/**
 * Load and validate configuration from multiple sources:
 * 1. Default config
 * 2. Environment-specific config (production.json)
 * 3. Environment variables
 */
export async function loadConfig(configDir = './config') {
  // 1. Load defaults
  let config = {};
  try {
    const { readFileSync } = await import('fs');
    const defaultPath = `${configDir}/default.json`;
    const defaultContent = readFileSync(defaultPath, 'utf-8');
    config = JSON.parse(defaultContent);
  } catch (e) {
    // Use schema defaults
  }

  // 2. Load environment-specific overrides
  const env = process.env.NODE_ENV ?? 'development';
  try {
    const { readFileSync } = await import('fs');
    const envPath = `${configDir}/${env}.json`;
    const envContent = readFileSync(envPath, 'utf-8');
    const envConfig = JSON.parse(envContent);
    config = deepMerge(config, envConfig);
  } catch (e) {
    // No env config file, that's ok
  }

  // 3. Apply environment variable overrides
  config = applyEnvOverrides(config);

  // 4. Validate with Zod schema
  try {
    return AgentConfigSchema.parse(config);
  } catch (e) {
    console.error('Configuration validation failed:', e.errors);
    return AgentConfigSchema.parse({}); // Fall back to defaults
  }
}

function deepMerge(target, source) {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      result[key] = deepMerge(result[key] ?? {}, source[key]);
    } else {
      result[key] = source[key];
    }
  }
  return result;
}

function applyEnvOverrides(config) {
  const envMap = {
    'LLM_PROVIDER':           ['llm', 'provider'],
    'LLM_MODEL':              ['llm', 'model'],
    'LLM_TEMPERATURE':        ['llm', 'temperature', Number],
    'LLM_MAX_TOKENS':         ['llm', 'maxTokens', Number],
    'LLM_ENDPOINT':           ['llm', 'endpoint'],
    'PREDATOR_DMODEL':        ['agent', 'dModel', Number],
    'PREDATOR_NHEADS':        ['agent', 'nHeads', Number],
    'PREDATOR_MAX_STEPS':     ['agent', 'maxSteps', Number],
    'MEMORY_DIR':             ['memory', 'storageDir'],
    'MEMORY_PERSIST':         ['memory', 'enablePersistence', v => v === 'true'],
    'SAFETY_LEVEL':           ['safety', 'safetyLevel'],
    'SAFETY_MAX_FILE_SIZE':   ['safety', 'maxFileSize', Number],
    'PORT':                   ['web', 'port', Number],
    'LOG_LEVEL':              ['logging', 'level'],
  };

  for (const [envVar, path] of Object.entries(envMap)) {
    const value = process.env[envVar];
    if (value !== undefined) {
      let obj = config;
      for (let i = 0; i < path.length - 1; i++) {
        if (!obj[path[i]]) obj[path[i]] = {};
        obj = obj[path[i]];
      }
      const key = path[path.length - 1];
      const transformer = typeof path[path.length - 1] === 'function'
        ? path.pop()
        : (v) => v;
      obj[key] = transformer(value);
    }
  }

  return config;
}
