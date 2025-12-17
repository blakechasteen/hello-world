import * as vscode from 'vscode';
import type { ReasoningMode } from '../api/holoLoomClient';

export interface HoloLoomConfig {
  apiEndpoint: string;
  apiTimeout: number;
  defaultReasoningMode: ReasoningMode;
  includeFileContext: boolean;
  maxContextLines: number;
  showStatusBar: boolean;
  autoConnect: boolean;
}

/**
 * Get the current HoloLoom configuration
 */
export function getConfig(): HoloLoomConfig {
  const config = vscode.workspace.getConfiguration('hololoom');

  return {
    apiEndpoint: config.get<string>('apiEndpoint', 'http://localhost:8000'),
    apiTimeout: config.get<number>('apiTimeout', 30000),
    defaultReasoningMode: config.get<ReasoningMode>('defaultReasoningMode', 'verify'),
    includeFileContext: config.get<boolean>('includeFileContext', true),
    maxContextLines: config.get<number>('maxContextLines', 100),
    showStatusBar: config.get<boolean>('showStatusBar', true),
    autoConnect: config.get<boolean>('autoConnect', true),
  };
}

/**
 * Update a configuration value
 */
export async function updateConfig<K extends keyof HoloLoomConfig>(
  key: K,
  value: HoloLoomConfig[K],
  global: boolean = true
): Promise<void> {
  const config = vscode.workspace.getConfiguration('hololoom');
  await config.update(key, value, global);
}

/**
 * Listen for configuration changes
 */
export function onConfigChange(
  callback: (config: HoloLoomConfig) => void
): vscode.Disposable {
  return vscode.workspace.onDidChangeConfiguration((event) => {
    if (event.affectsConfiguration('hololoom')) {
      callback(getConfig());
    }
  });
}
