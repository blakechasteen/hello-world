import * as vscode from 'vscode';
import { HoloLoomClient } from './api/holoLoomClient';
import { StatusBarProvider } from './providers/statusBarProvider';
import { ChatViewProvider } from './providers/chatViewProvider';
import { MemoryViewProvider } from './providers/memoryViewProvider';
import { registerCommands } from './commands';
import { getConfig } from './utils/config';

let client: HoloLoomClient;
let statusBarProvider: StatusBarProvider;

export async function activate(context: vscode.ExtensionContext) {
  console.log('HoloLoom extension is activating...');

  // Initialize API client
  const config = getConfig();
  client = new HoloLoomClient(config.apiEndpoint, config.apiTimeout);

  // Create status bar provider
  statusBarProvider = new StatusBarProvider(client);
  context.subscriptions.push(statusBarProvider);

  // Register webview providers
  const chatViewProvider = new ChatViewProvider(context.extensionUri, client);
  const memoryViewProvider = new MemoryViewProvider(context.extensionUri, client);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'hololoom.chatView',
      chatViewProvider,
      {
        webviewOptions: {
          retainContextWhenHidden: true,
        },
      }
    )
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'hololoom.memoryView',
      memoryViewProvider,
      {
        webviewOptions: {
          retainContextWhenHidden: true,
        },
      }
    )
  );

  // Register commands
  registerCommands(context, client, chatViewProvider);

  // Auto-connect if enabled
  if (config.autoConnect) {
    try {
      const health = await client.checkHealth();
      if (health.status === 'healthy') {
        statusBarProvider.setConnected(true);
        vscode.window.showInformationMessage('HoloLoom: Connected to server');
      }
    } catch (error) {
      statusBarProvider.setConnected(false);
      console.log('HoloLoom: Failed to connect on startup', error);
    }
  }

  console.log('HoloLoom extension activated successfully');
}

export function deactivate() {
  console.log('HoloLoom extension is deactivating...');
}

export function getClient(): HoloLoomClient {
  return client;
}

export function getStatusBarProvider(): StatusBarProvider {
  return statusBarProvider;
}
