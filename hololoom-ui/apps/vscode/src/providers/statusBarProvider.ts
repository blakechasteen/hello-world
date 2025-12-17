import * as vscode from 'vscode';
import { HoloLoomClient } from '../api/holoLoomClient';
import { getConfig, onConfigChange } from '../utils/config';

export class StatusBarProvider implements vscode.Disposable {
  private statusBarItem: vscode.StatusBarItem;
  private client: HoloLoomClient;
  private isConnected: boolean = false;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private disposables: vscode.Disposable[] = [];

  constructor(client: HoloLoomClient) {
    this.client = client;

    // Create status bar item
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.statusBarItem.command = 'hololoom.openChat';
    this.updateStatusBar();

    // Show if enabled in config
    const config = getConfig();
    if (config.showStatusBar) {
      this.statusBarItem.show();
    }

    // Listen for config changes
    this.disposables.push(
      onConfigChange((config) => {
        if (config.showStatusBar) {
          this.statusBarItem.show();
        } else {
          this.statusBarItem.hide();
        }
      })
    );

    // Start health check polling
    this.startHealthCheck();
  }

  /**
   * Update connection status
   */
  setConnected(connected: boolean): void {
    this.isConnected = connected;
    this.updateStatusBar();
  }

  /**
   * Set loading state
   */
  setLoading(loading: boolean): void {
    if (loading) {
      this.statusBarItem.text = '$(loading~spin) HoloLoom';
      this.statusBarItem.tooltip = 'Processing query...';
    } else {
      this.updateStatusBar();
    }
  }

  /**
   * Update the status bar display
   */
  private updateStatusBar(): void {
    if (this.isConnected) {
      this.statusBarItem.text = '$(check) HoloLoom';
      this.statusBarItem.tooltip = 'HoloLoom: Connected - Click to open chat';
      this.statusBarItem.backgroundColor = undefined;
    } else {
      this.statusBarItem.text = '$(warning) HoloLoom';
      this.statusBarItem.tooltip = 'HoloLoom: Disconnected - Click to reconnect';
      this.statusBarItem.backgroundColor = new vscode.ThemeColor(
        'statusBarItem.warningBackground'
      );
    }
  }

  /**
   * Start periodic health checks
   */
  private startHealthCheck(): void {
    // Check every 30 seconds
    this.healthCheckInterval = setInterval(async () => {
      try {
        const isConnected = await this.client.testConnection();
        if (isConnected !== this.isConnected) {
          this.setConnected(isConnected);
        }
      } catch {
        if (this.isConnected) {
          this.setConnected(false);
        }
      }
    }, 30000);
  }

  /**
   * Stop health checks
   */
  private stopHealthCheck(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }

  /**
   * Manually check connection
   */
  async checkConnection(): Promise<boolean> {
    const isConnected = await this.client.testConnection();
    this.setConnected(isConnected);
    return isConnected;
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.stopHealthCheck();
    this.statusBarItem.dispose();
    this.disposables.forEach((d) => d.dispose());
  }
}
