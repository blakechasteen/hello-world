import * as vscode from 'vscode';
import * as path from 'path';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

/**
 * HoloLoom LSP Client
 *
 * Manages the Language Server Protocol client for HoloLoom neural memory system.
 * Provides:
 * - Automatic server startup/shutdown
 * - Configuration management
 * - Graceful fallback to HTTP API if server unavailable
 * - Auto-detection of HoloLoom installation
 */
export class HoloLoomLSPClient {
    private client: LanguageClient | undefined;
    private logger: vscode.OutputChannel;
    private context: vscode.ExtensionContext;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
        this.logger = vscode.window.createOutputChannel('HoloLoom LSP');
    }

    /**
     * Start the LSP client and connect to HoloLoom LSP server
     */
    async start(): Promise<boolean> {
        try {
            this.logger.appendLine('Starting HoloLoom LSP client...');

            // Check if LSP is enabled in configuration
            const config = vscode.workspace.getConfiguration('hololoom');
            const enabled = config.get<boolean>('lsp.enabled', true);

            if (!enabled) {
                this.logger.appendLine('LSP client disabled in configuration');
                return false;
            }

            // Get HoloLoom path (auto-detect or use configured path)
            const hololoomPath = this.getHoloLoomPath();
            if (!hololoomPath) {
                this.logger.appendLine('HoloLoom installation not found. Falling back to HTTP API.');
                vscode.window.showWarningMessage(
                    'HoloLoom LSP: Installation not found. Using HTTP API fallback.',
                    'Configure Path'
                ).then(selection => {
                    if (selection === 'Configure Path') {
                        vscode.commands.executeCommand('workbench.action.openSettings', 'hololoom.lsp.hololoomPath');
                    }
                });
                return false;
            }

            this.logger.appendLine(`Using HoloLoom path: ${hololoomPath}`);

            // Get Python interpreter path
            const pythonPath = this.getPythonPath();
            this.logger.appendLine(`Using Python: ${pythonPath}`);

            // Server options (start HoloLoom LSP server)
            const serverOptions: ServerOptions = {
                command: pythonPath,
                args: ['-m', 'HoloLoom.lsp.server', '--log-level', 'INFO'],
                options: {
                    cwd: hololoomPath,
                    env: {
                        ...process.env,
                        PYTHONPATH: hololoomPath
                    }
                }
            };

            // Client options (all file types for knowledge graph integration)
            const clientOptions: LanguageClientOptions = {
                documentSelector: [
                    { scheme: 'file', pattern: '**/*' }
                ],
                synchronize: {
                    fileEvents: vscode.workspace.createFileSystemWatcher('**/*')
                },
                outputChannel: this.logger,
                initializationOptions: {
                    workspacePath: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
                }
            };

            // Create and start client
            this.client = new LanguageClient(
                'hololoom',
                'HoloLoom Language Server',
                serverOptions,
                clientOptions
            );

            await this.client.start();
            this.logger.appendLine('HoloLoom LSP client started successfully');

            // Show status in status bar
            vscode.window.setStatusBarMessage('$(check) HoloLoom LSP connected', 5000);

            return true;

        } catch (error: any) {
            this.logger.appendLine(`Failed to start LSP client: ${error.message}`);
            this.logger.appendLine(error.stack || '');

            vscode.window.showErrorMessage(
                `HoloLoom LSP failed to start: ${error.message}. Falling back to HTTP API.`,
                'View Logs'
            ).then(selection => {
                if (selection === 'View Logs') {
                    this.logger.show();
                }
            });

            return false;
        }
    }

    /**
     * Stop the LSP client gracefully
     */
    async stop(): Promise<void> {
        if (this.client) {
            this.logger.appendLine('Stopping HoloLoom LSP client...');
            try {
                await this.client.stop();
                this.logger.appendLine('HoloLoom LSP client stopped');
            } catch (error: any) {
                this.logger.appendLine(`Error stopping client: ${error.message}`);
            }
            this.client = undefined;
        }
    }

    /**
     * Restart the LSP client
     */
    async restart(): Promise<boolean> {
        await this.stop();
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1s
        return await this.start();
    }

    /**
     * Get the active LSP client (or undefined if not started)
     */
    getClient(): LanguageClient | undefined {
        return this.client;
    }

    /**
     * Check if LSP client is running
     */
    isRunning(): boolean {
        return this.client !== undefined;
    }

    /**
     * Auto-detect HoloLoom installation path
     */
    private getHoloLoomPath(): string | undefined {
        const config = vscode.workspace.getConfiguration('hololoom');
        const configuredPath = config.get<string>('lsp.hololoomPath');

        if (configuredPath) {
            return configuredPath;
        }

        // Auto-detection: check workspace folders
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders) {
            for (const folder of workspaceFolders) {
                // Check if HoloLoom package exists in workspace
                const hololoomPath = path.join(folder.uri.fsPath, 'HoloLoom');
                try {
                    const fs = require('fs');
                    if (fs.existsSync(hololoomPath)) {
                        const initPath = path.join(hololoomPath, '__init__.py');
                        if (fs.existsSync(initPath)) {
                            return folder.uri.fsPath;
                        }
                    }
                } catch (error) {
                    // Continue searching
                }
            }
        }

        // Check parent directory of workspace
        if (workspaceFolders && workspaceFolders.length > 0) {
            const parentDir = path.dirname(workspaceFolders[0].uri.fsPath);
            const hololoomPath = path.join(parentDir, 'HoloLoom');
            try {
                const fs = require('fs');
                if (fs.existsSync(hololoomPath)) {
                    return parentDir;
                }
            } catch (error) {
                // Not found
            }
        }

        return undefined;
    }

    /**
     * Get Python interpreter path from configuration or detect
     */
    private getPythonPath(): string {
        const config = vscode.workspace.getConfiguration('hololoom');
        const configuredPython = config.get<string>('lsp.pythonPath');

        if (configuredPython) {
            return configuredPython;
        }

        // Try to use Python extension's interpreter
        try {
            const pythonExtension = vscode.extensions.getExtension('ms-python.python');
            if (pythonExtension && pythonExtension.isActive) {
                const pythonPath = pythonExtension.exports?.settings?.getExecutionDetails?.()?.execCommand;
                if (pythonPath) {
                    return Array.isArray(pythonPath) ? pythonPath[0] : pythonPath;
                }
            }
        } catch (error) {
            // Python extension not available
        }

        // Default to 'python' or 'python3'
        return process.platform === 'win32' ? 'python' : 'python3';
    }

    /**
     * Send a custom request to the LSP server
     */
    async sendRequest<T>(method: string, params: any): Promise<T> {
        if (!this.client) {
            throw new Error('LSP client not started');
        }
        return await this.client.sendRequest(method, params);
    }

    /**
     * Send a custom notification to the LSP server
     */
    sendNotification(method: string, params?: any): void {
        if (!this.client) {
            throw new Error('LSP client not started');
        }
        this.client.sendNotification(method, params);
    }
}
