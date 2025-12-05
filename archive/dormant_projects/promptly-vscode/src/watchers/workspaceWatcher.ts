import * as vscode from 'vscode';
import { HoloLoomCommands } from '../commands/hololoomCommands';

interface IndexingProgress {
    filesProcessed: number;
    filesTotal: number;
    currentFile: string;
}

export class WorkspaceWatcher {
    private fileWatcher: vscode.FileSystemWatcher | undefined;
    private hololoomCommands: HoloLoomCommands;
    private debounceTimers: Map<string, NodeJS.Timeout> = new Map();
    private isIndexing: boolean = false;
    private indexingProgress: IndexingProgress | null = null;

    // Debounce delay (ms) - wait this long after last change before re-indexing
    private static readonly DEBOUNCE_DELAY = 2000; // 2 seconds

    // File patterns to watch
    private static readonly WATCH_PATTERNS = [
        '**/*.{ts,tsx,js,jsx,py,md}'
    ];

    constructor() {
        this.hololoomCommands = new HoloLoomCommands();
    }

    public async start() {
        console.log('WorkspaceWatcher: Starting file system watcher...');

        // Create file system watcher for code files
        const watchPattern = `{${WorkspaceWatcher.WATCH_PATTERNS.join(',')}}`;
        this.fileWatcher = vscode.workspace.createFileSystemWatcher(watchPattern);

        // Watch for file changes
        this.fileWatcher.onDidChange(async (uri) => {
            await this.onFileChanged(uri);
        });

        // Watch for file creation
        this.fileWatcher.onDidCreate(async (uri) => {
            await this.onFileCreated(uri);
        });

        // Watch for file deletion
        this.fileWatcher.onDidDelete(async (uri) => {
            await this.onFileDeleted(uri);
        });

        console.log('WorkspaceWatcher: File system watcher started');

        // Index workspace on startup
        await this.indexWorkspace();
    }

    public stop() {
        console.log('WorkspaceWatcher: Stopping file system watcher...');

        if (this.fileWatcher) {
            this.fileWatcher.dispose();
            this.fileWatcher = undefined;
        }

        // Clear all debounce timers
        this.debounceTimers.forEach(timer => clearTimeout(timer));
        this.debounceTimers.clear();

        console.log('WorkspaceWatcher: File system watcher stopped');
    }

    private async onFileChanged(uri: vscode.Uri) {
        console.log(`WorkspaceWatcher: File changed: ${uri.fsPath}`);

        // Debounce: Wait for user to stop typing
        this.debounceFileIndexing(uri);
    }

    private async onFileCreated(uri: vscode.Uri) {
        console.log(`WorkspaceWatcher: File created: ${uri.fsPath}`);

        // Index immediately (no debounce for new files)
        await this.indexFile(uri);
    }

    private async onFileDeleted(uri: vscode.Uri) {
        console.log(`WorkspaceWatcher: File deleted: ${uri.fsPath}`);

        // TODO: Call /api/forget endpoint to remove from knowledge graph
        // For now, just log
    }

    private debounceFileIndexing(uri: vscode.Uri) {
        const filePath = uri.fsPath;

        // Clear existing timer for this file
        const existingTimer = this.debounceTimers.get(filePath);
        if (existingTimer) {
            clearTimeout(existingTimer);
        }

        // Set new timer
        const timer = setTimeout(async () => {
            await this.indexFile(uri);
            this.debounceTimers.delete(filePath);
        }, WorkspaceWatcher.DEBOUNCE_DELAY);

        this.debounceTimers.set(filePath, timer);
    }

    private async indexFile(uri: vscode.Uri) {
        try {
            // Read file content
            const content = await vscode.workspace.fs.readFile(uri);
            const text = Buffer.from(content).toString('utf-8');

            // Get workspace folder
            const workspaceFolder = vscode.workspace.getWorkspaceFolder(uri);
            const workspace = workspaceFolder?.name || 'unknown';

            // Get relative path
            const relativePath = workspaceFolder
                ? uri.fsPath.replace(workspaceFolder.uri.fsPath, '')
                : uri.fsPath;

            // Store to HoloLoom
            await this.hololoomCommands.remember(
                `File indexed: ${relativePath}\n\n${text.substring(0, 500)}...`,
                {
                    workspace,
                    file: relativePath,
                    timestamp: new Date().toISOString(),
                    source: 'file_watcher'
                }
            );

            console.log(`WorkspaceWatcher: Indexed file: ${relativePath}`);
        } catch (error: any) {
            console.error(`WorkspaceWatcher: Failed to index file: ${error.message}`);
        }
    }

    public async indexWorkspace(showProgress: boolean = true) {
        if (this.isIndexing) {
            vscode.window.showWarningMessage('Workspace indexing already in progress');
            return;
        }

        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders || workspaceFolders.length === 0) {
            console.log('WorkspaceWatcher: No workspace folders to index');
            return;
        }

        this.isIndexing = true;

        try {
            for (const folder of workspaceFolders) {
                await this.indexWorkspaceFolder(folder, showProgress);
            }

            if (showProgress) {
                vscode.window.showInformationMessage(
                    `HoloLoom: Workspace indexed successfully! (${this.indexingProgress?.filesProcessed || 0} files)`
                );
            }
        } catch (error: any) {
            vscode.window.showErrorMessage(`Workspace indexing failed: ${error.message}`);
        } finally {
            this.isIndexing = false;
            this.indexingProgress = null;
        }
    }

    private async indexWorkspaceFolder(
        folder: vscode.WorkspaceFolder,
        showProgress: boolean
    ) {
        console.log(`WorkspaceWatcher: Indexing workspace folder: ${folder.name}`);

        // Find all matching files
        const files = await vscode.workspace.findFiles(
            `{${WorkspaceWatcher.WATCH_PATTERNS.join(',')}}`,
            '{**/node_modules/**,**/.git/**,**/dist/**,**/build/**,**/__pycache__/**}'
        );

        console.log(`WorkspaceWatcher: Found ${files.length} files to index`);

        this.indexingProgress = {
            filesProcessed: 0,
            filesTotal: files.length,
            currentFile: ''
        };

        if (showProgress) {
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Indexing workspace with HoloLoom",
                cancellable: false
            }, async (progress) => {
                for (const file of files) {
                    this.indexingProgress!.currentFile = file.fsPath;

                    progress.report({
                        message: `${this.indexingProgress!.filesProcessed}/${files.length} files`,
                        increment: (1 / files.length) * 100
                    });

                    await this.indexFile(file);
                    this.indexingProgress!.filesProcessed++;

                    // Small delay to avoid overwhelming the server
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
            });
        } else {
            // Index without progress UI (background)
            for (const file of files) {
                this.indexingProgress!.currentFile = file.fsPath;
                await this.indexFile(file);
                this.indexingProgress!.filesProcessed++;

                // Small delay
                await new Promise(resolve => setTimeout(resolve, 50));
            }
        }

        console.log(`WorkspaceWatcher: Indexed ${files.length} files from ${folder.name}`);
    }

    public getIndexingProgress(): IndexingProgress | null {
        return this.indexingProgress;
    }

    public isCurrentlyIndexing(): boolean {
        return this.isIndexing;
    }
}

export function registerWorkspaceCommands(context: vscode.ExtensionContext, watcher: WorkspaceWatcher) {
    // Command: Index workspace manually
    context.subscriptions.push(
        vscode.commands.registerCommand('promptly.indexWorkspace', async () => {
            await watcher.indexWorkspace(true);
        })
    );

    // Command: Show indexing status
    context.subscriptions.push(
        vscode.commands.registerCommand('promptly.indexingStatus', () => {
            const progress = watcher.getIndexingProgress();

            if (!progress) {
                vscode.window.showInformationMessage('No indexing in progress');
                return;
            }

            vscode.window.showInformationMessage(
                `Indexing: ${progress.filesProcessed}/${progress.filesTotal} files\n` +
                `Current: ${progress.currentFile}`
            );
        })
    );
}
