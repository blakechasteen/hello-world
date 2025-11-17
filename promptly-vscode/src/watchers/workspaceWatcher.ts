import * as vscode from 'vscode';
import { HoloLoomCommands } from '../commands/hololoomCommands';

interface IndexingProgress {
    filesProcessed: number;
    filesTotal: number;
    currentFile: string;
    status: 'idle' | 'indexing' | 'error';
    lastError?: string;
}

interface HoloLoomIngestResponse {
    success: boolean;
    files_indexed?: number;
    code_elements?: number;
    comments?: number;
    todos?: number;
    entities_created?: number;
    error?: string;
    stats?: {
        files: number;
        entities: number;
        [key: string]: any;
    };
}

export class WorkspaceWatcher {
    private fileWatcher: vscode.FileSystemWatcher | undefined;
    private hololoomCommands: HoloLoomCommands;
    private isIndexing: boolean = false;
    private indexingProgress: IndexingProgress | null = null;

    // Batch-based debouncing
    private pendingFiles: Set<string> = new Set();
    private batchDebounceTimer: NodeJS.Timeout | undefined;

    // Debounce delay (ms) - wait this long after last change before batch indexing
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

        // Clear batch debounce timer
        if (this.batchDebounceTimer) {
            clearTimeout(this.batchDebounceTimer);
            this.batchDebounceTimer = undefined;
        }

        // Clear pending files
        this.pendingFiles.clear();

        console.log('WorkspaceWatcher: File system watcher stopped');
    }

    private async onFileChanged(uri: vscode.Uri) {
        const filePath = uri.fsPath;
        console.log(`WorkspaceWatcher: File changed: ${filePath}`);

        // Add to batch (deduplicated via Set)
        this.pendingFiles.add(filePath);

        // Reset batch debounce timer
        this.resetBatchDebounceTimer();
    }

    private async onFileCreated(uri: vscode.Uri) {
        const filePath = uri.fsPath;
        console.log(`WorkspaceWatcher: File created: ${filePath}`);

        // New files are high priority - index immediately
        this.pendingFiles.add(filePath);

        // Use shorter debounce for new files (100ms)
        if (this.batchDebounceTimer) {
            clearTimeout(this.batchDebounceTimer);
        }

        this.batchDebounceTimer = setTimeout(async () => {
            await this.indexPendingFiles();
        }, 100);
    }

    private async onFileDeleted(uri: vscode.Uri) {
        const filePath = uri.fsPath;
        console.log(`WorkspaceWatcher: File deleted: ${filePath}`);

        // Remove from pending batch if it was there
        this.pendingFiles.delete(filePath);

        // TODO: Call /api/forget endpoint to remove from knowledge graph
        // For now, just log
    }

    private resetBatchDebounceTimer(): void {
        // Clear existing timer
        if (this.batchDebounceTimer) {
            clearTimeout(this.batchDebounceTimer);
        }

        // Set new timer
        this.batchDebounceTimer = setTimeout(async () => {
            await this.indexPendingFiles();
        }, WorkspaceWatcher.DEBOUNCE_DELAY);
    }

    private async indexPendingFiles(): Promise<void> {
        if (this.pendingFiles.size === 0) {
            return;
        }

        const files = Array.from(this.pendingFiles);
        this.pendingFiles.clear();

        if (this.isIndexing) {
            console.log('WorkspaceWatcher: Indexing already in progress, queueing files');
            // Re-add files back to pending for next batch
            files.forEach(f => this.pendingFiles.add(f));
            return;
        }

        this.isIndexing = true;

        try {
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Updating HoloLoom knowledge graph (${files.length} file${files.length === 1 ? '' : 's'})...`,
                cancellable: false
            }, async (progress) => {
                // Get workspace root
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
                if (!workspaceFolder) {
                    throw new Error('No workspace folder open');
                }

                // Call batch update endpoint
                const response = await this.hololoomCommands.ingestFilesIncremental(
                    files,
                    workspaceFolder.uri.fsPath
                );

                if (!response.success) {
                    throw new Error(response.error || 'Unknown error');
                }

                // Log statistics
                const filesIndexed = response.files_indexed || files.length;
                const entities = response.entities_created || 0;
                console.log(
                    `WorkspaceWatcher: Indexed ${filesIndexed} file(s), ` +
                    `created ${entities} entities, ` +
                    `${response.todos || 0} TODOs found`
                );

                // Show success notification only for significant updates
                if (entities > 10 || (response.todos || 0) > 2) {
                    vscode.window.showInformationMessage(
                        `HoloLoom: Updated ${entities} entities from ${filesIndexed} file(s)`
                    );
                }
            });
        } catch (error: any) {
            const errorMsg = error.message || String(error);
            console.error('WorkspaceWatcher: Failed to index files:', errorMsg);

            // Update progress with error
            if (this.indexingProgress) {
                this.indexingProgress.status = 'error';
                this.indexingProgress.lastError = errorMsg;
            }

            // Show warning (not error - don't disrupt user workflow)
            if (errorMsg.includes('ECONNREFUSED') || errorMsg.includes('connection')) {
                vscode.window.showWarningMessage(
                    'HoloLoom: Server not running. Knowledge graph updates paused.',
                    'Start Server'
                ).then(selection => {
                    if (selection === 'Start Server') {
                        vscode.commands.executeCommand('promptly.startHoloLoomServer');
                    }
                });
            } else {
                vscode.window.showWarningMessage(
                    `HoloLoom: Indexing failed (${errorMsg.substring(0, 50)}...)`
                );
            }
        } finally {
            this.isIndexing = false;
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
            // Index first workspace folder (most common case)
            const folder = workspaceFolders[0];
            await this.indexWorkspaceFolder(folder, showProgress);

            if (showProgress) {
                vscode.window.showInformationMessage(
                    `HoloLoom: Workspace indexed successfully! (${this.indexingProgress?.filesProcessed || 0} files)`
                );
            }

            // Index additional folders if present
            if (workspaceFolders.length > 1) {
                for (const folder of workspaceFolders.slice(1)) {
                    await this.indexWorkspaceFolder(folder, false);
                }
            }
        } catch (error: any) {
            const errorMsg = error.message || String(error);
            console.error('WorkspaceWatcher: Workspace indexing failed:', errorMsg);
            vscode.window.showErrorMessage(`Workspace indexing failed: ${errorMsg.substring(0, 100)}`);
        } finally {
            this.isIndexing = false;
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
            currentFile: '',
            status: 'indexing'
        };

        if (showProgress) {
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Indexing workspace with HoloLoom (${files.length} files)`,
                cancellable: false
            }, async (progress) => {
                try {
                    // Call batch API endpoint
                    const response = await this.hololoomCommands.ingestWorkspaceFull(
                        folder.uri.fsPath
                    );

                    if (response.success) {
                        this.indexingProgress!.filesProcessed = response.files_indexed || files.length;
                        console.log(
                            `WorkspaceWatcher: Full workspace indexed - ` +
                            `${response.files_indexed} files, ` +
                            `${response.code_elements} elements, ` +
                            `${response.todos} TODOs`
                        );
                    } else {
                        throw new Error(response.error || 'Unknown error during indexing');
                    }
                } catch (error: any) {
                    this.indexingProgress!.status = 'error';
                    this.indexingProgress!.lastError = error.message;
                    throw error;
                }
            });
        } else {
            try {
                // Background indexing without progress UI
                const response = await this.hololoomCommands.ingestWorkspaceFull(
                    folder.uri.fsPath
                );

                if (response.success) {
                    this.indexingProgress!.filesProcessed = response.files_indexed || files.length;
                } else {
                    throw new Error(response.error || 'Workspace indexing failed');
                }
            } catch (error: any) {
                console.error(`WorkspaceWatcher: Background indexing failed: ${error.message}`);
                this.indexingProgress!.status = 'error';
                this.indexingProgress!.lastError = error.message;
            }
        }
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
