/**
 * Squad VS Code Extension - Main Entry Point
 *
 * AI-powered code intelligence with HoloLoom backend
 *
 * Features:
 * - Explain code
 * - Find similar code
 * - Generate unit tests
 * - Add documentation
 * - Suggest refactorings
 * - Review changes
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { HoloLoomBridge } from './lib/HoloLoomBridge';
import { CodeContextExtractor } from './lib/CodeContextExtractor';
import { CacheManager } from './lib/CacheManager';
import { SquadCommands } from './commands';

let bridge: HoloLoomBridge;
let extractor: CodeContextExtractor;
let cache: CacheManager;
let commands: SquadCommands;

export function activate(context: vscode.ExtensionContext) {
    console.log('Squad extension activated');

    // Load configuration
    const config = vscode.workspace.getConfiguration('squad');
    const hololoomUrl = config.get<string>('hololoomUrl') || 'http://localhost:8000';
    const enableCache = config.get<boolean>('enableCache') ?? true;
    const cacheMaxSize = config.get<number>('cacheMaxSize') || 10000;
    const autoIndexWorkspace = config.get<boolean>('autoIndexWorkspace') ?? false;

    // Initialize components
    bridge = new HoloLoomBridge(hololoomUrl);
    extractor = new CodeContextExtractor();

    // Initialize cache if enabled
    if (enableCache) {
        const cachePath = path.join(context.globalStorageUri.fsPath, 'squad.cache.db');
        cache = new CacheManager(cachePath, cacheMaxSize);
    } else {
        // Dummy cache that doesn't persist
        cache = createDummyCache();
    }

    // Initialize commands
    commands = new SquadCommands(bridge, extractor, cache);

    // Register commands
    context.subscriptions.push(
        // Code intelligence commands
        vscode.commands.registerCommand('squad.explainCode', () => commands.explainCode()),
        vscode.commands.registerCommand('squad.findSimilar', () => commands.findSimilar()),
        vscode.commands.registerCommand('squad.generateTests', () => commands.generateTests()),
        vscode.commands.registerCommand('squad.addDocumentation', () => commands.addDocumentation()),
        vscode.commands.registerCommand('squad.refactorCode', () => commands.refactorCode()),
        vscode.commands.registerCommand('squad.reviewChanges', () => commands.reviewChanges()),

        // Workspace commands
        vscode.commands.registerCommand('squad.indexWorkspace', () => commands.indexWorkspace()),
        vscode.commands.registerCommand('squad.clearCache', () => commands.clearCache()),
        vscode.commands.registerCommand('squad.showStats', () => commands.showStats())
    );

    // Configuration change listener
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('squad.hololoomUrl')) {
                bridge.updateConfig();
            }
        })
    );

    // File watcher for cache invalidation
    if (enableCache) {
        const watcher = vscode.workspace.createFileSystemWatcher('**/*');
        context.subscriptions.push(watcher);
    }

    // Auto-index workspace on startup if enabled
    if (autoIndexWorkspace) {
        setTimeout(() => {
            commands.indexWorkspace().catch(err => {
                console.error('Failed to auto-index workspace:', err);
            });
        }, 5000); // Wait 5 seconds after activation
    }

    // Show welcome message on first activation
    const hasShownWelcome = context.globalState.get('hasShownWelcome');
    if (!hasShownWelcome) {
        vscode.window.showInformationMessage(
            'Welcome to Squad! 🚀 Select code and right-click to access AI-powered features.',
            'Show Commands',
            'View Stats'
        ).then(selection => {
            if (selection === 'Show Commands') {
                vscode.commands.executeCommand('workbench.action.showCommands');
            } else if (selection === 'View Stats') {
                commands.showStats();
            }
        });
        context.globalState.update('hasShownWelcome', true);
    }

    console.log('Squad extension fully initialized');
}

export function deactivate() {
    // Cleanup
    if (bridge) {
        bridge.dispose();
    }

    if (cache && cache.dispose) {
        cache.dispose();
    }

    if (commands) {
        commands.dispose();
    }

    console.log('Squad extension deactivated');
}

/**
 * Create a dummy cache that doesn't persist (for when cache is disabled)
 */
function createDummyCache(): CacheManager {
    const inMemoryCache = new Map();

    return {
        get: (key: string) => inMemoryCache.get(key) || null,
        set: (key: string, value: any) => inMemoryCache.set(key, value),
        needsUpdate: () => true,
        invalidateFile: () => {},
        getStats: () => ({
            totalEntries: inMemoryCache.size,
            totalHits: 0,
            totalMisses: 0,
            hitRate: 0,
            size: 0,
            maxSize: 0
        }),
        clear: () => inMemoryCache.clear(),
        clearOlderThan: () => {},
        getEntries: () => [],
        dispose: () => inMemoryCache.clear()
    } as any;
}
