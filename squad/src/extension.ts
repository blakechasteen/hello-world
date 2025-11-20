/**
 * extension.ts
 * ============
 *
 * HoloLoom Squad VS Code Extension
 *
 * Main entry point for the extension.
 * Initializes all personas and features.
 *
 * Created: 2025-01-20
 */

import * as vscode from 'vscode';
import { HoloLoomIntegration } from './HoloLoomIntegration';

let hololoom: HoloLoomIntegration;

export function activate(context: vscode.ExtensionContext) {
    console.log('🚀 HoloLoom Squad is activating...');

    // Load configuration
    const config = vscode.workspace.getConfiguration('hololoom');
    const backendUrl = config.get<string>('backendUrl') || 'http://localhost:8000';

    // Initialize HoloLoom with all features
    hololoom = new HoloLoomIntegration({
        backendUrl,
        personas: {
            proto: config.get<boolean>('proto.enabled') ?? true,
            trough: config.get<boolean>('trough.enabled') ?? true,
            edwin: config.get<boolean>('edwin.enabled') ?? true
        },
        features: {
            promptly: true,
            eduverse: config.get<boolean>('eduverse.enabled') ?? true,
            chatops: config.get<boolean>('chatops.enabled') ?? true,
            voice: config.get<boolean>('voice.enabled') ?? true,
            workflows: true
        }
    });

    // Register all commands
    hololoom.registerCommands(context);

    // Register tree views
    const trough = hololoom.getTrough();
    if (trough) {
        const issuesProvider = trough.getIssuesProvider();
        vscode.window.registerTreeDataProvider('hololoom.issues', issuesProvider);
    }

    // Status bar welcome
    vscode.window.showInformationMessage(
        '🎉 HoloLoom Squad activated! Use Ctrl+Shift+P and search for "HoloLoom" to get started.'
    );

    console.log('✅ HoloLoom Squad activated successfully');
}

export function deactivate() {
    console.log('👋 HoloLoom Squad is deactivating...');
    hololoom?.dispose();
}
