import * as vscode from 'vscode';
import { HoloLoomCommands } from '../commands/hololoomCommands';

interface RelatedMemory {
    content: string;
    confidence: number;
    timestamp?: string;
    source?: string;
}

export class HoloLoomCodeLensProvider implements vscode.CodeLensProvider {
    private hololoomCommands: HoloLoomCommands;
    private _onDidChangeCodeLenses: vscode.EventEmitter<void> = new vscode.EventEmitter<void>();
    public readonly onDidChangeCodeLenses: vscode.Event<void> = this._onDidChangeCodeLenses.event;

    constructor() {
        this.hololoomCommands = new HoloLoomCommands();

        // Refresh CodeLens when configuration changes
        vscode.workspace.onDidChangeConfiguration(() => {
            this._onDidChangeCodeLenses.fire();
        });
    }

    public async provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): Promise<vscode.CodeLens[]> {
        const lenses: vscode.CodeLens[] = [];
        const text = document.getText();

        // Pattern to match NOTE, TODO, FIXME comments
        // Supports: // NOTE:, /* NOTE: */, # NOTE:, <!-- NOTE: -->
        const commentPatterns = [
            /\/\/\s*(NOTE|TODO|FIXME):\s*(.+)/g,      // JS/TS/C/C++ style
            /\/\*\s*(NOTE|TODO|FIXME):\s*(.+?)\*\//g, // Block comment
            /#\s*(NOTE|TODO|FIXME):\s*(.+)/g,         // Python/Ruby style
            /<!--\s*(NOTE|TODO|FIXME):\s*(.+?)-->/g   // HTML/XML style
        ];

        for (const pattern of commentPatterns) {
            let match;
            while ((match = pattern.exec(text)) !== null) {
                const commentType = match[1];
                const commentText = match[2].trim();
                const line = document.positionAt(match.index).line;
                const range = new vscode.Range(line, 0, line, 0);

                // Query HoloLoom for related knowledge (async)
                try {
                    const related = await this.findRelated(commentText, token);

                    if (related.length > 0) {
                        // Add CodeLens showing number of related items
                        lenses.push(new vscode.CodeLens(range, {
                            title: `💡 ${related.length} related note${related.length > 1 ? 's' : ''}`,
                            command: 'promptly.showRelated',
                            arguments: [commentText, related, line]
                        }));
                    } else if (commentType === 'TODO' || commentType === 'FIXME') {
                        // For TODOs/FIXMEs with no related notes, offer to create one
                        lenses.push(new vscode.CodeLens(range, {
                            title: `📝 Capture this ${commentType}`,
                            command: 'promptly.captureComment',
                            arguments: [commentText, commentType]
                        }));
                    }
                } catch (error) {
                    // Silently fail if HoloLoom server not available
                    console.error('CodeLens error:', error);
                }
            }
        }

        return lenses;
    }

    private async findRelated(
        commentText: string,
        token: vscode.CancellationToken
    ): Promise<RelatedMemory[]> {
        // Don't query if comment is too short
        if (commentText.length < 10) {
            return [];
        }

        try {
            // Query HoloLoom for related memories
            const result = await this.hololoomCommands.recall(commentText);

            // Parse result (it's a formatted string from recall command)
            return this.parseRecallResult(result);
        } catch (error) {
            // Server not available or other error
            return [];
        }
    }

    private parseRecallResult(result: string): RelatedMemory[] {
        // Parse markdown-formatted recall results
        const memories: RelatedMemory[] = [];

        // Match pattern: **1.** content
        //                _Confidence: 75% | timestamp_
        const memoryPattern = /\*\*\d+\.\*\* (.+?)\n\s+_Confidence: (\d+)%.*?_/gs;
        let match;

        while ((match = memoryPattern.exec(result)) !== null) {
            memories.push({
                content: match[1].trim(),
                confidence: parseInt(match[2]) / 100
            });
        }

        // Filter low confidence results (< 60%)
        return memories.filter(m => m.confidence >= 0.6);
    }
}

export function registerCodeLensCommands(context: vscode.ExtensionContext) {
    const hololoomCommands = new HoloLoomCommands();

    // Command: Show related notes
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'promptly.showRelated',
            async (commentText: string, related: RelatedMemory[], line: number) => {
                const quickPick = vscode.window.createQuickPick();
                quickPick.title = `Related to: ${commentText}`;
                quickPick.placeholder = 'Select a related note to view details';

                quickPick.items = related.map((mem, i) => ({
                    label: `$(note) ${mem.content.substring(0, 60)}...`,
                    description: `${Math.round(mem.confidence * 100)}% confident`,
                    detail: mem.source || mem.timestamp || `Memory #${i + 1}`,
                    memory: mem
                })) as any;

                quickPick.onDidChangeSelection(selection => {
                    if (selection[0]) {
                        const memory = (selection[0] as any).memory;
                        vscode.window.showInformationMessage(
                            memory.content,
                            { modal: false }
                        );
                    }
                });

                quickPick.onDidAccept(() => {
                    quickPick.hide();
                });

                quickPick.show();
            }
        )
    );

    // Command: Capture comment as note
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'promptly.captureComment',
            async (commentText: string, commentType: string) => {
                const editor = vscode.window.activeTextEditor;
                if (!editor) return;

                const fileName = editor.document.fileName;
                const lineNumber = editor.selection.active.line + 1;

                const enhancedContent = `[${commentType}] ${commentText}\n\nFile: ${fileName}:${lineNumber}`;

                const result = await hololoomCommands.remember(enhancedContent);

                vscode.window.showInformationMessage(
                    `${commentType} captured to HoloLoom!`
                );
            }
        )
    );
}
