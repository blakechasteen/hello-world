/**
 * Code Context Provider
 * Extracts relevant code context for Squad queries
 */

import * as vscode from 'vscode';
import { CodeContext } from './HoloLoomBridge';

export class CodeContextProvider {
    getCurrentContext(): CodeContext {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return {
                workspace: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
            };
        }

        const document = editor.document;
        const selection = editor.selection;

        return {
            currentFile: document.getText(),
            fileName: document.fileName,
            languageId: document.languageId,
            selection: selection.isEmpty ? undefined : document.getText(selection),
            diagnostics: vscode.languages.getDiagnostics(document.uri),
            workspace: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
        };
    }

    async getDocumentSymbols(): Promise<vscode.DocumentSymbol[]> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return [];

        const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
            'vscode.executeDocumentSymbolProvider',
            editor.document.uri
        );

        return symbols || [];
    }
}
