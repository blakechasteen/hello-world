/**
 * Code Context Provider
 * Extracts code context from VS Code for Squad queries
 */

import * as vscode from 'vscode';
import { CodeContext } from './HoloLoomBridge';

export class CodeContextProvider {
    getCurrentContext(): CodeContext {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return {};
        }

        const document = editor.document;
        const selection = editor.selection;

        return {
            currentFile: document.getText(),
            fileName: document.fileName,
            languageId: document.languageId,
            selection: selection.isEmpty ? undefined : document.getText(selection),
            workspace: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
        };
    }

    getSelectionContext(): CodeContext | undefined {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.selection.isEmpty) {
            return undefined;
        }

        const document = editor.document;
        const selection = editor.selection;

        return {
            fileName: document.fileName,
            languageId: document.languageId,
            selection: document.getText(selection),
            workspace: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
        };
    }
}
