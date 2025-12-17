import * as vscode from 'vscode';
import { getConfig } from './config';

export interface EditorContext {
  file?: string;
  language?: string;
  selection?: string;
  lineNumber?: number;
  surroundingCode?: string;
  workspaceFolder?: string;
}

/**
 * Get context from the active text editor
 */
export function getEditorContext(): EditorContext | null {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return null;
  }

  const document = editor.document;
  const selection = editor.selection;
  const config = getConfig();

  const context: EditorContext = {
    file: document.fileName,
    language: document.languageId,
    lineNumber: selection.start.line + 1,
  };

  // Get selected text
  if (!selection.isEmpty) {
    context.selection = document.getText(selection);
  }

  // Get surrounding code for context
  if (config.includeFileContext) {
    const maxLines = config.maxContextLines;
    const startLine = Math.max(0, selection.start.line - Math.floor(maxLines / 2));
    const endLine = Math.min(
      document.lineCount - 1,
      selection.end.line + Math.floor(maxLines / 2)
    );

    const range = new vscode.Range(startLine, 0, endLine, Number.MAX_VALUE);
    context.surroundingCode = document.getText(range);
  }

  // Get workspace folder
  const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
  if (workspaceFolder) {
    context.workspaceFolder = workspaceFolder.name;
  }

  return context;
}

/**
 * Get the currently selected text
 */
export function getSelectedText(): string | null {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    return null;
  }
  return editor.document.getText(editor.selection);
}

/**
 * Get the word at cursor position
 */
export function getWordAtCursor(): string | null {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return null;
  }

  const position = editor.selection.active;
  const wordRange = editor.document.getWordRangeAtPosition(position);
  if (!wordRange) {
    return null;
  }

  return editor.document.getText(wordRange);
}

/**
 * Get the current line
 */
export function getCurrentLine(): string | null {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return null;
  }

  const line = editor.selection.active.line;
  return editor.document.lineAt(line).text;
}

/**
 * Format context for display
 */
export function formatContextForPrompt(context: EditorContext): string {
  const parts: string[] = [];

  if (context.file) {
    const fileName = context.file.split(/[\\/]/).pop() || context.file;
    parts.push(`File: ${fileName}`);
  }

  if (context.language) {
    parts.push(`Language: ${context.language}`);
  }

  if (context.lineNumber) {
    parts.push(`Line: ${context.lineNumber}`);
  }

  if (context.selection) {
    parts.push(`\nSelected Code:\n\`\`\`${context.language || ''}\n${context.selection}\n\`\`\``);
  }

  if (context.surroundingCode && !context.selection) {
    parts.push(`\nContext:\n\`\`\`${context.language || ''}\n${context.surroundingCode}\n\`\`\``);
  }

  return parts.join('\n');
}

/**
 * Get a brief summary of the current context
 */
export function getContextSummary(): string {
  const context = getEditorContext();
  if (!context) {
    return 'No active editor';
  }

  const fileName = context.file?.split(/[\\/]/).pop() || 'Unknown file';
  const lang = context.language || 'Unknown';
  const hasSelection = context.selection ? ` (${context.selection.split('\n').length} lines selected)` : '';

  return `${fileName} (${lang})${hasSelection}`;
}
