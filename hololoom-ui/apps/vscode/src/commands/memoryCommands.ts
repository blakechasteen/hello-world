/**
 * Memory-related commands
 *
 * These commands handle memory operations with HoloLoom.
 */

import * as vscode from 'vscode';
import { HoloLoomClient } from '../api/holoLoomClient';
import { getSelectedText, getEditorContext } from '../utils/context';

/**
 * Register memory-related commands
 */
export function registerMemoryCommands(
  context: vscode.ExtensionContext,
  client: HoloLoomClient
): void {
  // Remember Selection
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.remember', async () => {
      const selectedText = getSelectedText();

      if (!selectedText) {
        vscode.window.showWarningMessage('Please select some content to remember');
        return;
      }

      const editorContext = getEditorContext();

      // Ask for optional note
      const note = await vscode.window.showInputBox({
        prompt: 'Add a note (optional)',
        placeHolder: 'What should I remember about this?',
      });

      const content = note
        ? `${note}\n\n${selectedText}`
        : selectedText;

      try {
        await client.storeMemory({
          content,
          type: editorContext?.language || 'code',
          metadata: {
            file: editorContext?.file,
            language: editorContext?.language,
            line: editorContext?.lineNumber,
          },
        });

        vscode.window.showInformationMessage('Content saved to memory');
      } catch (error) {
        vscode.window.showErrorMessage(
          `Failed to save memory: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    })
  );

  // Open Memory Browser
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.openMemory', () => {
      vscode.commands.executeCommand('hololoom.memoryView.focus');
    })
  );

  // Search Memories
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.searchMemory', async () => {
      const query = await vscode.window.showInputBox({
        prompt: 'Search your memories',
        placeHolder: 'What are you looking for?',
      });

      if (!query) {
        return;
      }

      try {
        const memories = await client.searchMemory({ query, limit: 10 });

        if (memories.length === 0) {
          vscode.window.showInformationMessage('No memories found');
          return;
        }

        // Show quick pick with results
        const items = memories.map((m) => ({
          label: m.content.substring(0, 60) + (m.content.length > 60 ? '...' : ''),
          description: m.type,
          detail: m.content,
          memory: m,
        }));

        const selected = await vscode.window.showQuickPick(items, {
          placeHolder: 'Select a memory to insert',
          matchOnDescription: true,
          matchOnDetail: true,
        });

        if (selected) {
          const editor = vscode.window.activeTextEditor;
          if (editor) {
            editor.edit((editBuilder) => {
              editBuilder.insert(editor.selection.active, selected.memory.content);
            });
          } else {
            // Copy to clipboard if no editor
            await vscode.env.clipboard.writeText(selected.memory.content);
            vscode.window.showInformationMessage('Memory copied to clipboard');
          }
        }
      } catch (error) {
        vscode.window.showErrorMessage(
          `Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    })
  );
}
