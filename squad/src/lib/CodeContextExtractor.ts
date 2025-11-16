/**
 * CodeContextExtractor - AST parsing and code context extraction
 *
 * Uses tree-sitter for multi-language AST parsing to extract:
 * - Function signatures
 * - Class definitions
 * - Import statements
 * - Dependencies
 * - Code structure
 */

import * as vscode from 'vscode';
import Parser from 'tree-sitter';
import { CodeContext, CodeEntity, EntityType, FileContext } from '../types';

// Language parsers (lazy-loaded)
let pythonParser: typeof import('tree-sitter-python') | null = null;
let typescriptParser: typeof import('tree-sitter-typescript') | null = null;
let javascriptParser: typeof import('tree-sitter-javascript') | null = null;

export class CodeContextExtractor {
    private parsers: Map<string, Parser> = new Map();

    constructor() {
        this.initializeParsers();
    }

    /**
     * Initialize tree-sitter parsers for supported languages
     */
    private async initializeParsers() {
        try {
            // Python
            pythonParser = await import('tree-sitter-python');
            const pythonParserInstance = new Parser();
            pythonParserInstance.setLanguage(pythonParser);
            this.parsers.set('python', pythonParserInstance);

            // TypeScript
            typescriptParser = await import('tree-sitter-typescript');
            const tsParser = new Parser();
            tsParser.setLanguage(typescriptParser.typescript);
            this.parsers.set('typescript', tsParser);

            // JavaScript
            javascriptParser = await import('tree-sitter-javascript');
            const jsParser = new Parser();
            jsParser.setLanguage(javascriptParser);
            this.parsers.set('javascript', jsParser);

        } catch (error) {
            console.warn('Failed to initialize some tree-sitter parsers:', error);
        }
    }

    /**
     * Extract code context from active editor
     */
    async extractContext(editor: vscode.TextEditor): Promise<CodeContext> {
        const document = editor.document;
        const selection = editor.selection;

        const context: CodeContext = {
            fileName: document.fileName,
            languageId: document.languageId,
            workspace: vscode.workspace.name,
            selection: document.getText(selection),
            diagnostics: this.extractDiagnostics(document)
        };

        return context;
    }

    /**
     * Extract detailed file context with AST analysis
     */
    async extractFileContext(document: vscode.TextDocument): Promise<FileContext> {
        const content = document.getText();
        const parser = this.parsers.get(document.languageId);

        const fileContext: FileContext = {
            path: document.fileName,
            languageId: document.languageId,
            content: content,
            entities: [],
            imports: [],
            dependencies: [],
            lastModified: Date.now()
        };

        if (!parser) {
            // Fallback: basic text analysis if no parser available
            return this.extractWithoutParser(fileContext, content);
        }

        try {
            const tree = parser.parse(content);
            fileContext.entities = this.extractEntities(tree.rootNode, content, document.languageId);
            fileContext.imports = this.extractImports(tree.rootNode, content, document.languageId);
            fileContext.dependencies = this.extractDependencies(tree.rootNode, content, document.languageId);
        } catch (error) {
            console.error('AST parsing failed:', error);
            return this.extractWithoutParser(fileContext, content);
        }

        return fileContext;
    }

    /**
     * Extract relevant context for a code selection
     * Returns only the code elements needed to understand the selection
     */
    async extractRelevantContext(
        document: vscode.TextDocument,
        selection: vscode.Selection
    ): Promise<{ entities: CodeEntity[], context: string }> {
        const fullContext = await this.extractFileContext(document);
        const selectionStart = document.offsetAt(selection.start);
        const selectionEnd = document.offsetAt(selection.end);

        // Find entities that overlap with selection
        const relevantEntities = fullContext.entities.filter(entity => {
            const entityStart = document.offsetAt(new vscode.Position(entity.lineNumber, entity.column));
            // Approximate entity end (could be improved with actual ranges)
            const entityEnd = entityStart + (entity.signature?.length || 100);

            return (entityStart <= selectionEnd && entityEnd >= selectionStart);
        });

        // Extract minimal context (parent scopes, relevant imports)
        const contextLines: string[] = [];
        const content = document.getText();

        // Add imports
        if (fullContext.imports.length > 0) {
            contextLines.push('// Imports:');
            fullContext.imports.slice(0, 5).forEach(imp => contextLines.push(imp));
            contextLines.push('');
        }

        // Add parent entities (classes, functions)
        relevantEntities
            .filter(e => [EntityType.Class, EntityType.Function].includes(e.type))
            .forEach(entity => {
                if (entity.signature) {
                    contextLines.push(`// ${entity.type}: ${entity.name}`);
                    contextLines.push(entity.signature);
                    contextLines.push('');
                }
            });

        return {
            entities: relevantEntities,
            context: contextLines.join('\n')
        };
    }

    /**
     * Extract diagnostics (errors, warnings) from document
     */
    private extractDiagnostics(document: vscode.TextDocument) {
        const diagnostics = vscode.languages.getDiagnostics(document.uri);

        return diagnostics.map(d => ({
            line: d.range.start.line,
            column: d.range.start.character,
            severity: d.severity === vscode.DiagnosticSeverity.Error ? 'error' as const :
                     d.severity === vscode.DiagnosticSeverity.Warning ? 'warning' as const : 'info' as const,
            message: d.message,
            source: d.source
        }));
    }

    /**
     * Extract code entities from AST
     */
    private extractEntities(node: Parser.SyntaxNode, content: string, language: string): CodeEntity[] {
        const entities: CodeEntity[] = [];

        const traverse = (n: Parser.SyntaxNode, parent?: string) => {
            // Language-specific entity extraction
            if (language === 'python') {
                this.extractPythonEntities(n, content, entities, parent);
            } else if (language === 'typescript' || language === 'javascript') {
                this.extractTypeScriptEntities(n, content, entities, parent);
            }

            // Recursively traverse children
            for (const child of n.children) {
                traverse(child, parent);
            }
        };

        traverse(node);
        return entities;
    }

    private extractPythonEntities(
        node: Parser.SyntaxNode,
        content: string,
        entities: CodeEntity[],
        parent?: string
    ) {
        if (node.type === 'function_definition') {
            const nameNode = node.childForFieldName('name');
            if (nameNode) {
                entities.push({
                    name: content.substring(nameNode.startIndex, nameNode.endIndex),
                    type: parent ? EntityType.Method : EntityType.Function,
                    lineNumber: node.startPosition.row,
                    column: node.startPosition.column,
                    signature: content.substring(node.startIndex, Math.min(node.endIndex, node.startIndex + 200)),
                    parent
                });
            }
        } else if (node.type === 'class_definition') {
            const nameNode = node.childForFieldName('name');
            if (nameNode) {
                const className = content.substring(nameNode.startIndex, nameNode.endIndex);
                entities.push({
                    name: className,
                    type: EntityType.Class,
                    lineNumber: node.startPosition.row,
                    column: node.startPosition.column,
                    signature: content.substring(node.startIndex, Math.min(node.endIndex, node.startIndex + 200)),
                    children: []
                });
            }
        }
    }

    private extractTypeScriptEntities(
        node: Parser.SyntaxNode,
        content: string,
        entities: CodeEntity[],
        parent?: string
    ) {
        if (node.type === 'function_declaration' || node.type === 'method_definition') {
            const nameNode = node.childForFieldName('name');
            if (nameNode) {
                entities.push({
                    name: content.substring(nameNode.startIndex, nameNode.endIndex),
                    type: parent ? EntityType.Method : EntityType.Function,
                    lineNumber: node.startPosition.row,
                    column: node.startPosition.column,
                    signature: content.substring(node.startIndex, Math.min(node.endIndex, node.startIndex + 200)),
                    parent
                });
            }
        } else if (node.type === 'class_declaration') {
            const nameNode = node.childForFieldName('name');
            if (nameNode) {
                entities.push({
                    name: content.substring(nameNode.startIndex, nameNode.endIndex),
                    type: EntityType.Class,
                    lineNumber: node.startPosition.row,
                    column: node.startPosition.column,
                    signature: content.substring(node.startIndex, Math.min(node.endIndex, node.startIndex + 200)),
                    children: []
                });
            }
        } else if (node.type === 'interface_declaration') {
            const nameNode = node.childForFieldName('name');
            if (nameNode) {
                entities.push({
                    name: content.substring(nameNode.startIndex, nameNode.endIndex),
                    type: EntityType.Interface,
                    lineNumber: node.startPosition.row,
                    column: node.startPosition.column,
                    signature: content.substring(node.startIndex, Math.min(node.endIndex, node.startIndex + 200))
                });
            }
        }
    }

    /**
     * Extract import statements
     */
    private extractImports(node: Parser.SyntaxNode, content: string, language: string): string[] {
        const imports: string[] = [];

        const traverse = (n: Parser.SyntaxNode) => {
            if (language === 'python' && n.type === 'import_statement' || n.type === 'import_from_statement') {
                imports.push(content.substring(n.startIndex, n.endIndex));
            } else if ((language === 'typescript' || language === 'javascript') && n.type === 'import_statement') {
                imports.push(content.substring(n.startIndex, n.endIndex));
            }

            for (const child of n.children) {
                traverse(child);
            }
        };

        traverse(node);
        return imports;
    }

    /**
     * Extract dependencies (imported modules)
     */
    private extractDependencies(node: Parser.SyntaxNode, content: string, language: string): string[] {
        const deps: Set<string> = new Set();
        const imports = this.extractImports(node, content, language);

        imports.forEach(imp => {
            // Extract module name from import statement
            if (language === 'python') {
                const match = imp.match(/(?:from|import)\s+([\w.]+)/);
                if (match) deps.add(match[1].split('.')[0]);
            } else {
                const match = imp.match(/from\s+['"]([^'"]+)['"]/);
                if (match) deps.add(match[1]);
            }
        });

        return Array.from(deps);
    }

    /**
     * Fallback extraction without parser (basic regex)
     */
    private extractWithoutParser(fileContext: FileContext, content: string): FileContext {
        const lines = content.split('\n');

        // Basic function/class detection
        lines.forEach((line, index) => {
            // Python
            if (line.match(/^\s*def\s+(\w+)/)) {
                const match = line.match(/def\s+(\w+)/);
                if (match) {
                    fileContext.entities.push({
                        name: match[1],
                        type: EntityType.Function,
                        lineNumber: index,
                        column: line.indexOf('def')
                    });
                }
            }
            if (line.match(/^\s*class\s+(\w+)/)) {
                const match = line.match(/class\s+(\w+)/);
                if (match) {
                    fileContext.entities.push({
                        name: match[1],
                        type: EntityType.Class,
                        lineNumber: index,
                        column: line.indexOf('class')
                    });
                }
            }

            // TypeScript/JavaScript
            if (line.match(/^\s*function\s+(\w+)/)) {
                const match = line.match(/function\s+(\w+)/);
                if (match) {
                    fileContext.entities.push({
                        name: match[1],
                        type: EntityType.Function,
                        lineNumber: index,
                        column: line.indexOf('function')
                    });
                }
            }

            // Imports
            if (line.match(/^(import|from|require)/)) {
                fileContext.imports.push(line.trim());
            }
        });

        return fileContext;
    }
}
