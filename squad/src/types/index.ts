/**
 * Type definitions for Squad VS Code extension
 *
 * Matches HoloLoom backend API interfaces and adds extension-specific types
 */

// ============================================================================
// Code Context Types
// ============================================================================

export interface CodeContext {
    currentFile?: string;
    fileName?: string;
    languageId?: string;
    selection?: string;
    workspace?: string;
    diagnostics?: Diagnostic[];
}

export interface Diagnostic {
    line: number;
    column: number;
    severity: 'error' | 'warning' | 'info';
    message: string;
    source?: string;
}

export interface CodeEntity {
    name: string;
    type: EntityType;
    lineNumber: number;
    column: number;
    signature?: string;
    docstring?: string;
    children?: CodeEntity[];
    parent?: string;
}

export enum EntityType {
    Function = 'function',
    Class = 'class',
    Method = 'method',
    Variable = 'variable',
    Import = 'import',
    Module = 'module',
    Interface = 'interface',
    Type = 'type',
    Enum = 'enum'
}

export interface FileContext {
    path: string;
    languageId: string;
    content: string;
    entities: CodeEntity[];
    imports: string[];
    dependencies: string[];
    lastModified: number;
}

// ============================================================================
// HoloLoom API Types (matches backend)
// ============================================================================

export interface QueryRequest {
    text: string;
    context?: CodeContext;
    mode?: ReasoningMode;
    max_steps?: number;
}

export type ReasoningMode = 'direct' | 'verify' | 'research' | 'plan_execute';

export interface AgenticResponse {
    response: string;
    confidence: number;
    reasoning_mode: string;
    steps_taken: ReasoningStep[];
    total_queries: number;
    total_duration_ms: number;
    verification?: VerificationResponse;
    timestamp: string;
    query_id: string;
}

export interface ReasoningStep {
    type: string;
    query?: string;
    confidence?: number;
    finding?: string;
    completed?: boolean;
    tool?: string;
}

export interface VerificationResponse {
    verified: boolean;
    confidence: number;
    contradictions: string[];
    supporting_evidence: string[];
    suggested_refinements: string[];
}

// ============================================================================
// Cache Types
// ============================================================================

export interface CacheEntry {
    key: string;
    value: any;
    timestamp: number;
    hits: number;
    size: number;
}

export interface CacheStats {
    totalEntries: number;
    totalHits: number;
    totalMisses: number;
    hitRate: number;
    size: number;
    maxSize: number;
}

// ============================================================================
// Extension-Specific Types
// ============================================================================

export interface ExplanationResult {
    explanation: string;
    confidence: number;
    codeStructure: CodeEntity[];
    relatedConcepts: string[];
    suggestedReadings: string[];
}

export interface SimilarCodeResult {
    items: SimilarCodeItem[];
    totalFound: number;
    searchDuration: number;
}

export interface SimilarCodeItem {
    filePath: string;
    lineNumber: number;
    code: string;
    similarity: number;
    reason: string;
}

export interface TestGenerationResult {
    tests: string;
    framework: string;
    coverage: number;
    suggestions: string[];
}

export interface DocumentationResult {
    documentation: string;
    format: 'docstring' | 'jsdoc' | 'markdown';
    completeness: number;
}

export interface RefactoringResult {
    suggestions: RefactoringSuggestion[];
    totalIssues: number;
}

export interface RefactoringSuggestion {
    type: RefactoringType;
    severity: 'info' | 'warning' | 'error';
    line: number;
    description: string;
    originalCode: string;
    suggestedCode: string;
    rationale: string;
}

export enum RefactoringType {
    ExtractFunction = 'extract_function',
    ExtractVariable = 'extract_variable',
    RenameSymbol = 'rename_symbol',
    SimplifyConditional = 'simplify_conditional',
    RemoveDuplication = 'remove_duplication',
    ImproveNaming = 'improve_naming',
    AddErrorHandling = 'add_error_handling',
    OptimizePerformance = 'optimize_performance'
}

export interface WorkspaceIndexStats {
    filesIndexed: number;
    entitiesFound: number;
    duration: number;
    languages: string[];
}
