# TypeScript Autofix System Design

**Date**: November 16, 2025
**Author**: mythRL Team
**Status**: Implementation Ready

## Overview

Extension of the xTerminator autofix system to support TypeScript codebases, specifically targeting the Trough VS Code extension (squad/ directory).

## Architecture

### Design Principles

1. **Language Parity**: Mirror Python SimpleLLMFixer architecture for consistency
2. **TypeScript-Specific Patterns**: Detect and fix TS-specific issues
3. **Integration First**: Seamless integration with existing autofix pipeline
4. **Graceful Degradation**: Falls back to rule-based fixes if LLM unavailable

### Component Architecture

```
TypeScriptFixer (new)
├── Rule-based fixes (no dependencies)
│   ├── Unused imports detection
│   ├── Any type detection
│   ├── Non-null assertions (!.) detection
│   ├── Missing type annotations
│   └── Console.log cleanup
│
├── Pattern-based fixes (regex)
│   ├── Interface naming (I-prefix)
│   ├── Type vs Interface preference
│   ├── Explicit return types
│   └── Readonly properties
│
└── LLM-enhanced fixes (optional)
    ├── Complex refactorings
    ├── Type inference
    └── Best practices alignment
```

### Integration Points

```
Existing System           New TypeScript Support
─────────────────────────────────────────────────
autofix_tracker.py  →  TypeScriptFixer integration
autofix_policy.py   →  TS-specific risk levels
simple_llm_fixer.py →  TypeScriptFixer (parallel)
classification.py   →  TS category detection
```

## TypeScript Issue Categories

### 1. Dead Code (Unused Imports)

**Detection Pattern**:
```typescript
// Unused import detection
import { Foo, Bar } from './module';  // Bar unused
import * as vscode from 'vscode';     // vscode unused
```

**Fix Strategy**:
- **AST-based** (preferred): Parse TypeScript AST, track usage
- **Regex-based** (fallback): Pattern matching for simple cases

**Confidence**: 0.95+ (high confidence for unused imports)

### 2. Type Safety Issues

#### Any Types
```typescript
// Before
function process(data: any) {
    return data.value;
}

// After
interface DataType {
    value: string;
}
function process(data: DataType) {
    return data.value;
}
```

**Confidence**: 0.75 (requires type inference context)

#### Non-Null Assertions
```typescript
// Before
const value = obj!.property;  // Risky!

// After
const value = obj?.property;  // Safe optional chaining
```

**Confidence**: 0.90 (mechanical transformation)

### 3. Missing Type Annotations

```typescript
// Before
function calculate(a, b) {
    return a + b;
}

// After
function calculate(a: number, b: number): number {
    return a + b;
}
```

**Confidence**: 0.70 (requires type inference)

### 4. Console.log Cleanup

```typescript
// Before
console.log('Debug:', someVar);
console.error('Error:', err);

// After
// Removed or wrapped in debug flag
```

**Confidence**: 0.95 (simple removal)

### 5. Interface vs Type Preference

```typescript
// Before (inconsistent)
type UserData = { name: string };
interface Config { timeout: number; }

// After (consistent - prefer interface for objects)
interface UserData { name: string; }
interface Config { timeout: number; }
```

**Confidence**: 0.85 (style preference)

## Implementation Plan

### Phase 1: Core TypeScript Fixer (3 hours)

**File**: `xterminator/typescript_fixer.py`

```python
class TypeScriptFixer:
    """
    TypeScript-specific code fixer.

    Mirrors SimpleLLMFixer architecture for consistency.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.categories = {
            'dead_code': self._fix_unused_import,
            'any_types': self._fix_any_type,
            'non_null_assertion': self._fix_non_null_assertion,
            'missing_types': self._fix_missing_type_annotation,
            'console_logs': self._fix_console_log
        }

    async def fix_issue(
        self,
        issue: Dict[str, Any],
        full_code: str,
        file_path: str
    ) -> Optional[Tuple[str, str]]:
        """Generate and apply fix for TypeScript issue."""
        ...
```

**Key Methods**:
- `_fix_unused_import()` - Remove unused imports
- `_fix_any_type()` - Replace `any` with proper types
- `_fix_non_null_assertion()` - Replace `!.` with `?.`
- `_fix_missing_type_annotation()` - Add type annotations
- `_fix_console_log()` - Remove or wrap console statements

### Phase 2: TypeScript Scanner (2 hours)

**File**: `xterminator/typescript_scanner.py`

```python
class TypeScriptScanner:
    """
    Scans TypeScript files for common issues.

    Integrates with classification_engine.py.
    """

    def scan_file(self, file_path: str) -> List[Issue]:
        """Scan TypeScript file for issues."""
        ...

    def scan_directory(
        self,
        dir_path: str,
        max_files: int = 50
    ) -> List[Issue]:
        """Scan directory of TypeScript files."""
        ...
```

### Phase 3: Integration & Testing (2 hours)

1. **Configuration Update**:
   ```python
   # In autofix_policy.py
   class TypeScriptPolicy(AutofixPolicy):
       """TypeScript-specific policy."""

       typescript_strict_mode: bool = True
       allow_any_types: bool = False
       require_return_types: bool = True
   ```

2. **Test Suite**:
   ```python
   # xterminator/test_typescript_fixer.py
   class TestTypeScriptFixer:
       def test_unused_import_removal(self):
       def test_any_type_replacement(self):
       def test_non_null_assertion_fix(self):
       def test_type_annotation_addition(self):
   ```

3. **Demo Script**:
   ```python
   # xterminator/demo_typescript_autofix.py
   async def demo():
       fixer = TypeScriptFixer()
       scanner = TypeScriptScanner()

       # Scan trough/ directory
       issues = scanner.scan_directory('./trough/src')

       # Auto-fix with policy
       policy = TypeScriptPolicy.balanced()
       for issue in issues:
           decision, reason = policy.decide(...)
           if decision == FixDecision.AUTO:
               fixed_code, diff = await fixer.fix_issue(issue)
   ```

### Phase 4: Documentation (1 hour)

- **README**: TypeScript autofix capabilities
- **Examples**: Before/after code samples
- **Integration Guide**: How to run on Trough codebase

## Risk Assessment

### TypeScript-Specific Risks

| Issue Category | Risk Level | Auto-Fix Safe? | Rationale |
|----------------|------------|----------------|-----------|
| Unused imports | LOW | ✅ Yes | Safe removal, no runtime impact |
| Any types | MEDIUM | 🟡 Review | Type inference may be wrong |
| Non-null assertions | MEDIUM | ✅ Yes | Optional chaining is safer |
| Missing types | MEDIUM | 🟡 Review | Type inference required |
| Console.log | LOW | ✅ Yes | Safe removal in production |
| Interface style | LOW | ✅ Yes | Cosmetic, no runtime impact |

### Safety Guardrails

1. **Syntax Validation**: Validate fixed TypeScript with `tsc --noEmit`
2. **Test Coverage**: Require tests for auto-fix (same as Python)
3. **Progressive Rollout**: Start with LOW risk categories
4. **Human Review**: MEDIUM+ risk requires review

## Performance Characteristics

### Expected Performance

- **Scan Speed**: ~50-100 files/second (regex-based)
- **Fix Speed**: ~10-20 fixes/second (simple transformations)
- **LLM Enhancement**: +500ms per fix (if enabled)

### Scalability

- **Trough Codebase**: ~20 TypeScript files → <1 second scan
- **Large Projects**: 500+ files → ~5-10 seconds scan

## Success Metrics

### Phase 1 Success Criteria

✅ TypeScriptFixer class implemented
✅ 5 core fix categories working
✅ Integration with autofix_tracker.py
✅ Test coverage ≥80%

### Phase 2 Success Criteria

✅ TypeScriptScanner scans trough/ successfully
✅ Detects ≥10 real issues in Trough codebase
✅ Auto-fixes ≥5 issues with high confidence
✅ Zero false positives on manual review

## Future Enhancements

### Phase 5: Advanced TypeScript Support

1. **Generic Type Inference**:
   ```typescript
   // Infer generic constraints
   function wrap<T>(value: T): Wrapper<T> { ... }
   ```

2. **Discriminated Unions**:
   ```typescript
   // Detect incomplete union handling
   type Result = Success | Error;
   ```

3. **Async/Promise Patterns**:
   ```typescript
   // Add missing await keywords
   const data = await fetchData();
   ```

4. **React Component Types**:
   ```typescript
   // Add React.FC or props interface
   interface ButtonProps { onClick: () => void; }
   const Button: React.FC<ButtonProps> = ...
   ```

### Integration with Trough Extension

**VS Code Extension Integration**:
```typescript
// trough/src/AutofixProvider.ts
import { TypeScriptAutofix } from './AutofixProvider';

class TypeScriptCodeAction implements vscode.CodeActionProvider {
    async provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range,
        context: vscode.CodeActionContext
    ): Promise<vscode.CodeAction[]> {
        // Call Python TypeScriptFixer via HoloLoom bridge
        const fixes = await bridge.getAutofixes(document, range);
        return fixes.map(fix => this.createCodeAction(fix));
    }
}
```

## Migration Strategy

### Existing Python Autofix → TypeScript Autofix

**Shared Infrastructure**:
- ✅ `autofix_tracker.py` - Already language-agnostic
- ✅ `autofix_policy.py` - Add TypeScript profile
- ✅ `git_applicator.py` - Works with any text files

**New TypeScript-Specific**:
- 🆕 `typescript_fixer.py` - Core TypeScript transformations
- 🆕 `typescript_scanner.py` - TypeScript issue detection
- 🆕 `typescript_policy.py` - TypeScript-specific policies

**No Breaking Changes**:
- Python autofix continues working unchanged
- TypeScript support is additive

## Appendix: TypeScript Patterns Reference

### Common Import Patterns

```typescript
// Named imports
import { Foo, Bar } from './module';

// Namespace imports
import * as vscode from 'vscode';

// Default imports
import React from 'react';

// Side-effect imports
import './styles.css';

// Type-only imports (TS 3.8+)
import type { SomeType } from './types';
```

### Type Annotation Patterns

```typescript
// Function signatures
function foo(a: string, b: number): boolean { ... }

// Arrow functions
const bar = (x: number): number => x * 2;

// Interface vs Type
interface User { name: string; }  // Prefer for objects
type ID = string | number;         // Prefer for unions

// Generics
function wrap<T>(value: T): T[] { return [value]; }

// Optional and readonly
interface Config {
    readonly apiKey: string;
    timeout?: number;
}
```

### Common Anti-Patterns to Fix

```typescript
// ❌ Anti-pattern: any type
function process(data: any) { ... }

// ✅ Fixed
interface ProcessData { id: string; value: number; }
function process(data: ProcessData) { ... }

// ❌ Anti-pattern: non-null assertion
const value = obj!.property;

// ✅ Fixed
const value = obj?.property ?? defaultValue;

// ❌ Anti-pattern: implicit any
function calculate(a, b) { return a + b; }

// ✅ Fixed
function calculate(a: number, b: number): number { return a + b; }
```

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 3 hours | TypeScriptFixer implementation |
| Phase 2 | 2 hours | TypeScriptScanner integration |
| Phase 3 | 2 hours | Testing & integration |
| Phase 4 | 1 hour | Documentation |
| **Total** | **8 hours** | **Complete TypeScript autofix system** |

## References

- **TypeScript Handbook**: https://www.typescriptlang.org/docs/
- **ESLint TypeScript**: https://typescript-eslint.io/
- **VS Code API**: https://code.visualstudio.com/api
- **Python SimpleLLMFixer**: `/home/user/hello-world/xterminator/simple_llm_fixer.py`
- **Trough Extension**: `/home/user/hello-world/trough/src/`
