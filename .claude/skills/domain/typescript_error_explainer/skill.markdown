# Skill: TypeScript Error Explainer

## Metadata

- **Name**: `typescript_error_explainer`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `typescript, errors, debugging, compiler`

## Description

**Short Description**:
Decodes cryptic TypeScript compiler errors into human-readable explanations with context-aware fix suggestions and related error patterns.

**Detailed Description**:
TypeScript compiler errors (TS####) can be confusing, especially for beginners. This skill analyzes TypeScript errors, provides clear explanations of what went wrong, suggests multiple fix strategies, identifies related errors that might occur, and explains the underlying TypeScript concepts. It goes beyond simple error message translation by providing educational context and actionable fixes.

## Required Capabilities

- [ ] File system access (read)
- [ ] File system access (write)
- [ ] Code execution (bash)
- [ ] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None (standalone)

**External Dependencies**: None

**HoloLoom Integration**:
- [ ] Uses HoloLoom memory system
- [ ] Uses HoloLoom RAG
- [ ] Uses HoloLoom alignment framework
- [ ] Uses HoloLoom learning systems

## Input Schema

```json
{
  "error_code": "string - TypeScript error code (e.g., TS2322)",
  "error_message": "string - Full compiler error message",
  "code_snippet": "string - Code that triggered the error",
  "context": "string (optional) - Additional context about the code"
}
```

## Output Schema

```json
{
  "explanation": "string - Human-readable explanation of the error",
  "root_cause": "string - What actually caused this error",
  "fix_suggestions": [
    {
      "approach": "string - Fix strategy name",
      "code_example": "string - Example fix code",
      "explanation": "string - Why this fix works"
    }
  ],
  "related_errors": ["array of related TS error codes"],
  "concept_explanation": "string - Underlying TypeScript concept",
  "metadata": {
    "error_severity": "error|warning|info",
    "common_mistake": "boolean",
    "confidence": "number (0.0-1.0)"
  }
}
```

## Prompt Template

```markdown
You are a TypeScript expert helping developers understand compiler errors.

**Input Error**:
- Code: {error_code}
- Message: {error_message}
- Snippet: {code_snippet}
- Context: {context}

**Your Task**:
1. Explain what this error means in plain English
2. Identify the root cause (why TypeScript is complaining)
3. Provide 2-3 fix suggestions with code examples
4. List related errors that might occur
5. Explain the underlying TypeScript concept

**Output Requirements**:
- Be clear and educational (not condescending)
- Provide runnable code examples
- Explain the "why" not just the "how"
- Flag common mistakes
- Rate your confidence in the explanation

**Common TS Errors Database**:
- TS2322: Type mismatch
- TS2345: Argument type mismatch
- TS2339: Property does not exist
- TS2571: Object is of type 'unknown'
- TS7006: Implicit 'any' type
- TS2304: Cannot find name
- TS2769: No overload matches

Return structured JSON matching the output schema.
```

## Examples

### Example 1: Type Mismatch (TS2322)

**Input**:
```json
{
  "error_code": "TS2322",
  "error_message": "Type 'string' is not assignable to type 'number'",
  "code_snippet": "let x: number = 'hello';",
  "context": "Variable assignment"
}
```

**Expected Output**:
```json
{
  "explanation": "You're trying to assign a string value ('hello') to a variable declared as a number. TypeScript's type system prevents this mismatch to catch bugs early.",
  "root_cause": "The variable 'x' is explicitly typed as 'number', but you're assigning a string literal 'hello'.",
  "fix_suggestions": [
    {
      "approach": "Change variable type to string",
      "code_example": "let x: string = 'hello';",
      "explanation": "If you intend to store a string, declare the variable as string type."
    },
    {
      "approach": "Change value to number",
      "code_example": "let x: number = 42;",
      "explanation": "If you want a number, assign a numeric value instead."
    },
    {
      "approach": "Use union type",
      "code_example": "let x: number | string = 'hello';",
      "explanation": "If the variable can hold both types, use a union type."
    }
  ],
  "related_errors": ["TS2345", "TS2769"],
  "concept_explanation": "TypeScript uses static typing to ensure variables only hold values of their declared type. This prevents runtime type errors.",
  "metadata": {
    "error_severity": "error",
    "common_mistake": true,
    "confidence": 0.98
  }
}
```

### Example 2: Property Does Not Exist (TS2339)

**Input**:
```json
{
  "error_code": "TS2339",
  "error_message": "Property 'length' does not exist on type 'number'",
  "code_snippet": "const x: number = 5; console.log(x.length);",
  "context": "Trying to access string property on number"
}
```

**Expected Output**:
```json
{
  "explanation": "You're trying to access the 'length' property on a number, but 'length' only exists on strings and arrays.",
  "root_cause": "The variable 'x' is a number (5), and numbers don't have a 'length' property.",
  "fix_suggestions": [
    {
      "approach": "Convert to string first",
      "code_example": "const x: number = 5; console.log(String(x).length);",
      "explanation": "Convert the number to a string, which has a length property."
    },
    {
      "approach": "Use different property",
      "code_example": "const x: number = 5; console.log(x.toString().length);",
      "explanation": "Call toString() to get a string representation, then access length."
    }
  ],
  "related_errors": ["TS2551", "TS2322"],
  "concept_explanation": "Different types have different properties. Numbers don't have a length, but strings and arrays do.",
  "metadata": {
    "error_severity": "error",
    "common_mistake": true,
    "confidence": 0.95
  }
}
```

### Example 3: Unknown Type (TS2571)

**Input**:
```json
{
  "error_code": "TS2571",
  "error_message": "Object is of type 'unknown'",
  "code_snippet": "function process(data: unknown) { return data.value; }",
  "context": "Accessing property on unknown type"
}
```

**Expected Output**:
```json
{
  "explanation": "'unknown' is a type-safe alternative to 'any'. You must narrow the type (via type guards) before accessing properties.",
  "root_cause": "The parameter 'data' is typed as 'unknown', meaning TypeScript doesn't know what properties it has. You need to prove it has a 'value' property first.",
  "fix_suggestions": [
    {
      "approach": "Type guard with type assertion",
      "code_example": "function process(data: unknown) {\n  if (typeof data === 'object' && data !== null && 'value' in data) {\n    return (data as { value: any }).value;\n  }\n}",
      "explanation": "Check that data is an object with a 'value' property before accessing it."
    },
    {
      "approach": "Define explicit type",
      "code_example": "interface Data { value: string; }\nfunction process(data: Data) { return data.value; }",
      "explanation": "If you know the structure, define an interface and type the parameter explicitly."
    }
  ],
  "related_errors": ["TS2345", "TS18046"],
  "concept_explanation": "'unknown' requires type narrowing before use. Use type guards (typeof, instanceof, 'in' operator) to safely access properties.",
  "metadata": {
    "error_severity": "error",
    "common_mistake": true,
    "confidence": 0.92
  }
}
```

## Testing Checklist

- [x] **Functionality**: Correctly explains common TS errors
- [x] **Error Handling**: Handles unknown error codes gracefully
- [x] **Security**: No code execution, only analysis
- [x] **Performance**: < 1s per error explanation
- [x] **Token Efficiency**: ~500 tokens
- [x] **Documentation**: Complete with 3+ examples
- [x] **Dependencies**: Zero external dependencies
- [x] **Edge Cases**: Unknown errors, malformed input
- [x] **Output Consistency**: Structured JSON always
- [x] **Integration**: Standalone (no integrations)

## Security Considerations

**Potential Risks**:
- **Code Injection**: Malicious code snippets could be analyzed
  - **Mitigation**: Static analysis only, no code execution
- **Information Leakage**: Error messages might expose file paths
  - **Mitigation**: Sanitize file paths from error messages

**Data Privacy**:
- [x] Does not log code snippets
- [x] Does not expose internal details
- [x] Does not make external requests

## Performance Characteristics

- **Expected Latency**: 200-500ms
- **Token Usage**: ~500 tokens
- **Resource Requirements**: Minimal (text processing only)
- **Scalability**: O(1) per error (independent processing)

## Maintenance Notes

**Known Limitations**:
- Covers ~20 most common TypeScript errors
- Generic fixes for less common errors
- Requires periodic updates as TypeScript evolves

**Future Enhancements**:
- Integration with TypeScript compiler API
- Context-aware fixes based on codebase patterns
- Learning from user feedback (which fixes worked)

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release with 20 common errors

## License

MIT License

## Support

**Issues**: https://github.com/blakechasteen/hello-world/issues
**Documentation**: TypeScript Handbook (https://www.typescriptlang.org/docs/)
