# Skill: Continuous Learning Capture

## Metadata

- **Name**: `continuous_learning_capture`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-01-18`
- **Last Updated**: `2025-01-18`
- **Category**: `meta`
- **Tags**: `learning, automation, skill-creation, meta-skill, continuous-improvement`

## Description

**Short Description**:
Captures "learning moments" from user sessions and automatically proposes new skills based on patterns, successful interactions, and user feedback.

**Detailed Description**:
The Continuous Learning Capture is a meta-skill that creates a self-improving skills ecosystem. It monitors user interactions to identify learning moments—situations where Claude successfully solved a problem that could be codified into a reusable skill. When a pattern emerges (e.g., user repeatedly asks for similar help), the system proposes creating a dedicated skill. It generates skill.markdown scaffolds pre-filled with examples from real interactions, leveraging HoloLoom's memory system to find similar past successes. This creates a virtuous cycle: interactions → learning → new skills → better interactions.

## Required Capabilities

- [x] File system access (read) - to read interaction logs
- [x] File system access (write) - to create skill proposals
- [ ] Code execution (bash)
- [ ] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [x] User interaction (questions) - to confirm skill creation

## Dependencies

**Required Skills**:
- `skill_gap_analyzer` (optional) - to validate proposed skills don't duplicate existing ones

**External Dependencies**: None

**HoloLoom Integration**:
- [x] Uses HoloLoom memory system - retrieves similar past interactions
- [x] Uses HoloLoom RAG - finds relevant context for skill creation
- [ ] Uses HoloLoom alignment framework
- [x] Uses HoloLoom learning systems - learns what makes a good skill candidate

## Input Schema

**Expected Input Format**:
```json
{
  "monitoring_mode": "active|passive",
  "session_history": [
    {
      "timestamp": "2025-01-18T10:30:00Z",
      "user_query": "Help me optimize this React component",
      "claude_response": "Here's an optimized version...",
      "success": true,
      "user_feedback": "Very helpful!"
    }
  ],
  "min_pattern_frequency": 3,
  "auto_create_threshold": 5,
  "output_format": "json|markdown"
}
```

**Example Input**:
```json
{
  "monitoring_mode": "active",
  "session_history": [
    {
      "timestamp": "2025-01-18T10:30:00Z",
      "user_query": "Explain this TypeScript error",
      "claude_response": "This error means...",
      "success": true,
      "user_feedback": null
    },
    {
      "timestamp": "2025-01-18T11:15:00Z",
      "user_query": "What does this TypeScript type error mean?",
      "claude_response": "The type error indicates...",
      "success": true,
      "user_feedback": "Thanks!"
    },
    {
      "timestamp": "2025-01-18T14:20:00Z",
      "user_query": "Help me understand this TS compilation error",
      "claude_response": "This compilation error...",
      "success": true,
      "user_feedback": "Perfect"
    }
  ],
  "min_pattern_frequency": 3,
  "auto_create_threshold": 5,
  "output_format": "json"
}
```

## Output Schema

**Expected Output Format**:
```json
{
  "learning_summary": {
    "total_interactions": 50,
    "learning_moments": 5,
    "patterns_identified": 3,
    "skills_proposed": 2
  },
  "patterns": [
    {
      "pattern_id": "typescript_error_explainer",
      "description": "User frequently asks for TypeScript error explanations",
      "frequency": 3,
      "confidence": 0.88,
      "sample_queries": [
        "Explain this TypeScript error",
        "What does this TS type error mean?",
        "Help me understand this TS compilation error"
      ],
      "avg_success_rate": 1.0,
      "user_satisfaction": 0.95
    }
  ],
  "skill_proposals": [
    {
      "proposed_skill_name": "typescript_error_explainer",
      "category": "domain",
      "description": "Explains TypeScript compilation errors in plain language with fix suggestions",
      "priority": "high",
      "evidence": {
        "frequency": 3,
        "success_rate": 1.0,
        "user_feedback": ["Thanks!", "Perfect", "Very helpful!"]
      },
      "draft_skill": {
        "metadata": {
          "name": "typescript_error_explainer",
          "version": "1.0.0",
          "category": "domain",
          "tags": ["typescript", "errors", "debugging"]
        },
        "description": "Auto-generated from successful user interactions",
        "prompt_template": "You are a TypeScript error explainer...",
        "examples": [
          {
            "input": "Explain this TypeScript error",
            "output": "This error means...",
            "source": "User interaction 2025-01-18T10:30:00Z"
          }
        ]
      },
      "recommended_next_steps": [
        "Review draft_skill and refine prompt",
        "Add more examples from interaction history",
        "Run skill_security_analyzer and skill_tester",
        "Deploy to skills/domain/ directory"
      ]
    }
  ],
  "recommendations": [
    "Create typescript_error_explainer skill (3 requests, 100% success rate)",
    "Monitor for React optimization pattern (2 requests so far, needs 1 more)"
  ],
  "metadata": {
    "execution_time_ms": 400,
    "confidence": 0.85,
    "warnings": []
  }
}
```

## Prompt Template

```markdown
You are the **Continuous Learning Capture** meta-skill that creates new skills from successful user interactions.

**Your Task**:
Analyze session history to identify learning moments and propose new skills based on recurring patterns.

**Input Data**:
{input_data}

**Session History**:
{session_history}

**Analysis Steps**:

1. **Pattern Detection**:
   - Group similar queries using semantic similarity
   - Count frequency of each pattern
   - Filter patterns by min_pattern_frequency threshold

2. **Success Analysis**:
   - Calculate success rate per pattern (successful interactions / total)
   - Extract user feedback sentiment
   - Identify high-value patterns (frequent + successful + positive feedback)

3. **Skill Candidacy**:
   - For each pattern, determine if it's skill-worthy:
     - Frequency ≥ min_pattern_frequency
     - Success rate ≥ 0.75
     - Task is repeatable (not one-off)
     - Scope is well-defined
   - Prioritize by: frequency × success_rate × user_satisfaction

4. **Draft Skill Generation**:
   - Extract skill name from pattern (e.g., "typescript_error_explainer")
   - Generate description from successful examples
   - Create prompt template from common structure
   - Populate examples from actual interactions
   - Suggest capabilities based on what was used

5. **Validation**:
   - Check against existing skills (avoid duplicates)
   - Ensure sufficient examples (min 2-3)
   - Verify skill scope is neither too narrow nor too broad

**Monitoring Modes**:
- **active**: Proactively suggest skills after min_frequency met
- **passive**: Only report patterns, let user decide

**Auto-Creation**:
If pattern frequency ≥ auto_create_threshold:
- Auto-generate full skill.markdown
- Save to skills/proposals/ directory
- Notify user for review

**Output Format**:
Return learning analysis as JSON (see Output Schema above).

**Quality Standards**:
- Only propose skills with ≥75% success rate
- Require min 2-3 real examples from interactions
- Ensure skill scope is well-defined
- Avoid over-fitting to specific user queries

**Error Handling**:
- If session history empty, report "no data to analyze"
- If no patterns found, suggest increasing monitoring period
- If proposed skill duplicates existing, merge examples instead
```

## Examples

### Example 1: Pattern Detected → Skill Proposed

**Input**:
```json
{
  "monitoring_mode": "active",
  "session_history": [
    {
      "timestamp": "2025-01-18T10:30:00Z",
      "user_query": "Explain this TypeScript error: TS2322",
      "claude_response": "TS2322 means type mismatch...",
      "success": true,
      "user_feedback": "Thanks!"
    },
    {
      "timestamp": "2025-01-18T11:15:00Z",
      "user_query": "What does TS2345 error mean?",
      "claude_response": "TS2345 indicates argument type error...",
      "success": true,
      "user_feedback": "Perfect"
    },
    {
      "timestamp": "2025-01-18T14:20:00Z",
      "user_query": "Help with TS7053 error",
      "claude_response": "TS7053 is an index signature error...",
      "success": true,
      "user_feedback": "Very helpful!"
    }
  ],
  "min_pattern_frequency": 3,
  "auto_create_threshold": 5,
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "learning_summary": {
    "total_interactions": 3,
    "learning_moments": 1,
    "patterns_identified": 1,
    "skills_proposed": 1
  },
  "patterns": [
    {
      "pattern_id": "typescript_error_explainer",
      "description": "Explaining TypeScript error codes",
      "frequency": 3,
      "confidence": 0.95,
      "sample_queries": [
        "Explain this TypeScript error: TS2322",
        "What does TS2345 error mean?",
        "Help with TS7053 error"
      ],
      "avg_success_rate": 1.0,
      "user_satisfaction": 1.0
    }
  ],
  "skill_proposals": [
    {
      "proposed_skill_name": "typescript_error_explainer",
      "category": "domain",
      "description": "Explains TypeScript error codes in plain language with fix suggestions",
      "priority": "high",
      "evidence": {
        "frequency": 3,
        "success_rate": 1.0,
        "user_feedback": ["Thanks!", "Perfect", "Very helpful!"]
      },
      "draft_skill": {
        "metadata": {
          "name": "typescript_error_explainer",
          "version": "1.0.0",
          "category": "domain",
          "tags": ["typescript", "errors", "debugging", "compiler"]
        },
        "description": "Explains TypeScript compiler errors in clear language with examples and fix suggestions. Auto-generated from 3 successful user interactions.",
        "prompt_template": "You are a TypeScript error explainer. When given a TS error code (e.g., TS2322), provide:\n1. Plain language explanation\n2. Common causes\n3. Example that triggers the error\n4. How to fix it\n\nInput: {error_code}\nOutput: Explanation + fix",
        "examples": [
          {
            "input": {"error_code": "TS2322"},
            "output": "TS2322 means type mismatch...",
            "source": "User interaction 2025-01-18T10:30:00Z"
          },
          {
            "input": {"error_code": "TS2345"},
            "output": "TS2345 indicates argument type error...",
            "source": "User interaction 2025-01-18T11:15:00Z"
          },
          {
            "input": {"error_code": "TS7053"},
            "output": "TS7053 is an index signature error...",
            "source": "User interaction 2025-01-18T14:20:00Z"
          }
        ]
      },
      "recommended_next_steps": [
        "Review draft skill and refine prompt template",
        "Add schema for input/output",
        "Run skill_security_analyzer (check for code injection)",
        "Run skill_tester with examples",
        "Deploy to skills/domain/typescript_error_explainer/"
      ]
    }
  ],
  "recommendations": [
    "Create typescript_error_explainer skill immediately (3/3 threshold, 100% success)",
    "Skill is ready for deployment after security/testing review"
  ],
  "metadata": {
    "execution_time_ms": 320,
    "confidence": 0.92,
    "warnings": []
  }
}
```

**Explanation**:
Pattern detected after 3 successful interactions, skill auto-proposed with draft.

### Example 2: Pattern Emerging (Not Ready Yet)

**Input**:
```json
{
  "monitoring_mode": "active",
  "session_history": [
    {
      "timestamp": "2025-01-18T10:00:00Z",
      "user_query": "Optimize this React component",
      "claude_response": "Use React.memo()...",
      "success": true,
      "user_feedback": null
    },
    {
      "timestamp": "2025-01-18T15:00:00Z",
      "user_query": "Make this React code faster",
      "claude_response": "Apply useMemo and useCallback...",
      "success": true,
      "user_feedback": "Good advice"
    }
  ],
  "min_pattern_frequency": 3,
  "auto_create_threshold": 5,
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "learning_summary": {
    "total_interactions": 2,
    "learning_moments": 1,
    "patterns_identified": 1,
    "skills_proposed": 0
  },
  "patterns": [
    {
      "pattern_id": "react_performance_optimizer",
      "description": "React component performance optimization",
      "frequency": 2,
      "confidence": 0.75,
      "sample_queries": [
        "Optimize this React component",
        "Make this React code faster"
      ],
      "avg_success_rate": 1.0,
      "user_satisfaction": 0.85
    }
  ],
  "skill_proposals": [],
  "recommendations": [
    "Pattern 'react_performance_optimizer' emerging (2/3 requests)",
    "1 more similar request needed to propose skill creation",
    "Continue monitoring for React optimization requests"
  ],
  "metadata": {
    "execution_time_ms": 180,
    "confidence": 0.70,
    "warnings": ["Pattern frequency (2) below threshold (3) - continuing to monitor"]
  }
}
```

**Explanation**:
Pattern emerging but not yet at threshold, system continues monitoring.

### Example 3: Auto-Creation Threshold Met

**Input**:
```json
{
  "monitoring_mode": "active",
  "session_history": [
    {
      "user_query": "Convert this JSON to TypeScript interface",
      "success": true
    },
    {
      "user_query": "Generate TS types from this JSON",
      "success": true
    },
    {
      "user_query": "Create TypeScript interface for this API response",
      "success": true
    },
    {
      "user_query": "JSON to TypeScript converter",
      "success": true
    },
    {
      "user_query": "Make TS types from this JSON payload",
      "success": true
    }
  ],
  "min_pattern_frequency": 3,
  "auto_create_threshold": 5,
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "learning_summary": {
    "total_interactions": 5,
    "learning_moments": 1,
    "patterns_identified": 1,
    "skills_proposed": 1
  },
  "patterns": [
    {
      "pattern_id": "json_to_typescript_converter",
      "description": "Converting JSON to TypeScript interfaces/types",
      "frequency": 5,
      "confidence": 0.98,
      "sample_queries": [
        "Convert this JSON to TypeScript interface",
        "Generate TS types from this JSON"
      ],
      "avg_success_rate": 1.0,
      "user_satisfaction": 0.90
    }
  ],
  "skill_proposals": [
    {
      "proposed_skill_name": "json_to_typescript_converter",
      "category": "domain",
      "description": "Auto-converts JSON to TypeScript interfaces with proper typing",
      "priority": "critical",
      "evidence": {
        "frequency": 5,
        "success_rate": 1.0,
        "user_feedback": ["Good!", "Perfect", "Thanks!"]
      },
      "draft_skill": {},
      "auto_created": true,
      "file_path": "skills/proposals/json_to_typescript_converter/skill.markdown",
      "recommended_next_steps": [
        "Review auto-generated skill at skills/proposals/json_to_typescript_converter/",
        "Run skill_security_analyzer and skill_tester",
        "Move to skills/domain/ if tests pass"
      ]
    }
  ],
  "recommendations": [
    "AUTO-CREATED: json_to_typescript_converter skill (5 requests, 100% success)",
    "Skill saved to skills/proposals/ - review and test before deploying"
  ],
  "metadata": {
    "execution_time_ms": 450,
    "confidence": 0.95,
    "warnings": []
  }
}
```

**Explanation**:
Threshold met (5 requests), skill automatically created and saved to proposals/ directory.

## Testing Checklist

- [x] **Functionality**: Both monitoring modes work (active/passive)
- [x] **Error Handling**: Handles empty history, no patterns gracefully
- [x] **Security**: Self-test passes (doesn't create skills recursively!)
- [x] **Performance**: Completes analysis in <1s for 100 interactions
- [x] **Token Efficiency**: Efficient pattern detection algorithms
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Works standalone or with HoloLoom integration
- [x] **Edge Cases**: Handles duplicate patterns, low-quality proposals
- [x] **Output Consistency**: Consistent JSON format
- [x] **Integration**: Integrates with skill_gap_analyzer, HoloLoom memory

## Security Considerations

**Potential Risks**:
- **Auto-creation**: Could create harmful skills - always require human review
- **Prompt injection**: Malicious user queries shouldn't be codified into skills

**Data Privacy**:
- [x] Anonymizes user data in draft skills
- [x] User feedback not logged externally
- [x] No external requests

**Sandboxing**:
- [x] Writes only to skills/proposals/ (not directly to skills/domain/)
- [x] Requires human approval before deployment
- [x] No code execution

## Performance Characteristics

- **Expected Latency**: 200-800ms depending on history size
- **Token Usage**: ~1500-4000 tokens per analysis
- **Resource Requirements**: Moderate (pattern matching + semantic similarity)
- **Scalability**: Linear with session history size

## Maintenance Notes

**Known Limitations**:
- Requires sufficient interaction history (min 50-100 interactions recommended)
- Pattern detection may have false positives (requires human review)
- Cannot guarantee skill quality (only success rate)

**Future Enhancements**:
- ML-based pattern detection (more sophisticated than frequency counting)
- Automatic testing of proposed skills before deployment
- Integration with skill_gap_analyzer to merge with planned skills
- Real-time monitoring (not just batch analysis)
- Skill evolution (update existing skills based on new patterns)

**Changelog**:
- **v1.0.0** (2025-01-18): Initial release

## License

MIT License (part of HoloLoom ecosystem)

## Support

**Issues**: https://github.com/yourusername/hello-world/issues
**Documentation**: See skills/docs/continuous_learning_guide.md
**Contributors**: HoloLoom Team

---

## Development Notes (Internal)

**Design Decisions**:
- Two-phase approach: monitoring → proposal (with human review)
- Auto-creation only at high threshold (5+) to avoid low-quality skills
- Draft skills saved to proposals/ directory (not auto-deployed)
- Real interaction examples embedded in skill.markdown for authenticity

**Alternative Approaches Considered**:
- Fully automated skill creation (too risky, removed human oversight)
- Manual skill creation (doesn't scale, misses opportunities)
- Rule-based pattern detection (chosen for v1, ML for future)

**Integration Points**:
- Runs continuously in background (or on-demand)
- Feeds into skill_gap_analyzer for roadmap planning
- Uses HoloLoom memory to find similar past interactions
- Outputs to skills/proposals/ for human review

**Testing Strategy**:
- Bootstrap test: Run on empty history (should report "no data")
- Self-test: Ensure it doesn't recursively propose meta-skills
- Corpus test: Analyze 1000 synthetic interactions, measure precision/recall
- Validation: Compare proposed skills with manually created skills
