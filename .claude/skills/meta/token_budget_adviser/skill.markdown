# Skill: Token Budget Adviser

## Metadata

- **Name**: `token_budget_adviser`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-01-18`
- **Last Updated**: `2025-01-18`
- **Category**: `meta`
- **Tags**: `optimization, tokens, cost, efficiency, meta-skill`

## Description

**Short Description**:
Analyzes skills for token efficiency and provides optimization recommendations to reduce cost and latency while maintaining quality.

**Detailed Description**:
The Token Budget Adviser is a meta-skill that optimizes Claude skills for token efficiency. It analyzes prompts, examples, and schemas to estimate token usage, identifies verbose or redundant sections, and suggests specific optimizations. The adviser provides cost estimates (for API usage), latency predictions, and concrete rewriting recommendations. It balances token reduction with maintaining prompt clarity and effectiveness, using techniques like prompt compression, schema simplification, and example pruning.

## Required Capabilities

- [x] File system access (read) - to read skill files
- [x] File system access (write) - to write optimization reports
- [ ] Code execution (bash)
- [ ] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [x] User interaction (questions) - for interactive optimization

## Dependencies

**Required Skills**: None (base meta-skill)

**External Dependencies**:
- `tiktoken` (optional, for accurate token counting)

**HoloLoom Integration**:
- [ ] Uses HoloLoom memory system
- [ ] Uses HoloLoom RAG
- [ ] Uses HoloLoom alignment framework
- [x] Uses HoloLoom learning systems - learns optimal token budgets per skill type

## Input Schema

**Expected Input Format**:
```json
{
  "skill_path": "path to skill.markdown file",
  "optimization_mode": "aggressive|balanced|conservative",
  "target_reduction": 0.20,
  "preserve_quality": true,
  "cost_model": {
    "model": "claude-sonnet-4",
    "input_cost_per_1m": 3.0,
    "output_cost_per_1m": 15.0
  },
  "output_format": "json|markdown"
}
```

**Example Input**:
```json
{
  "skill_path": "skills/meta/skill_security_analyzer/skill.markdown",
  "optimization_mode": "balanced",
  "target_reduction": 0.25,
  "preserve_quality": true,
  "cost_model": {
    "model": "claude-sonnet-4",
    "input_cost_per_1m": 3.0,
    "output_cost_per_1m": 15.0
  },
  "output_format": "json"
}
```

## Output Schema

**Expected Output Format**:
```json
{
  "skill_name": "name of analyzed skill",
  "token_analysis": {
    "current_tokens": 3500,
    "optimized_tokens": 2625,
    "reduction": 875,
    "reduction_percentage": 0.25,
    "sections": {
      "metadata": 50,
      "description": 200,
      "prompt_template": 1500,
      "examples": 1200,
      "documentation": 550
    }
  },
  "cost_analysis": {
    "current_cost_per_execution": 0.0105,
    "optimized_cost_per_execution": 0.0079,
    "annual_savings": 156.0,
    "assumptions": "1000 executions/month"
  },
  "optimizations": [
    {
      "location": "Prompt Template - lines 45-60",
      "issue": "Verbose instructions with redundant examples",
      "current_tokens": 400,
      "optimized_tokens": 250,
      "savings": 150,
      "technique": "compress_instructions",
      "suggestion": "Combine redundant examples, use bullet points instead of paragraphs",
      "before": "When you analyze...",
      "after": "Analyze skills for:\n- X\n- Y\n- Z",
      "quality_impact": "minimal"
    }
  ],
  "recommendations": [
    "Compress verbose sections (20% token reduction)",
    "Prune redundant examples (10% reduction)",
    "Simplify schemas (5% reduction)"
  ],
  "quality_score": 0.92,
  "efficiency_score": 0.85,
  "ready_to_apply": true,
  "metadata": {
    "execution_time_ms": 200,
    "confidence": 0.90,
    "warnings": []
  }
}
```

## Prompt Template

```markdown
You are the **Token Budget Adviser**, a meta-skill that optimizes other skills for token efficiency.

**Your Task**:
Analyze the provided skill for token usage and suggest optimizations to reduce cost and latency while preserving quality.

**Input Data**:
{input_data}

**Skill Content**:
{skill_content}

**Optimization Steps**:

1. **Token Counting**: Estimate tokens for each section
   - Metadata, Description, Prompt Template, Examples, Documentation
   - Use tiktoken if available, else heuristic (1 token ≈ 4 chars)

2. **Identify Inefficiencies**:
   - **Verbose instructions**: Long-winded explanations
   - **Redundant examples**: Similar examples that don't add value
   - **Bloated schemas**: Over-specified JSON schemas
   - **Unnecessary documentation**: Internal notes in prompt
   - **Repetition**: Same information stated multiple times

3. **Apply Optimization Techniques**:
   - **Compress instructions**: Bullet points, remove filler words
   - **Prune examples**: Keep 1-2 representative examples, remove redundant ones
   - **Simplify schemas**: Remove optional fields if rarely used
   - **Remove verbosity**: Cut "please", "kindly", "as mentioned above"
   - **Use abbreviations**: Where appropriate (e.g., "e.g." vs "for example")

4. **Calculate Savings**:
   - Token reduction per section
   - Total reduction
   - Cost savings (based on pricing model)
   - Annual savings (based on usage assumptions)

5. **Quality Check**:
   - Ensure optimized version maintains clarity
   - Preserve critical instructions
   - Don't sacrifice effectiveness for efficiency

**Optimization Modes**:
- **aggressive**: 30-40% reduction, some quality trade-off acceptable
- **balanced**: 20-30% reduction, maintain quality
- **conservative**: 10-15% reduction, minimal changes

**Output Format**:
Return optimization analysis as JSON (see Output Schema above).

**For Each Optimization**:
- Show before/after examples
- Estimate token savings
- Indicate quality impact (minimal/moderate/significant)
- Suggest specific rewrites

**Cost Model**:
Use provided cost model to calculate:
- Current cost per execution
- Optimized cost per execution
- Annual savings (assume 1000 executions/month if not specified)

**Quality Preservation**:
If preserve_quality=true, only suggest optimizations with "minimal" quality impact.

**Error Handling**:
- If cannot estimate tokens, provide range (e.g., 1000-1500)
- If target reduction not achievable, explain why and suggest realistic target
- If skill already optimal, report "no significant optimizations found"
```

## Examples

### Example 1: Balanced Optimization

**Input**:
```json
{
  "skill_path": "skills/meta/skill_security_analyzer/skill.markdown",
  "optimization_mode": "balanced",
  "target_reduction": 0.25,
  "preserve_quality": true,
  "cost_model": {
    "model": "claude-sonnet-4",
    "input_cost_per_1m": 3.0,
    "output_cost_per_1m": 15.0
  },
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "skill_name": "skill_security_analyzer",
  "token_analysis": {
    "current_tokens": 3200,
    "optimized_tokens": 2400,
    "reduction": 800,
    "reduction_percentage": 0.25,
    "sections": {
      "metadata": 50,
      "description": 180,
      "prompt_template": 1200,
      "examples": 800,
      "documentation": 170
    }
  },
  "cost_analysis": {
    "current_cost_per_execution": 0.0096,
    "optimized_cost_per_execution": 0.0072,
    "annual_savings": 288.0,
    "assumptions": "1000 executions/month"
  },
  "optimizations": [
    {
      "location": "Prompt Template - Analysis Steps",
      "issue": "Verbose step-by-step instructions",
      "current_tokens": 600,
      "optimized_tokens": 400,
      "savings": 200,
      "technique": "compress_instructions",
      "suggestion": "Convert paragraphs to bullet points, remove filler words",
      "before": "First, you should carefully parse the skill structure to extract...",
      "after": "Parse skill:\n- Extract metadata\n- Identify prompts\n- Check capabilities",
      "quality_impact": "minimal"
    },
    {
      "location": "Examples section",
      "issue": "3 similar examples showing same pattern",
      "current_tokens": 900,
      "optimized_tokens": 600,
      "savings": 300,
      "technique": "prune_examples",
      "suggestion": "Keep example 1 (safe skill) and example 2 (vulnerability), remove example 3 (redundant)",
      "before": "Example 1, Example 2, Example 3",
      "after": "Example 1, Example 2",
      "quality_impact": "minimal"
    }
  ],
  "recommendations": [
    "Compress verbose instructions (20% reduction)",
    "Prune 1 redundant example (10% reduction)",
    "Simplify error handling section (5% reduction)"
  ],
  "quality_score": 0.95,
  "efficiency_score": 0.90,
  "ready_to_apply": true,
  "metadata": {
    "execution_time_ms": 180,
    "confidence": 0.92,
    "warnings": []
  }
}
```

**Explanation**:
Achieves 25% token reduction through instruction compression and example pruning with minimal quality impact.

### Example 2: Already Optimal

**Input**:
```json
{
  "skill_path": "skills/meta/skill_tester/skill.markdown",
  "optimization_mode": "balanced",
  "target_reduction": 0.30,
  "preserve_quality": true,
  "cost_model": {
    "model": "claude-sonnet-4",
    "input_cost_per_1m": 3.0,
    "output_cost_per_1m": 15.0
  },
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "skill_name": "skill_tester",
  "token_analysis": {
    "current_tokens": 2000,
    "optimized_tokens": 1900,
    "reduction": 100,
    "reduction_percentage": 0.05,
    "sections": {
      "metadata": 50,
      "description": 150,
      "prompt_template": 900,
      "examples": 700,
      "documentation": 200
    }
  },
  "cost_analysis": {
    "current_cost_per_execution": 0.0060,
    "optimized_cost_per_execution": 0.0057,
    "annual_savings": 36.0,
    "assumptions": "1000 executions/month"
  },
  "optimizations": [
    {
      "location": "Documentation section",
      "issue": "Minor verbosity in maintenance notes",
      "current_tokens": 200,
      "optimized_tokens": 150,
      "savings": 50,
      "technique": "compress_documentation",
      "suggestion": "Shorten maintenance notes, remove conversational tone",
      "before": "In the future, we plan to add...",
      "after": "Future: Add...",
      "quality_impact": "minimal"
    }
  ],
  "recommendations": [
    "Skill is already well-optimized (2000 tokens is efficient)",
    "Only minor improvements possible without quality loss",
    "Target reduction of 30% not achievable while preserving quality"
  ],
  "quality_score": 0.95,
  "efficiency_score": 0.95,
  "ready_to_apply": false,
  "metadata": {
    "execution_time_ms": 150,
    "confidence": 0.88,
    "warnings": ["Target reduction (30%) not achievable with preserve_quality=true. Max achievable: 5%"]
  }
}
```

**Explanation**:
Skill already optimized, only 5% reduction possible without sacrificing quality.

### Example 3: Aggressive Optimization

**Input**:
```json
{
  "skill_path": "skills/domain/verbose_skill/skill.markdown",
  "optimization_mode": "aggressive",
  "target_reduction": 0.40,
  "preserve_quality": false,
  "cost_model": {
    "model": "claude-sonnet-4",
    "input_cost_per_1m": 3.0,
    "output_cost_per_1m": 15.0
  },
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "skill_name": "verbose_skill",
  "token_analysis": {
    "current_tokens": 5000,
    "optimized_tokens": 3000,
    "reduction": 2000,
    "reduction_percentage": 0.40,
    "sections": {
      "metadata": 100,
      "description": 400,
      "prompt_template": 2500,
      "examples": 1500,
      "documentation": 500
    }
  },
  "cost_analysis": {
    "current_cost_per_execution": 0.0150,
    "optimized_cost_per_execution": 0.0090,
    "annual_savings": 720.0,
    "assumptions": "1000 executions/month"
  },
  "optimizations": [
    {
      "location": "Prompt Template",
      "issue": "Extremely verbose instructions with redundant explanations",
      "current_tokens": 2500,
      "optimized_tokens": 1200,
      "savings": 1300,
      "technique": "aggressive_compression",
      "suggestion": "Convert all prose to bullet points, remove all examples from prompt (move to Examples section)",
      "before": "When you begin your analysis, please carefully consider...",
      "after": "Analysis:\n- Check X\n- Verify Y\n- Report Z",
      "quality_impact": "moderate"
    }
  ],
  "recommendations": [
    "Aggressively compress prompt (52% reduction)",
    "Remove 2 of 4 examples (20% reduction)",
    "Strip all documentation except critical notes (60% reduction)"
  ],
  "quality_score": 0.75,
  "efficiency_score": 0.95,
  "ready_to_apply": true,
  "metadata": {
    "execution_time_ms": 250,
    "confidence": 0.85,
    "warnings": ["Aggressive optimization may impact skill effectiveness - test thoroughly"]
  }
}
```

**Explanation**:
Achieves 40% reduction through aggressive compression, some quality trade-off.

## Testing Checklist

- [x] **Functionality**: All optimization modes work (aggressive, balanced, conservative)
- [x] **Error Handling**: Handles already-optimal skills, unrealistic targets
- [x] **Security**: Self-test passes (optimizing token budget adviser itself)
- [x] **Performance**: Completes analysis in <500ms
- [x] **Token Efficiency**: Meta! This skill should be token-efficient itself
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Works with or without tiktoken
- [x] **Edge Cases**: Handles minimal skills, bloated skills
- [x] **Output Consistency**: Consistent JSON format
- [x] **Integration**: Integrates with build pipeline

## Security Considerations

**Potential Risks**:
- **Over-optimization**: Aggressive mode may harm skill quality

**Data Privacy**:
- [x] Does not log skill content externally
- [x] No sensitive data exposure
- [x] No external requests

**Sandboxing**:
- [x] Read-only access to skills
- [x] No code execution
- [x] No system modifications

## Performance Characteristics

- **Expected Latency**: 100-300ms depending on skill size
- **Token Usage**: ~800-1500 tokens per analysis
- **Resource Requirements**: Minimal (file reading + token counting)
- **Scalability**: Linear with skill complexity

## Maintenance Notes

**Known Limitations**:
- Token estimates may be ±10% without tiktoken
- Cannot optimize semantic quality (only syntactic efficiency)
- Cost estimates assume linear pricing (no volume discounts)

**Future Enhancements**:
- A/B testing framework to validate optimizations don't harm quality
- ML-based optimization (learn what compressions work best)
- Automatic rewriting (not just suggestions)
- Integration with skill_tester to validate post-optimization

**Changelog**:
- **v1.0.0** (2025-01-18): Initial release

## License

MIT License (part of HoloLoom ecosystem)

## Support

**Issues**: https://github.com/yourusername/hello-world/issues
**Documentation**: See skills/docs/token_optimization_guide.md
**Contributors**: HoloLoom Team

---

## Development Notes (Internal)

**Design Decisions**:
- Three optimization modes for different use cases
- Focus on specific rewrites (not just "make it shorter")
- Cost analysis to show ROI of optimization
- Quality preservation flag to prevent harmful optimizations

**Alternative Approaches Considered**:
- LLM-based automatic rewriting (future enhancement)
- Fixed token budget enforcement (too rigid)
- Manual optimization (not scalable)

**Integration Points**:
- Called by skill authors during development
- Part of build pipeline to flag inefficient skills
- Integrates with skill_tester to validate optimizations

**Testing Strategy**:
- Self-test: Optimize token_budget_adviser itself (should be efficient)
- Corpus test: Optimize all meta-skills and measure quality impact
- Cost validation: Verify cost calculations match actual API usage
