# Example Interactions for Promptly Analytics

> Example tool calls and responses for LLM training and prompting.

---

## Example 1: Analyzing Prompt Performance

**User:** How is my summarization prompt performing?

**Assistant:** Let me check the analytics for your summarization prompt.

```json
{
  "tool": "promptly_analytics_stats",
  "arguments": {
    "prompt_name": "summarize_article",
    "days": 30
  }
}
```

**Tool Response:**
```json
{
  "prompt_name": "summarize_article",
  "total_executions": 147,
  "avg_quality": 0.82,
  "success_rate": 0.89,
  "avg_latency_ms": 1250,
  "quality_trend": "stable",
  "task_type_distribution": {
    "news": {"count": 78, "avg_quality": 0.85},
    "research": {"count": 45, "avg_quality": 0.79},
    "blog": {"count": 24, "avg_quality": 0.80}
  },
  "version_performance": {
    "1": {"executions": 50, "avg_quality": 0.75},
    "2": {"executions": 97, "avg_quality": 0.86}
  },
  "thompson_expected_quality": 0.84,
  "recommendation": "Good performance. Version 2 shows +14.7% improvement over v1. Consider A/B testing with enhanced version."
}
```

**Assistant:** Your summarization prompt is performing well:
- **147 executions** in the last 30 days
- **82% average quality** with 89% success rate
- Performance is **stable** over time
- Version 2 outperforms v1 by **14.7%**
- Thompson Sampling predicts **84% expected quality**

The prompt works best for news articles (85% quality) and could be improved for research papers (79%).

---

## Example 2: Getting Best Prompt for Task

**User:** Which prompt should I use for code review?

**Assistant:** Let me get the Thompson Sampling recommendation for code review tasks.

```json
{
  "tool": "promptly_recommend",
  "arguments": {
    "task_type": "code_review"
  }
}
```

**Tool Response:**
```json
{
  "recommended_prompt": "code_review_detailed",
  "expected_quality": 0.88,
  "confidence": 0.92,
  "total_candidates": 3,
  "thompson_samples": {
    "code_review_detailed": 0.88,
    "code_review_quick": 0.75,
    "code_review_security": 0.82
  },
  "recommendation_reason": "Highest expected quality with high confidence based on 156 executions"
}
```

**Assistant:** Based on Thompson Sampling optimization, I recommend using **code_review_detailed**:

| Prompt | Expected Quality | Use Case |
|--------|-----------------|----------|
| **code_review_detailed** | 88% ⭐ | Comprehensive reviews |
| code_review_security | 82% | Security-focused reviews |
| code_review_quick | 75% | Fast, basic reviews |

The recommendation has **92% confidence** based on 156 historical executions.

---

## Example 3: Comparing Two Prompts

**User:** Compare my two explanation prompts to see which is better.

**Assistant:** I'll run a statistical comparison between your explanation prompts.

```json
{
  "tool": "promptly_analytics_compare",
  "arguments": {
    "prompt_a": "explain_concept_v1",
    "prompt_b": "explain_concept_v2",
    "days": 30
  }
}
```

**Tool Response:**
```json
{
  "prompt_a": {
    "name": "explain_concept_v1",
    "executions": 89,
    "avg_quality": 0.72,
    "success_rate": 0.78,
    "avg_latency_ms": 980
  },
  "prompt_b": {
    "name": "explain_concept_v2",
    "executions": 67,
    "avg_quality": 0.85,
    "success_rate": 0.91,
    "avg_latency_ms": 1150
  },
  "winner": "explain_concept_v2",
  "quality_difference": 0.13,
  "statistical_significance": true,
  "p_value": 0.003,
  "effect_size": 0.72,
  "recommendation": "explain_concept_v2 is significantly better (+18% quality). The 17% latency increase is justified by quality gains. Recommend migrating to v2."
}
```

**Assistant:** Here's the statistical comparison:

| Metric | v1 | v2 | Difference |
|--------|----|----|------------|
| Quality | 72% | **85%** | +18% ⬆️ |
| Success Rate | 78% | **91%** | +17% ⬆️ |
| Latency | **980ms** | 1150ms | +17% ⬇️ |

**Winner: explain_concept_v2** with statistical significance (p=0.003)

The quality improvement (+18%) outweighs the latency increase. I recommend migrating fully to v2.

---

## Example 4: Enhancing Prompt with HoloLoom Context

**User:** Can you enhance my explanation prompt with relevant context for machine learning topics?

**Assistant:** I'll enhance your prompt with RAG context from hololoom memory.

```json
{
  "tool": "promptly_hololoom_enhance",
  "arguments": {
    "prompt_name": "explain_concept",
    "task_description": "explaining machine learning algorithms and concepts",
    "context_k": 5,
    "injection_method": "prefix"
  }
}
```

**Tool Response:**
```json
{
  "original_prompt": "Explain {concept} in simple terms...",
  "enhanced_prompt": "You have access to the following context about machine learning:\n\n1. Neural networks use layers of interconnected nodes...\n2. Gradient descent optimizes by following the negative gradient...\n3. Backpropagation efficiently computes gradients using chain rule...\n4. Regularization techniques like L1/L2 prevent overfitting...\n5. Cross-validation helps estimate model generalization...\n\n---\n\nExplain {concept} in simple terms...",
  "context_retrieved": 5,
  "context_relevance": 0.87,
  "injection_method": "prefix",
  "token_overhead": 245,
  "recommendation": "Context successfully injected. Relevance score 0.87 indicates strong topical match."
}
```

**Assistant:** Your prompt has been enhanced with 5 relevant context items:

- **Context relevance**: 87% match to machine learning topics
- **Token overhead**: +245 tokens
- **Injection method**: Prefix (context appears before your prompt)

The enhanced prompt now includes foundational ML concepts (neural networks, gradient descent, backpropagation, regularization, cross-validation) that will help provide more accurate explanations.

---

## Example 5: Agentic Execution with Verification

**User:** Run my fact-checking prompt with verification mode to ensure accuracy.

**Assistant:** I'll execute your prompt with agentic verification to double-check the output.

```json
{
  "tool": "promptly_hololoom_agentic",
  "arguments": {
    "prompt_name": "fact_check_claim",
    "mode": "verify",
    "variables": {
      "claim": "Python was created in 1991 by Guido van Rossum"
    },
    "max_steps": 5
  }
}
```

**Tool Response:**
```json
{
  "mode": "verify",
  "steps_taken": 3,
  "initial_response": "The claim is accurate. Python was indeed created by Guido van Rossum, with the first version released in February 1991.",
  "verification": {
    "verified": true,
    "checks_performed": [
      {"check": "creator_name", "result": "confirmed", "source": "memory"},
      {"check": "creation_year", "result": "confirmed", "source": "memory"},
      {"check": "consistency", "result": "no contradictions found"}
    ],
    "confidence": 0.95,
    "discrepancies": []
  },
  "final_response": "✓ VERIFIED: Python was created by Guido van Rossum, with the first version (0.9.0) released in February 1991.",
  "execution_time_ms": 620
}
```

**Assistant:** The fact-check completed with **verification mode**:

**Result:** ✓ VERIFIED

The claim "Python was created in 1991 by Guido van Rossum" is **accurate**.

**Verification checks performed:**
1. ✓ Creator name: Guido van Rossum (confirmed)
2. ✓ Creation year: 1991 (confirmed)
3. ✓ Consistency: No contradictions found

**Confidence:** 95%

---

## Example 6: Research Mode for Complex Topics

**User:** Research the tradeoffs between different bandit algorithms for my decision-making prompt.

**Assistant:** I'll use research mode to explore this topic from multiple angles.

```json
{
  "tool": "promptly_hololoom_agentic",
  "arguments": {
    "prompt_name": "analyze_algorithms",
    "mode": "research",
    "variables": {
      "topic": "multi-armed bandit algorithms",
      "focus": "exploration-exploitation tradeoffs"
    },
    "max_steps": 10
  }
}
```

**Tool Response:**
```json
{
  "mode": "research",
  "steps_taken": 7,
  "queries_explored": [
    "What are the main multi-armed bandit algorithms?",
    "How does Thompson Sampling balance exploration and exploitation?",
    "What are the regret bounds for UCB vs Thompson Sampling?",
    "When should you use epsilon-greedy vs Bayesian approaches?",
    "What are the computational costs of different bandit strategies?",
    "How do contextual bandits extend basic algorithms?",
    "What are real-world applications of each approach?"
  ],
  "key_findings": [
    {
      "finding": "Thompson Sampling often achieves near-optimal regret in practice",
      "confidence": 0.92,
      "sources": 3
    },
    {
      "finding": "UCB provides theoretical guarantees but may over-explore",
      "confidence": 0.88,
      "sources": 2
    },
    {
      "finding": "Epsilon-greedy is simplest but wastes exploration on suboptimal arms",
      "confidence": 0.95,
      "sources": 4
    }
  ],
  "synthesized_response": "Multi-armed bandit algorithms present different exploration-exploitation tradeoffs:\n\n**Thompson Sampling**: Uses Bayesian priors (Beta distributions) to naturally balance exploration and exploitation. Achieves near-optimal regret in practice with O(√T) bounds.\n\n**UCB (Upper Confidence Bound)**: Deterministic approach using confidence intervals. Strong theoretical guarantees but can over-explore in early stages.\n\n**Epsilon-Greedy**: Simplest approach with fixed exploration rate. Easy to implement but inefficient - explores uniformly across all arms regardless of their estimated quality.\n\n**Recommendation**: For prompt optimization, Thompson Sampling is recommended due to natural uncertainty quantification and strong empirical performance.",
  "execution_time_ms": 890
}
```

**Assistant:** Here's my research on bandit algorithm tradeoffs:

### Key Findings

| Algorithm | Exploration Strategy | Best For |
|-----------|---------------------|----------|
| **Thompson Sampling** | Bayesian (Beta priors) | Prompt optimization ⭐ |
| **UCB** | Confidence bounds | Theoretical guarantees |
| **Epsilon-Greedy** | Fixed random rate | Simple implementations |

### Detailed Comparison

1. **Thompson Sampling** (Recommended)
   - Natural exploration via posterior sampling
   - Near-optimal O(√T) regret in practice
   - Handles uncertainty gracefully

2. **UCB**
   - Deterministic upper confidence bounds
   - Strong theoretical guarantees
   - May over-explore initially

3. **Epsilon-Greedy**
   - Fixed ε% random exploration
   - Simple to implement
   - Wastes exploration on known bad options

**For Promptly**: Thompson Sampling is ideal because it naturally quantifies uncertainty in prompt quality and adapts exploration based on confidence.

---

## Example 7: Finding Underperforming Prompts

**User:** Which of my prompts need improvement?

**Assistant:** I'll identify prompts performing below the quality threshold.

```json
{
  "tool": "promptly_analytics_underperforming",
  "arguments": {
    "threshold": 0.7,
    "days": 30
  }
}
```

**Tool Response:**
```json
{
  "threshold": 0.7,
  "underperforming_prompts": [
    {
      "name": "translate_technical",
      "avg_quality": 0.52,
      "executions": 34,
      "issues": ["High failure rate on domain-specific terms", "Inconsistent output format"],
      "suggestion": "Add glossary context via RAG enhancement"
    },
    {
      "name": "generate_tests_legacy",
      "avg_quality": 0.61,
      "executions": 23,
      "issues": ["Outdated testing patterns", "Missing edge cases"],
      "suggestion": "Update to modern testing frameworks, consider v2 creation"
    },
    {
      "name": "summarize_legal",
      "avg_quality": 0.65,
      "executions": 18,
      "issues": ["Missing key legal terms", "Overly simplified"],
      "suggestion": "Enhance with legal domain context using hololoom_enhance"
    }
  ],
  "total_underperforming": 3,
  "total_prompts_analyzed": 15
}
```

**Assistant:** Found **3 prompts** performing below 70% quality:

| Prompt | Quality | Executions | Main Issue |
|--------|---------|------------|------------|
| translate_technical | **52%** ⚠️ | 34 | Domain terms failing |
| generate_tests_legacy | **61%** | 23 | Outdated patterns |
| summarize_legal | **65%** | 18 | Missing legal terms |

### Recommended Actions

1. **translate_technical** (Critical)
   - Add domain glossary via `hololoom_enhance`
   - Consider creating v2 with structured term handling

2. **generate_tests_legacy**
   - Update to modern testing frameworks
   - Create new version with current best practices

3. **summarize_legal**
   - Enhance with legal domain context
   - Add key term preservation requirements

---

## Example 8: Recording Prompt Execution

**User:** I just ran a prompt and want to record the result for analytics.

**Assistant:** I'll record this execution to update the analytics and Thompson Sampling priors.

```json
{
  "tool": "promptly_record_execution",
  "arguments": {
    "prompt_name": "code_review_detailed",
    "version": 2,
    "task_type": "code_review",
    "input_data": "Review this Python function for best practices...",
    "output": "The function has good structure but could benefit from...",
    "quality_score": 0.88,
    "latency_ms": 1250,
    "llm_provider": "anthropic",
    "llm_model": "claude-3-5-sonnet-20241022",
    "token_count": 856
  }
}
```

**Tool Response:**
```json
{
  "recorded": true,
  "execution_id": "exec_20241115_143052_a7b3c",
  "prompt_name": "code_review_detailed",
  "version": 2,
  "quality_score": 0.88,
  "is_success": true,
  "thompson_prior_updated": {
    "previous": {"alpha": 45.2, "beta": 8.3},
    "updated": {"alpha": 46.08, "beta": 8.3},
    "expected_quality_change": "+0.005"
  },
  "prompt_stats_updated": {
    "total_executions": 157,
    "avg_quality": 0.845,
    "success_rate": 0.898
  }
}
```

**Assistant:** Execution recorded successfully:

- **Execution ID:** `exec_20241115_143052_a7b3c`
- **Quality Score:** 88% (counted as success ≥70%)
- **Thompson Prior Updated:** α increased from 45.2 → 46.08

**Updated Prompt Stats:**
- Total executions: 157
- Average quality: 84.5%
- Success rate: 89.8%

The prompt's expected quality increased slightly (+0.5%) based on this successful execution.

---

## Example 9: Finding Similar Prompts

**User:** I need a prompt similar to my summarization prompt but for different content types.

**Assistant:** I'll search for semantically similar prompts in your library.

```json
{
  "tool": "promptly_hololoom_similar",
  "arguments": {
    "query": "content summarization and condensing long text into key points",
    "limit": 5,
    "min_quality": 0.7
  }
}
```

**Tool Response:**
```json
{
  "query": "content summarization and condensing long text into key points",
  "results": [
    {
      "name": "summarize_article",
      "similarity": 0.94,
      "avg_quality": 0.82,
      "description": "Summarize news articles into bullet points"
    },
    {
      "name": "extract_key_points",
      "similarity": 0.89,
      "avg_quality": 0.78,
      "description": "Extract main arguments from documents"
    },
    {
      "name": "tldr_generator",
      "similarity": 0.85,
      "avg_quality": 0.75,
      "description": "Generate TL;DR summaries for any text"
    },
    {
      "name": "meeting_notes_summary",
      "similarity": 0.82,
      "avg_quality": 0.80,
      "description": "Condense meeting transcripts into action items"
    },
    {
      "name": "research_abstract",
      "similarity": 0.79,
      "avg_quality": 0.71,
      "description": "Create abstracts for research papers"
    }
  ],
  "total_matches": 5
}
```

**Assistant:** Found 5 similar prompts with quality ≥70%:

| Prompt | Similarity | Quality | Use Case |
|--------|------------|---------|----------|
| **summarize_article** | 94% | 82% | News articles |
| extract_key_points | 89% | 78% | Document arguments |
| tldr_generator | 85% | 75% | General TL;DR |
| meeting_notes_summary | 82% | 80% | Meeting transcripts |
| research_abstract | 79% | 71% | Research papers |

You might want to:
- Use **summarize_article** as a template for new content types
- Combine techniques from **meeting_notes_summary** for action-oriented summaries
- Improve **research_abstract** (lowest quality) using RAG enhancement

---

## Example 10: Getting Quality Trend

**User:** Show me how my code review prompt quality has changed over the last month.

**Assistant:** I'll fetch the quality trend data for your prompt.

```json
{
  "tool": "promptly_analytics_trend",
  "arguments": {
    "prompt_name": "code_review_detailed",
    "days": 30,
    "granularity": "daily"
  }
}
```

**Tool Response:**
```json
{
  "prompt_name": "code_review_detailed",
  "period": {
    "start": "2024-10-16",
    "end": "2024-11-15",
    "granularity": "daily"
  },
  "trend_data": [
    {"date": "2024-10-16", "avg_quality": 0.78, "executions": 3},
    {"date": "2024-10-17", "avg_quality": 0.82, "executions": 5},
    {"date": "2024-10-18", "avg_quality": 0.80, "executions": 4},
    {"date": "2024-10-25", "avg_quality": 0.85, "executions": 6},
    {"date": "2024-11-01", "avg_quality": 0.88, "executions": 7},
    {"date": "2024-11-08", "avg_quality": 0.90, "executions": 8},
    {"date": "2024-11-15", "avg_quality": 0.89, "executions": 6}
  ],
  "trend_direction": "improving",
  "improvement_rate": 0.14,
  "peak_quality": {"date": "2024-11-08", "value": 0.90},
  "lowest_quality": {"date": "2024-10-16", "value": 0.78}
}
```

**Assistant:** Here's the quality trend for **code_review_detailed** over the last 30 days:

```
Quality Trend (Oct 16 - Nov 15)

0.90 |                    ●
0.85 |              ●  ●     ●
0.80 |    ●  ●
0.78 |●
     └──────────────────────────
      Oct 16        Nov 1     Nov 15
```

**Summary:**
- **Trend:** 📈 Improving (+14% over 30 days)
- **Peak:** 90% on Nov 8
- **Current:** 89%

The prompt shows steady improvement, likely due to:
- Learning from recorded executions
- Thompson Sampling optimization
- Version improvements

---

## Workflow Patterns

### Pattern A: New Task Setup

```
1. promptly_list(pattern="*") → See available prompts
2. promptly_recommend(task_type="my_task") → Get recommendation
3. If no good match → promptly_create(name, content)
4. Test with promptly_hololoom_agentic(mode="verify")
5. Record results with promptly_record_execution()
```

### Pattern B: Continuous Improvement

```
1. Weekly: promptly_analytics_underperforming(threshold=0.7)
2. For each underperformer:
   - promptly_analytics_stats(name) → Understand issues
   - promptly_hololoom_enhance(name) → Try RAG improvement
   - Create v2 with improvements
3. Compare: promptly_analytics_compare(v1, v2)
4. Deploy winner
```

### Pattern C: Critical Task Execution

```
1. promptly_recommend(task_type) → Get best prompt
2. promptly_hololoom_enhance(name, task_description) → Add context
3. promptly_hololoom_agentic(name, mode="verify") → Execute with verification
4. promptly_record_execution() → Update analytics
```

---

## Error Handling Examples

### Prompt Not Found

```json
{
  "error": "prompt_not_found",
  "message": "Prompt 'nonexistent_prompt' not found",
  "suggestion": "Use promptly_list() to see available prompts"
}
```

### Insufficient Data for Comparison

```json
{
  "error": "insufficient_data",
  "message": "prompt_a has only 5 executions (minimum 10 required for statistical comparison)",
  "suggestion": "Wait for more executions or reduce comparison requirements"
}
```

### HoloLoom Connection Failed

```json
{
  "error": "hololoom_unavailable",
  "message": "Could not connect to HoloLoom memory system",
  "fallback": "Basic prompt execution available without RAG enhancement",
  "suggestion": "Check HoloLoom service status or use non-enhanced prompts"
}
```

---

## Best Practices Summary

1. **Always use Thompson Sampling recommendations** for task-based prompt selection
2. **Record all executions** to improve future recommendations
3. **Use verify mode** for critical/safety-sensitive outputs
4. **Enhance with RAG** for domain-specific tasks
5. **Check underperforming prompts weekly** and iterate
6. **Compare versions statistically** before switching
7. **Use research mode** for complex, multi-faceted questions
