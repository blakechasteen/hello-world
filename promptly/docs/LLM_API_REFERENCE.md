# Promptly Analytics API Reference for LLMs

> Complete API documentation designed for LLM tool calling.
> All functions return JSON-serializable results suitable for function calling.

## Overview

Promptly is a prompt management and analytics system with:
- **Prompt Management**: Create, version, and organize prompts
- **Analytics**: Track execution quality, trends, and recommendations
- **Thompson Sampling**: Bayesian optimization for prompt selection
- **HoloLoom Integration**: RAG context and agentic reasoning

---

## Function Signatures (Tool Definitions)

### 1. analytics_get_stats

Get comprehensive analytics for a prompt.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_name": {
      "type": "string",
      "description": "Name of the prompt to analyze"
    },
    "days": {
      "type": "integer",
      "default": 30,
      "description": "Analysis window in days (1-365)"
    },
    "format": {
      "type": "string",
      "enum": ["json", "text", "html"],
      "default": "json",
      "description": "Output format"
    }
  },
  "required": ["prompt_name"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_name": {"type": "string"},
    "total_executions": {"type": "integer", "minimum": 0},
    "avg_quality": {"type": "number", "minimum": 0, "maximum": 1},
    "success_rate": {"type": "number", "minimum": 0, "maximum": 1},
    "avg_latency_ms": {"type": "number", "minimum": 0},
    "quality_trend": {
      "type": "string",
      "enum": ["improving", "stable", "declining"]
    },
    "thompson_expected_quality": {"type": "number", "minimum": 0, "maximum": 1},
    "task_type_distribution": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "count": {"type": "integer"},
          "avg_quality": {"type": "number"}
        }
      }
    },
    "version_performance": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "count": {"type": "integer"},
          "avg_quality": {"type": "number"},
          "avg_latency_ms": {"type": "number"}
        }
      }
    },
    "recommendation": {"type": "string"}
  }
}
```

**Example Call:**
```python
result = analytics_get_stats(
    prompt_name="summarize_article",
    days=30,
    format="json"
)
```

**Example Response:**
```json
{
  "prompt_name": "summarize_article",
  "total_executions": 147,
  "avg_quality": 0.82,
  "success_rate": 0.89,
  "avg_latency_ms": 1250,
  "quality_trend": "stable",
  "thompson_expected_quality": 0.84,
  "task_type_distribution": {
    "summarization": {"count": 120, "avg_quality": 0.85},
    "extraction": {"count": 27, "avg_quality": 0.72}
  },
  "version_performance": {
    "1": {"count": 50, "avg_quality": 0.78, "avg_latency_ms": 1400},
    "2": {"count": 97, "avg_quality": 0.84, "avg_latency_ms": 1180}
  },
  "recommendation": "Good performance. Consider A/B testing with enhanced version."
}
```

---

### 2. analytics_compare_prompts

Compare two prompts statistically.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_a": {
      "type": "string",
      "description": "First prompt name"
    },
    "prompt_b": {
      "type": "string",
      "description": "Second prompt name"
    },
    "task_type": {
      "type": "string",
      "description": "Optional task type filter"
    },
    "days": {
      "type": "integer",
      "default": 30,
      "description": "Analysis window in days"
    }
  },
  "required": ["prompt_a", "prompt_b"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_a": {"type": "string"},
    "prompt_b": {"type": "string"},
    "prompt_a_stats": {
      "type": "object",
      "properties": {
        "avg_quality": {"type": "number"},
        "success_rate": {"type": "number"},
        "total_executions": {"type": "integer"},
        "avg_latency_ms": {"type": "number"}
      }
    },
    "prompt_b_stats": {
      "type": "object",
      "properties": {
        "avg_quality": {"type": "number"},
        "success_rate": {"type": "number"},
        "total_executions": {"type": "integer"},
        "avg_latency_ms": {"type": "number"}
      }
    },
    "winner": {"type": "string", "nullable": true},
    "statistical_significance": {"type": "number", "minimum": 0, "maximum": 1},
    "recommendation": {"type": "string"}
  }
}
```

**Example Call:**
```python
result = analytics_compare_prompts(
    prompt_a="summarize_v1",
    prompt_b="summarize_v2",
    task_type="summarization"
)
```

---

### 3. analytics_recommend_prompt

Get Thompson Sampling recommendation for best prompt for a task type.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "task_type": {
      "type": "string",
      "description": "Type of task (e.g., summarization, code_review, explanation)"
    },
    "candidates": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Optional list of candidate prompt names to consider"
    }
  },
  "required": ["task_type"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "recommended_prompt": {"type": "string"},
    "expected_quality": {"type": "number", "minimum": 0, "maximum": 1},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "alternatives": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "expected_quality": {"type": "number"}
        }
      }
    },
    "reasoning": {"type": "string"}
  }
}
```

**Example Call:**
```python
result = analytics_recommend_prompt(
    task_type="code_review",
    candidates=["code_review_detailed", "code_review_quick", "code_review_security"]
)
```

**Example Response:**
```json
{
  "recommended_prompt": "code_review_detailed",
  "expected_quality": 0.88,
  "confidence": 0.92,
  "alternatives": [
    {"name": "code_review_quick", "expected_quality": 0.75},
    {"name": "code_review_security", "expected_quality": 0.82}
  ],
  "reasoning": "Highest expected quality based on 45 executions with Thompson Sampling priors alpha=38, beta=7"
}
```

---

### 4. analytics_get_trend

Get quality trend over time.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_name": {
      "type": "string",
      "description": "Prompt to analyze"
    },
    "days": {
      "type": "integer",
      "default": 30,
      "description": "Analysis window in days"
    },
    "granularity": {
      "type": "string",
      "enum": ["hourly", "daily", "weekly"],
      "default": "daily",
      "description": "Time granularity"
    }
  },
  "required": ["prompt_name"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_name": {"type": "string"},
    "granularity": {"type": "string"},
    "data_points": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "date": {"type": "string", "format": "date"},
          "avg_quality": {"type": "number"},
          "count": {"type": "integer"},
          "success_rate": {"type": "number"}
        }
      }
    },
    "overall_trend": {
      "type": "string",
      "enum": ["improving", "stable", "declining"]
    }
  }
}
```

---

### 5. analytics_identify_underperforming

Find prompts below quality threshold.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "threshold": {
      "type": "number",
      "default": 0.6,
      "minimum": 0,
      "maximum": 1,
      "description": "Quality threshold (0.0-1.0)"
    },
    "days": {
      "type": "integer",
      "default": 30,
      "description": "Analysis window in days"
    }
  }
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "threshold": {"type": "number"},
    "underperforming": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "prompt_name": {"type": "string"},
          "avg_quality": {"type": "number"},
          "total_executions": {"type": "integer"},
          "reason": {"type": "string"},
          "suggestion": {"type": "string"}
        }
      }
    },
    "total_count": {"type": "integer"}
  }
}
```

---

### 6. hololoom_enhance_prompt

Enhance a prompt with RAG context from HoloLoom memory.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_name": {
      "type": "string",
      "description": "Name of prompt to enhance"
    },
    "task_description": {
      "type": "string",
      "description": "Description of the task for context retrieval"
    },
    "context_k": {
      "type": "integer",
      "default": 5,
      "minimum": 1,
      "maximum": 20,
      "description": "Number of context items to retrieve"
    },
    "injection_method": {
      "type": "string",
      "enum": ["prefix", "suffix", "inline"],
      "default": "prefix",
      "description": "How to inject context into prompt"
    }
  },
  "required": ["prompt_name"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "original_prompt": {"type": "string"},
    "enhanced_prompt": {"type": "string"},
    "context_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "content": {"type": "string"},
          "source": {"type": "string"},
          "relevance": {"type": "number"}
        }
      }
    },
    "injection_method": {"type": "string"},
    "total_context_tokens": {"type": "integer"},
    "avg_relevance": {"type": "number"}
  }
}
```

**Example Call:**
```python
result = hololoom_enhance_prompt(
    prompt_name="explain_concept",
    task_description="explaining machine learning algorithms",
    context_k=5,
    injection_method="prefix"
)
```

---

### 7. hololoom_run_agentic

Execute a prompt with agentic reasoning for verification or research.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_name": {
      "type": "string",
      "description": "Name of prompt to execute"
    },
    "variables": {
      "type": "object",
      "description": "Variables to substitute in prompt",
      "additionalProperties": {"type": "string"}
    },
    "mode": {
      "type": "string",
      "enum": ["direct", "verify", "research", "plan_execute"],
      "default": "direct",
      "description": "Reasoning mode"
    },
    "max_steps": {
      "type": "integer",
      "default": 5,
      "minimum": 1,
      "maximum": 20,
      "description": "Maximum reasoning steps"
    }
  },
  "required": ["prompt_name", "mode"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "response": {"type": "string"},
    "mode": {"type": "string"},
    "steps_taken": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "step_index": {"type": "integer"},
          "step_type": {"type": "string"},
          "input_text": {"type": "string"},
          "output_text": {"type": "string"},
          "confidence": {"type": "number"},
          "duration_ms": {"type": "number"}
        }
      }
    },
    "total_steps": {"type": "integer"},
    "total_duration_ms": {"type": "number"},
    "aggregated_confidence": {"type": "number"},
    "verification": {
      "type": "object",
      "nullable": true,
      "properties": {
        "verified": {"type": "boolean"},
        "confidence": {"type": "number"},
        "checks_passed": {"type": "array", "items": {"type": "string"}},
        "checks_failed": {"type": "array", "items": {"type": "string"}},
        "discrepancies": {"type": "array", "items": {"type": "object"}},
        "suggestions": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

**Example Call:**
```python
result = hololoom_run_agentic(
    prompt_name="analyze_code",
    variables={"code": "def foo(): return 42"},
    mode="verify",
    max_steps=5
)
```

---

### 8. hololoom_find_similar

Find similar prompts based on semantic similarity.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query or concept"
    },
    "limit": {
      "type": "integer",
      "default": 5,
      "minimum": 1,
      "maximum": 20,
      "description": "Maximum results to return"
    },
    "min_quality": {
      "type": "number",
      "default": 0.7,
      "minimum": 0,
      "maximum": 1,
      "description": "Minimum quality threshold"
    }
  },
  "required": ["query"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "prompt_name": {"type": "string"},
          "similarity": {"type": "number"},
          "avg_quality": {"type": "number"},
          "total_executions": {"type": "integer"},
          "preview": {"type": "string"}
        }
      }
    },
    "total_found": {"type": "integer"}
  }
}
```

---

### 9. promptly_get

Retrieve a prompt by name.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Prompt name"
    },
    "version": {
      "type": "integer",
      "description": "Optional version number (latest if not specified)"
    }
  },
  "required": ["name"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "content": {"type": "string"},
    "version": {"type": "integer"},
    "metadata": {"type": "object"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"}
  }
}
```

---

### 10. promptly_list

List available prompts.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Optional glob pattern to filter (e.g., 'code_*')"
    },
    "tag": {
      "type": "string",
      "description": "Optional tag to filter by"
    },
    "limit": {
      "type": "integer",
      "default": 50,
      "description": "Maximum results"
    }
  }
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "current_version": {"type": "integer"},
          "tags": {"type": "array", "items": {"type": "string"}},
          "created_at": {"type": "string", "format": "date-time"}
        }
      }
    },
    "total_count": {"type": "integer"}
  }
}
```

---

### 11. promptly_create

Create a new prompt.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Prompt name (alphanumeric and underscores)"
    },
    "content": {
      "type": "string",
      "description": "Prompt content with {variable} placeholders"
    },
    "metadata": {
      "type": "object",
      "description": "Optional metadata",
      "properties": {
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "author": {"type": "string"}
      }
    }
  },
  "required": ["name", "content"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "version": {"type": "integer"},
    "created": {"type": "boolean"},
    "message": {"type": "string"}
  }
}
```

---

### 12. promptly_record_execution

Record a prompt execution for analytics.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "prompt_name": {
      "type": "string",
      "description": "Name of executed prompt"
    },
    "version": {
      "type": "integer",
      "description": "Version that was executed"
    },
    "task_type": {
      "type": "string",
      "description": "Type of task (e.g., 'summarization', 'code_review')"
    },
    "input_data": {
      "type": "string",
      "description": "Input provided to prompt"
    },
    "output": {
      "type": "string",
      "description": "Generated output"
    },
    "quality_score": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Quality score (0.0-1.0)"
    },
    "latency_ms": {
      "type": "number",
      "description": "Execution latency in milliseconds"
    },
    "llm_provider": {
      "type": "string",
      "description": "LLM provider used (e.g., 'anthropic', 'openai')"
    },
    "llm_model": {
      "type": "string",
      "description": "Model name used"
    },
    "token_count": {
      "type": "integer",
      "description": "Total tokens used"
    }
  },
  "required": ["prompt_name", "quality_score"]
}
```

**Output Schema:**
```json
{
  "type": "object",
  "properties": {
    "execution_id": {"type": "integer"},
    "recorded": {"type": "boolean"},
    "thompson_updated": {"type": "boolean"}
  }
}
```

---

## Error Responses

All functions may return error responses:

```json
{
  "type": "object",
  "properties": {
    "error": {"type": "boolean", "const": true},
    "error_type": {
      "type": "string",
      "enum": ["not_found", "validation_error", "internal_error"]
    },
    "message": {"type": "string"},
    "details": {"type": "object"}
  }
}
```

**Error Types:**
- `not_found`: Resource (prompt, execution) not found
- `validation_error`: Invalid input parameters
- `internal_error`: System error (database, network)

---

## Rate Limits

| Function | Rate Limit | Burst |
|----------|------------|-------|
| analytics_get_stats | 100/min | 20 |
| analytics_compare_prompts | 50/min | 10 |
| analytics_recommend_prompt | 100/min | 20 |
| hololoom_enhance_prompt | 30/min | 5 |
| hololoom_run_agentic | 20/min | 3 |
| promptly_record_execution | 500/min | 100 |

---

## Best Practices

### 1. Use Thompson Sampling for Selection

```python
# Good: Let Thompson Sampling choose based on performance
result = analytics_recommend_prompt(task_type="summarization")
prompt_to_use = result["recommended_prompt"]

# Then record execution for learning
promptly_record_execution(
    prompt_name=prompt_to_use,
    quality_score=measured_quality
)
```

### 2. Record All Executions

```python
# Record every execution to improve recommendations
promptly_record_execution(
    prompt_name="my_prompt",
    task_type="code_review",
    quality_score=0.85,  # Always include quality
    latency_ms=1250
)
```

### 3. Use Verification for Critical Tasks

```python
# For important tasks, use verify mode
result = hololoom_run_agentic(
    prompt_name="critical_analysis",
    mode="verify",
    max_steps=5
)

if not result["verification"]["verified"]:
    # Handle verification failure
    issues = result["verification"]["checks_failed"]
```

### 4. Check Underperforming Prompts Regularly

```python
# Weekly check for prompts needing attention
underperforming = analytics_identify_underperforming(threshold=0.6)

for prompt in underperforming["underperforming"]:
    # Review and improve these prompts
    print(f"{prompt['prompt_name']}: {prompt['suggestion']}")
```

---

## Thompson Sampling Details

Promptly uses Thompson Sampling with Beta distribution priors for prompt selection:

**Prior Update Formula:**
- Success (quality >= 0.7): `alpha += quality_score`
- Failure (quality < 0.7): `beta += (1 - quality_score)`

**Expected Quality:**
```
E[quality] = alpha / (alpha + beta)
```

**Selection Process:**
1. For each candidate prompt, sample from Beta(alpha, beta)
2. Select prompt with highest sampled value
3. Execute and record outcome
4. Update priors based on quality score

This balances exploration (trying less-used prompts) with exploitation (using proven prompts).
