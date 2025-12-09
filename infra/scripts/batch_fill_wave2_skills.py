#!/usr/bin/env python3
"""Batch fill Wave 2 skills with complete content."""

import re
from pathlib import Path

# Define complete content for each skill
SKILLS_CONTENT = {
    "pattern_discovery_engine": {
        "description_short": "Discover emergent patterns (loops, clusters, threads, resonance) in HoloLoom's knowledge graph for insight generation.",
        "description_long": "Knowledge graphs contain hidden structures that reveal deep insights when discovered. This skill exposes UnifiedMemory.discover_patterns() to find 4 types of emergent patterns: LOOP (cycles/feedback), CLUSTER (tightly connected groups), THREAD (narrative chains), and RESONANCE (highly activated memories). Each pattern type reveals different aspects of knowledge structure: loops show feedback systems, clusters reveal topics, threads expose causality, and resonance highlights current focus.",
        "capabilities": ["read", "python"],
        "dependencies": "HoloLoom.memory.unified (UnifiedMemory.discover_patterns())",
        "input_schema": '''{
  "pattern_types": ["loop", "cluster", "thread", "resonance"],
  "min_strength": "number - 0.0-1.0, minimum pattern strength (default: 0.4)",
  "max_patterns": "number - Max patterns to return (default: 10)"
}''',
        "output_schema": '''{
  "patterns": [
    {
      "type": "loop|cluster|thread|resonance",
      "memories": ["array of memory IDs in pattern"],
      "strength": "number - 0.0-1.0",
      "description": "string - Human-readable pattern description"
    }
  ],
  "insights": ["array of meta-insights about discovered patterns"],
  "metadata": {
    "total_patterns_found": "number",
    "avg_strength": "number",
    "pattern_type_distribution": {"loop": 0, "cluster": 0, "thread": 0, "resonance": 0}
  }
}''',
        "prompt": '''You are a pattern discovery expert analyzing HoloLoom knowledge graphs.

**Pattern Discovery Request**:
- Pattern types: {pattern_types}
- Min strength: {min_strength}
- Max patterns: {max_patterns}

**4 Pattern Types**:
- **LOOP**: Cycles and feedback loops (A -> B -> C -> A). Reveals self-reinforcing systems.
- **CLUSTER**: Tightly connected groups (high modularity). Reveals coherent topics.
- **THREAD**: Narrative chains (A -> B -> C -> D). Reveals causal sequences.
- **RESONANCE**: Highly activated memories (activation >= min_strength). Reveals current focus.

**Your Task**:
1. Use UnifiedMemory.discover_patterns() to find patterns
2. For each pattern, calculate strength:
   - LOOP: cycle_length / max_cycle_length (shorter = stronger)
   - CLUSTER: modularity score (0.0-1.0)
   - THREAD: path_length / 10 (longer = stronger, capped at 1.0)
   - RESONANCE: avg_activation across members
3. Generate human-readable descriptions
4. Provide meta-insights about pattern distribution

**Output Format**: Return structured JSON matching output schema.''',
        "example": '''{"pattern_types": ["loop", "cluster"], "min_strength": 0.5}''',
        "token_count": "~700",
        "latency": "100-300ms"
    },

    "recursive_refiner": {
        "description_short": "Multi-pass recursive refinement using ELEGANCE and VERIFY strategies for quality improvement.",
        "description_long": "Great answers aren't written, they're refined. This skill exposes AdvancedRefiner to iteratively improve results through multi-pass refinement. Two strategies: ELEGANCE (clarity -> simplicity -> beauty) and VERIFY (accuracy -> completeness -> consistency). Each pass improves a specific quality dimension with measurable trajectory. Perfect for low-confidence results that need polishing or complex tasks requiring multiple perspectives.",
        "capabilities": ["read", "python"],
        "dependencies": "HoloLoom.recursive.advanced_refiner (AdvancedRefiner)",
        "input_schema": '''{
  "initial_result": "string - Result to refine",
  "strategy": "elegance|verify|auto (auto-select based on initial_result)",
  "max_iterations": "number - Max refinement passes (default: 3, max: 5)",
  "quality_threshold": "number - Stop when quality >= threshold (default: 0.9)"
}''',
        "output_schema": '''{
  "refined_result": "string - Final improved result",
  "quality_trajectory": ["array of quality scores per iteration"],
  "improvements": ["array of improvements made"],
  "iterations_used": "number",
  "strategy_used": "elegance|verify",
  "metadata": {
    "initial_quality": "number",
    "final_quality": "number",
    "improvement_delta": "number"
  }
}''',
        "prompt": '''You are a recursive refinement expert improving results iteratively.

**Refinement Request**:
- Initial result: {initial_result}
- Strategy: {strategy}
- Max iterations: {max_iterations}
- Quality threshold: {quality_threshold}

**Refinement Strategies**:
- **ELEGANCE**: Clarity (pass 1) -> Simplicity (pass 2) -> Beauty (pass 3)
- **VERIFY**: Accuracy (pass 1) -> Completeness (pass 2) -> Consistency (pass 3)
- **AUTO**: Choose ELEGANCE if initial quality < 0.7, else VERIFY

**Quality Scoring**:
quality = 0.7 * confidence + 0.2 * context_richness + 0.1 * completeness

**Your Task**:
1. Assess initial quality (0.0-1.0)
2. Select strategy (if auto)
3. For each iteration:
   - Apply strategy-specific improvements
   - Calculate new quality score
   - Stop if quality >= threshold
4. Track quality trajectory and improvements

**Output Format**: Return structured JSON matching output schema.''',
        "example": '''{"initial_result": "Thompson Sampling is a bandit algorithm.", "strategy": "elegance", "max_iterations": 3}''',
        "token_count": "~750",
        "latency": "200-500ms"
    },

    "semantic_search_explainer": {
        "description_short": "Explain why specific memories were retrieved using Semantic Calculus 16-axis analysis.",
        "description_long": "Transparency builds trust. This skill explains retrieval results by analyzing queries along 16 interpretable semantic axes (sentiment, formality, technicality, certainty, urgency, abstraction, etc.). Shows which axes drove the match, highlights unexpected connections, and reveals hidden query intent. Perfect for debugging poor retrieval, understanding why certain memories surfaced, or explaining AI decisions to users.",
        "capabilities": ["read", "python"],
        "dependencies": "HoloLoom.semantic_calculus (228D projections, 16 axes)",
        "input_schema": '''{
  "query": "string - Original search query",
  "retrieved_memories": [
    {"id": "string", "content": "string", "relevance": "number"}
  ],
  "top_k": "number - Analyze top K results (default: 5)"
}''',
        "output_schema": '''{
  "explanations": [
    {
      "memory_id": "string",
      "why_retrieved": "string - Human-readable explanation",
      "matching_axes": ["array of axes that matched"],
      "axis_scores": {"sentiment": 0.0, "formality": 0.0, ...},
      "surprisal": "number - How unexpected this match was (0.0-1.0)"
    }
  ],
  "query_intent": {
    "primary_axes": ["array of dominant axes in query"],
    "query_type": "factual|procedural|analytical|exploratory",
    "semantic_profile": "string - Brief characterization"
  },
  "insights": ["array of insights about retrieval patterns"]
}''',
        "prompt": '''You are a semantic search explainability expert using 16-axis analysis.

**Explanation Request**:
- Query: {query}
- Retrieved memories: {retrieved_memories}
- Top K to analyze: {top_k}

**16 Semantic Axes**:
sentiment, formality, technicality, certainty, urgency, abstraction,
specificity, temporality, objectivity, complexity, scope, directness,
emotionality, actionability, novelty, controversy

**Your Task**:
1. Project query onto 16 axes (0.0-1.0 each)
2. For each retrieved memory:
   - Project memory onto 16 axes
   - Calculate axis-wise distances
   - Identify matching axes (distance < 0.2)
   - Calculate surprisal (how unexpected given query profile)
3. Determine query intent (primary axes, query type)
4. Generate human-readable explanations

**Output Format**: Return structured JSON matching output schema.''',
        "example": '''{"query": "Urgently need help debugging!", "retrieved_memories": [...], "top_k": 3}''',
        "token_count": "~600",
        "latency": "50-150ms"
    },

    "thompson_sampling_advisor": {
        "description_short": "Explain Thompson Sampling bandit decisions using Bayesian prior analysis.",
        "description_long": "Thompson Sampling balances exploration/exploitation through Bayesian priors (α, β). This skill explains bandit decisions by showing expected rewards, uncertainty levels, exploration probabilities, and why specific tools were selected. Visualizes α/β distributions, compares tools, and identifies when exploration vs exploitation dominated. Perfect for debugging policy behavior, understanding tool selection, or explaining AI exploration to users.",
        "capabilities": ["read", "python"],
        "dependencies": "HoloLoom.policy.unified (Thompson Sampling bandit)",
        "input_schema": '''{
  "tool_stats": [
    {"tool": "string", "alpha": "number", "beta": "number", "n_pulls": "number"}
  ],
  "selected_tool": "string - Tool that was actually selected",
  "context": "string (optional) - Query/decision context"
}''',
        "output_schema": '''{
  "decision_explanation": "string - Why selected_tool was chosen",
  "tool_analysis": [
    {
      "tool": "string",
      "expected_reward": "number - alpha/(alpha+beta)",
      "uncertainty": "number - Variance of Beta distribution",
      "exploration_probability": "number - P(selected via exploration)",
      "sample": "number - Thompson sample drawn this decision"
    }
  ],
  "exploration_vs_exploitation": {
    "decision_type": "exploration|exploitation",
    "confidence": "number - 0.0-1.0 confidence in classification",
    "rationale": "string"
  },
  "recommendations": ["array of suggestions for tuning"]
}''',
        "prompt": '''You are a Thompson Sampling expert explaining bandit decisions.

**Decision Context**:
- Tool stats: {tool_stats}
- Selected tool: {selected_tool}
- Context: {context}

**Thompson Sampling Mechanics**:
- Sample reward ~ Beta(α, β) for each tool
- Select tool with highest sample
- Update: Success → α++, Failure → β++
- Expected reward: E[X] = α/(α+β)
- Uncertainty: Var[X] = αβ/((α+β)²(α+β+1))

**Your Task**:
1. Calculate expected reward and uncertainty for each tool
2. Determine why selected_tool won (highest sample? highest E[X]? high uncertainty?)
3. Classify decision (exploration if high-uncertainty tool chosen, else exploitation)
4. Provide tuning recommendations (e.g., "Tool X needs more pulls for better estimates")

**Output Format**: Return structured JSON matching output schema.''',
        "example": '''{"tool_stats": [{"tool": "answer", "alpha": 10, "beta": 2}, {"tool": "research", "alpha": 3, "beta": 5}], "selected_tool": "research"}''',
        "token_count": "~550",
        "latency": "30-100ms"
    },

    "alignment_safety_checker": {
        "description_short": "Pre-flight safety checks using HoloLoom alignment framework for risk-aware action gating.",
        "description_long": "Safety first. This skill performs comprehensive pre-flight checks before executing potentially risky actions. Integrates with Safety Guardrails to assess risk levels (LOW/MEDIUM/HIGH/CRITICAL), detects adversarial patterns, checks instrumental convergence (power-seeking), and requires human-in-the-loop for high-risk actions. Returns actionable safety recommendations and automatic abort triggers.",
        "capabilities": ["read", "python"],
        "dependencies": "HoloLoom.alignment.safety_guardrails (SafetyGuardrails)",
        "input_schema": '''{
  "action": "string - Action to execute",
  "context": {"object - Execution context (code, data, permissions, etc.)"},
  "epistemic_confidence": "number (optional) - 0.0-1.0, system certainty"
}''',
        "output_schema": '''{
  "allowed": "boolean - Whether action is safe to proceed",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "safety_score": "number - 0.0-1.0 (1.0 = perfectly safe)",
  "checks_performed": [
    {"check": "string", "passed": "boolean", "details": "string"}
  ],
  "warnings": ["array of safety warnings"],
  "recommendations": ["array of mitigation strategies"],
  "requires_human_approval": "boolean",
  "abort_reason": "string (if allowed=false)"
}''',
        "prompt": '''You are a safety alignment expert performing pre-flight checks.

**Safety Check Request**:
- Action: {action}
- Context: {context}
- Epistemic confidence: {epistemic_confidence}

**Safety Checks**:
1. **Adversarial Pattern Detection**: Prompt injection, code injection, SQL injection
2. **Instrumental Convergence**: Power-seeking, resource acquisition, self-preservation
3. **Deception Detection**: Goal transparency, hidden objectives
4. **Epistemic Humility**: If confidence < 0.3, escalate risk to HIGH

**Risk Levels**:
- LOW: Safe actions (read-only, well-tested, reversible)
- MEDIUM: Moderate risk (file writes, network calls, computation)
- HIGH: Risky actions (code execution, database writes, API calls)
- CRITICAL: Dangerous (system modification, privilege escalation, irreversible)

**Your Task**:
1. Perform all safety checks
2. Calculate risk level (max of base risk and epistemic risk)
3. Determine if action allowed (LOW/MEDIUM: yes, HIGH/CRITICAL: human approval)
4. Provide actionable recommendations

**Output Format**: Return structured JSON matching output schema.''',
        "example": '''{"action": "execute_code", "context": {"code": "import os; os.system('ls')"}, "epistemic_confidence": 0.8}''',
        "token_count": "~600",
        "latency": "20-80ms"
    },

    "visual_compression_helper": {
        "description_short": "Compress knowledge graphs to PNG images for 5-20x token savings in LLM context.",
        "description_long": "Context is expensive. This skill uses visual compression to convert knowledge graphs into PNG images, achieving 5-20x token savings while preserving entity relationships. Automatic compression when sources > threshold (default: 10), supports knowledge graphs and retrieval results, returns PNG bytes + compression metrics. Perfect for RAG systems, multimodal LLMs, or any scenario where graph context exceeds token budgets.",
        "capabilities": ["read", "python"],
        "dependencies": "HoloLoom.memory.visual_compression (compress_graph_to_image)",
        "input_schema": '''{
  "graph": "object - Knowledge graph or list of memory nodes",
  "compression_threshold": "number - Min nodes to trigger compression (default: 10)",
  "image_size": "tuple - (width, height) in pixels (default: (800, 600))",
  "include_labels": "boolean - Show node labels (default: true)"
}''',
        "output_schema": '''{
  "compressed": "boolean - Whether compression was applied",
  "image_bytes": "base64 - PNG image data (if compressed=true)",
  "compression_ratio": "number - Token savings ratio (e.g., 12.5 = 12.5x savings)",
  "original_tokens": "number - Estimated tokens without compression",
  "compressed_tokens": "number - Tokens after compression",
  "metadata": {
    "node_count": "number",
    "edge_count": "number",
    "image_size_bytes": "number"
  }
}''',
        "prompt": '''You are a visual compression expert converting graphs to images.

**Compression Request**:
- Graph: {graph}
- Compression threshold: {compression_threshold}
- Image size: {image_size}
- Include labels: {include_labels}

**Compression Algorithm**:
1. Check node count >= threshold
2. If yes:
   - Convert graph to NetworkX/force-directed layout
   - Render to PNG using matplotlib/PIL
   - Calculate token savings (text_tokens / image_tokens)
3. If no: return compressed=false

**Token Calculation**:
- Text: ~4 chars/token * (node_text_length + edge_text_length)
- Image: ~85 tokens per 512x512 image (GPT-4 Vision pricing)

**Your Task**:
1. Assess if compression worthwhile (node_count >= threshold)
2. If yes, generate force-directed graph layout
3. Render to PNG with optional labels
4. Calculate compression metrics

**Output Format**: Return structured JSON matching output schema.''',
        "example": '''{"graph": [{"id": "A", "edges": [{"to": "B", "type": "USES"}]}, {"id": "B", "edges": []}], "compression_threshold": 2}''',
        "token_count": "~500",
        "latency": "100-300ms"
    },

    "awareness_metrics_dashboard": {
        "description_short": "Visualize awareness graph metrics (activation, coherence, temporal) for memory monitoring.",
        "description_long": "Monitor the mind. This skill visualizes awareness graph metrics to understand memory state: activation levels (which memories are 'awake'), coherence (how well-connected active memories are), temporal dynamics (activation decay over time), and network-wide statistics. Generates HTML dashboards with interactive charts, tracks activation trajectories, detects anomalies, and provides recommendations for memory health.",
        "capabilities": ["read", "python"],
        "dependencies": "HoloLoom.memory.awareness_graph (AwarenessGraph.get_metrics())",
        "input_schema": '''{
  "include_history": "boolean - Include historical metrics (default: false)",
  "history_window": "number - Hours of history (default: 24)",
  "anomaly_detection": "boolean - Detect unusual patterns (default: true)"
}''',
        "output_schema": '''{
  "current_metrics": {
    "active_nodes": "number",
    "mean_activation": "number - 0.0-1.0",
    "global_coherence": "number - 0.0-1.0",
    "activation_std": "number",
    "network_density": "number - 0.0-1.0"
  },
  "historical_trends": [
    {"timestamp": "string", "active_nodes": "number", "coherence": "number"}
  ],
  "anomalies": [
    {"type": "sudden_drop|prolonged_low|high_variance", "severity": "LOW|MEDIUM|HIGH", "description": "string"}
  ],
  "visualizations": {
    "activation_heatmap_html": "string - HTML chart",
    "coherence_trajectory_html": "string - HTML chart",
    "network_graph_html": "string - HTML interactive graph"
  },
  "recommendations": ["array of actionable insights"]
}''',
        "prompt": '''You are an awareness metrics visualization expert.

**Dashboard Request**:
- Include history: {include_history}
- History window: {history_window} hours
- Anomaly detection: {anomaly_detection}

**Awareness Metrics**:
- **Activation**: Memory "wakefulness" (0.0-1.0), tracks spreading activation
- **Coherence**: How well-connected active memories are (graph modularity)
- **Temporal**: Activation decay over time (5% per hour default)
- **Network**: Density, centrality, clustering coefficient

**Anomaly Types**:
- SUDDEN_DROP: Coherence drops >0.2 in <1 hour
- PROLONGED_LOW: Coherence <0.4 for >3 hours
- HIGH_VARIANCE: Activation std dev >0.15

**Your Task**:
1. Retrieve current metrics via AwarenessGraph.get_metrics()
2. If include_history, fetch historical data
3. If anomaly_detection, detect patterns
4. Generate HTML visualizations (activation heatmap, coherence trajectory, network graph)
5. Provide recommendations (e.g., "Coherence low - consider memory consolidation")

**Output Format**: Return structured JSON matching output schema.''',
        "example": '''{"include_history": true, "history_window": 24, "anomaly_detection": true}''',
        "token_count": "~650",
        "latency": "50-200ms"
    }
}

def fill_skill(skill_name: str, skill_dir: Path):
    """Fill a skill with complete content."""
    content = SKILLS_CONTENT[skill_name]
    skill_file = skill_dir / "skill.markdown"

    # Read template
    text = skill_file.read_text(encoding="utf-8")

    # Replace description
    text = re.sub(
        r'\*\*Short Description\*\* \(1-2 sentences\):.*?(?=\n\n)',
        f'**Short Description**:\n{content["description_short"]}',
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r'\*\*Detailed Description\*\*:.*?\n\n## Required Capabilities',
        f'**Detailed Description**:\n{content["description_long"]}\n\n## Required Capabilities',
        text,
        flags=re.DOTALL
    )

    # Update capabilities
    for cap in ["read", "write", "bash", "python", "web_fetch", "web_search", "mcp", "api", "questions"]:
        if cap in content["capabilities"]:
            # Find and check the capability
            cap_map = {
                "read": "File system access (read)",
                "write": "File system access (write)",
                "bash": "Code execution (bash)",
                "python": "Code execution (python)",
                "web_fetch": "Network access (web fetch)",
                "web_search": "Network access (web search)",
                "mcp": "MCP server access",
                "api": "External API access",
                "questions": "User interaction (questions)"
            }
            pattern = rf'- \[ \] {re.escape(cap_map[cap])}'
            replacement = f'- [x] {cap_map[cap]}'
            text = text.replace(pattern, replacement)

    # Replace dependencies
    text = re.sub(
        r'\*\*Required Skills\*\*.*?\*\*HoloLoom Integration\*\*.*?(?=\n\n## Input Schema)',
        f'**Required Skills**: None\n**External Dependencies**: {content["dependencies"]}\n**HoloLoom Integration**: See dependencies',
        text,
        flags=re.DOTALL
    )

    # Replace input schema
    text = re.sub(
        r'## Input Schema.*?## Output Schema',
        f'## Input Schema\n\n```json\n{content["input_schema"]}\n```\n\n## Output Schema',
        text,
        flags=re.DOTALL
    )

    # Replace output schema
    text = re.sub(
        r'## Output Schema.*?## Prompt Template',
        f'## Output Schema\n\n```json\n{content["output_schema"]}\n```\n\n## Prompt Template',
        text,
        flags=re.DOTALL
    )

    # Replace prompt template
    text = re.sub(
        r'## Prompt Template.*?## Examples',
        f'## Prompt Template\n\n```markdown\n{content["prompt"]}\n```\n\n## Examples',
        text,
        flags=re.DOTALL
    )

    # Add single concise example
    text = re.sub(
        r'### Example 1:.*?### Example 3:.*?(?=## Testing Checklist)',
        f'''### Example 1: Basic Usage

**Input**:
```json
{content["example"]}
```

**Explanation**:
Demonstrates core functionality. See skill description for expected output structure.

''',
        text,
        flags=re.DOTALL
    )

    # Update performance characteristics
    text = re.sub(
        r'## Performance Characteristics.*?## Maintenance Notes',
        f'''## Performance Characteristics

- **Expected Latency**: {content["latency"]}
- **Token Usage**: {content["token_count"]}
- **Resource Requirements**: HoloLoom integration, minimal overhead
- **Scalability**: Depends on graph/data size

## Maintenance Notes''',
        text,
        flags=re.DOTALL
    )

    # Simplify maintenance notes
    text = re.sub(
        r'\*\*Known Limitations\*\*:.*?\*\*Changelog\*\*:.*?(?=\n\n## License)',
        f'''**Known Limitations**:
- Requires HoloLoom integration (graceful degradation if unavailable)

**Future Enhancements**:
- Enhanced visualization options
- Additional export formats
- Performance optimizations

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release''',
        text,
        flags=re.DOTALL
    )

    # Set license and remove development notes
    text = re.sub(
        r'## License.*',
        '## License\n\nMIT License\n',
        text,
        flags=re.DOTALL
    )

    # Write updated content
    skill_file.write_text(text, encoding="utf-8")
    print(f"[OK] Filled {skill_name}")

def main():
    """Fill all Wave 2 skills."""
    base_dir = Path("skills/domain")

    for skill_name in SKILLS_CONTENT.keys():
        skill_dir = base_dir / skill_name
        if not skill_dir.exists():
            print(f"[SKIP] {skill_name} directory not found")
            continue
        fill_skill(skill_name, skill_dir)

    print("\n[DONE] All Wave 2 skills filled!")

if __name__ == "__main__":
    main()
