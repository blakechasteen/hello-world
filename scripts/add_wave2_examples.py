#!/usr/bin/env python3
"""
Add missing examples to Wave 2 skills for validation gate compliance.
Each skill needs 3+ examples to pass testing gate.
"""

import re
from pathlib import Path

# Additional examples for each skill (adding 2 more to existing 1)
SKILL_EXAMPLES = {
    "pattern_discovery_engine": {
        "example2": {
            "title": "Thread Discovery",
            "input": '{"pattern_types": ["thread"], "min_strength": 0.4}',
            "explanation": "Discovers narrative chains in knowledge graph showing causal sequences like question→hypothesis→experiment→data→conclusion."
        },
        "example3": {
            "title": "Resonance Detection",
            "input": '{"pattern_types": ["resonance"], "min_strength": 0.7}',
            "explanation": "Finds highly activated memories (hot topics) with activation ≥0.7 indicating current focus areas in the system."
        }
    },
    "recursive_refiner": {
        "example2": {
            "title": "Verify Mode Refinement",
            "input": '{"initial_result": {"response": "Python uses dynamic typing", "confidence": 0.65}, "strategy": "verify", "max_iterations": 3}',
            "explanation": "Uses VERIFY strategy (accuracy→completeness→consistency passes) to improve factual answer quality through multi-pass refinement."
        },
        "example3": {
            "title": "Auto-Strategy Selection",
            "input": '{"initial_result": {"response": "Short answer", "confidence": 0.55}, "strategy": null, "max_iterations": 5}',
            "explanation": "Automatically selects best refinement strategy based on query characteristics and iterates until quality threshold reached or max iterations."
        }
    },
    "semantic_search_explainer": {
        "example2": {
            "title": "Low Relevance Analysis",
            "input": '{"query": "machine learning", "retrieved_memories": [{"id": "mem_1", "text": "breakfast recipes", "relevance": 0.15}], "top_k": 1}',
            "explanation": "Explains why low-relevance results were retrieved - useful for debugging semantic search quality and identifying index issues."
        },
        "example3": {
            "title": "Multi-Result Comparison",
            "input": '{"query": "neural networks", "retrieved_memories": [{"id": "mem_1", "text": "deep learning basics", "relevance": 0.92}, {"id": "mem_2", "text": "backpropagation", "relevance": 0.85}], "top_k": 2}',
            "explanation": "Compares multiple results showing why first result scored higher - analyzes semantic overlap, abstraction level, and axis alignment differences."
        }
    },
    "thompson_sampling_advisor": {
        "example2": {
            "title": "Exploration Decision",
            "input": '{"action": "answer", "context": {"alpha": 5, "beta": 15, "epsilon": 0.1, "bandit_strategy": "pure_thompson"}}',
            "explanation": "Explains why exploration action was chosen using Thompson Sampling - Beta(5, 15) distribution suggests low success rate, triggering exploration."
        },
        "example3": {
            "title": "Bayesian Blend Analysis",
            "input": '{"action": "research", "context": {"alpha": 20, "beta": 5, "bandit_strategy": "bayesian_blend", "neural_confidence": 0.65}}',
            "explanation": "Shows how Bayesian blend combines neural prediction (0.65) with bandit prior (0.8) to make final decision - weighted 70/30 split."
        }
    },
    "alignment_safety_checker": {
        "example2": {
            "title": "Medium Risk Action",
            "input": '{"action": "query_database", "context": {"query": "SELECT * FROM users WHERE role=\'admin\'"}}',
            "explanation": "Classifies database query as MEDIUM risk - SQL injection potential but read-only operation, allows with warning logged to audit trail."
        },
        "example3": {
            "title": "Safe Action",
            "input": '{"action": "answer", "context": {"question": "What is 2+2?"}}',
            "explanation": "Simple answer action classified as LOW risk - no external effects, no sensitive data, allows immediately without logging overhead."
        }
    },
    "visual_compression_helper": {
        "example2": {
            "title": "Small Graph Skip",
            "input": '{"graph": {"nodes": 5, "edges": 4}, "compression_threshold": 10}',
            "explanation": "Skips compression for small graph (<10 nodes) - overhead not worth it, returns original text representation instead of PNG."
        },
        "example3": {
            "title": "Maximum Compression",
            "input": '{"graph": {"nodes": 100, "edges": 250}, "compression_threshold": 10, "image_size": [800, 600]}',
            "explanation": "Achieves 20x token savings for large graph - 100 nodes × 50 tokens/node = 5000 tokens → 250 token image description = 20x compression."
        }
    },
    "awareness_metrics_dashboard": {
        "example2": {
            "title": "Low Coherence Warning",
            "input": '{"metrics": {"activation_level": 0.85, "coherence": 0.25, "active_nodes": 50}}',
            "explanation": "High activation but low coherence indicates fragmented knowledge - many memories active but poorly connected, suggests need for integration."
        },
        "example3": {
            "title": "Temporal Analysis",
            "input": '{"metrics": {"activation_level": 0.65, "coherence": 0.80, "active_nodes": 20, "temporal_context": {"recent_shift": 0.45}}}',
            "explanation": "Recent shift of 0.45 indicates major context change - awareness graph adapting to new topic, shows learning in progress."
        }
    }
}


def add_examples_to_skill(skill_name: str, examples: dict) -> bool:
    """Add two more examples to a skill's markdown file."""

    skill_dir = Path(f"skills/domain/{skill_name}")
    skill_file = skill_dir / "skill.markdown"

    if not skill_file.exists():
        print(f"[FAIL] Skill file not found: {skill_file}")
        return False

    # Read existing content
    content = skill_file.read_text(encoding='utf-8')

    # Find the position after Example 1 (before Testing Checklist)
    example_pattern = r'(### Example 1:.*?(?=## Testing Checklist))'
    match = re.search(example_pattern, content, re.DOTALL)

    if not match:
        print(f"[FAIL] Could not find Example 1 in {skill_name}")
        return False

    # Build additional examples
    additional_examples = ""

    # Example 2
    additional_examples += f"\n### Example 2: {examples['example2']['title']}\n\n"
    additional_examples += "**Input**:\n"
    additional_examples += f"```json\n{examples['example2']['input']}\n```\n\n"
    additional_examples += "**Explanation**:\n"
    additional_examples += f"{examples['example2']['explanation']}\n\n"

    # Example 3
    additional_examples += f"### Example 3: {examples['example3']['title']}\n\n"
    additional_examples += "**Input**:\n"
    additional_examples += f"```json\n{examples['example3']['input']}\n```\n\n"
    additional_examples += "**Explanation**:\n"
    additional_examples += f"{examples['example3']['explanation']}\n\n"

    # Insert examples after Example 1
    insertion_point = match.end()
    new_content = content[:insertion_point] + additional_examples + content[insertion_point:]

    # Write back
    skill_file.write_text(new_content, encoding='utf-8')

    return True


def main():
    """Add examples to all Wave 2 skills that failed testing gate."""

    print("[*] Adding examples to Wave 2 skills...\n")

    success_count = 0
    fail_count = 0

    for skill_name, examples in SKILL_EXAMPLES.items():
        if add_examples_to_skill(skill_name, examples):
            print(f"[OK] Added examples to {skill_name}")
            success_count += 1
        else:
            print(f"[FAIL] Failed to add examples to {skill_name}")
            fail_count += 1

    print(f"\n[DONE] Added examples to {success_count}/{len(SKILL_EXAMPLES)} skills")

    if fail_count > 0:
        print(f"[WARN] {fail_count} skills failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
