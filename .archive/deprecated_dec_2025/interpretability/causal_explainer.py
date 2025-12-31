"""
Causal Explainer for HoloLoom
=============================
Provides causal explanations for system decisions.

Philosophy:
- Explain WHY decisions were made
- Trace causal chains (input → features → decision → output)
- Counterfactual reasoning (what if X was different?)
- Natural language explanations (human-readable)

Explanation Types:
1. FEATURE_BASED: "Selected tool X because feature Y had high value Z"
2. CONTEXT_BASED: "Selected tool X because context item Y was relevant"
3. RULE_BASED: "Selected tool X because rule Y was triggered"
4. CONFIDENCE_BASED: "Selected tool X with confidence Y because Z"
5. COUNTERFACTUAL: "If feature X was different, would have selected tool Y"

Example Explanations:
- "I selected 'search_memory' because the query motif 'Thompson' matched a high-confidence memory"
- "I chose 'answer' with 92% confidence due to strong semantic similarity (0.95) to retrieved context"
- "If the embedding similarity was <0.7, I would have selected 'search_web' instead"