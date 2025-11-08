#!/usr/bin/env python3
"""
Dark Trace - Entity Extraction Demo (No vLLM Required)
=======================================================

Demonstrates the semantic memory extraction pipeline without requiring
the full vLLM installation (14GB+ download).

This shows:
1. Medical entity extraction from text
2. Qdrant vector storage
3. Similarity search for related memories
"""

import numpy as np
from typing import List
from dataclasses import asdict
import json

# Import components from main pipeline
import sys
sys.path.append('.')
from dark_trace_pipeline_REFACTORED import (
    SemanticMemoryExtractor,
    SemanticMemoryStore,
    MedicalEntity,
    setup_determinism
)


def demo_entity_extraction():
    """Demo entity extraction on medical scenarios"""

    print("\n" + "="*70)
    print("DARK TRACE: Entity Extraction Demo (No LLM Required)")
    print("="*70)

    # Setup
    setup_determinism(seed=42)
    extractor = SemanticMemoryExtractor()
    store = SemanticMemoryStore(use_local=True)

    # Test scenarios (from ER domain)
    scenarios = [
        {
            "name": "Scenario 1: Penicillin Allergy + Amoxicillin",
            "text": """
            Patient: 42M, allergic to penicillin
            Presenting: bacterial infection (pneumonia suspected)
            Recommended: amoxicillin 500mg TID

            ALERT: Amoxicillin is a penicillin derivative.
            Contraindicated for patients with penicillin allergy.
            Risk: Anaphylaxis
            """
        },
        {
            "name": "Scenario 2: Aspirin + Warfarin Interaction",
            "text": """
            Patient: 68F, on warfarin for atrial fibrillation
            Presenting: headache, minor pain
            Recommended: aspirin for pain relief

            WARNING: Aspirin + warfarin = dangerous interaction
            Risk: Excessive bleeding, hemorrhage
            Alternative: acetaminophen
            """
        },
        {
            "name": "Scenario 3: Hypertension + Diabetes",
            "text": """
            Patient: 55M, history of hypertension and diabetes
            Presenting: elevated blood pressure (160/95)
            Recommended: lisinopril (ACE inhibitor)

            NOTE: Lisinopril is appropriate for hypertensive diabetic patients
            Benefit: Renal protective effects
            """
        },
        {
            "name": "Scenario 4: Sepsis Recognition",
            "text": """
            Patient: 72F, fever, confusion, low blood pressure
            Presenting: suspected sepsis
            Immediate action: Blood cultures, IV antibiotics within 1 hour

            CRITICAL: Sepsis requires immediate intervention
            Delay = increased mortality risk
            """
        }
    ]

    print(f"\nProcessing {len(scenarios)} medical scenarios...\n")

    all_entities = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario['name']}")
        print("-" * 70)

        # Extract entities
        entities = extractor.extract_entities(scenario['text'])

        if entities:
            print(f"  Found {len(entities)} entities:")
            for entity in entities:
                print(f"    - {entity.entity_type:18s}: {entity.content}")
                all_entities.append(entity)

                # Store in Qdrant (with random feature vector for demo)
                feature_vector = np.random.rand(4096).tolist()
                store.store_memory(entity, feature_vector)
        else:
            print("  No entities detected")

        print()

    # Demonstrate similarity search
    print("\n" + "="*70)
    print("SEMANTIC MEMORY SEARCH DEMO")
    print("="*70)

    print("\nQuery: 'Find memories related to allergies and contraindications'")
    print("-" * 70)

    # Generate query vector (in production, this would be from SAE)
    query_vector = np.random.rand(4096).tolist()

    results = store.query_similar(query_vector, top_k=5)

    print(f"\nTop {len(results)} similar memories:")
    for j, result in enumerate(results, 1):
        print(f"\n  {j}. {result.payload['entity_type']}: {result.payload['content']}")
        print(f"     Confidence: {result.payload['confidence']:.2f}")
        print(f"     Timestamp: {result.payload['timestamp']}")
        print(f"     Similarity: {result.score:.4f}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    entity_counts = {}
    for entity in all_entities:
        entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1

    print(f"\nTotal entities extracted: {len(all_entities)}")
    print("\nBreakdown by type:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type:18s}: {count}")

    print(f"\nMemories stored in Qdrant: {len(all_entities)}")
    print("Vector dimension: 4096")
    print("Distance metric: COSINE")

    # Export results
    results_file = "entity_extraction_results.json"
    results_data = {
        "scenarios_processed": len(scenarios),
        "total_entities": len(all_entities),
        "entity_breakdown": entity_counts,
        "entities": [asdict(e) for e in all_entities]
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, indent=2, fp=f)

    print(f"\n[OK] Results exported to: {results_file}")

    # Show what's next
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
Current (MVP):
  [OK] Simple keyword-based entity extraction
  [OK] Qdrant vector storage working
  [OK] Similarity search functional
  [OK] Complete provenance tracking

Production Upgrades:
  1. Replace keyword matching with BioBERT NER
  2. Add Goodfire SAE feature mapping
  3. Implement real activation capture
  4. Integrate with vLLM for deterministic inference
  5. Add contraindication detection logic
  6. Build audit trail visualization

Medical AI Safety:
  - Deterministic: Same input → same output (vLLM)
  - Interpretable: Know which features fired (SAE)
  - Auditable: Complete provenance trail (timestamps, layers)
  - Verifiable: Can replay any decision
    """)


if __name__ == "__main__":
    demo_entity_extraction()
