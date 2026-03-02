"""
Voice Correction Demo - Conversational Schema Improvement
=========================================================

Demonstrates the voice correction system:
1. Extract receipt data (some errors expected)
2. Apply voice corrections ("merchant is Whole Foods")
3. System learns patterns automatically
4. Future receipts benefit from learned patterns

This shows the "conversational wool->yarn" paradigm.

Usage:
    python demos/demo_voice_correction.py

Author: Claude Code
Date: January 2025
"""

import asyncio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


async def demo_voice_correction():
    """
    Demonstrate voice-based correction and self-tuning.
    """
    print("=" * 70)
    print("VOICE CORRECTION & SELF-TUNING DEMO")
    print("=" * 70)
    print("\nThis demo shows:")
    print("  1. Extract receipt with imperfect OCR")
    print("  2. Apply voice corrections")
    print("  3. System learns patterns")
    print("  4. Future receipts auto-corrected")

    from hololoom.spinningWheel.voice_correction import (
        VoiceCorrector,
        SelfTuningEngine,
        IntentParser
    )

    # ========================================================================
    # Setup
    # ========================================================================

    print("\n" + "-" * 70)
    print("SETUP: Creating voice corrector...")
    print("-" * 70)

    tuning_engine = SelfTuningEngine(
        storage_path=Path("./learned_patterns.json"),
        min_confidence=0.7
    )

    intent_parser = IntentParser(use_llm=False)  # Rule-based for now

    corrector = VoiceCorrector(
        tuning_engine=tuning_engine,
        intent_parser=intent_parser
    )

    print("[OK] Voice corrector ready")
    print(f"  Patterns loaded: {len(tuning_engine.patterns)}")

    # ========================================================================
    # Scenario 1: First Receipt (Imperfect Extraction)
    # ========================================================================

    print("\n" + "-" * 70)
    print("SCENARIO 1: First receipt (OCR errors)")
    print("-" * 70)

    # Simulate imperfect OCR extraction
    receipt_1_data = {
        'merchant': 'WH FOODS',  # Should be "Whole Foods Market"
        'date': '01/15/25',      # Should be normalized
        'total': 4599,           # Should be 45.99
        'category': None,
        'items': [
            {'name': 'BANANAS', 'price': 399},
            {'name': 'MILK', 'price': 449}
        ]
    }

    print("\nExtracted data (with OCR errors):")
    print(f"  Merchant: {receipt_1_data['merchant']}")
    print(f"  Total: ${receipt_1_data['total']}")
    print(f"  Items: {len(receipt_1_data['items'])}")

    print("\nProblems:")
    print("  - Merchant name abbreviated (WH FOODS)")
    print("  - Total is wrong format (4599 instead of 45.99)")
    print("  - Category missing")

    # ========================================================================
    # Apply Voice Corrections
    # ========================================================================

    print("\n" + "-" * 70)
    print("APPLYING VOICE CORRECTIONS...")
    print("-" * 70)

    # Correction 1: Fix merchant
    print("\n1. User says: \"the merchant is Whole Foods Market\"")

    correction_1 = await corrector.apply_correction(
        transformation_id="tx_001",
        voice_command="the merchant is Whole Foods Market",
        original_data=receipt_1_data,
        schema_name="expenses"
    )

    print(f"   Intent: {correction_1.intent.type.value}")
    print(f"   Field: {correction_1.intent.field_name}")
    print(f"   New value: {correction_1.intent.field_value}")

    if correction_1.pattern_learned:
        pattern = correction_1.pattern_learned
        print(f"\n   Pattern learned:")
        print(f"     '{pattern.source_pattern}' -> '{pattern.target_action}'")
        print(f"     Confidence: {pattern.confidence:.2f}")

    # Correction 2: Fix total
    print("\n2. User says: \"total should be 45.99\"")

    receipt_1_data['merchant'] = 'Whole Foods Market'  # Apply first correction

    correction_2 = await corrector.apply_correction(
        transformation_id="tx_001",
        voice_command="total should be 45.99",
        original_data=receipt_1_data,
        schema_name="expenses"
    )

    print(f"   Intent: {correction_2.intent.type.value}")
    print(f"   Field: {correction_2.intent.field_name}")
    print(f"   New value: {correction_2.intent.field_value}")

    # Correction 3: Add category
    print("\n3. User says: \"category grocery\"")

    receipt_1_data['total'] = 45.99  # Apply second correction

    correction_3 = await corrector.apply_correction(
        transformation_id="tx_001",
        voice_command="category grocery",
        original_data=receipt_1_data,
        schema_name="expenses"
    )

    print(f"   Intent: {correction_3.intent.type.value}")
    print(f"   Category: {correction_3.intent.category_name}")

    # ========================================================================
    # Show Learned Patterns
    # ========================================================================

    print("\n" + "-" * 70)
    print("LEARNED PATTERNS")
    print("-" * 70)

    patterns = list(tuning_engine.patterns.values())
    print(f"\nTotal patterns learned: {len(patterns)}")

    for i, pattern in enumerate(patterns[:3], 1):
        print(f"\n{i}. {pattern.pattern_type.value}")
        print(f"   Source: '{pattern.source_pattern}'")
        print(f"   Target: '{pattern.target_action}'")
        print(f"   Confidence: {pattern.confidence:.2f}")
        print(f"   Context: {pattern.context}")

    # ========================================================================
    # Scenario 2: Second Receipt (Auto-Correction!)
    # ========================================================================

    print("\n" + "-" * 70)
    print("SCENARIO 2: Second receipt (auto-correction)")
    print("-" * 70)

    # Simulate another receipt with same OCR errors
    receipt_2_data = {
        'merchant': 'WH FOODS',  # Same error!
        'date': '01/16/25',
        'total': 3299,
        'category': None,
        'items': [
            {'name': 'BREAD', 'price': 599},
            {'name': 'EGGS', 'price': 699}
        ]
    }

    print("\nExtracted data (before patterns):")
    print(f"  Merchant: {receipt_2_data['merchant']}")
    print(f"  Total: ${receipt_2_data['total']}")

    # Apply learned patterns
    print("\nApplying learned patterns...")

    improved_data = await tuning_engine.apply_learned_patterns(
        receipt_2_data,
        schema_name="expenses"
    )

    print(f"\nImproved data (after patterns):")
    print(f"  Merchant: {improved_data['merchant']}")
    print(f"  Total: ${improved_data['total']}")

    if tuning_engine.last_patterns_applied:
        print(f"\nPatterns applied: {len(tuning_engine.last_patterns_applied)}")
        for pattern in tuning_engine.last_patterns_applied:
            print(f"  - {pattern.source_pattern} -> {pattern.target_action}")

    # ========================================================================
    # Scenario 3: Pattern Confidence Over Time
    # ========================================================================

    print("\n" + "-" * 70)
    print("SCENARIO 3: Pattern confidence increases with usage")
    print("-" * 70)

    # Simulate 5 more receipts with same pattern
    for i in range(5):
        test_data = {'merchant': 'WH FOODS'}
        improved = await tuning_engine.apply_learned_patterns(
            test_data,
            schema_name="expenses"
        )

        if improved['merchant'] == 'Whole Foods Market':
            # Pattern worked - record success
            for pattern in tuning_engine.last_patterns_applied:
                # Success already recorded in apply_learned_patterns
                pass

    # Show updated pattern
    merchant_patterns = tuning_engine.get_patterns_for_field('merchant')

    if merchant_patterns:
        pattern = merchant_patterns[0]
        print(f"\nMerchant normalization pattern:")
        print(f"  Source: '{pattern.source_pattern}'")
        print(f"  Target: '{pattern.target_action}'")
        print(f"  Usage count: {pattern.usage_count}")
        print(f"  Success rate: {pattern.success_rate:.1%}")
        print(f"  Confidence: {pattern.confidence:.2f}")

    # ========================================================================
    # Show Correction Suggestions
    # ========================================================================

    print("\n" + "-" * 70)
    print("CORRECTION SUGGESTIONS")
    print("-" * 70)

    # New receipt with known pattern
    receipt_3_data = {
        'merchant': 'WH FOODS',
        'date': '01/17/25',
        'total': 2999
    }

    suggestions = corrector.get_suggestions(
        receipt_3_data,
        schema_name="expenses"
    )

    print(f"\nSuggestions for new receipt:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("\nWhat happened:")
    print("  1. First receipt had OCR errors")
    print("  2. User corrected via voice commands")
    print("  3. System learned patterns from corrections")
    print("  4. Second receipt auto-corrected using patterns")
    print("  5. Pattern confidence increased with usage")
    print("  6. System suggests corrections proactively")

    print("\nKey Benefits:")
    print("  [OK] Zero configuration - just voice corrections")
    print("  [OK] Self-tuning - learns from every correction")
    print("  [OK] Pattern propagation - applies to future data")
    print("  [OK] Confidence tracking - patterns improve over time")
    print("  [OK] Proactive suggestions - helps user catch errors")

    print("\nProduction Usage:")
    print("  # Setup once")
    print("  corrector = VoiceCorrector()")
    print("")
    print("  # Correct via voice")
    print("  await corrector.apply_correction(tx_id, 'merchant is Whole Foods')")
    print("")
    print("  # Future receipts auto-corrected!")
    print("  improved = await tuning_engine.apply_learned_patterns(data, schema)")

    print("\n" + "=" * 70)


async def main():
    """Run demo."""
    await demo_voice_correction()


if __name__ == "__main__":
    asyncio.run(main())
