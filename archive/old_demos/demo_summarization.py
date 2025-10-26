"""
Demo: Text Summarization with Entity/Measurement Preservation
==============================================================

Shows how summarization fits into the ExpertLoom pipeline:
1. Extract entities and measurements (existing)
2. Generate summary preserving key info (new!)
3. Store both full text + summary
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mythRL_core.entity_resolution import EntityRegistry, EntityResolver
from mythRL_core.entity_resolution.extractor import EntityExtractor
from mythRL_core.summarization import TextSummarizer, SummarizerConfig, SummarizationStrategy, summarize_text

print("="*80)
print("Text Summarization Demo")
print("="*80 + "\n")

# Load automotive domain
print("Loading automotive domain...")
registry_path = Path("mythRL_core/domains/automotive/registry.json")
registry = EntityRegistry.load(registry_path)
resolver = EntityResolver(registry)
extractor = EntityExtractor(resolver, custom_patterns=registry.measurement_patterns)
print("  Domain loaded: automotive_repair\n")

# Create summarizer
summarizer = TextSummarizer(SummarizerConfig(
    strategy=SummarizationStrategy.EXTRACTIVE,
    max_sentences=2,  # Keep only 2 most important sentences
    preserve_entities=True,
    preserve_measurements=True
))
print("Summarizer ready with extractive strategy\n")

# Test notes (realistic automotive maintenance)
test_notes = [
    {
        "title": "Oil Change Day",
        "text": "Today I decided to change the oil in my Corolla since it was looking pretty dirty during my last inspection. I bought 4.4 quarts of 5W-30 synthetic oil from the auto parts store. The old oil came out black and thick, which confirms it was overdue. After draining, I replaced the filter and filled with fresh oil. The engine started smoothly and the oil pressure light went off immediately. I reset the maintenance reminder on the dashboard. Total time was about 45 minutes. The car should be good for another 5,000 miles now."
    },
    {
        "title": "Low Tire Pressure Warning",
        "text": "Got a tire pressure warning light this morning on the way to work. Stopped at a gas station and checked all four tires. Front left was at 25 PSI, which is definitely too low. Front right was at 31 PSI, rear left at 32 PSI, and rear right at 33 PSI. I inflated the front left tire to 32 PSI to match the others. Couldn't find any obvious punctures or damage, but I'll keep monitoring it over the next few days to see if it's a slow leak. The warning light turned off after driving for a few minutes."
    },
    {
        "title": "Brake Inspection",
        "text": "During my routine maintenance today, I checked the brake pads on the Corolla. The front pads are getting thin - measured at about 3mm remaining. That's right at the replacement threshold. The rear pads still look good at around 7mm. I also noticed a slight squealing sound when braking, which is probably the wear indicator. I should schedule a brake pad replacement within the next week or two before they get too worn down. The rotors look fine, no deep grooves or warping that I can see."
    }
]

print("="*80)
print("Processing Notes with Full Pipeline")
print("="*80 + "\n")

for i, note_data in enumerate(test_notes, 1):
    print(f"Note {i}: {note_data['title']}")
    print("-" * 80)

    text = note_data["text"]
    print(f"\nOriginal ({len(text)} chars, {len(text.split())} words):")
    print(f"  \"{text}\"\n")

    # Step 1: Extract entities and measurements
    extracted = extractor.extract(text)

    print(f"Extracted Data:")
    print(f"  Entities: {len(extracted.entities)}")
    for entity in extracted.entities:
        print(f"    - {entity['matched_text']} -> {entity['canonical_id']}")

    if extracted.measurements:
        print(f"  Measurements:")
        for key, value in extracted.measurements.items():
            print(f"    - {key}: {value}")
    print()

    # Step 2: Generate summary
    summary_result = summarizer.summarize(
        text,
        entities=[e['canonical_id'] for e in extracted.entities],
        measurements=extracted.measurements
    )

    summary_text = summary_result["summary"]
    compression = summary_result["compression"]

    print(f"Summary ({len(summary_text)} chars, {len(summary_text.split())} words):")
    print(f"  \"{summary_text}\"")
    print()

    print(f"Compression: {compression:.0%} of original")
    print(f"Preserved Entities: {len(summary_result['preserved_entities'])}/{len(extracted.entities)}")
    if summary_result['preserved_entities']:
        for entity_id in summary_result['preserved_entities']:
            print(f"  - {entity_id}")

    print(f"Preserved Measurements: {len(summary_result['preserved_measurements'])}/{len(extracted.measurements or {})}")
    if summary_result['preserved_measurements']:
        for key, value in summary_result['preserved_measurements'].items():
            print(f"  - {key}: {value}")

    print("\n" + "="*80 + "\n")

# Demo: Different summarization strategies
print("="*80)
print("Comparing Summarization Strategies")
print("="*80 + "\n")

long_note = """
This morning I performed a comprehensive inspection of the Corolla at 87,450 miles.
Started by checking all fluid levels. Engine oil was dark and dirty, indicating it's
time for a change. Coolant level was slightly low, so I topped it off with the pink
long-life coolant. Brake fluid was at the minimum line but still acceptable.

Next, I checked the tires. Front left tire pressure was concerningly low at 28 PSI.
Front right was at 31 PSI. Both rear tires were at 32 PSI. I inflated the front left
to match the spec of 32 PSI. Tread depth on all tires measured around 6mm, which is
still safe but getting worn.

Examined the brakes through the wheels. Front pads look thin, probably around 3mm
remaining. That's at the replacement threshold. I can hear a slight squealing during
braking which confirms the wear indicators are starting to contact the rotors.

Under the hood, I noticed the engine air filter is pretty dirty. Pulled it out and
held it up to the light - barely any light coming through. Should replace that soon
for better fuel economy and performance.

Battery terminals had some minor corrosion, so I cleaned them with a wire brush and
baking soda solution. Battery voltage read 12.4V, which is on the lower end but still
acceptable. Might need replacement in the next year.

Overall the car is in decent shape but needs some attention soon - oil change, brake
pads, and air filter are the priorities. I'll schedule those for next weekend.
"""

print("Long note example (1,442 chars, 240 words)\n")

strategies = [
    ("First 2 Sentences", SummarizationStrategy.FIRST_N_SENTENCES, 2),
    ("Extractive (2 sentences)", SummarizationStrategy.EXTRACTIVE, 2),
    ("Extractive (3 sentences)", SummarizationStrategy.EXTRACTIVE, 3),
    ("30% Compression", SummarizationStrategy.EXTRACTIVE, None),
]

# Extract from long note first
extracted_long = extractor.extract(long_note)
entities_long = [e['canonical_id'] for e in extracted_long.entities]

for name, strategy, max_sent in strategies:
    print(f"{name}:")
    print("-" * 80)

    if max_sent:
        config = SummarizerConfig(
            strategy=strategy,
            max_sentences=max_sent,
            preserve_entities=True,
            preserve_measurements=True
        )
    else:
        config = SummarizerConfig(
            strategy=strategy,
            compression_ratio=0.3,  # 30% of original
            preserve_entities=True,
            preserve_measurements=True
        )

    summarizer_test = TextSummarizer(config)
    result = summarizer_test.summarize(
        long_note,
        entities=entities_long,
        measurements=extracted_long.measurements
    )

    print(f"  Summary: \"{result['summary']}\"")
    print(f"  Compression: {result['compression']:.0%}")
    print(f"  Entities preserved: {len(result['preserved_entities'])}/{len(entities_long)}")
    print(f"  Measurements preserved: {len(result['preserved_measurements'])}/{len(extracted_long.measurements or {})}")
    print()

print("="*80)
print("\nKey Benefits of Summarization:")
print("  1. Faster search - scan summaries first, full text for details")
print("  2. Timeline views - show summaries in list view")
print("  3. Mobile UX - summaries fit on small screens")
print("  4. Trend detection - easier to spot patterns across many notes")
print("  5. Cost savings - search smaller summaries (if using paid APIs)")
print()

print("Storage Strategy:")
print("  - Full text: Always stored (never lost)")
print("  - Summary: Stored in metadata['summary']")
print("  - Both searchable via Qdrant")
print("  - Summary gets higher weight in search ranking (TBD)")
print()

print("Ready to integrate with Qdrant storage!")
