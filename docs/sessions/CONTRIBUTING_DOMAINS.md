# Contributing a Domain to ExpertLoom

Welcome! This guide will help you package your expertise into a reusable domain module that others can benefit from.

## What is a Domain?

A **domain** is a packaged collection of expertise from a specific field (beekeeping, automotive repair, cooking, woodworking, etc.). It includes:

- **Entities**: The "nouns" of your domain (hives, cars, recipes, tools)
- **Measurements**: How you quantify things (8 frames, 50 PSI, 2 cups)
- **Relationships**: How entities connect (hive USES treatment, car REQUIRES oil)
- **Micropolicies**: Rules of thumb experts use ("if X then Y")
- **Reverse Queries**: Questions experts ask to diagnose problems

## Why Contribute?

### For the Community
- Preserve valuable expertise that would otherwise be lost
- Help others learn from your experience
- Build a library of human knowledge

### For You (Expert Mode)
- **Earn ongoing royalties** from your expertise (70/30 split)
- Build passive income while helping others
- Get verified credentials for your domain expertise
- Track how your knowledge is being used
- Retain ownership of your intellectual property

## Getting Started

### 1. Understand Your Domain

Before coding, map out your expertise:

**Answer these questions:**
- What are the core "things" in your field? (entities)
- How do you refer to them? (aliases - formal vs casual names)
- What do you measure? (frames of bees, tire pressure, baking time)
- What patterns do you recognize? (categorical states like "calm" vs "aggressive")
- What rules do you follow? (policies - "if temp drops below 40°F, check for clustering")
- What questions do you ask when troubleshooting? (reverse queries)

**Example - Beekeeping:**
- Entities: Hives, treatments, beekeepers, forage sources
- Measurements: Frames of brood, weight, dosage, temperature
- Patterns: Temperament (calm/aggressive), brood pattern (solid/spotty)
- Rules: "If frames < 4 in October → feed syrup"
- Questions: "Did you notice any mites during last inspection?"

### 2. Create Your Domain Structure

Copy the template:

```bash
cp -r mythRL_core/domains/DOMAIN_TEMPLATE mythRL_core/domains/your_domain_name
cd mythRL_core/domains/your_domain_name
```

You'll be editing `registry.json` - the single file that defines your domain.

### 3. Define Your Entities

Start with 5-10 core entities. Quality > quantity.

**Example - Automotive Domain:**

```json
{
  "canonical_id": "component-engine-oil",
  "canonical_name": "Engine Oil",
  "aliases": [
    "oil",
    "motor oil",
    "engine oil",
    "5w-30",
    "synthetic oil"
  ],
  "entity_type": "fluid",
  "attributes": {
    "typical_capacity_quarts": 5.0,
    "change_interval_miles": 5000,
    "viscosity": "5W-30"
  },
  "metadata": {
    "importance": "high",
    "frequency": "common",
    "notes": "Most common maintenance item"
  }
}
```

**Good Practices:**
- Use descriptive canonical IDs: `component-engine-oil` not `c001`
- Include common misspellings in aliases
- Add both formal and casual terms ("engine oil" + "oil")
- Include brand names if commonly used to refer to the thing
- Set realistic attribute values based on your experience

### 4. Define Measurement Patterns

Use regex to extract numeric and categorical data from text.

**Numeric Example - Tire Pressure:**

```json
"tire_pressure_psi": {
  "pattern": "(\\d+(?:\\.\\d+)?)\\s*(?:PSI|psi|pounds?)",
  "unit": "PSI",
  "description": "Tire pressure in pounds per square inch"
}
```

This extracts:
- "Set to 32 PSI" -> 32.0
- "35 psi in front tires" -> 35.0
- "Inflated to 28 pounds" -> 28.0

**Categorical Example - Oil Condition:**

```json
"oil_condition": {
  "pattern": "\\b(clean|dirty|black|contaminated|milky)\\b",
  "values": ["clean", "dirty", "black", "contaminated", "milky"],
  "description": "Visual assessment of oil condition"
}
```

This extracts:
- "Oil looks dirty" -> "dirty"
- "Clean oil after change" -> "clean"
- "Found milky oil" -> "milky"

**Tips:**
- Test your patterns on real notes from your domain
- Use `\\b` word boundaries to avoid partial matches
- Make patterns case-insensitive (we handle that automatically)
- Include common variations (PSI, psi, pounds)

### 5. Define Relationships

How do entities connect?

```json
"relationships": [
  {
    "from_type": "vehicle",
    "to_type": "fluid",
    "relationship": "REQUIRES",
    "description": "Vehicles require fluids for operation"
  },
  {
    "from_type": "fluid",
    "to_type": "maintenance",
    "relationship": "TRIGGERS",
    "description": "Fluid condition triggers maintenance actions"
  }
]
```

Common relationship types:
- REQUIRES (dependency)
- USES (consumption)
- CONTAINS (composition)
- AFFECTS (influence)
- TRIGGERS (causation)
- LOCATED_AT (spatial)

### 6. Encode Micropolicies

These are your expert rules - the "if X then Y" knowledge.

**Example - Low Tire Pressure Policy:**

```json
{
  "policy_id": "policy-tire-low-pressure",
  "name": "Low Tire Pressure Alert",
  "trigger": {
    "condition": "tire_pressure_psi < threshold",
    "thresholds": {
      "tire_pressure_psi": 28.0
    }
  },
  "action": {
    "type": "alert",
    "message": "Tire pressure below 28 PSI - check for leaks and inflate to manufacturer spec"
  },
  "priority": "high"
}
```

**Example - Oil Change Interval Policy:**

```json
{
  "policy_id": "policy-oil-change-interval",
  "name": "Oil Change Due",
  "trigger": {
    "condition": "miles_since_change > 5000 OR oil_condition = black",
    "thresholds": {
      "miles_since_change": 5000
    }
  },
  "action": {
    "type": "suggest",
    "message": "Oil change recommended - either due to mileage or condition"
  },
  "priority": "medium"
}
```

### 7. Create Reverse Queries

What questions do YOU ask when something seems off?

**Example - Automotive:**

```json
{
  "query_id": "rq-engine-noise",
  "trigger": "mentions of unusual sounds (knocking, ticking, squealing)",
  "question_template": "You mentioned a {sound_type} sound - when does it occur? (startup, acceleration, idle, braking)",
  "expected_answer_type": "categorical",
  "purpose": "Narrow down source of engine noise based on occurrence timing"
}
```

**Example - Beekeeping:**

```json
{
  "query_id": "rq-sudden-strength-drop",
  "trigger": "frames_of_brood decreased by > 3 in < 2 weeks",
  "question_template": "{hive_name} dropped from {old_frames} to {new_frames} frames - did you notice any signs of disease, pests, or queen issues?",
  "expected_answer_type": "text",
  "purpose": "Identify cause of rapid colony decline"
}
```

### 8. Add Validation Tests

Include real example phrases from your domain to validate the system works.

```json
"validation": {
  "required_entity_types": ["vehicle", "fluid", "component"],
  "minimum_entities": 8,
  "required_measurements": ["tire_pressure_psi", "oil_condition"],
  "test_phrases": [
    "Checked the Corolla today - oil looks dirty and tire pressure at 28 PSI",
    "Changed engine oil and filter, reset maintenance light",
    "Front passenger tire low at 25 psi, found small nail"
  ]
}
```

## Testing Your Domain

### 1. Run the Entity Resolver Demo

```bash
cd mythRL_core/domains/your_domain_name
python -c "
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent.parent.parent))

from mythRL_core.entity_resolution import EntityRegistry, EntityResolver

registry = EntityRegistry.load(Path('registry.json'))
resolver = EntityResolver(registry)

test_note = 'YOUR TEST NOTE HERE'
result = resolver.tag_text(test_note)
print(f'Found {result[\"entity_count\"]} entities:')
for e in result['entities']:
    print(f'  {e[\"matched_text\"]} -> {e[\"canonical_id\"]}')
"
```

### 2. Run the Auto-Tagger Demo

```bash
PYTHONPATH=. python demo_auto_tagger.py
```

Replace the test notes with your domain's examples.

### 3. Run the Full Pipeline

```bash
PYTHONPATH=. python demo_full_pipeline.py
```

This tests: entity resolution + measurement extraction + embeddings + Qdrant storage + querying.

## Submitting Your Domain

### Community (Free) Tier

1. Fork the repository
2. Add your domain to `mythRL_core/domains/`
3. Submit a pull request with:
   - Your registry.json
   - README.md describing your domain
   - At least 3 test notes demonstrating extraction
4. Maintainers will review and merge

Your domain will be:
- Free to use
- Credited to you in docs
- Part of the core ExpertLoom library

### Expert (Paid) Tier

Want to monetize your expertise?

1. Complete all Community tier requirements
2. Apply for Expert Verification:
   - Submit credentials (certifications, years of experience)
   - Provide portfolio of your work in the domain
   - Pass expert review panel assessment
3. Set your licensing terms:
   - Free tier (limited queries/month)
   - Pro tier (unlimited queries, priority support)
   - Enterprise tier (custom integration)
4. Configure revenue split (default 70% you, 30% platform)

**Your Expert Domain includes:**
- **Verified Expert Badge** - users know it's from a real practitioner
- **Priority Support** - you provide guidance to users of your domain
- **Usage Analytics** - see how your knowledge is being used
- **Revenue Dashboard** - track earnings in real-time
- **Update Control** - you can improve your domain over time
- **IP Protection** - your licensing terms are enforced

**Example Pricing:**
- Free: 50 queries/month
- Pro ($9.99/month): Unlimited queries, $7.00 to you per subscriber
- Enterprise (custom): Volume licensing, negotiate rates

**Example Expert Earnings:**
- 200 Pro subscribers × $7.00 = $1,400/month
- 5 Enterprise licenses × $200 = $1,000/month
- **Total: $2,400/month passive income**

And you're helping 200+ people learn from your expertise!

## Domain Quality Guidelines

### Great Domains Have:

**Comprehensive Entity Coverage**
- 10+ entities covering core concepts
- 3+ aliases per entity
- Rich attributes (not just empty objects)

**Accurate Measurement Patterns**
- Tested regex patterns that work on real notes
- Both numeric AND categorical extraction
- Clear descriptions of what each measurement means

**Practical Micropolicies**
- Based on real rules you actually follow
- Clear trigger conditions
- Actionable recommendations

**Thoughtful Reverse Queries**
- Questions you genuinely ask when troubleshooting
- Help diagnose problems faster
- Specific enough to be useful

**Validated Test Cases**
- 5+ realistic test phrases
- Cover different scenarios (normal, problem, edge case)
- Demonstrate entity resolution + measurement extraction working

### Avoid:

- Generic entities without domain specificity
- Too few aliases (limits matching)
- Untested regex patterns (will fail on real data)
- Vague policies ("check on things sometimes")
- Reverse queries that are too broad ("what's wrong?")
- Test phrases that don't match your actual patterns

## Examples to Learn From

### 1. Beekeeping (mythRL_core/domains/beekeeping/)

A complete, production-ready domain with:
- 10 entities (hives, treatments, beekeepers, forage)
- 34 total aliases
- Numeric measurements (frames, weight, dosage, temperature)
- Categorical measurements (temperament, brood pattern, strength)
- Temporal correlation (seasonal patterns)

**Best practices demonstrated:**
- Casual aliases ("jodi" for "Jodi's Hive")
- Realistic attributes (typical_strength: 8.0 frames)
- Practical regex patterns for beekeeping measurements

### 2. Automotive (coming soon)

Will demonstrate:
- Vehicle components and fluids
- Maintenance intervals
- Diagnostic patterns
- Multi-level relationships (vehicle > system > component)

## Community Guidelines

### When Contributing:

**DO:**
- Use clear, descriptive entity names
- Include comprehensive aliases (formal + casual)
- Test your patterns on real-world notes
- Write helpful descriptions and examples
- Respect intellectual property of others
- Credit sources if you're adapting existing knowledge

**DON'T:**
- Copy proprietary information without permission
- Include personally identifiable information
- Use offensive or discriminatory language
- Submit untested/broken patterns
- Claim expertise you don't have

### Expert Verification Process:

For paid Expert tier, we verify:
1. **Credentials**: Certifications, licenses, education
2. **Experience**: Years in field, portfolio of work
3. **Quality**: Domain completeness, accuracy, testing
4. **Communication**: Ability to explain concepts clearly
5. **Ethics**: Commitment to helping others learn

Verified experts get:
- Badge on their profile and domains
- Revenue sharing opportunities
- Priority support allocation
- Community recognition

## Getting Help

### Resources:

- **Documentation**: See `docs/` folder for detailed guides
- **Examples**: Check `mythRL_core/domains/` for working domains
- **Tests**: Run `demo_*.py` scripts to validate your domain
- **Discord**: Join our community for questions and feedback

### Common Issues:

**"My entities aren't being detected"**
- Check your aliases match how you actually write notes
- Try adding more variations (singular/plural, abbreviations)
- Test with actual notes from your domain, not made-up examples

**"Measurements aren't extracting"**
- Validate your regex patterns at regex101.com
- Check for typos in pattern strings
- Make sure units are included in pattern (PSI, grams, etc.)

**"I don't know what micropolicies to add"**
- Think about decisions you make regularly
- What rules do you follow without thinking?
- Ask yourself: "If X happens, what do I always do?"

**"Reverse queries seem hard"**
- Start simple: what's the first question you ask when troubleshooting?
- Think about your diagnostic process step by step
- Focus on questions that narrow down possibilities

## Recognition and Credits

### All Contributors:

- Named in CONTRIBUTORS.md
- Profile on community page
- Contribution statistics

### Expert Contributors:

- Verified Expert badge
- Featured on Expert Marketplace
- Revenue dashboard access
- Direct user feedback
- Speaking opportunities at ExpertLoom events

## Expertise as Equity

You've spent years building your knowledge. ExpertLoom helps you:

1. **Preserve** your expertise in structured, reusable form
2. **Share** it with others who need it
3. **Profit** from the value you've created
4. **Maintain ownership** of your intellectual property
5. **Keep learning** from how others use your domain

**Your knowledge is valuable. Let's make sure you benefit from it.**

---

## Quick Start Checklist

- [ ] Copy DOMAIN_TEMPLATE to your domain name
- [ ] Define 5-10 core entities with aliases
- [ ] Add measurement patterns (numeric + categorical)
- [ ] Create at least 3 micropolicies
- [ ] Write 2-3 reverse queries
- [ ] Add 5+ validation test phrases
- [ ] Run entity resolver demo
- [ ] Run auto-tagger demo
- [ ] Run full pipeline demo
- [ ] Write domain README.md
- [ ] Submit pull request (Community) or apply for Expert verification

**Ready to share your expertise? Let's build something amazing together.**
