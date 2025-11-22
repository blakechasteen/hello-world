# BossPig: Business Documentation AI Slop Detector

**Created**: 2025-11-22
**Status**: Planning
**Target**: Q1 2026 (After Trough click-through testing)
**Tagline**: *"Because 'synergize our core competencies' isn't a plan."*

## Overview

BossPig is a **business documentation quality assurance system** that detects AI slop, corporate jargon, and quality issues in business writing. Think Trough, but for:
- Business documents (proposals, reports, emails)
- Marketing copy
- Product specs
- Meeting notes
- Strategic plans

**Core Philosophy**: Business writing should be **clear, specific, and actionable**—not filled with buzzwords and meaningless fluff.

## The Problem

### AI Slop in Business Docs (15+ Categories)

1. **Corporate Jargon**
   - "synergize", "leverage", "circle back", "touch base"
   - "low-hanging fruit", "move the needle", "deep dive"
   - "paradigm shift", "game changer", "thought leadership"

2. **Vague Commitments**
   - "We will try to..." (no commitment)
   - "Hopefully we can..." (no ownership)
   - "It would be nice if..." (wishful thinking)
   - "We should probably..." (indecision)

3. **Meaningless Metrics**
   - "Significant improvement" (no number)
   - "Considerable growth" (compared to what?)
   - "Substantial increase" (how much?)
   - "Better performance" (measured how?)

4. **Passive Voice Overload**
   - "Mistakes were made" (by whom?)
   - "A decision was reached" (who decided?)
   - "It has been determined" (determined by whom?)

5. **Missing Dates/Deadlines**
   - "Soon", "Shortly", "In the near future"
   - "As soon as possible"
   - "When we have time"
   - "Pending review"

6. **Unclear Ownership**
   - "Someone should handle this"
   - "The team will take care of it"
   - "We need to follow up" (who specifically?)

7. **AI Hallucination Markers**
   - "As an AI language model..."
   - "I don't have personal opinions but..."
   - Generic placeholder text
   - Copy-pasted templates with [INSERT X HERE]

8. **Redundant Phrasing**
   - "Each and every"
   - "First and foremost"
   - "In order to" (just "to")
   - "Due to the fact that" (just "because")

9. **Weasel Words**
   - "Some people say..."
   - "Studies show..." (which studies?)
   - "Experts believe..." (which experts?)
   - "Many think..." (how many?)

10. **Inconsistent Formatting**
    - Mixed date formats (2025-01-15 vs Jan 15, 2025)
    - Mixed capitalization
    - Inconsistent bullet points
    - Mixed number formats ($1,000 vs $1000)

11. **Empty Headers**
    - Section headers with no content
    - TODO sections left blank
    - Placeholder text still present

12. **Data Quality Issues**
    - Mismatched totals (numbers don't add up)
    - Outdated information (references to old dates)
    - Broken references ([See Section X] but Section X doesn't exist)

13. **Compliance Red Flags**
    - Missing required disclosures
    - Incomplete risk statements
    - Vague legal language
    - No version/date information

14. **Meeting Notes Anti-Patterns**
    - No action items
    - No decisions recorded
    - No attendees listed
    - No next steps

15. **Email Anti-Patterns**
    - CC overload (>10 people)
    - No clear ask/action
    - Subject line doesn't match content
    - Reply-all chains with no substance

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   BossPig System                     │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │  Document Ingestion (SpinningWheel)         │   │
│  │  • PDF, DOCX, Markdown, HTML                │   │
│  │  • Email (EML, MSG)                         │   │
│  │  • Google Docs, Notion exports              │   │
│  │  • Slack/Teams chat exports                 │   │
│  └─────────────────────────────────────────────┘   │
│                       ↓                              │
│  ┌─────────────────────────────────────────────┐   │
│  │  BossPig Detector                           │   │
│  │  • 15 business slop categories              │   │
│  │  • Jargon detection (300+ phrases)          │   │
│  │  • Vagueness scoring                        │   │
│  │  • Clarity metrics                          │   │
│  └─────────────────────────────────────────────┘   │
│                       ↓                              │
│  ┌─────────────────────────────────────────────┐   │
│  │  BossPig Scorer                             │   │
│  │  • Overall quality score (0-100)            │   │
│  │  • Clarity score                            │   │
│  │  • Actionability score                      │   │
│  │  • Professionalism score                    │   │
│  └─────────────────────────────────────────────┘   │
│                       ↓                              │
│  ┌─────────────────────────────────────────────┐   │
│  │  BossPig Fixer                              │   │
│  │  • Replace jargon with plain language       │   │
│  │  • Add missing dates                        │   │
│  │  • Clarify vague statements                 │   │
│  │  • Fix formatting inconsistencies           │   │
│  └─────────────────────────────────────────────┘   │
│                       ↓                              │
│  ┌─────────────────────────────────────────────┐   │
│  │  Interactive Report (Click-through)         │   │
│  │  • Findings by category                     │   │
│  │  • Document preview with highlights         │   │
│  │  • Fix suggestions                          │   │
│  │  • Export cleaned version                   │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Scoring Algorithm

### Quality Score (0-100)

```python
def calculate_quality_score(doc):
    # Component scores (0-1)
    clarity = 1.0 - (jargon_count / total_words)
    specificity = 1.0 - (vague_phrases / total_sentences)
    actionability = action_items_count / total_paragraphs
    professionalism = 1.0 - (ai_hallucinations + formatting_issues) / total_sections
    completeness = filled_sections / total_sections

    # Weighted average
    score = (
        0.30 * clarity +           # 30% weight - most important
        0.25 * specificity +       # 25% weight
        0.20 * actionability +     # 20% weight
        0.15 * professionalism +   # 15% weight
        0.10 * completeness        # 10% weight
    )

    return int(score * 100)
```

### Grade Thresholds

| Score | Grade | Label | Meaning |
|-------|-------|-------|---------|
| **90-100** | A | Excellent | Clear, specific, actionable |
| **80-89** | B | Good | Mostly clear, minor issues |
| **70-79** | C | Fair | Some jargon, needs improvement |
| **60-69** | D | Poor | Vague, full of buzzwords |
| **0-59** | F | Critical | AI slop, unusable |

## Detection Examples

### Example 1: Corporate Jargon

**Before (Score: 45/100 - F)**:
```markdown
# Q4 Strategic Initiative

We need to synergize our core competencies to leverage the low-hanging fruit
and move the needle on our key performance indicators. This paradigm shift
will be a game changer for our thought leadership in the space.

Action Items:
- Circle back on the deep dive
- Touch base with stakeholders
- Reach out to the team
```

**BossPig Findings**:
```
🔴 Critical: Corporate Jargon Overload (8 instances)
  • Line 3: "synergize our core competencies" → Replace with "combine our strengths"
  • Line 3: "leverage the low-hanging fruit" → Replace with "achieve easy wins"
  • Line 4: "move the needle" → Replace with "improve"
  • Line 4: "paradigm shift" → Replace with "major change"
  • Line 5: "game changer" → Replace with "significant improvement"
  • Line 5: "thought leadership" → Replace with "expertise"
  • Line 8: "Circle back" → Replace with "Follow up"
  • Line 9: "Touch base" → Replace with "Meet"

🟡 Warning: Vague Action Items (3 instances)
  • Line 8: "Circle back on the deep dive" → No owner, no deadline
  • Line 9: "Touch base with stakeholders" → Which stakeholders? When?
  • Line 10: "Reach out to the team" → Which team members? What about?

Overall Score: 45/100 (F - Critical)
Clarity: 30/100
Actionability: 20/100
```

**After BossPig Fix (Score: 88/100 - B)**:
```markdown
# Q4 Strategic Initiative

We need to combine our strengths in product development and marketing
to achieve easy wins and improve our customer satisfaction scores by 15%.
This major change will establish our expertise in customer experience.

Action Items:
- @alice: Review customer feedback analysis by Dec 1, 2025
- @bob: Meet with product team on Nov 25, 2025 to discuss roadmap
- @carol: Send email to engineering leads by Nov 23, 2025 with timeline
```

### Example 2: Vague Commitments

**Before (Score: 52/100 - F)**:
```markdown
# Project Update

We will try to finish the migration soon. Hopefully we can get it done
before the end of the quarter. It would be nice if we could avoid downtime.
We should probably test it first.

Progress:
- Some work has been completed
- A few issues were identified
- The team is working on it
```

**BossPig Findings**:
```
🔴 Critical: Vague Commitments (4 instances)
  • Line 3: "try to finish" → Weak commitment, use "will finish by [date]"
  • Line 3: "soon" → Ambiguous deadline, specify date
  • Line 3: "Hopefully we can get it done" → No ownership
  • Line 4: "It would be nice if" → Wishful thinking, not a requirement

🟡 Warning: Missing Dates/Deadlines (2 instances)
  • Line 3: "soon" → Replace with specific date
  • Line 4: "before the end of the quarter" → Specify exact date (Dec 31, 2025)

🟡 Warning: Passive Voice (3 instances)
  • Line 8: "work has been completed" → Who completed it?
  • Line 9: "issues were identified" → Who identified them?
  • Line 10: "The team is working on it" → Which team members specifically?

Overall Score: 52/100 (F - Critical)
Clarity: 40/100
Specificity: 25/100
Ownership: 30/100
```

**After BossPig Fix (Score: 92/100 - A)**:
```markdown
# Project Update

@alice will complete the migration by November 30, 2025, 5:00 PM EST.
We have scheduled a maintenance window on November 29, 2025, 10:00 PM - 2:00 AM
to minimize downtime. @bob completed load testing on November 20, 2025.

Progress:
- Database schema migration: 100% complete (completed by @alice on Nov 18)
- API endpoint updates: 75% complete (@bob, on track for Nov 28)
- Integration testing: 3 issues found by @carol (see JIRA-1234, JIRA-1235, JIRA-1236)
- @david is resolving JIRA-1234 (critical), ETA Nov 24
```

### Example 3: AI Hallucination

**Before (Score: 15/100 - F)**:
```markdown
# Product Specification

As an AI language model, I don't have personal opinions, but here are some
features that could be implemented:

[INSERT PRODUCT NAME HERE]
- Feature 1: [TO BE DETERMINED]
- Feature 2: TBD
- Feature 3: See Section 5 for details

Studies show that users prefer intuitive interfaces. Many experts believe
that performance is important.
```

**BossPig Findings**:
```
🔴 Critical: AI Hallucination Markers (1 instance)
  • Line 3: "As an AI language model, I don't have personal opinions" →
    This is LLM boilerplate, remove entirely

🔴 Critical: Placeholder Text (4 instances)
  • Line 5: "[INSERT PRODUCT NAME HERE]" → Incomplete document
  • Line 6: "[TO BE DETERMINED]" → No actual feature description
  • Line 7: "TBD" → Incomplete
  • Line 8: "See Section 5" → Section 5 doesn't exist (broken reference)

🟡 Warning: Weasel Words (2 instances)
  • Line 10: "Studies show" → Which studies? Citation needed
  • Line 10: "Many experts believe" → Which experts? Names/sources needed

Overall Score: 15/100 (F - Critical)
Completeness: 10/100
Professionalism: 5/100
```

## Implementation Plan

### Phase 1: Core Detector (Week 1-2)

**Files to Create**:
- `bosspig/detector.py` (800 lines)
  - 15 business slop detection algorithms
  - Jargon dictionary (300+ phrases)
  - Vagueness scoring
  - Quality metrics

**Detection Algorithms**:
```python
class BossPigDetector:
    def __init__(self):
        self.jargon_phrases = load_jargon_dictionary()
        self.vague_patterns = compile_vague_patterns()
        self.ai_hallucination_markers = compile_ai_markers()

    def detect_jargon(self, text: str) -> List[JargonFinding]:
        """Detect corporate jargon and buzzwords"""
        findings = []
        for phrase in self.jargon_phrases:
            if phrase in text.lower():
                findings.append(JargonFinding(
                    phrase=phrase,
                    replacement=self.jargon_phrases[phrase],
                    severity="critical"
                ))
        return findings

    def detect_vague_commitments(self, text: str) -> List[VagueFinding]:
        """Detect weak/vague commitments"""
        patterns = [
            r'we will try to',
            r'hopefully we can',
            r'it would be nice if',
            r'we should probably'
        ]
        # ... detection logic

    def detect_missing_dates(self, text: str) -> List[DateFinding]:
        """Detect vague time references"""
        vague_dates = ['soon', 'shortly', 'asap', 'pending', 'tbd']
        # ... detection logic
```

### Phase 2: Scoring System (Week 3)

**Metrics**:
- Clarity score (jargon density)
- Specificity score (vagueness)
- Actionability score (action items per section)
- Professionalism score (AI hallucinations, formatting)
- Completeness score (TODOs, placeholders)

### Phase 3: Auto-Fixer (Week 4)

**Fixes**:
- Replace jargon with plain language
- Add placeholder dates (with warning to fill in)
- Convert passive → active voice
- Remove AI hallucination markers
- Fix formatting inconsistencies

### Phase 4: Interactive Report (Week 5)

**Features**:
- Document preview with inline highlights
- Click to see fix suggestions
- Export cleaned version
- PDF/DOCX download

## Jargon Dictionary (Sample)

```python
JARGON_REPLACEMENTS = {
    # Corporate jargon
    "synergize": "combine",
    "leverage": "use",
    "circle back": "follow up",
    "touch base": "meet",
    "low-hanging fruit": "easy wins",
    "move the needle": "improve",
    "paradigm shift": "major change",
    "game changer": "significant improvement",
    "thought leadership": "expertise",
    "deep dive": "detailed analysis",
    "bandwidth": "time" or "capacity",
    "ping me": "contact me",
    "loop in": "include",
    "run it up the flagpole": "propose to leadership",
    "boil the ocean": "attempt too much",

    # Vague commitments
    "we will try to": "we will [specific action] by [date]",
    "hopefully we can": "we will [specific action]",
    "it would be nice if": "requirement: [specific action]",
    "we should probably": "action item: [owner] will [action] by [date]",

    # Meaningless metrics
    "significant improvement": "[X]% improvement",
    "considerable growth": "[X]% growth compared to [baseline]",
    "substantial increase": "[X] unit increase from [Y] to [Z]",
    "better performance": "[metric] improved from [X] to [Y]",

    # Vague dates
    "soon": "by [specific date]",
    "shortly": "by [specific date]",
    "asap": "by [specific date and time]",
    "when we have time": "scheduled for [specific date]",

    # Weasel words
    "some people say": "[specific source] states",
    "studies show": "[specific study, citation] shows",
    "experts believe": "[specific expert name] believes",
    "many think": "[X]% of [population] think",
}
```

## Usage Examples

### CLI Usage

```bash
# Analyze a document
python -m bosspig analyze proposal.docx

# Output:
# BossPig Analysis: proposal.docx
# Overall Score: 62/100 (D - Poor)
#
# 🔴 Critical Issues (8):
#   - Corporate jargon: 5 instances
#   - Vague commitments: 3 instances
#
# 🟡 Warnings (12):
#   - Missing dates: 7 instances
#   - Passive voice: 5 instances
#
# Recommendations:
#   1. Replace jargon with plain language (5 fixes available)
#   2. Add specific dates for all commitments
#   3. Identify owners for all action items

# Auto-fix and export
python -m bosspig fix proposal.docx --output proposal_fixed.docx

# Generate interactive report
python -m bosspig analyze proposal.docx --interactive
```

### Programmatic Usage

```python
from bosspig import BossPigDetector, BossPigFixer

# Analyze document
detector = BossPigDetector()
findings = detector.analyze("proposal.docx")

print(f"Quality Score: {findings.quality_score}/100")
print(f"Grade: {findings.grade}")

# Apply fixes
fixer = BossPigFixer()
fixed_doc = fixer.fix_document(
    "proposal.docx",
    findings=findings,
    auto_fix_jargon=True,
    auto_fix_formatting=True
)

# Export
fixed_doc.save("proposal_fixed.docx")
```

### API Integration

```python
# Add to HoloLoom departments
from HoloLoom.departments import register_department
from bosspig import BossPigDepartment

register_department("bosspig", BossPigDepartment())

# Use in workflows
dept = get_department("bosspig")
result = await dept.process({
    "action": "analyze",
    "document": "proposal.docx",
    "auto_fix": True
})
```

## Market Positioning

### Target Customers

1. **Enterprise B2B** (Gold tier)
   - Large companies with compliance requirements
   - Need for consistent, professional documentation
   - 100-1000+ documents per month
   - Pricing: $500-2000/month

2. **Consulting Firms** (Silver tier)
   - Proposal writing quality is critical
   - Client-facing documents must be polished
   - 50-100 documents per month
   - Pricing: $200-500/month

3. **Startups** (Bronze tier)
   - Need professional docs but limited budget
   - Pitch decks, investor updates, product specs
   - 10-50 documents per month
   - Pricing: $50-200/month

### Competitive Advantages

| Feature | BossPig | Grammarly Business | Hemingway Editor |
|---------|---------|-------------------|------------------|
| **AI Slop Detection** | ✅ Yes | ❌ No | ❌ No |
| **Jargon Replacement** | ✅ 300+ phrases | 🟡 Some | 🟡 Some |
| **Vagueness Scoring** | ✅ Proprietary | ❌ No | 🟡 Readability only |
| **Action Item Detection** | ✅ Yes | ❌ No | ❌ No |
| **Date/Owner Validation** | ✅ Yes | ❌ No | ❌ No |
| **Quality Score** | ✅ 0-100 | 🟡 Generic | 🟡 Reading level |
| **Auto-Fix** | ✅ Yes | 🟡 Suggestions only | ❌ No |
| **Interactive Report** | ✅ Click-through | ❌ No | ❌ No |

## Revenue Model

### SaaS Tiers

**Bronze** ($50/month):
- 50 documents/month
- 15 detection categories
- Basic auto-fix
- Email support

**Silver** ($200/month):
- 200 documents/month
- 15 detection categories
- Advanced auto-fix
- Custom jargon dictionary
- Priority support

**Gold** ($500/month):
- Unlimited documents
- 15 detection categories
- Full auto-fix + custom rules
- API access
- Dedicated account manager
- White-label reports

**Enterprise** (Custom pricing):
- On-premise deployment
- Custom detection rules
- Compliance modules (HIPAA, SOC2, etc.)
- SSO integration
- Training and onboarding

### API Pricing

- $0.01 per document (< 1000 words)
- $0.03 per document (1000-5000 words)
- $0.10 per document (> 5000 words)
- Free tier: 100 documents/month

## Success Metrics

**Technical**:
- Detection accuracy: >90%
- False positive rate: <5%
- Processing speed: <2 seconds per document
- Auto-fix success rate: >80%

**Business**:
- Customer quality score improvement: +30 points average
- Time saved per document: 15-20 minutes
- Customer retention: >85%
- NPS score: >50

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1** | 2 weeks | Core detector (15 categories) |
| **Phase 2** | 1 week | Scoring system |
| **Phase 3** | 1 week | Auto-fixer |
| **Phase 4** | 1 week | Interactive report |
| **Phase 5** | 2 weeks | API + SaaS deployment |
| **Total** | 7 weeks | Production launch |

**Target Launch**: Q1 2026 (February-March)

## Next Steps

1. **Validate market** (2 weeks)
   - Interview 20 potential customers
   - Survey 100 business doc writers
   - Identify top 10 pain points

2. **Build MVP** (4 weeks)
   - Core detector (top 5 categories)
   - Simple scoring
   - CLI interface

3. **Beta testing** (4 weeks)
   - 10 beta customers
   - Collect feedback
   - Iterate on detection rules

4. **Production launch** (2 weeks)
   - SaaS deployment
   - Marketing campaign
   - Customer onboarding

---

**BossPig**: *Turning business slop into clear, actionable communication.*